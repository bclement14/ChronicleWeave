# tests/test_whisperx_docker.py
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from chronicleweave.pipeline import (
    PipelineConfig,
    build_whisperx_command,
    redact_command,
    run_whisperx_docker,
)


def _session(tmp_path):
    chunks = tmp_path / "chunked_tracks"
    chunks.mkdir()
    files = [chunks / "GM-01.flac", chunks / "Pen-01.flac"]
    for f in files:
        f.write_bytes(b"x")
    return files


def test_default_model_is_large_v3():
    assert PipelineConfig().whisperx_model == "large-v3"


def test_command_mounts_cache_and_session(tmp_path):
    files = _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "cache")
    cmd = build_whisperx_command(tmp_path, files, cfg, environ={})
    assert f"{tmp_path.resolve()}:/app" in cmd
    assert f"{(tmp_path / 'cache').resolve()}:/root/.cache" in cmd
    assert (tmp_path / "cache").is_dir()  # created before docker run
    assert cmd[cmd.index("--model") + 1] == "large-v3"
    assert "/app/chunked_tracks/GM-01.flac" in cmd


def test_local_model_folder_is_mounted_read_only(tmp_path):
    files = _session(tmp_path)
    model = tmp_path / "models" / "ctranslate2"
    model.mkdir(parents=True)
    (model / "model.bin").write_bytes(b"m")
    cfg = PipelineConfig(base_path=tmp_path, whisperx_model=str(model), whisperx_cache_dir=tmp_path / "c")
    cmd = build_whisperx_command(tmp_path, files, cfg, environ={})
    assert f"{model.resolve()}:/models/ctranslate2:ro" in cmd
    assert cmd[cmd.index("--model") + 1] == "/models/ctranslate2"


def test_offline_flag_passed_through(tmp_path):
    files = _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    cmd = build_whisperx_command(tmp_path, files, cfg, environ={"HF_HUB_OFFLINE": "1"})
    assert "HF_HUB_OFFLINE=1" in cmd and cmd[cmd.index("HF_HUB_OFFLINE=1") - 1] == "-e"


def test_redaction():
    text = redact_command(["whisperx", "a.flac", "--hf_token", "hf_secret", "--model", "x"])
    assert "hf_secret" not in text and "--hf_token ****" in text


def test_no_chunks_raises(tmp_path):
    (tmp_path / "chunked_tracks").mkdir()
    with pytest.raises(FileNotFoundError):
        run_whisperx_docker(tmp_path, "chunked_tracks", PipelineConfig(base_path=tmp_path))


def test_container_output_goes_to_logger(tmp_path, caplog):
    _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")

    def fake_cmd(*_a, **_k):
        return ["sh", "-c", "echo out-line; echo err-line 1>&2"]

    with patch("chronicleweave.pipeline.build_whisperx_command", side_effect=fake_cmd):
        with caplog.at_level(logging.INFO, logger="chronicleweave.docker"):
            run_whisperx_docker(tmp_path, "chunked_tracks", cfg)
    assert "out-line" in caplog.text and "err-line" in caplog.text


def test_nonzero_exit_raises(tmp_path):
    _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    with patch("chronicleweave.pipeline.build_whisperx_command", return_value=["sh", "-c", "exit 3"]):
        with pytest.raises(RuntimeError, match="exit code 3"):
            run_whisperx_docker(tmp_path, "chunked_tracks", cfg)


# --- final review F8: a named container, stopped and reaped when streaming fails ---

import os
import re
import subprocess
import time

from chronicleweave import pipeline as pipeline_module
from chronicleweave.pipeline import container_name


def test_container_has_unique_name(tmp_path):
    files = _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    cmd = build_whisperx_command(tmp_path, files, cfg, environ={})
    i = cmd.index("--name")
    assert cmd[i + 1] == f"chronicleweave-{tmp_path.name}-{os.getpid()}" == container_name(tmp_path)
    assert i < cmd.index(cfg.whisperx_docker_image)


def test_container_name_is_valid_for_docker():
    name = container_name(Path("/data/sessions/session5 - Copie (2)"))
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", name)
    assert name.startswith("chronicleweave-session5") and name.endswith(f"-{os.getpid()}")


@pytest.mark.parametrize("exc", [OSError("stdout broke"), KeyboardInterrupt()])
def test_streaming_failure_stops_container_and_reaps_client(tmp_path, exc):
    _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    procs = []
    real_popen = subprocess.Popen

    def spy(*a, **k):
        procs.append(real_popen(*a, **k))
        return procs[-1]

    started = time.monotonic()
    with patch("chronicleweave.pipeline.build_whisperx_command", return_value=["sh", "-c", "echo started; exec sleep 30"]), \
            patch("chronicleweave.pipeline.subprocess.Popen", side_effect=spy), \
            patch("chronicleweave.pipeline.subprocess.run") as run_mock, \
            patch.object(pipeline_module.docker_log, "info", side_effect=exc):
        with pytest.raises(type(exc)):
            run_whisperx_docker(tmp_path, "chunked_tracks", cfg)
    assert time.monotonic() - started < 15  # the client was terminated, not waited for 30 s
    assert any(c.args[0][:2] == ["docker", "stop"] and c.args[0][-1] == container_name(tmp_path)
               for c in run_mock.call_args_list)
    assert procs[0].returncode is not None  # reaped
    assert procs[0].stdout.closed


def test_docker_stop_failure_is_ignored(tmp_path):
    _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    with patch("chronicleweave.pipeline.build_whisperx_command", return_value=["sh", "-c", "echo started; exec sleep 30"]), \
            patch("chronicleweave.pipeline.subprocess.run", side_effect=OSError("no docker")), \
            patch.object(pipeline_module.docker_log, "info", side_effect=OSError("stdout broke")):
        with pytest.raises(OSError, match="stdout broke"):  # the original error, not the cleanup one
            run_whisperx_docker(tmp_path, "chunked_tracks", cfg)


# --- F17: longer Hugging Face download timeouts inside the container ---

def _env_pairs(cmd):
    return {cmd[i + 1].split("=", 1)[0]: cmd[i + 1].split("=", 1)[1] for i, a in enumerate(cmd) if a == "-e"}


def test_hub_timeouts_passed_by_default(tmp_path):
    files = _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    cmd = build_whisperx_command(tmp_path, files, cfg, environ={})
    pairs = _env_pairs(cmd)
    assert pairs["HF_HUB_DOWNLOAD_TIMEOUT"] == "120" and pairs["HF_HUB_ETAG_TIMEOUT"] == "60"
    assert max(i for i, a in enumerate(cmd) if a == "-e") < cmd.index(cfg.whisperx_docker_image)


def test_hub_timeouts_from_caller_environment_win(tmp_path):
    files = _session(tmp_path)
    cfg = PipelineConfig(base_path=tmp_path, whisperx_cache_dir=tmp_path / "c")
    cmd = build_whisperx_command(tmp_path, files, cfg,
                                 environ={"HF_HUB_DOWNLOAD_TIMEOUT": "300", "HF_HUB_ETAG_TIMEOUT": "30"})
    pairs = _env_pairs(cmd)
    assert pairs["HF_HUB_DOWNLOAD_TIMEOUT"] == "300" and pairs["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert cmd.count("-e") == 2  # no default added next to the caller's value
