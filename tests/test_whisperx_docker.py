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
