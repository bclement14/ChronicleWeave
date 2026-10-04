# tests/test_cli.py
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chronicleweave import cli
from chronicleweave.pipeline import PipelineError, StepRange, run_pipeline


def test_main_success_returns_0(tmp_path):
    with patch("chronicleweave.cli.run_pipeline") as rp, patch("chronicleweave.cli.load_env"):
        assert cli.main(["-b", str(tmp_path), "--steps", "1-7"]) == 0
    kwargs = rp.call_args.kwargs
    assert kwargs["base_path"] == str(tmp_path) and kwargs["steps_to_run"] == StepRange(1, 7)
    assert "use_low_ram" not in kwargs and "speaker_mapping_path" not in kwargs


def test_main_failure_returns_1(tmp_path, capsys):
    with patch("chronicleweave.cli.run_pipeline", side_effect=PipelineError(3, "bad")), patch("chronicleweave.cli.load_env"):
        assert cli.main(["-b", str(tmp_path)]) == 1
    assert "Step 3 failed" in capsys.readouterr().err


def test_main_bad_steps_returns_1(tmp_path):
    with patch("chronicleweave.cli.load_env"):
        assert cli.main(["-b", str(tmp_path), "--steps", "1,3"]) == 1


def test_main_missing_env_file_returns_1(tmp_path):
    assert cli.main(["-b", str(tmp_path), "--env-file", str(tmp_path / "missing.env")]) == 1


def test_main_default_steps_exclude_9(tmp_path):
    with patch("chronicleweave.cli.run_pipeline") as rp, patch("chronicleweave.cli.load_env"):
        cli.main(["-b", str(tmp_path)])
    assert rp.call_args.kwargs["steps_to_run"] == StepRange(1, 8)


def test_main_model_flag(tmp_path):
    with patch("chronicleweave.cli.run_pipeline") as rp, patch("chronicleweave.cli.load_env"):
        cli.main(["-b", str(tmp_path), "--whisperx-model", "/models/dec16"])
    assert rp.call_args.kwargs["whisperx_model"] == "/models/dec16"


def test_batch_main_exit_codes(tmp_path):
    from chronicleweave.batch import BatchResult, SessionDir
    (tmp_path / "session1").mkdir()
    (tmp_path / "session1" / "craig-X.flac.zip").write_bytes(b"z")  # an empty root now exits 1 (F14)
    ok = [BatchResult(SessionDir(1, tmp_path), "OK", 1.0, None, None)]
    bad = [BatchResult(SessionDir(1, tmp_path), "FAILED", 1.0, None, "x")]
    no_docker = ["--steps", "3-7"]  # steps without transcription: no docker preflight (F11)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=ok):
        assert cli.batch_main([str(tmp_path), *no_docker]) == 0
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=bad):
        assert cli.batch_main([str(tmp_path), "--only", "1", *no_docker]) == 1


def _batch_root(tmp_path):
    session = tmp_path / "session10"
    session.mkdir()
    (session / "craig-X.flac.zip").write_bytes(b"z")
    return tmp_path


def test_batch_force_requires_only(tmp_path, capsys):
    root = _batch_root(tmp_path)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch") as rb:
        assert cli.batch_main([str(root), "--force", "--steps", "3-7"]) == 1
    rb.assert_not_called()
    assert "--only" in capsys.readouterr().err


def test_batch_force_with_only_runs(tmp_path):
    root = _batch_root(tmp_path)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=[]) as rb:
        assert cli.batch_main([str(root), "--force", "--only", "10", "--steps", "3-7"]) == 0
    assert rb.call_args.kwargs["force"] is True and rb.call_args.kwargs["only"] == {10}


# --- final review F11: docker preflight before any session of the batch ---

def _docker_on_path(tmp_path, monkeypatch, exit_code):
    bin_dir = tmp_path / "dockerbin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(f'#!/bin/sh\necho "$@" > "{bin_dir}/args"\necho "Error: No such image" >&2\nexit {exit_code}\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir / "args"


def test_batch_preflight_failure_processes_nothing(tmp_path, capsys):
    root = _batch_root(tmp_path)  # the test fake docker on PATH exits 97
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch") as rb:
        assert cli.batch_main([str(root)]) == 1
    rb.assert_not_called()
    err = capsys.readouterr().err
    assert "chronicleweave-whisperx" in err and "97" in err


def test_batch_preflight_inspects_the_image_once(tmp_path, monkeypatch):
    (tmp_path / "root").mkdir()
    root = _batch_root(tmp_path / "root")
    args = _docker_on_path(tmp_path, monkeypatch, 0)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=[]) as rb:
        assert cli.batch_main([str(root)]) == 0
    rb.assert_called_once()
    assert args.read_text().split() == ["image", "inspect", "chronicleweave-whisperx"]


@pytest.mark.parametrize("extra", [["--steps", "3-7"], ["--no-whisperx"]])
def test_batch_no_preflight_without_transcription(tmp_path, extra):
    root = _batch_root(tmp_path)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=[]) as rb:
        assert cli.batch_main([str(root), *extra]) == 0
    rb.assert_called_once()


# --- final review F12: print which .env was loaded and the speaker Tags ---

def test_main_prints_env_file_and_tags(tmp_path, capsys):
    env = tmp_path / "my.env"
    env.write_text("CW_SPEAKERS=gamemaster:GM, penny42:Pen\n", encoding="utf-8")
    with patch.dict(os.environ):
        os.environ.pop("CW_SPEAKERS", None)
        with patch("chronicleweave.cli.run_pipeline"):
            assert cli.main(["-b", str(tmp_path), "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if str(env) in line]
    assert len(lines) == 1 and "GM, Pen" in lines[0]
    assert "gamemaster" not in out and "penny42" not in out


def test_batch_prints_no_env_found_and_missing_speakers(tmp_path, capsys):
    root = _batch_root(tmp_path)
    with patch.dict(os.environ):
        os.environ.pop("CW_SPEAKERS", None)
        with patch("chronicleweave.cli.load_env", return_value=None), \
                patch("chronicleweave.cli.run_batch", return_value=[]):
            assert cli.batch_main([str(root), "--steps", "3-7"]) == 0
    out = capsys.readouterr().out
    assert "no .env found" in out and "CW_SPEAKERS" in out


# --- final review F13: the session folder and the batch root must exist ---

@pytest.mark.parametrize("make_file", [False, True])
def test_main_missing_base_path_exits_1_without_creating_it(tmp_path, capsys, make_file):
    target = tmp_path / "sesion26"  # a typo
    if make_file:
        target.write_text("not a folder")
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_pipeline") as rp:
        assert cli.main(["-b", str(target)]) == 1
    rp.assert_not_called()
    assert target.is_dir() is False
    assert "sesion26" in capsys.readouterr().err


def test_run_pipeline_does_not_create_a_missing_session(tmp_path):
    with pytest.raises(PipelineError, match="not found"):
        run_pipeline(base_path=str(tmp_path / "nope"), steps_to_run="8", log_level="WARNING")
    assert not (tmp_path / "nope").exists()


def test_batch_missing_root_exits_1(tmp_path, capsys):
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch") as rb:
        assert cli.batch_main([str(tmp_path / "Sesions"), "--steps", "3-7"]) == 1
    rb.assert_not_called()
    assert "Sesions" in capsys.readouterr().err


# --- final review F14: nothing selected is an error ---

@pytest.mark.parametrize("args", [["--only", "35"], []])
def test_batch_nothing_selected_exits_1(tmp_path, capsys, args):
    root = tmp_path / "root"
    root.mkdir()
    if args:
        _batch_root(root)  # session10 exists, 35 does not
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch") as rb:
        assert cli.batch_main([str(root), "--steps", "3-7", *args]) == 1
    rb.assert_not_called()
    assert "No session" in capsys.readouterr().err


def test_batch_reports_requested_sessions_not_found(tmp_path, capsys):
    root = _batch_root(tmp_path)
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=[]) as rb:
        assert cli.batch_main([str(root), "--only", "10,35", "--steps", "3-7"]) == 0
    rb.assert_called_once()
    assert "35" in capsys.readouterr().err
