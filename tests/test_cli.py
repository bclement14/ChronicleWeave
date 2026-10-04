# tests/test_cli.py
from pathlib import Path
from unittest.mock import patch

from chronicleweave import cli
from chronicleweave.pipeline import PipelineError, StepRange


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
    ok = [BatchResult(SessionDir(1, tmp_path), "OK", 1.0, None, None)]
    bad = [BatchResult(SessionDir(1, tmp_path), "FAILED", 1.0, None, "x")]
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=ok):
        assert cli.batch_main([str(tmp_path)]) == 0
    with patch("chronicleweave.cli.load_env"), patch("chronicleweave.cli.run_batch", return_value=bad):
        assert cli.batch_main([str(tmp_path), "--only", "1"]) == 1


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
