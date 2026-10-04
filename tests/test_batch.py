# tests/test_batch.py
from pathlib import Path

import pytest

from chronicleweave.batch import (
    discover_sessions,
    format_summary,
    is_session_done,
    parse_only,
    run_batch,
)
from chronicleweave.pipeline import DONE_MARKER_NAME, LOG_FILE_NAME, RUNNING_MARKER_NAME, PipelineError


def _session(root: Path, name: str, archive=True, tracks=False) -> Path:
    p = root / name
    p.mkdir()
    if archive:
        (p / "craig-X.flac.zip").write_bytes(b"z")
    if tracks:
        (p / "tracks").mkdir()
        (p / "tracks" / "GM.flac").write_bytes(b"x")
    return p


def test_discovery_excludes_copies_and_work_folders(tmp_path):
    _session(tmp_path, "session2")
    _session(tmp_path, "Session13")
    _session(tmp_path, "session5 - Copie")
    _session(tmp_path, "session5 - Copie (2)")
    _session(tmp_path, "_asr_ab")
    _session(tmp_path, "session30", archive=False, tracks=True)
    _session(tmp_path, "session31", archive=False)  # nothing to process
    found = discover_sessions(tmp_path)
    assert [(s.number, s.path.name) for s in found] == [(2, "session2"), (13, "Session13"), (30, "session30")]


def test_archive_directory_does_not_count(tmp_path):
    p = tmp_path / "session27"
    p.mkdir()
    (p / "craig-X.flac.zip").mkdir()
    assert discover_sessions(tmp_path) == []


def test_done_rules(tmp_path):
    legacy = _session(tmp_path, "session10")
    (legacy / "final_outputs").mkdir()
    (legacy / "final_outputs" / "final_script.txt").write_text("x")
    assert is_session_done(legacy)

    marked = _session(tmp_path, "session20")
    (marked / "final_outputs").mkdir()
    (marked / "final_outputs" / DONE_MARKER_NAME).write_text("{}")
    (marked / LOG_FILE_NAME).write_text("log")
    assert is_session_done(marked)

    legacy_with_log = _session(tmp_path, "session22")
    (legacy_with_log / "final_outputs").mkdir()
    (legacy_with_log / "final_outputs" / "final_script.txt").write_text("x")
    (legacy_with_log / LOG_FILE_NAME).write_text("step 9 only")
    assert is_session_done(legacy_with_log)


def test_crashed_run_is_not_done(tmp_path):
    crashed = _session(tmp_path, "session21")
    (crashed / "final_outputs").mkdir()
    (crashed / "final_outputs" / "final_script.txt").write_text("old")
    (crashed / "final_outputs" / RUNNING_MARKER_NAME).write_text("")
    (crashed / LOG_FILE_NAME).write_text("started")
    (crashed / "wx_output").mkdir()
    assert not is_session_done(crashed)


def test_legacy_session_with_log_still_done(tmp_path):
    p = _session(tmp_path, "session12")
    (p / "final_outputs").mkdir()
    (p / "final_outputs" / "final_script.txt").write_text("accepted")
    (p / LOG_FILE_NAME).write_text("step 9 only run")
    assert is_session_done(p)


def test_parse_only():
    assert parse_only(None) is None
    assert parse_only("13,15-17, 20") == {13, 15, 16, 17, 20}
    with pytest.raises(ValueError):
        parse_only("x")


def test_run_batch_skips_done_continues_after_failure(tmp_path):
    done = _session(tmp_path, "session10")
    (done / "final_outputs").mkdir()
    (done / "final_outputs" / "final_script.txt").write_text("x")
    _session(tmp_path, "session20")
    _session(tmp_path, "session21")
    calls = []

    def runner(base_path, **kw):
        calls.append(Path(base_path).name)
        if Path(base_path).name == "session20":
            raise PipelineError(2, "boom")

    results = run_batch(tmp_path, only=None, force=False, pipeline_kwargs={"log_level": "WARNING"}, runner=runner)
    assert calls == ["session20", "session21"]
    assert [(r.session.number, r.status) for r in results] == [(10, "SKIPPED"), (20, "FAILED"), (21, "OK")]
    assert "boom" in results[1].error
    summary = format_summary(results)
    assert "session20" in summary and "FAILED" in summary


def test_run_batch_only_and_force(tmp_path):
    done = _session(tmp_path, "session10")
    (done / "final_outputs").mkdir()
    (done / "final_outputs" / "final_script.txt").write_text("x")
    _session(tmp_path, "session11")
    calls = []
    run_batch(tmp_path, only={10}, force=True, pipeline_kwargs={}, runner=lambda base_path, **kw: calls.append(base_path))
    assert [Path(c).name for c in calls] == ["session10"]
