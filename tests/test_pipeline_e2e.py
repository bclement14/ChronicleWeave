# tests/test_pipeline_e2e.py
"""End-to-end tests for the chronicleweave pipeline.

Builds a synthetic session: one Craig zip → tracks → fake WhisperX JSON →
final script. Mocks the WhisperX Docker step (no GPU/Docker in CI) by writing
the JSON outputs directly. Covers run control: completion marker, session log,
failure propagation, clean re-runs and an explicitly requested step 9."""

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from chronicleweave.pipeline import DONE_MARKER_NAME, RUNNING_MARKER_NAME, PipelineError, run_pipeline

pytestmark = pytest.mark.slow


def _make_craig_zip(target: Path, speakers: dict) -> None:
    """speakers: {raw_username: AudioSegment} -> single zip file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as zf:
        for i, (uname, seg) in enumerate(speakers.items(), start=1):
            tmp = target.parent / f"{i}-{uname}.flac"
            seg.export(tmp, format="flac")
            zf.write(tmp, arcname=f"{i}-{uname}.flac")
            tmp.unlink()
        zf.writestr("info.txt", "fake metadata")
        zf.writestr("raw.dat", b"\x00\x00\x00")


def _fake_whisperx(session_path: Path, chunks_folder_name: str, config) -> None:
    """Stand-in for run_whisperx_docker: writes one JSON per chunked flac."""
    chunks_dir = session_path / chunks_folder_name
    out_dir = session_path / config.whisperx_output_folderName
    out_dir.mkdir(parents=True, exist_ok=True)
    for flac in sorted(chunks_dir.glob("*.flac")):
        out = out_dir / f"{flac.stem}.json"
        out.write_text(json.dumps({
            "segments": [
                {
                    "start": 0.0, "end": 1.0,
                    "text": f"hello from {flac.stem}",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
                        {"word": "from", "start": 0.4, "end": 0.6, "score": 0.9},
                        {"word": flac.stem, "start": 0.6, "end": 1.0, "score": 0.9},
                    ],
                },
                {
                    "start": 1.0, "end": 2.0,
                    "text": "second sentence",
                    "words": [
                        {"word": "second", "start": 1.0, "end": 1.4, "score": 0.9},
                        {"word": "sentence", "start": 1.4, "end": 2.0, "score": 0.9},
                    ],
                },
            ],
            "language": "fr",
        }))


SPEAKERS_ENV = {"CW_SPEAKERS": "player_one:Titar,gamemaster:GM", "CW_IGNORE_TRACKS": "Spoticord"}


@pytest.fixture
def synthetic_session(tmp_path, monkeypatch):
    for k, v in SPEAKERS_ENV.items():
        monkeypatch.setenv(k, v)
    session = tmp_path / "session_test"
    session.mkdir()
    speech = AudioSegment.silent(duration=2000, frame_rate=16000)
    _make_craig_zip(session / "craig-FAKE12345.flac.zip", {
        "player_one": speech, "gamemaster": speech, "Spoticord_Music_4270": speech,
    })
    return session


def _run(session, **kw):
    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=_fake_whisperx):
        run_pipeline(base_path=str(session), steps_to_run="1-7", log_level="WARNING", **kw)


def test_e2e_steps_1_to_7(synthetic_session):
    _run(synthetic_session)
    text = (synthetic_session / "final_outputs" / "final_script.txt").read_text(encoding="utf-8")
    assert "[Titar]" in text and "[GM]" in text
    assert "player_one" not in text and "Spoticord" not in text
    marker = json.loads((synthetic_session / "final_outputs" / DONE_MARKER_NAME).read_text())
    assert marker["steps"] == [1, 7] and marker["whisperx_model"] == "large-v3"
    assert (synthetic_session / "pipeline.log").read_text(encoding="utf-8")
    assert not (synthetic_session / "final_outputs" / RUNNING_MARKER_NAME).exists()


def test_e2e_failure_raises_and_writes_no_marker(synthetic_session):
    def broken(*a, **k):
        raise RuntimeError("docker exploded")
    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=broken):
        with pytest.raises(PipelineError):
            run_pipeline(base_path=str(synthetic_session), steps_to_run="1-7", log_level="WARNING")
    assert not (synthetic_session / "final_outputs" / DONE_MARKER_NAME).exists()
    assert (synthetic_session / "final_outputs" / RUNNING_MARKER_NAME).exists()  # failed after cleanup
    assert "docker exploded" in (synthetic_session / "pipeline.log").read_text(encoding="utf-8")


def test_e2e_no_prepare_with_raw_names_fails(synthetic_session):
    import zipfile as _zf
    archive = next(synthetic_session.glob("craig-*.flac.zip"))
    tracks = synthetic_session / "tracks"
    tracks.mkdir()
    with _zf.ZipFile(archive) as zf:
        zf.extract("1-player_one.flac", tracks)
    with pytest.raises(PipelineError):
        _run(synthetic_session, auto_prepare_tracks=False)


def test_rerun_after_partial_outputs_starts_clean(synthetic_session):
    _run(synthetic_session)
    (synthetic_session / "wx_output" / "Ghost-01.json").write_text("{}")  # stale leftover
    (synthetic_session / "final_outputs" / DONE_MARKER_NAME).unlink()
    _run(synthetic_session)
    assert not (synthetic_session / "wx_output" / "Ghost-01.json").exists()
    assert (synthetic_session / "final_outputs" / DONE_MARKER_NAME).exists()


def test_rerun_with_new_transcripts_uses_new_text(synthetic_session):
    _run(synthetic_session)

    def other_text(session_path, chunks_folder_name, config):
        _fake_whisperx(session_path, chunks_folder_name, config)
        for p in (session_path / config.whisperx_output_folderName).glob("*.json"):
            p.write_text(p.read_text().replace("hello", "THIRD"))  # in text AND words (step 3 rebuilds from words)

    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=other_text):
        run_pipeline(base_path=str(synthetic_session), steps_to_run="2-7", log_level="WARNING")
    text = (synthetic_session / "final_outputs" / "final_script.txt").read_text(encoding="utf-8")
    assert "THIRD" in text and "hello" not in text


def test_step9_requested_failure_raises(synthetic_session, monkeypatch):
    _run(synthetic_session)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="Step 9"):
        run_pipeline(base_path=str(synthetic_session), steps_to_run="9", log_level="WARNING")
    assert not (synthetic_session / "final_outputs" / RUNNING_MARKER_NAME).exists()


def test_step8_only_rerun_keeps_marker(synthetic_session):
    _run(synthetic_session)
    run_pipeline(base_path=str(synthetic_session), steps_to_run="8", log_level="WARNING")
    assert (synthetic_session / "final_outputs" / DONE_MARKER_NAME).exists()
    assert not (synthetic_session / "final_outputs" / RUNNING_MARKER_NAME).exists()


def test_missing_speakers_fails_before_cleanup(synthetic_session, monkeypatch):
    _run(synthetic_session)
    monkeypatch.delenv("CW_SPEAKERS")
    with pytest.raises(PipelineError):
        _run(synthetic_session)
    assert (synthetic_session / "final_outputs" / "final_script.txt").exists()
    assert (synthetic_session / "wx_output").is_dir()
    assert (synthetic_session / "final_outputs" / DONE_MARKER_NAME).exists()


def _assert_untouched(session):
    assert (session / "final_outputs" / "final_script.txt").exists()
    assert (session / "final_outputs" / DONE_MARKER_NAME).exists()
    assert not (session / "final_outputs" / RUNNING_MARKER_NAME).exists()


@pytest.mark.parametrize("steps,remove", [
    ("3", "chunked_tracks"),
    ("3-8", "final_outputs/cut_points.txt"),
    ("7", "json_files"),
])
def test_missing_earlier_input_fails_before_cleanup(synthetic_session, steps, remove):
    _run(synthetic_session)
    target = synthetic_session / remove
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=_fake_whisperx):
        with pytest.raises(PipelineError, match="missing input"):
            run_pipeline(base_path=str(synthetic_session), steps_to_run=steps, log_level="WARNING")
    _assert_untouched(synthetic_session)


def test_guard_refusal_keeps_done_marker(synthetic_session):
    _run(synthetic_session)
    with pytest.raises(PipelineError, match="tracks"):
        _run(synthetic_session, script_chunks_in_final_folder=False, script_chunks_folderName="tracks")
    _assert_untouched(synthetic_session)


def test_step0_failure_keeps_outputs(synthetic_session):
    _run(synthetic_session)
    (synthetic_session / "tracks" / "9-stranger.flac").write_bytes(b"x")
    with pytest.raises(PipelineError, match="stranger"):
        _run(synthetic_session)
    _assert_untouched(synthetic_session)


def test_cleanup_oserror_becomes_logged_pipeline_error(synthetic_session):
    _run(synthetic_session)
    with patch("chronicleweave.pipeline.shutil.rmtree", side_effect=OSError("disk says no")):
        with pytest.raises(PipelineError, match="Removing previous outputs"):
            _run(synthetic_session)
    assert "disk says no" in (synthetic_session / "pipeline.log").read_text(encoding="utf-8")
    assert (synthetic_session / "final_outputs" / RUNNING_MARKER_NAME).exists()


@pytest.mark.parametrize("folder", ["../other/tracks", "."])
def test_tracks_folder_outside_session_refused_before_step0(synthetic_session, folder):
    other = synthetic_session.parent / "other" / "tracks"
    other.mkdir(parents=True)
    music = other / "3-Spoticord_Music_4270.flac"
    music.write_bytes(b"music")
    with pytest.raises(PipelineError, match="tracks folder"):
        _run(synthetic_session, input_audio_folderName=folder)
    assert music.exists()
    assert not list(synthetic_session.glob("*.flac"))  # nothing extracted into the session root
    assert not (synthetic_session / "tracks").exists()
