# tests/test_pipeline_e2e.py
"""End-to-end smoke test for the chronicleweave pipeline.

Builds a synthetic session: one Craig zip → tracks → fake WhisperX JSON →
final script. Mocks the WhisperX Docker step (no GPU/Docker in CI) by writing
the JSON outputs directly. Catches wiring regressions across stages."""

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydub import AudioSegment

from chronicleweave.pipeline import run_pipeline

pytestmark = pytest.mark.slow


def _silent_flac(path: Path, duration_ms: int = 2000) -> None:
    """Write a tiny silent FLAC. Two seconds is enough for the chunker
    to treat the whole file as a single chunk."""
    AudioSegment.silent(duration=duration_ms, frame_rate=16000).export(path, format="flac")


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


@pytest.fixture
def synthetic_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_test"
    session.mkdir()
    silence = AudioSegment.silent(duration=2000, frame_rate=16000)
    _make_craig_zip(session / "craig-FAKE12345.flac.zip", {
        "player_one": silence,
        "gamemaster": silence,
    })
    return session


def test_e2e_steps_1_to_7_produces_final_script(synthetic_session, monkeypatch):
    """
    Run the full transcription chain (steps 1-7) on a synthetic session.
    Verifies: prepare_tracks → chunking → (faked) WhisperX → SRT → script,
    with the speaker mapping applied so the final script tags Titar/GM,
    not the raw Discord usernames.
    """
    # Mock the docker call only.
    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=_fake_whisperx):
        run_pipeline(
            base_path=str(synthetic_session),
            steps_to_run=[1, 2, 3, 4, 5, 6, 7],
            run_whisperx=True,  # uses our patched docker fn
            log_level="WARNING",
        )

    final = synthetic_session / "final_outputs" / "final_script.txt"
    assert final.is_file(), "final_script.txt was not produced"
    text = final.read_text(encoding="utf-8")

    # Speaker mapping was applied (Step 0): raw Craig usernames replaced.
    assert "[Titar]" in text or "Titar" in text, f"Expected mapped speaker Titar in:\n{text}"
    assert "[GM]" in text or "GM" in text, f"Expected mapped speaker GM in:\n{text}"
    # Raw usernames must NOT leak into the final script.
    assert "player_one" not in text
    assert "gamemaster" not in text


def test_e2e_no_prepare_keeps_raw_names(synthetic_session, monkeypatch):
    """With auto_prepare_tracks=False the user must extract themselves;
    if they do but skip mapping, raw stems propagate to the script."""
    # Manually unzip without mapping.
    import zipfile as _zf
    archive = next(synthetic_session.glob("craig-*.flac.zip"))
    tracks = synthetic_session / "tracks"
    tracks.mkdir()
    with _zf.ZipFile(archive) as zf:
        for m in zf.namelist():
            if m.endswith(".flac"):
                zf.extract(m, tracks)

    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=_fake_whisperx):
        run_pipeline(
            base_path=str(synthetic_session),
            steps_to_run=[1, 2, 3, 4, 5, 6, 7],
            run_whisperx=True,
            log_level="WARNING",
            auto_prepare_tracks=False,
        )

    text = (synthetic_session / "final_outputs" / "final_script.txt").read_text()
    # Raw stems leak through when prep is disabled — proves prepare_tracks is doing real work.
    assert "player_one" in text or "1-player_one" in text


def test_e2e_idempotent_rerun(synthetic_session):
    """Running the pipeline twice must not crash and must produce the same final output."""
    with patch("chronicleweave.pipeline.run_whisperx_docker", side_effect=_fake_whisperx):
        run_pipeline(
            base_path=str(synthetic_session),
            steps_to_run=[1, 2, 3, 4, 5, 6, 7],
            run_whisperx=True, log_level="WARNING",
        )
        first = (synthetic_session / "final_outputs" / "final_script.txt").read_text()
        run_pipeline(
            base_path=str(synthetic_session),
            steps_to_run=[1, 2, 3, 4, 5, 6, 7],
            run_whisperx=True, log_level="WARNING",
        )
        second = (synthetic_session / "final_outputs" / "final_script.txt").read_text()
    assert first == second
