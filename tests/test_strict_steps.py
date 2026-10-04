# tests/test_strict_steps.py
import json
import os
from pathlib import Path

import pytest

from chronicleweave.modules.convert_json_to_srt import convert_json_folder_to_srt
from chronicleweave.modules.convert_srt_to_script import srt_to_script
from chronicleweave.modules.whisperx_corrector_core import correct_whisperx_outputs

GOOD = {"segments": [{"start": 0.0, "end": 1.0, "text": "bonjour",
                      "words": [{"word": "bonjour", "start": 0.0, "end": 1.0, "score": 0.9}]}]}


def test_corrector_raises_when_a_file_fails(tmp_path):
    src, dst = tmp_path / "wx", tmp_path / "json"
    src.mkdir()
    (src / "GM-01.json").write_text(json.dumps(GOOD))
    (src / "GM-02.json").write_text("{broken")
    with pytest.raises(RuntimeError, match="1 file"):
        correct_whisperx_outputs(src, dst, overwrite=True)


def test_corrector_accepts_empty_segments(tmp_path):
    src, dst = tmp_path / "wx", tmp_path / "json"
    src.mkdir()
    (src / "GM-01.json").write_text(json.dumps({"segments": []}))
    correct_whisperx_outputs(src, dst, overwrite=True)
    assert (dst / "GM-01.json").exists()


def test_json_to_srt_raises_on_bad_file(tmp_path):
    src = tmp_path / "json"
    src.mkdir()
    (src / "GM-01.json").write_text(json.dumps(GOOD))
    (src / "GM-02.json").write_text("{broken")
    with pytest.raises(RuntimeError, match="GM-02"):
        convert_json_folder_to_srt(src, tmp_path / "srt")


def test_json_to_srt_raises_when_empty_folder(tmp_path):
    (tmp_path / "json").mkdir()
    with pytest.raises(FileNotFoundError):
        convert_json_folder_to_srt(tmp_path / "json", tmp_path / "srt")


def test_json_to_srt_empty_segments_gives_empty_srt(tmp_path):
    src = tmp_path / "json"
    src.mkdir()
    (src / "GM-01.json").write_text(json.dumps({"segments": []}))
    convert_json_folder_to_srt(src, tmp_path / "srt")
    assert (tmp_path / "srt" / "GM-01.srt").read_text() == ""


def test_script_written_atomically(tmp_path, monkeypatch):
    srt_file = tmp_path / "cleaned.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\n[GM] bonjour\n", encoding="utf-8")
    out = tmp_path / "final_script.txt"
    calls = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append((Path(a).name, Path(b).name)), real_replace(a, b)))
    srt_to_script(srt_file, out)
    assert out.read_text(encoding="utf-8") == "[GM] bonjour"
    assert calls and calls[-1][1] == "final_script.txt"
