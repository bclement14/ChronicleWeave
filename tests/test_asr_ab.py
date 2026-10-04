# tests/test_asr_ab.py
import json
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from tools.asr_ab.ab import (
    build_page,
    build_windows,
    normalise,
    repetition_loops,
    score,
    select_items,
    window_text,
)


def _json(path: Path, segs):
    path.write_text(json.dumps({"segments": [{"start": s, "end": e, "text": t} for s, e, t in segs]}), encoding="utf-8")


def test_windows_include_speech_present_in_only_one_model():
    a = [(1.0, 2.0, "bonjour")]
    b = [(1.0, 2.0, "bonjour"), (10.0, 12.0, "phrase oubliée")]
    windows = build_windows(a, b)
    assert (10.0, 12.0) in windows


def test_windows_are_capped():
    a = [(i * 1.0, i * 1.0 + 0.9, f"m{i}") for i in range(60)]
    windows = build_windows(a, a, max_len=25.0)
    assert all(e - s <= 25.0 + 1e-6 for s, e in windows)


def test_window_text_by_midpoint():
    segs = [(0.0, 1.0, "un"), (5.0, 6.0, "deux")]
    assert window_text(segs, (0.0, 2.0)) == "un"


def test_normalise():
    assert normalise("Bonjour, Kashkyst !") == "bonjour kashkyst"


def test_select_items_only_differences(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _json(a / "GM-01.json", [(0, 1, "pareil"), (5, 6, "le cash kist")])
    _json(b / "GM-01.json", [(0, 1, "Pareil."), (5, 6, "le Kashkyst")])
    items = select_items(a, b, n=30)
    assert len(items) == 1 and items[0]["stem"] == "GM-01"
    assert items[0]["a_text"] == "le cash kist" and items[0]["b_text"] == "le Kashkyst"


def test_repetition_loops():
    assert repetition_loops("merci merci beaucoup " * 1) == 0
    assert repetition_loops("sous-titrage ST 501 " * 3) == 1


def test_build_page_and_score(tmp_path):
    chunks = tmp_path / "chunked_tracks"
    chunks.mkdir()
    t = np.zeros(48000 * 8, dtype=np.int16)
    AudioSegment(t.tobytes(), frame_rate=48000, sample_width=2, channels=1).export(chunks / "GM-01.flac", format="flac")
    items = [{"id": "i1", "stem": "GM-01", "start": 2.0, "end": 4.0, "a_text": "x", "b_text": "y"}]
    page, mapping_path = build_page(items, "large-v3", "dec16", chunks, tmp_path / "out")
    html = page.read_text(encoding="utf-8")
    assert "large-v3" not in html and "dec16" not in html  # blind
    assert (tmp_path / "out" / "clips" / "i1.ogg").exists()
    mapping = json.loads(mapping_path.read_text())
    winner_label = [k for k, v in mapping["i1"].items() if v == "dec16"][0]
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"i1": winner_label}))
    a_script, b_script = tmp_path / "a.txt", tmp_path / "b.txt"
    a_script.write_text("[GM] ok"); b_script.write_text("[GM] ok")
    result = score(answers, mapping_path, a_script, b_script)
    assert result["wins"]["dec16"] == 1 and result["non_tied"] == 1
    assert result["decision"] == "inconclusive"  # fewer than 20 non-tied judgments
