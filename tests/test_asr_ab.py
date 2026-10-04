# tests/test_asr_ab.py
import json
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from tools.asr_ab.ab import (
    build_page,
    build_windows,
    main,
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
    result = score(answers, mapping_path, {"large-v3": a_script, "dec16": b_script})
    assert result["wins"]["dec16"] == 1 and result["non_tied"] == 1
    assert result["decision"] == "inconclusive"  # fewer than 20 non-tied judgments


LOOP = "sous-titrage ST 501 " * 3


def _score_cli(tmp_path, capsys, n_cand, n_base, n_tie=0, cand_script="ok", base_script="ok", candidate="dec16", b_name="dec16"):
    mapping, answers = {}, {}
    total = n_cand + n_base + n_tie
    for i in range(total):
        mapping[f"i{i}"] = {"X": "large-v3", "Y": "dec16"} if i % 2 else {"X": "dec16", "Y": "large-v3"}
        if i < n_cand:
            winner = "dec16"
        elif i < n_cand + n_base:
            winner = "large-v3"
        else:
            answers[f"i{i}"] = "same"
            continue
        answers[f"i{i}"] = [k for k, v in mapping[f"i{i}"].items() if v == winner][0]
    (tmp_path / "m.json").write_text(json.dumps(mapping))
    (tmp_path / "a.json").write_text(json.dumps(answers))
    (tmp_path / "base.txt").write_text(base_script)
    (tmp_path / "cand.txt").write_text(cand_script)
    code = main(["score", "--answers", str(tmp_path / "a.json"), "--mapping", str(tmp_path / "m.json"),
                 "--a-name", "large-v3", "--a-script", str(tmp_path / "base.txt"),
                 "--b-name", b_name, "--b-script", str(tmp_path / "cand.txt"), "--candidate", candidate])
    out = capsys.readouterr()
    return code, (json.loads(out.out) if out.out else None), out.err


def test_cli_19_non_tied_is_inconclusive(tmp_path, capsys):
    code, res, _ = _score_cli(tmp_path, capsys, 19, 0)
    assert code == 0 and res["non_tied"] == 19 and res["decision"] == "inconclusive -> keep large-v3"


def test_cli_exactly_two_thirds_switches(tmp_path, capsys):
    code, res, _ = _score_cli(tmp_path, capsys, 14, 7)
    assert res["non_tied"] == 21 and res["decision"] == "switch to dec16"


def test_cli_one_fewer_win_keeps_baseline(tmp_path, capsys):
    code, res, _ = _score_cli(tmp_path, capsys, 13, 8)
    assert res["decision"] == "keep large-v3"


def test_cli_more_loops_in_candidate_keeps_baseline(tmp_path, capsys):
    code, res, _ = _score_cli(tmp_path, capsys, 21, 0, cand_script=LOOP)
    assert res["loops"] == {"large-v3": 0, "dec16": 1} and res["decision"] == "keep large-v3"


def test_cli_ties_do_not_count(tmp_path, capsys):
    code, res, _ = _score_cli(tmp_path, capsys, 19, 0, n_tie=10)
    assert res["non_tied"] == 19 and res["ties"] == 10 and res["decision"].startswith("inconclusive")


def test_cli_unknown_candidate_exits_2(tmp_path, capsys):
    code, res, err = _score_cli(tmp_path, capsys, 21, 0, candidate="nope")
    assert code == 2 and res is None and "nope" in err


def test_cli_names_not_matching_mapping_exit_2(tmp_path, capsys):
    code, res, err = _score_cli(tmp_path, capsys, 21, 0, b_name="other")
    assert code == 2 and res is None and err


def test_score_reports_ignored_answers(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"i1": {"X": "a", "Y": "b"}}))
    (tmp_path / "a.json").write_text(json.dumps({"i1": "X", "zz": "X", "i2": "maybe"}))
    (tmp_path / "s.txt").write_text("ok")
    res = score(tmp_path / "a.json", tmp_path / "m.json", {"a": tmp_path / "s.txt", "b": tmp_path / "s.txt"})
    assert res["ignored"] == 2 and res["non_tied"] == 1
