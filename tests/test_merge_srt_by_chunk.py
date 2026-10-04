# tests/test_merge_srt_by_chunk.py
from datetime import timedelta
from pathlib import Path

import pytest
import srt

from chronicleweave.modules.merge_srt_by_chunk import (
    load_srt_file,
    merge_srt_by_chunk,
    parse_srt_filename,
    read_cut_points,
)


def _cut_points(path: Path, boundaries):
    lines = [f"# Audio Cut Points (Total Chunks: {len(boundaries) - 1})", "# Format: Chunk Index, Start Time (seconds)"]
    lines += [f"Chunk {i}, {b:.3f}" for i, b in enumerate(boundaries)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _srt(path: Path, entries):
    subs = [srt.Subtitle(i + 1, timedelta(seconds=s), timedelta(seconds=e), t) for i, (s, e, t) in enumerate(entries)]
    path.write_text(srt.compose(subs), encoding="utf-8")


def test_read_cut_points_valid(tmp_path):
    assert read_cut_points(_cut_points(tmp_path / "c.txt", [0.0, 541.783, 1084.043])) == [0.0, 541.783, 1084.043]


@pytest.mark.parametrize("boundaries", [[1.0, 2.0], [0.0, 5.0, 5.0], [0.0, 5.0, 3.0], [0.0]])
def test_read_cut_points_invalid(tmp_path, boundaries):
    with pytest.raises(ValueError):
        read_cut_points(_cut_points(tmp_path / "c.txt", boundaries))


def test_read_cut_points_non_contiguous_index(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("Chunk 0, 0.000\nChunk 2, 10.000\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_cut_points(p)


def test_parse_srt_filename():
    assert parse_srt_filename("GM-01.srt") == ("GM", 1)
    assert parse_srt_filename("Khale-be-12.srt") == ("Khale-be", 12)
    with pytest.raises(ValueError):
        parse_srt_filename("GM.srt")


def test_offsets_come_from_cut_points(tmp_path):
    srt_dir = tmp_path / "srt"
    srt_dir.mkdir()
    _srt(srt_dir / "GM-01.srt", [(1.0, 2.0, "un")])
    _srt(srt_dir / "GM-02.srt", [(0.5, 1.0, "deux")])
    _srt(srt_dir / "Pen-02.srt", [(0.2, 0.4, "trois")])
    cps = _cut_points(tmp_path / "cut_points.txt", [0.0, 100.0, 250.0])
    count = merge_srt_by_chunk(srt_dir, tmp_path, "merged.srt", cps)
    subs = list(srt.parse((tmp_path / "merged.srt").read_text(encoding="utf-8")))
    assert count == 3
    assert [(s.start.total_seconds(), s.content) for s in subs] == [
        (1.0, "[GM] un"), (100.2, "[Pen] trois"), (100.5, "[GM] deux"),
    ]


def test_chunk_number_beyond_cut_points_fails(tmp_path):
    srt_dir = tmp_path / "srt"
    srt_dir.mkdir()
    _srt(srt_dir / "GM-03.srt", [(0.0, 1.0, "x")])
    cps = _cut_points(tmp_path / "cut_points.txt", [0.0, 10.0, 20.0])
    with pytest.raises(ValueError, match="chunk 3"):
        merge_srt_by_chunk(srt_dir, tmp_path, "merged.srt", cps)


def test_unparsable_srt_fails(tmp_path):
    srt_dir = tmp_path / "srt"
    srt_dir.mkdir()
    (srt_dir / "GM-01.srt").write_text("1\nnot a timestamp\nhello\n", encoding="utf-8")
    cps = _cut_points(tmp_path / "cut_points.txt", [0.0, 10.0])
    with pytest.raises(Exception):
        merge_srt_by_chunk(srt_dir, tmp_path, "merged.srt", cps)


def test_empty_srt_is_valid(tmp_path):
    srt_dir = tmp_path / "srt"
    srt_dir.mkdir()
    (srt_dir / "GM-01.srt").write_text("", encoding="utf-8")
    _srt(srt_dir / "Pen-01.srt", [(0.0, 1.0, "salut")])
    cps = _cut_points(tmp_path / "cut_points.txt", [0.0, 10.0])
    assert merge_srt_by_chunk(srt_dir, tmp_path, "merged.srt", cps) == 1


def test_no_srt_files_fails(tmp_path):
    (tmp_path / "srt").mkdir()
    cps = _cut_points(tmp_path / "cut_points.txt", [0.0, 10.0])
    with pytest.raises(FileNotFoundError):
        merge_srt_by_chunk(tmp_path / "srt", tmp_path, "merged.srt", cps)


def test_load_srt_file_tags_and_offsets(tmp_path):
    p = tmp_path / "GM-01.srt"
    _srt(p, [(1.0, 2.0, "bonjour")])
    subs = load_srt_file(p, "GM", timedelta(seconds=10))
    assert subs[0].start == timedelta(seconds=11) and subs[0].content == "[GM] bonjour"
