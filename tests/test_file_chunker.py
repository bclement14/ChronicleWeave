# tests/test_file_chunker.py

from pathlib import Path

import pytest

from chronicleweave.modules.file_chunker import chunk_text_file_by_lines, compact_lines


# --- compact_lines ---

def test_compact_lines_collapses_runs():
    lines = [
        "A: hi\n", "\n", "\n", "\n",
        "B: hello\n", "\n", "\n",
        "A: bye\n",
    ]
    out = compact_lines(lines)
    assert out == ["A: hi\n", "\n", "B: hello\n", "\n", "A: bye\n"]


def test_compact_lines_strips_trailing_blanks():
    lines = ["text\n", "\n", "\n"]
    assert compact_lines(lines) == ["text\n"]


def test_compact_lines_empty_input():
    assert compact_lines([]) == []


def test_compact_lines_no_blanks_unchanged():
    lines = ["a\n", "b\n", "c\n"]
    assert compact_lines(lines) == lines


def test_compact_lines_treats_whitespace_only_as_blank():
    lines = ["a\n", "   \n", "\t\n", "b\n"]
    assert compact_lines(lines) == ["a\n", "   \n", "b\n"]


# --- chunk_text_file_by_lines ---

def test_chunk_creates_files_of_expected_size(tmp_path: Path):
    src = tmp_path / "in.txt"
    src.write_text("".join(f"line{i}\n" for i in range(25)), encoding="utf-8")
    out = tmp_path / "out"

    chunks = chunk_text_file_by_lines(src, out, chunk_size=10)

    assert len(chunks) == 3
    assert chunks[0].read_text().count("\n") == 10
    assert chunks[2].read_text().count("\n") == 5


def test_chunk_compact_mode_reduces_size(tmp_path: Path):
    src = tmp_path / "in.txt"
    src.write_text("a\n\n\n\nb\n\n\n\nc\n", encoding="utf-8")
    out = tmp_path / "out"

    plain = chunk_text_file_by_lines(src, out / "plain", chunk_size=100)
    compact = chunk_text_file_by_lines(src, out / "compact", chunk_size=100, compact_mode=True)

    assert plain[0].read_text().count("\n") > compact[0].read_text().count("\n")


def test_chunk_missing_input_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        chunk_text_file_by_lines(tmp_path / "nope.txt", tmp_path / "out")


def test_chunk_filename_template_applied(tmp_path: Path):
    src = tmp_path / "in.txt"
    src.write_text("x\n" * 5, encoding="utf-8")
    out = tmp_path / "out"

    chunks = chunk_text_file_by_lines(
        src, out, chunk_size=2, output_filename_template="part_{:03d}.txt"
    )
    assert {c.name for c in chunks} == {"part_000.txt", "part_001.txt", "part_002.txt"}
