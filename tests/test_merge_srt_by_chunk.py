# tests/test_merge_srt_by_chunk.py

import pytest
from pathlib import Path
import srt # Required for creating/checking Subtitle objects
from datetime import timedelta
import logging

# Import the function and config to test
from chronicleweave.modules.merge_srt_by_chunk import (
    merge_srt_by_chunk,
    parse_srt_filename,
    load_srt_file,
    SRTMergeConfig,
    DEFAULT_MERGE_CONFIG
)

# --- Test Data ---

CHUNK1_SPEAKER_A = """1
00:00:01,500 --> 00:00:03,000
Alice chunk 1 line 1.
"""
CHUNK1_SPEAKER_B = """1
00:00:02,000 --> 00:00:04,000
Bob chunk 1 line 1.
"""
CHUNK2_SPEAKER_A = """1
00:00:00,500 --> 00:00:02,500
Alice chunk 2 line 1.

2
00:00:03,000 --> 00:00:04,000
Alice chunk 2 line 2.
"""
CHUNK2_SPEAKER_B = """1
00:00:01,000 --> 00:00:03,500
Bob chunk 2 line 1.
"""

# --- Tests for parse_srt_filename ---

@pytest.mark.parametrize("filename, expected_output", [
    ("SpeakerA-01.srt", ("SpeakerA", 1)),
    ("Speaker_B-10.srt", ("Speaker_B", 10)),
    ("Speaker-With-Hyphens-99.srt", ("Speaker-With-Hyphens", 99)),
    ("speaker_1-05.SRT", ("speaker_1", 5)), # Case insensitive suffix
    ("NoNumber.srt", None),
    ("SpeakerA-Chunk1.srt", None), # Non-digit chunk
    ("SpeakerA-00.srt", None), # Chunk 0 might be invalid? (Parser currently allows >0) Let's assume valid
    ("SpeakerA-0.srt", None), # Single digit without padding - current regex requires \d+ ending
    ("SpeakerA-123456.srt", None), # Exceeds arbitrary limit in parser
    ("SpeakerA-10.txt", None), # Wrong extension
    ("SpeakerA-.srt", None), # Missing number
    ("-10.srt", None), # Speaker name missing (or starts with -) - current regex needs something before hyphen
])
def test_parse_srt_filename(filename, expected_output):
    """Test filename parsing for various valid and invalid formats."""
    assert parse_srt_filename(filename) == expected_output

# Add a specific test for speaker name starting with hyphen if needed based on regex
def test_parse_srt_filename_hyphen_speaker():
     assert parse_srt_filename("Sp-A-10.srt") == ("Sp-A", 10)
     # This test assumes the regex `^(.*?)-(\d+)$` is greedy with the first part

# --- Tests for load_srt_file ---

def test_load_srt_file_success(tmp_path: Path):
    """Test loading a valid SRT, applying offset and speaker tag."""
    srt_path = tmp_path / "test.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
    offset = timedelta(seconds=10)
    speaker = "Tester"

    subtitles = load_srt_file(srt_path, speaker, offset)

    assert len(subtitles) == 1
    sub = subtitles[0]
    assert sub.index == 1
    assert sub.start == timedelta(seconds=11) # 1 + 10
    assert sub.end == timedelta(seconds=12)   # 2 + 10
    assert sub.content == "[Tester] Hello"

def test_load_srt_file_not_found():
    """Test loading a non-existent file."""
    offset = timedelta(seconds=0)
    speaker = "Tester"
    non_existent_path = Path("./non_existent_file.srt")
    subtitles = load_srt_file(non_existent_path, speaker, offset)
    assert subtitles == [] # Function should return empty list on error

def test_load_srt_file_invalid_content(tmp_path: Path):
    """Test loading an invalid SRT file."""
    srt_path = tmp_path / "invalid.srt"
    srt_path.write_text("This is not SRT content", encoding="utf-8")
    offset = timedelta(seconds=0)
    speaker = "Tester"
    subtitles = load_srt_file(srt_path, speaker, offset)
    assert subtitles == [] # Function should return empty list on error

# --- Tests for merge_srt_by_chunk ---

def test_merge_srt_by_chunk_standard(tmp_path: Path):
    """Test merging multiple chunks with multiple speakers."""
    input_dir = tmp_path / "input_srt"
    output_dir = tmp_path / "output_srt"
    input_dir.mkdir()
    output_filename = "merged.srt"
    output_file = output_dir / output_filename

    # Create dummy files
    (input_dir / "Alice-01.srt").write_text(CHUNK1_SPEAKER_A, encoding="utf-8")
    (input_dir / "Bob-01.srt").write_text(CHUNK1_SPEAKER_B, encoding="utf-8")
    (input_dir / "Alice-02.srt").write_text(CHUNK2_SPEAKER_A, encoding="utf-8")
    (input_dir / "Bob-02.srt").write_text(CHUNK2_SPEAKER_B, encoding="utf-8")
    # Add a skipped file
    (input_dir / "ignore_this.txt").touch()
    (input_dir / "nonumber.srt").touch()

    # Use default config (0.1s gap)
    config = SRTMergeConfig(chunk_gap_s=0.1)
    merge_srt_by_chunk(srt_folder=input_dir, output_dir=output_dir,
                       output_filename=output_filename, config=config)

    assert output_file.exists()
    output_content = output_file.read_text(encoding="utf-8")
    output_subs = list(srt.parse(output_content))

    # Expected count: 2 subs from chunk 1 + 3 subs from chunk 2 = 5 subs
    assert len(output_subs) == 5

    # Check chunk 1 timing (no offset)
    # Sub 1: Alice C1 L1 (orig 1.5-3.0)
    assert output_subs[0].content.startswith("[Alice]")# Check custom attribute if srt lib supports, otherwise parse content
    
    # Optionally check the rest of the content if needed, removing the tag:
    # assert output_subs[0].content.replace("[Alice] ", "") == "Alice chunk 1 line 1."
    assert output_subs[0].start == timedelta(seconds=1.5)
    assert output_subs[0].end == timedelta(seconds=3.0)

    # Sub 2: Bob C1 L1 (orig 2.0-4.0) - comes *after* Alice's start time
    # --- CORRECTED ASSERTION: Check content start ---
    assert output_subs[1].content.startswith("[Bob]")
    assert output_subs[1].start == timedelta(seconds=2.0)
    assert output_subs[1].end == timedelta(seconds=4.0)

    # Check chunk 2 timing (offset = max end of chunk 1 + gap = 4.0 + 0.1 = 4.1s)
    offset_s = 4.1
    # Sub 3: Alice C2 L1 (orig 0.5-2.5 -> 4.6 - 6.6)
    # --- CORRECTED ASSERTION: Check content start ---
    assert output_subs[2].content.startswith("[Alice]")
    assert output_subs[2].start == pytest.approx(timedelta(seconds=offset_s + 0.5), abs=timedelta(milliseconds=1))
    assert output_subs[2].end == pytest.approx(timedelta(seconds=offset_s + 2.5), abs=timedelta(milliseconds=1))

    # Sub 4: Bob C2 L1 (orig 1.0-3.5 -> 5.1 - 7.6) - interleaved correctly
    # --- CORRECTED ASSERTION: Check content start ---
    assert output_subs[3].content.startswith("[Bob]")
    assert output_subs[3].start == pytest.approx(timedelta(seconds=offset_s + 1.0), abs=timedelta(milliseconds=1))
    assert output_subs[3].end == pytest.approx(timedelta(seconds=offset_s + 3.5), abs=timedelta(milliseconds=1))

    # Sub 5: Alice C2 L2 (orig 3.0-4.0 -> 7.1 - 8.1)
    # --- CORRECTED ASSERTION: Check content start ---
    assert output_subs[4].content.startswith("[Alice]")
    assert output_subs[4].start == pytest.approx(timedelta(seconds=offset_s + 3.0), abs=timedelta(milliseconds=1))
    assert output_subs[4].end == pytest.approx(timedelta(seconds=offset_s + 4.0), abs=timedelta(milliseconds=1))

    # Check overall order
    for i in range(len(output_subs) - 1):
        assert output_subs[i].start <= output_subs[i+1].start

def test_merge_srt_by_chunk_empty_input_dir(tmp_path: Path):
    """Test merging with an empty input directory."""
    input_dir = tmp_path / "input_empty"
    output_dir = tmp_path / "output_empty"
    input_dir.mkdir()
    output_filename = "merged_empty.srt"
    output_file = output_dir / output_filename

    # Expect a warning log, but no error, and an empty output file
    merge_srt_by_chunk(srt_folder=input_dir, output_dir=output_dir,
                       output_filename=output_filename, config=DEFAULT_MERGE_CONFIG)

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == ""

def test_merge_srt_by_chunk_no_matching_files(tmp_path: Path):
    """Test merging when directory has files but none match pattern."""
    input_dir = tmp_path / "input_nomatch"
    output_dir = tmp_path / "output_nomatch"
    input_dir.mkdir()
    (input_dir / "some_other_file.txt").touch()
    (input_dir / "another.srt").touch() # Doesn't match pattern
    output_filename = "merged_nomatch.srt"
    output_file = output_dir / output_filename

    merge_srt_by_chunk(srt_folder=input_dir, output_dir=output_dir,
                       output_filename=output_filename, config=DEFAULT_MERGE_CONFIG)

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == ""

def test_merge_srt_by_chunk_one_invalid_srt(tmp_path: Path, caplog):
    """Test merging continues if one SRT file is invalid."""
    input_dir = tmp_path / "input_partial"
    output_dir = tmp_path / "output_partial"
    input_dir.mkdir()
    output_filename = "merged_partial.srt"
    output_file = output_dir / output_filename

    # Create files: one valid, one invalid
    (input_dir / "Alice-01.srt").write_text(CHUNK1_SPEAKER_A, encoding="utf-8")
    (input_dir / "Bob-01.srt").write_text("This is not valid SRT", encoding="utf-8") # Invalid
    (input_dir / "Alice-02.srt").write_text(CHUNK2_SPEAKER_A, encoding="utf-8") # Valid chunk 2

    caplog.set_level(logging.ERROR)
    merge_srt_by_chunk(srt_folder=input_dir, output_dir=output_dir,
                       output_filename=output_filename, config=DEFAULT_MERGE_CONFIG)

    # Check that an error was logged for the invalid file
    assert "Failed to parse SRT file" in caplog.text
    assert "Bob-01.srt" in caplog.text

    # Check that the output file contains subtitles from the valid files
    assert output_file.exists()
    output_content = output_file.read_text(encoding="utf-8")
    output_subs = list(srt.parse(output_content))

    # Expect 1 sub from Alice C1 + 2 subs from Alice C2 = 3 subs
    assert len(output_subs) == 3
    assert output_subs[0].content == "[Alice] Alice chunk 1 line 1."
    assert output_subs[0].start == timedelta(seconds=1.5) # Chunk 1, no offset yet
    assert output_subs[0].end == timedelta(seconds=3.0)

    # Offset for chunk 2 depends only on valid subs from chunk 1
    offset_s = 3.0 + DEFAULT_MERGE_CONFIG.chunk_gap_s # Max end of Alice C1 + gap
    assert output_subs[1].content == "[Alice] Alice chunk 2 line 1."
    assert output_subs[1].start == pytest.approx(timedelta(seconds=offset_s + 0.5), abs=timedelta(milliseconds=1))
    assert output_subs[2].content == "[Alice] Alice chunk 2 line 2."
    assert output_subs[2].start == pytest.approx(timedelta(seconds=offset_s + 3.0), abs=timedelta(milliseconds=1))

def test_merge_srt_input_not_found(tmp_path: Path):
    """Test FileNotFoundError if input folder doesn't exist."""
    with pytest.raises(FileNotFoundError):
        merge_srt_by_chunk("non_existent_folder", tmp_path, "output.srt")

def test_merge_srt_invalid_output_filename(tmp_path: Path):
    """Test ValueError if output filename is invalid."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="Output filename must.*.srt"):
        merge_srt_by_chunk(input_dir, tmp_path, "output.txt") # Wrong extension
    with pytest.raises(ValueError, match="Output filename must.*.srt"):
        merge_srt_by_chunk(input_dir, tmp_path, "") # Empty filename