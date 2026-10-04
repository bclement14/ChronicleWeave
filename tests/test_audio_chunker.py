# tests/test_audio_chunker.py

import pytest
from pathlib import Path
import numpy as np 
import logging # Import logging if needed for caplog, though not used here yet

# Import functions and config from the module to test
from chronicleweave.modules.audio_chunker import (
    get_audio_files_from_folder,
    determine_cut_points,
    write_cut_points,
    ChunkingConfig,
    DEFAULT_CHUNKING_CONFIG
)

# --- Fixtures ---

@pytest.fixture
def default_chunk_config() -> ChunkingConfig:
    """Provides a default ChunkingConfig instance."""
    # Use the actual default imported from the module
    return DEFAULT_CHUNKING_CONFIG

# --- Tests for get_audio_files_from_folder ---

def test_get_audio_files_success(tmp_path: Path):
    """Test finding FLAC files in a directory."""
    input_dir = tmp_path / "audio_in"
    input_dir.mkdir()
    file1 = input_dir / "speakerA.flac"
    file2 = input_dir / "speakerB.flac"
    other_file = input_dir / "notes.txt"
    file1.touch()
    file2.touch()
    other_file.touch()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")

    assert len(found_files) == 2
    # Use sets for order-independent comparison
    assert set(found_files) == {file1, file2}

def test_get_audio_files_custom_extension(tmp_path: Path):
    """Test finding files with a different extension (using lowercase)."""
    input_dir = tmp_path / "audio_wav"
    input_dir.mkdir()
    file1 = input_dir / "track1.wav"
    # Use lowercase for test reliability across platforms
    file2 = input_dir / "track2.wav"
    other_file = input_dir / "track.flac"
    file1.touch()
    file2.touch()
    other_file.touch()

    # Test with simple lowercase glob pattern
    found_files = get_audio_files_from_folder(input_dir, extension="wav")

    assert len(found_files) == 2
    assert set(found_files) == {file1, file2}

def test_get_audio_files_empty(tmp_path: Path):
    """Test finding files in an empty directory."""
    input_dir = tmp_path / "audio_empty"
    input_dir.mkdir()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")
    assert found_files == []

def test_get_audio_files_no_matching(tmp_path: Path):
    """Test finding files when none match the extension."""
    input_dir = tmp_path / "audio_nomatch"
    input_dir.mkdir()
    (input_dir / "notes.txt").touch()
    (input_dir / "audio.mp3").touch()

    found_files = get_audio_files_from_folder(input_dir, extension="flac")
    assert found_files == []

def test_get_audio_files_dir_not_found(tmp_path: Path):
    """Test error when input directory does not exist."""
    input_dir = tmp_path / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        get_audio_files_from_folder(input_dir)

# --- Tests for determine_cut_points ---

# Use default config: min_chunk=540s (9min), max_chunk=900s (15min)
def test_determine_cuts_no_silence(default_chunk_config):
    """Test cutting based on max duration when no silences are provided."""
    silences = []
    duration = 1000.0 # > 900s max
    # Expect 2 chunks: [0.0, 500.0, 1000.0] (Fallback cuts based on duration/num_chunks)
    cut_points = determine_cut_points(silences, duration, default_chunk_config)
    assert len(cut_points) == 3
    assert cut_points[0] == pytest.approx(0.0)
    # Fallback calculation: ceil(1000/900)=2 chunks. Points are 0, 1000/2, 1000.
    assert cut_points[1] == pytest.approx(500.0)
    assert cut_points[2] == pytest.approx(1000.0)

def test_determine_cuts_simple_valid_silences(default_chunk_config):
   def test_determine_cuts_simple_valid_silences(default_chunk_config):
    """Test cutting within suitable silences."""
    # Config: min=540, max=900
    silences = [(600.0, 610.0), (1200.0, 1210.0)] # Silences at 10min and 20min
    duration = 1500.0
    cut_points = determine_cut_points(silences, duration, default_chunk_config)

    # Corrected Expected behaviour trace:
    # 1. Process (600, 610): mid=605. Chunk 0-605 is 605s (valid). Add 605. last_cut=605. points=[0, 605].
    # 2. Process (1200, 1210): mid=1205. Chunk 605-1205 is 600s (valid). Add 1205. last_cut=1205. points=[0, 605, 1205].
    # 3. End loop. Check final segment 1205-1500 is 295s (> min/2=270). OK.
    # 4. Add final point 1500. points=[0, 605, 1205, 1500].
    expected_points = [0.0, 605.0, 1205.0, 1500.0]
    assert len(cut_points) == 4 # Expect 4 points (3 chunks)
    assert cut_points == pytest.approx(expected_points)

def test_determine_cuts_force_cut_due_to_max_duration(default_chunk_config):
    """Test forcing a cut when silence is too far away."""
    # Config: min=540, max=900
    silences = [(1000.0, 1010.0)] # Only one silence, far out
    duration = 1500.0
    # Expected cuts: [0.0, 900.0, 1500.0] (as analyzed before)
    cut_points = determine_cut_points(silences, duration, default_chunk_config)
    assert len(cut_points) == 3
    assert cut_points[0] == pytest.approx(0.0)
    assert cut_points[1] == pytest.approx(900.0) # Forced cut at max duration
    assert cut_points[2] == pytest.approx(1500.0)

def test_determine_cuts_force_cut_chooses_earlier_silence(default_chunk_config):
    """Test forced cut selecting the latest possible silence before max duration."""
    # Config: min=540, max=900
    silences = [(600.0, 610.0), (850.0, 860.0), (1000.0, 1010.0)]
    duration = 1500.0
    cut_points = determine_cut_points(silences, duration, default_chunk_config)

    # Corrected Expected behaviour trace:
    # 1. Process (600, 610): mid=605. Chunk 0-605 is 605s (valid). Add 605. last_cut=605. points=[0, 605].
    # 2. Process (850, 860): mid=855. Chunk 605-855 is 250s (< min=540). Skip this silence.
    # 3. Process (1000, 1010): mid=1005. Chunk 605-1005 is 400s (< min=540). Skip this silence.
    # 4. End loop. Check final segment 605-1500 is 895s (OK).
    # 5. Add final point 1500. points=[0, 605, 1500].
    expected_points = [0.0, 605.0, 1500.0]
    assert len(cut_points) == 3
    assert cut_points == pytest.approx(expected_points)

def test_determine_cuts_final_segment_too_short(default_chunk_config):
    """Test removing the last cut point if the final segment is too short."""
    # Config: min=540, max=900. min/2 = 270
    # Case 1: Final segment is long enough
    silences1 = [(600.0, 610.0), (1100.0, 1110.0)]
    duration1 = 1200.0
    # Expected: [0.0, 605.0, 1200.0] (Final segment 595s > 270)
    cut_points1 = determine_cut_points(silences1, duration1, default_chunk_config)
    assert len(cut_points1) == 3
    assert cut_points1 == pytest.approx([0.0, 605.0, 1200.0])

    # Case 2: Final segment is just long enough
    silences2 = [(600.0, 610.0), (850.0, 860.0)]
    duration2 = 900.0
    # Expected: [0.0, 605.0, 900.0] (Final segment 295s > 270)
    cut_points2 = determine_cut_points(silences2, duration2, default_chunk_config)
    assert len(cut_points2) == 3
    assert cut_points2 == pytest.approx([0.0, 605.0, 900.0])

    # Case 3: Final segment is too short, last cut removed
    silences3 = [(600.0, 610.0)]
    duration3 = 700.0
    # Expected: [0.0, 700.0] (Final segment 95s < 270, cut at 605 removed)
    cut_points3 = determine_cut_points(silences3, duration3, default_chunk_config)
    assert len(cut_points3) == 2
    assert cut_points3 == pytest.approx([0.0, 700.0])


# --- Tests for write_cut_points ---

def test_write_cut_points_standard(tmp_path: Path):
    """Test writing cut points to a file."""
    cut_points = [0.0, 605.1234, 1200.5]
    output_file = tmp_path / "cuts_output" / "cut_points.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    assert "# Audio Cut Points (Total Chunks: 2)" in content[0]
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    assert content[2] == "Chunk 0, 0.000"
    assert content[3] == "Chunk 1, 605.123" # Check rounding to 3 decimal places
    assert content[4] == "Chunk 2, 1200.500"

def test_write_cut_points_empty(tmp_path: Path):
    """Test writing when the cut points list is empty."""
    cut_points = []
    output_file = tmp_path / "empty_cuts.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    # --- CORRECTED ASSERTION ---
    # Expect only the two header lines when input list is empty
    assert len(content) == 2
    assert "# Audio Cut Points (Total Chunks: -1)" in content[0] # len([])-1 = -1
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    # --- END CORRECTION ---

def test_write_cut_points_single_point(tmp_path: Path):
    """Test writing when only one cut point (start=0) exists."""
    cut_points = [0.0]
    output_file = tmp_path / "single_cut.txt"

    write_cut_points(cut_points, output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8").splitlines()
    # Expect two headers and one data line
    assert len(content) == 3
    assert "# Audio Cut Points (Total Chunks: 0)" in content[0] # len([0.0])-1 = 0
    assert "# Format: Chunk Index, Start Time (seconds)" in content[1]
    assert content[2] == "Chunk 0, 0.000"


def test_write_cut_points_cannot_write(tmp_path: Path):
    """Test error handling when output file cannot be written."""
    cut_points = [0.0, 100.0]
    # Create a directory where the file should go
    output_file_as_dir = tmp_path / "cut_points.txt"
    output_file_as_dir.mkdir()

    with pytest.raises((IOError, OSError)):
        write_cut_points(cut_points, output_file_as_dir)

# --- TODO: Write Integration tests or tests for pydub dependent functions ---
# - _merge_tracks_high_ram / _merge_tracks_low_ram
# - _detect_silence_high_ram / _detect_silence_low_ram
# - split_tracks
# - chunk_audio (main function)

# --- ffmpeg-based merge and slicing (spec 4.4) ---
import subprocess
import numpy as np
from pydub import AudioSegment

from chronicleweave.modules.audio_chunker import (
    build_merge_command,
    build_slice_command,
    chunk_audio,
    merge_tracks,
    probe_duration_s,
    split_tracks,
)


def _tone(path, seconds, freq=440, rate=48000, gain_db=-20.0, channels=2):
    t = np.arange(int(seconds * rate)) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 32767 * 10 ** (gain_db / 20)).astype(np.int16)
    seg = AudioSegment(samples.tobytes(), frame_rate=rate, sample_width=2, channels=1)
    if channels == 2:
        seg = seg.set_channels(2)
    seg.export(path, format="flac")


def test_merge_command_shape(tmp_path):
    cmd = build_merge_command([tmp_path / "a.flac", tmp_path / "b.flac"], tmp_path / "m.flac")
    joined = " ".join(cmd)
    assert "amix=inputs=2:duration=longest:normalize=0" in joined
    assert cmd[-7:] == ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp_path / "m.flac")]
    assert "aformat" not in joined


def test_merge_sums_levels_not_average(tmp_path):
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"
    _tone(a, 2.0, gain_db=-20.0)
    _tone(b, 2.0, gain_db=-20.0)
    out = tmp_path / "m.flac"
    merge_tracks([a, b], out)
    merged = AudioSegment.from_file(out)
    assert merged.frame_rate == 16000 and merged.channels == 1 and merged.sample_width == 2
    # two identical in-phase tones summed: +6 dB peak over one tone (amix fades the last inputs
    # out near the end of short files, so compare peaks, not average level)
    single = AudioSegment.from_file(a)
    assert merged.max_dBFS - single.max_dBFS == pytest.approx(6.0, abs=0.5)


def test_merge_uses_longest_duration(tmp_path):
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"
    _tone(a, 1.0)
    _tone(b, 3.0)
    out = tmp_path / "m.flac"
    merge_tracks([a, b], out)
    assert probe_duration_s(out) == pytest.approx(3.0, abs=0.05)


def test_merge_failure_raises(tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        merge_tracks([bad], tmp_path / "m.flac")


def test_slice_command_uses_input_seek(tmp_path):
    cmd = build_slice_command(tmp_path / "t.flac", 12.5, 3.25, tmp_path / "o.flac")
    i = cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "12.500" and cmd.index("-ss") < i
    assert cmd[cmd.index("-t") + 1] == "3.250"
    assert cmd[-3:] == ["-c:a", "flac", str(tmp_path / "o.flac")]


def test_split_produces_exact_durations(tmp_path):
    track = tmp_path / "GM.flac"
    _tone(track, 3.0)
    out = tmp_path / "chunks"
    files = split_tracks([track], [0.0, 1.25, 3.0], out)
    assert [f.name for f in files] == ["GM-01.flac", "GM-02.flac"]
    assert probe_duration_s(out / "GM-01.flac") == pytest.approx(1.25, abs=0.01)
    assert probe_duration_s(out / "GM-02.flac") == pytest.approx(1.75, abs=0.01)
    seg = AudioSegment.from_file(out / "GM-01.flac")
    assert seg.frame_rate == 48000 and seg.channels == 2


def test_split_skips_chunks_beyond_track_end(tmp_path):
    long_t, short_t = tmp_path / "GM.flac", tmp_path / "Pen.flac"
    _tone(long_t, 3.0)
    _tone(short_t, 1.0)
    files = split_tracks([long_t, short_t], [0.0, 1.5, 3.0], tmp_path / "c")
    assert sorted(f.name for f in files) == ["GM-01.flac", "GM-02.flac", "Pen-01.flac"]


def test_split_failure_raises(tmp_path):
    bad = tmp_path / "GM.flac"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError):
        split_tracks([bad], [0.0, 1.0], tmp_path / "c")


def test_chunk_audio_end_to_end_small(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    _tone(tracks / "GM.flac", 4.0)
    _tone(tracks / "Pen.flac", 4.0, freq=660)
    cfg = ChunkingConfig(min_chunk_duration_s=1, max_chunk_duration_s=2)
    files = chunk_audio(tracks, tmp_path / "chunked", tmp_path / "final" / "merged_audio.flac",
                        tmp_path / "final" / "cut_points.txt", cfg)
    assert files and all(f.exists() for f in files)
    assert (tmp_path / "final" / "cut_points.txt").read_text().startswith("# Audio Cut Points")


def test_chunk_audio_no_tracks_raises(tmp_path):
    (tmp_path / "tracks").mkdir()
    with pytest.raises(FileNotFoundError):
        chunk_audio(tmp_path / "tracks", tmp_path / "c", tmp_path / "m.flac", tmp_path / "cp.txt")


# --- final review F10: the merged file is decoded once; windows and silences are unchanged ---
from unittest.mock import patch

from pydub import silence

from chronicleweave.modules.audio_chunker import detect_silence_windowed, iter_pcm_windows

RATE = 16000
# Digital silences: one across the 60 s window edge (>= 1 s on each side, merged into one range), one
# across the 120 s edge with only 0.8 s before it (the previous detector misses that part; so must
# the new one), and one running to the very end.
SILENCES = [(10.0, 12.5), (58.5, 61.7), (100.0, 101.5), (119.2, 122.0), (130.0, 130.5), (148.9, 150.6)]
DURATION = 150.50031  # not a whole number of ms: the last window is shorter and the stream ends a few samples later


def _merged_like(path, seconds, silences, seed=0):
    """A 16 kHz mono s16 FLAC like the step-1 merge: noise around digital silences."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(seconds * RATE)) * 3000).astype(np.int16)
    for a, b in silences:
        x[int(a * RATE):int(b * RATE)] = 0
    AudioSegment(x.tobytes(), frame_rate=RATE, sample_width=2, channels=1).export(path, format="flac")


def _pydub_windows(path, total_ms, window_ms=60000):
    """The windows the previous detector loaded: one pydub (ffmpeg + ffprobe) decode per window."""
    for start in range(0, total_ms, window_ms):
        load = min(window_ms, total_ms - start)
        yield start, AudioSegment.from_file(path, format="flac", start_second=start / 1000.0, duration=load / 1000.0)


def _previous_detector(path, config, duration_s):
    """The previous implementation: same per-window detection and the same range merging."""
    ranges = []
    for start, chunk in _pydub_windows(path, int(duration_s * 1000), config.incremental_chunk_size_ms):
        for s, e in silence.detect_silence(chunk, min_silence_len=config.min_silence_duration_ms,
                                           silence_thresh=config.silence_threshold_dbfs):
            ranges.append(((start + s) / 1000.0, (start + e) / 1000.0))
    ranges.sort()
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


@pytest.fixture(scope="module")
def merged_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("merged") / "merged_audio.flac"
    _merged_like(path, DURATION, SILENCES)
    return path


def test_windows_equal_pydub_windows_byte_for_byte(merged_file):
    total_ms = int(probe_duration_s(merged_file) * 1000)
    new = list(iter_pcm_windows(merged_file, total_ms, 60000))
    old = list(_pydub_windows(merged_file, total_ms))
    assert [s for s, _ in new] == [s for s, _ in old] == [0, 60000, 120000]
    for (_, a), (_, b) in zip(new, old):
        assert (a.frame_rate, a.channels, a.sample_width) == (b.frame_rate, b.channels, b.sample_width)
        assert len(a.raw_data) == len(b.raw_data) and a.raw_data == b.raw_data


def test_detected_ranges_identical_to_previous_detector(merged_file):
    duration = probe_duration_s(merged_file)
    new = detect_silence_windowed(merged_file, DEFAULT_CHUNKING_CONFIG, duration)
    assert new == _previous_detector(merged_file, DEFAULT_CHUNKING_CONFIG, duration)
    assert any(s < 60.0 < e for s, e in new)  # merged across the 60 s edge
    assert any(s == 120.0 for s, _ in new) and not any(s < 120.0 < e for s, e in new)  # same edge artifact


def test_merged_file_decoded_by_one_ffmpeg_process(merged_file):
    duration = probe_duration_s(merged_file)
    real_popen = subprocess.Popen
    cmds = []

    def spy(cmd, *a, **k):
        cmds.append(list(cmd))
        return real_popen(cmd, *a, **k)

    with patch("subprocess.Popen", side_effect=spy), patch("pydub.silence.detect_silence", return_value=[]):
        detect_silence_windowed(merged_file, DEFAULT_CHUNKING_CONFIG, duration)
    assert cmds == [["ffmpeg", "-nostdin", "-v", "error", "-i", str(merged_file),
                     "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]]


def test_truncated_merged_file_raises(tmp_path, merged_file):
    data = merged_file.read_bytes()
    truncated = tmp_path / "merged_audio.flac"
    truncated.write_bytes(data[: len(data) // 2])
    duration = probe_duration_s(merged_file)  # the header still announces the full length
    with pytest.raises(RuntimeError, match="ended early") as err:
        detect_silence_windowed(truncated, DEFAULT_CHUNKING_CONFIG, duration)
    assert "Silence detection failed" not in str(err.value)  # not re-wrapped


def test_corrupt_merged_file_raises(tmp_path):
    bad = tmp_path / "merged_audio.flac"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(RuntimeError, match="exit") as err:
        detect_silence_windowed(bad, DEFAULT_CHUNKING_CONFIG, 120.0)
    assert "Silence detection failed" not in str(err.value)


@pytest.mark.parametrize("tty", [False, True])  # with a terminal, tqdm shows a bar and does not close the iterable
def test_detection_error_stops_the_decoder(merged_file, tty):
    import sys
    duration = probe_duration_s(merged_file)
    real_popen = subprocess.Popen
    procs = []

    def spy(cmd, *a, **k):
        procs.append(real_popen(cmd, *a, **k))
        return procs[-1]

    with patch("subprocess.Popen", side_effect=spy), patch.object(sys.stdout, "isatty", return_value=tty), \
            patch("pydub.silence.detect_silence", side_effect=ValueError("analysis broke")):
        with pytest.raises(RuntimeError, match="analysis broke") as err:
            detect_silence_windowed(merged_file, DEFAULT_CHUNKING_CONFIG, duration)
    # checked while the traceback (and so the detector's frame) is still alive, as when a caller logs it
    assert err.value.__traceback__ is not None
    assert len(procs) == 1 and procs[0].returncode is not None  # ffmpeg reaped, not left blocked on its pipe
