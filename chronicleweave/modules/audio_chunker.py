# chronicleweave/modules/audio_chunker.py

import logging
import numpy as np
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple
import subprocess
from pydub import AudioSegment, silence
import os
from tqdm import tqdm
from dataclasses import dataclass, field

# --- Module Logger ---
log = logging.getLogger(__name__)

# --- Configuration ---
@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for the audio chunking process."""
    # Silence Detection
    silence_threshold_dbfs: int = -40
    min_silence_duration_ms: int = 1000
    # Segment Duration Constraints (seconds)
    min_chunk_duration_s: int = field(default=9 * 60)
    max_chunk_duration_s: int = field(default=15 * 60)
    # Silence detection analyses the merged file in windows of this length (milliseconds)
    incremental_chunk_size_ms: int = 60000 # 1 minute windows

    def __post_init__(self):
        # Validation
        if self.min_silence_duration_ms <= 0:
            raise ValueError("min_silence_duration_ms must be positive")
        if self.min_chunk_duration_s <= 0:
            raise ValueError("min_chunk_duration_s must be positive")
        if self.max_chunk_duration_s <= self.min_chunk_duration_s:
            raise ValueError("max_chunk_duration_s must be greater than min_chunk_duration_s")
        if self.incremental_chunk_size_ms <= 0:
             raise ValueError("incremental_chunk_size_ms must be positive")

# Create a default config instance
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()

# --- Helper Functions ---

def get_audio_files_from_folder(folder: Path, extension: str = "flac") -> List[Path]:
    """
    Retrieve all audio files with the specified extension from the folder.

    Args:
        folder: Path object representing the directory to search.
        extension: The file extension to look for (without the dot).

    Returns:
        A list of Path objects for the found audio files.

    Raises:
        FileNotFoundError: If the folder does not exist or is not a directory.
    """
    if not folder.is_dir():
        log.error(f"Input folder not found or is not a directory: {folder}")
        raise FileNotFoundError(f"Folder not found: {folder}")

    log.debug(f"Scanning folder '{folder}' for '*.{extension}' files...")
    audio_files = list(folder.glob(f"*.{extension}"))
    log.info(f"Found {len(audio_files)} '*.{extension}' files in '{folder}'.")
    return audio_files


def _run_ffmpeg(cmd: List[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg failed while {what} (exit {proc.returncode}): {tail}")


def probe_duration_s(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        value = float(proc.stdout.strip()) if proc.returncode == 0 else float("nan")
    except ValueError:  # ffprobe prints "N/A" for non-audio files and still exits 0
        value = float("nan")
    if not (value > 0 and value != float("inf")):
        raise RuntimeError(f"ffprobe could not read a valid duration for {path}: {proc.stdout.strip()} {proc.stderr.strip()}")
    return value


def build_merge_command(tracks: Sequence[Path], output: Path) -> List[str]:
    cmd: List[str] = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for track in tracks:
        cmd += ["-i", str(track)]
    cmd += [
        "-filter_complex", f"amix=inputs={len(tracks)}:duration=longest:normalize=0",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(output),
    ]
    return cmd


def merge_tracks(tracks: Sequence[Path], output: Path) -> None:
    if not tracks:
        raise ValueError("merge_tracks needs at least one track")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Merging {len(tracks)} tracks with ffmpeg into {output} (16 kHz mono, summed)...")
    _run_ffmpeg(build_merge_command(tracks, output), f"merging tracks into {output.name}")


# --- Silence Detection ---

PCM_FRAME_RATE = 16000  # the merged file's format (build_merge_command): 16 kHz mono s16
PCM_SAMPLE_WIDTH = 2
_PCM_BYTES_PER_MS = PCM_FRAME_RATE * PCM_SAMPLE_WIDTH // 1000


def build_pcm_decode_command(audio_file: Path) -> List[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_file),
            "-f", "s16le", "-ac", "1", "-ar", str(PCM_FRAME_RATE), "-"]


def _read_exact(stream, size: int) -> bytes:
    parts, remaining = [], size
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            break
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def _stderr_tail(errfile) -> str:
    errfile.seek(0)
    lines = errfile.read().decode("utf-8", errors="replace").strip().splitlines()
    return "\n".join(lines[-5:])


def iter_pcm_windows(audio_file: Path, total_length_ms: int, window_ms: int) -> Iterator[Tuple[int, AudioSegment]]:
    """Decode `audio_file` ONCE with one ffmpeg process and yield (start_ms, window) for consecutive
    windows of `window_ms` (the last one shorter) covering exactly `total_length_ms`. The windows are the
    ones the previous detector loaded with one pydub decode per window (from_file(start_second, duration)).

    Raises RuntimeError on a read error, a non-zero ffmpeg exit, or a stream shorter than announced."""
    with tempfile.TemporaryFile() as errfile:
        proc = subprocess.Popen(build_pcm_decode_command(audio_file), stdout=subprocess.PIPE, stderr=errfile)
        assert proc.stdout is not None
        try:
            for start_ms in range(0, total_length_ms, window_ms):
                load_ms = min(window_ms, total_length_ms - start_ms)
                wanted = load_ms * _PCM_BYTES_PER_MS
                try:
                    data = _read_exact(proc.stdout, wanted)
                except OSError as e:
                    raise RuntimeError(f"Could not read the decoded audio of {audio_file} at {start_ms} ms: {e}") from e
                if len(data) < wanted:
                    proc.stdout.close()
                    code = proc.wait()
                    if code != 0:
                        raise RuntimeError(f"ffmpeg failed decoding {audio_file} for silence detection "
                                           f"(exit {code}): {_stderr_tail(errfile)}")
                    raise RuntimeError(f"Decoding {audio_file} for silence detection ended early at "
                                       f"{start_ms + len(data) // _PCM_BYTES_PER_MS} ms of {total_length_ms} ms: "
                                       f"{_stderr_tail(errfile)}")
                yield start_ms, AudioSegment(data=data, sample_width=PCM_SAMPLE_WIDTH,
                                             frame_rate=PCM_FRAME_RATE, channels=1)
            while proc.stdout.read(1 << 16):  # the few samples past the last whole millisecond
                pass
            proc.stdout.close()
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed decoding {audio_file} for silence detection "
                                   f"(exit {code}): {_stderr_tail(errfile)}")
        finally:
            if proc.poll() is None:  # stopped early (error or abandoned generator): do not leave ffmpeg behind
                proc.kill()
            proc.stdout.close()
            proc.wait()


def detect_silence_windowed(audio_file: Path, config: ChunkingConfig, duration_s: float) -> List[Tuple[float, float]]:
    """Detect silences window by window (60 s by default) on one streamed decode of the merged file.
    Each window is analysed on its own, then overlapping or touching ranges are merged."""
    log.info("Detecting silences (one decode, windowed analysis)...")
    try:
        total_length_ms = int(duration_s * 1000)
        log.info(f"Audio duration: {duration_s:.2f}s ({total_length_ms}ms)")

        if total_length_ms == 0:
            log.warning(f"Audio file {audio_file} is empty or has zero duration.")
            return []

        all_silent_ranges_s: List[Tuple[float, float]] = []
        chunk_size_ms = config.incremental_chunk_size_ms
        n_windows = -(-total_length_ms // chunk_size_ms)
        log.info(f"Processing in {chunk_size_ms}ms windows...")
        disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG

        # closing(): if detection fails mid-way, ffmpeg is stopped now, not at garbage collection
        with closing(iter_pcm_windows(audio_file, total_length_ms, chunk_size_ms)) as windows:
            for chunk_start_ms, chunk in tqdm(windows, total=n_windows, desc="Analyzing chunks for silence",
                                              disable=disable_tqdm):
                log.debug(f"Detecting silence in window at {chunk_start_ms}ms ({len(chunk)}ms)...")
                chunk_silences_ms = silence.detect_silence(
                    chunk,
                    min_silence_len=config.min_silence_duration_ms,
                    silence_thresh=config.silence_threshold_dbfs
                )
                # Timestamps relative to the whole file, in seconds. Windows do not overlap, so a silence
                # crossing a window edge is found as two parts (each at least min_silence_duration_ms
                # long) that the merge below joins.
                for start_in_chunk_ms, end_in_chunk_ms in chunk_silences_ms:
                    all_silent_ranges_s.append(((chunk_start_ms + start_in_chunk_ms) / 1000.0,
                                                (chunk_start_ms + end_in_chunk_ms) / 1000.0))

        log.info(f"Detected {len(all_silent_ranges_s)} potential silent ranges (boundary accuracy may vary).")
        if not all_silent_ranges_s: return []
        # Sort and merge overlapping/adjacent intervals
        all_silent_ranges_s.sort()
        merged_ranges = [list(all_silent_ranges_s[0])] # Start with the first range as a mutable list
        for next_start, next_end in all_silent_ranges_s[1:]:
            last_start, last_end = merged_ranges[-1]
            # Merge if next start is before or exactly at the last end
            if next_start <= last_end:
                 merged_ranges[-1][1] = max(last_end, next_end) # Extend the end of the last range
            else:
                 merged_ranges.append([next_start, next_end]) # Start a new range
        # Convert back to tuples
        final_merged_ranges = [tuple(r) for r in merged_ranges]
        log.info(f"Merged into {len(final_merged_ranges)} final silent ranges.")
        return final_merged_ranges

    except RuntimeError:
        raise  # already says what failed; do not re-wrap
    except Exception as e:
        log.exception(f"Error during silence detection: {e}")
        raise RuntimeError(f"Silence detection failed: {e}") from e

# --- Cut Point Logic ---

def determine_cut_points(
    silence_timestamps: List[Tuple[float, float]],
    audio_duration_s: float,
    config: ChunkingConfig
    ) -> List[float]:
    """
    Select best cut points based on detected silences and duration constraints.

    Args:
        silence_timestamps: List of (start_sec, end_sec) tuples for silences.
        audio_duration_s: Total duration of the audio in seconds.
        config: Chunking configuration object.

    Returns:
        List of cut point timestamps in seconds, including 0.0 and audio_duration_s.
    """
    log.info("Determining optimal cut points...")
    if not silence_timestamps:
        log.warning("No silence timestamps provided. Cannot determine optimal cut points based on silence.")
        # Fallback: Cut based purely on max duration? Or return just start/end?
        # For now, let's create chunks based on max_duration if possible.
        num_chunks = max(1, int(np.ceil(audio_duration_s / config.max_chunk_duration_s)))
        cut_points = [i * audio_duration_s / num_chunks for i in range(num_chunks + 1)]
        log.info(f"Generating {len(cut_points)-1} fallback cuts based on max duration ({config.max_chunk_duration_s}s).")
        return cut_points

    # Sort silences just in case
    silence_timestamps.sort()

    cut_points = [0.0] # Start with the beginning of the audio
    last_cut_s = 0.0

    # Iterate through silences to find suitable cut points
    for silence_start_s, silence_end_s in silence_timestamps:
        silence_mid_point_s = (silence_start_s + silence_end_s) / 2.0
        current_chunk_duration = silence_mid_point_s - last_cut_s

        # Check if the current chunk duration meets the minimum requirement
        if current_chunk_duration >= config.min_chunk_duration_s:
            # If it also meets the maximum constraint, cut at the middle of the silence
            if current_chunk_duration <= config.max_chunk_duration_s:
                log.debug(f"Adding cut point at {silence_mid_point_s:.2f}s (mid-silence). Chunk duration: {current_chunk_duration:.2f}s")
                cut_points.append(silence_mid_point_s)
                last_cut_s = silence_mid_point_s
            else:
                # Chunk is too long, need to cut earlier within the allowed max duration.
                # Find the latest possible cut point within the max duration limit.
                # We aim for a point *before* the current silence if possible.
                ideal_cut_time = last_cut_s + config.max_chunk_duration_s

                # Find the *last* silence that ends *before* this ideal cut time.
                # This prioritizes cutting within silences if possible.
                best_earlier_cut = None
                for prev_start, prev_end in reversed(silence_timestamps):
                    if prev_end < ideal_cut_time and prev_end > last_cut_s: # Find suitable silence ending before ideal time
                         best_earlier_cut = (prev_start + prev_end) / 2.0 # Cut in middle of that silence
                         break

                if best_earlier_cut:
                    actual_cut = best_earlier_cut
                    log.debug(f"Chunk too long ({current_chunk_duration:.2f}s > {config.max_chunk_duration_s}s). Cutting earlier at {actual_cut:.2f}s (mid-silence).")
                else:
                    # No suitable earlier silence found, force cut at max duration from last cut
                    actual_cut = ideal_cut_time
                    log.debug(f"Chunk too long ({current_chunk_duration:.2f}s > {config.max_chunk_duration_s}s). No suitable silence found. Cutting at max duration: {actual_cut:.2f}s.")

                # Ensure we don't cut beyond the audio duration
                actual_cut = min(actual_cut, audio_duration_s)
                # Add only if it's meaningfully after the last cut
                if actual_cut > last_cut_s + 1.0: # Avoid tiny segments caused by forced cuts
                    cut_points.append(actual_cut)
                    last_cut_s = actual_cut

    # Ensure the final segment isn't too small after the last cut
    # If the remaining part is very short, merge it with the previous chunk by removing the last cut point.
    if len(cut_points) > 1 and (audio_duration_s - last_cut_s) < (config.min_chunk_duration_s / 2.0): # Heuristic: less than half min duration
        log.warning(f"Final segment is very short ({audio_duration_s - last_cut_s:.2f}s). Merging with previous chunk by removing last cut point at {last_cut_s:.2f}s.")
        last_cut_s = cut_points.pop() # Remove last cut
        # Update last_cut_s to the new last cut point for the final check
        last_cut_s = cut_points[-1] if cut_points else 0.0


    # Always add the end of the audio as the final cut point, if not already there
    if not np.isclose(last_cut_s, audio_duration_s):
         log.debug(f"Adding final cut point at audio end: {audio_duration_s:.2f}s")
         cut_points.append(audio_duration_s)

    # Remove potential duplicate points (e.g., if audio_duration was added when last_cut was already very close)
    unique_cut_points = sorted(list(set(round(cp, 3) for cp in cut_points))) # Round to avoid float precision issues

    log.info(f"Determined {len(unique_cut_points)-1} cut segments.")
    return unique_cut_points


def write_cut_points(cut_points: List[float], output_file: Path) -> None:
    """Write cut points (in seconds) to a text file."""
    log.info(f"Writing {len(cut_points)} cut points to {output_file}...")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Audio Cut Points (Total Chunks: {len(cut_points)-1})\n")
            f.write("# Format: Chunk Index, Start Time (seconds)\n")
            for i, timestamp_s in enumerate(cut_points):
                # Write chunk start time. Chunk 'i' starts at cut_points[i].
                # The file effectively lists the start time of each chunk.
                # Chunk 0 starts at 0.0, Chunk 1 starts at cut_points[1], etc.
                f.write(f"Chunk {i}, {timestamp_s:.3f}\n")
        log.info("Cut points saved successfully.")
    except IOError as e:
        log.exception(f"Failed to write cut points file: {output_file}")
        raise IOError(f"Failed to write cut points to {output_file}") from e

# --- Splitting ---

def build_slice_command(track: Path, start_s: float, duration_s: float, output: Path) -> List[str]:
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(track),
        "-c:a", "flac", str(output),
    ]


def split_tracks(speaker_files: Sequence[Path], cut_points_s: Sequence[float], output_dir: Path) -> List[Path]:
    if len(cut_points_s) < 2:
        raise ValueError("Need at least two cut points (start and end) to split tracks.")
    if not speaker_files:
        raise FileNotFoundError("No speaker files to split.")
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []
    for track in speaker_files:
        track_duration = probe_duration_s(track)
        for i in range(len(cut_points_s) - 1):
            start, end = cut_points_s[i], min(cut_points_s[i + 1], track_duration)
            if end - start <= 0.001:
                log.info(f"{track.name}: chunk {i + 1:02d} starts after the track ends; not produced.")
                continue
            out = output_dir / f"{track.stem}-{i + 1:02d}.flac"
            _run_ffmpeg(build_slice_command(track, start, end - start, out), f"cutting {out.name}")
            produced.append(out)
    log.info(f"Split {len(speaker_files)} tracks into {len(produced)} chunk files in {output_dir}.")
    return produced


def chunk_audio(
    input_folder: Path,
    output_dir: Path,
    merged_file: Path,
    cut_points_file: Path,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> List[Path]:
    speaker_files = sorted(get_audio_files_from_folder(input_folder))
    if not speaker_files:
        raise FileNotFoundError(f"No FLAC tracks in {input_folder}; nothing to chunk.")
    merge_tracks(speaker_files, merged_file)
    duration_s = probe_duration_s(merged_file)
    log.info(f"Merged audio duration: {duration_s:.2f} s")
    silences = detect_silence_windowed(merged_file, config, duration_s)
    cut_points = determine_cut_points(silences, duration_s, config)
    if len(cut_points) < 2:
        raise RuntimeError("No usable cut points were determined.")
    write_cut_points(cut_points, cut_points_file)
    return split_tracks(speaker_files, cut_points, output_dir)
