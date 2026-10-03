# chronicleweave/modules/treat_flac_tracks.py

import logging
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from pydub.utils import mediainfo

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
    # Incremental Processing (Low RAM mode)
    # Chunk size for processing large files incrementally (milliseconds)
    incremental_chunk_size_ms: int = 60000 # 1 minute chunks for low RAM processing

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


def _merge_tracks_high_ram(speaker_files: List[Path], output_file: Path) -> None:
    """Merge multiple speaker audio files into a single track (higher RAM usage)."""
    log.info("Merging tracks (high RAM mode)...")
    if not speaker_files:
        log.warning("No speaker files provided for merging.")
        # Create an empty file? Or raise error? For now, log warning.
        # Creating an empty valid FLAC is non-trivial, maybe raise error is better.
        raise ValueError("Cannot merge empty list of speaker files.")

    try:
        log.debug(f"Loading track: {speaker_files[0]}")
        merged = AudioSegment.from_file(speaker_files[0], format="flac")

        # Overlay subsequent tracks
        # Disable tqdm if logging level is DEBUG or lower to avoid interleaving
        disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG
        for track_path in tqdm(speaker_files[1:], desc="Merging tracks", disable=disable_tqdm):
            log.debug(f"Loading and overlaying track: {track_path}")
            track = AudioSegment.from_file(track_path, format="flac")
            merged = merged.overlay(track)

        log.info(f"Exporting merged track to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        merged.export(output_file, format="flac")
        log.info(f"Merged track saved successfully.")

    except CouldntDecodeError as e:
        log.exception(f"Error decoding audio file during merge: {e}")
        raise IOError(f"Failed to decode audio file: {e}") from e
    except Exception as e:
        log.exception(f"An unexpected error occurred during high RAM merge: {e}")
        raise


def _merge_tracks_low_ram(speaker_files: List[Path], output_file: Path, chunk_size_ms: int) -> None:
    """Incrementally merge multiple audio files to reduce RAM usage."""
    log.info("Merging tracks (low RAM mode)...")
    if not speaker_files:
        log.warning("No speaker files provided for merging.")
        raise ValueError("Cannot merge empty list of speaker files.")

    try:
        # Load tracks (still requires loading metadata, potentially some data)
        # Disable tqdm if logging level is DEBUG or lower
        disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG
        log.debug("Loading track metadata...")
        tracks = [AudioSegment.from_file(file, format="flac") for file in tqdm(speaker_files, desc="Loading tracks", disable=disable_tqdm)]
        log.debug("Track metadata loaded.")

        # Determine maximum length
        max_length_ms = 0
        for track in tracks:
            max_length_ms = max(max_length_ms, len(track))
        if max_length_ms == 0:
            log.warning("All input tracks appear to be empty.")
            # Export an empty file?
            AudioSegment.silent(duration=0).export(output_file, format="flac")
            return

        log.info(f"Max track length: {max_length_ms / 1000:.2f}s. Processing in {chunk_size_ms}ms chunks.")
        merged_output = AudioSegment.silent(duration=0)

        # Process in chunks
        for i in tqdm(range(0, max_length_ms, chunk_size_ms), desc="Merging chunks", disable=disable_tqdm):
            # Create silent chunk of the correct duration for this iteration
            current_chunk_duration = min(chunk_size_ms, max_length_ms - i)
            chunk_overlay = AudioSegment.silent(duration=current_chunk_duration)

            # Overlay the corresponding segment from each track
            for track in tracks:
                if i < len(track): # Check if track extends into this chunk
                    segment = track[i : i + current_chunk_duration]
                    chunk_overlay = chunk_overlay.overlay(segment)

            merged_output += chunk_overlay # Append processed chunk

        log.info(f"Exporting incrementally merged track to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_output.export(output_file, format="flac")
        log.info(f"Incrementally merged track saved successfully.")

    except CouldntDecodeError as e:
        log.exception(f"Error decoding audio file during low RAM merge: {e}")
        raise IOError(f"Failed to decode audio file: {e}") from e
    except Exception as e:
        log.exception(f"An unexpected error occurred during low RAM merge: {e}")
        raise

# --- Silence Detection ---

def _detect_silence_high_ram(audio_file: Path, config: ChunkingConfig) -> List[Tuple[float, float]]:
    """Detect silence in an audio file (higher RAM usage)."""
    log.info("Detecting silences (high RAM mode)...")
    try:
        log.debug(f"Loading audio file: {audio_file}")
        audio = AudioSegment.from_file(audio_file, format="flac")
        log.debug("Audio loaded, detecting silence...")
        silent_ranges_ms = silence.detect_silence(
            audio,
            min_silence_len=config.min_silence_duration_ms,
            silence_thresh=config.silence_threshold_dbfs
        )
        # Convert ms to seconds
        silent_ranges_s = [(start / 1000, end / 1000) for start, end in silent_ranges_ms]
        log.info(f"Detected {len(silent_ranges_s)} silent ranges.")
        return silent_ranges_s
    except CouldntDecodeError as e:
        log.exception(f"Could not decode audio file for silence detection: {audio_file}")
        raise IOError(f"Failed to decode audio file: {e}") from e
    except Exception as e:
        log.exception(f"Error during high RAM silence detection: {e}")
        raise RuntimeError("Silence detection failed") from e


def _detect_silence_low_ram(audio_file: Path, config: ChunkingConfig) -> List[Tuple[float, float]]:
    """Detect silence incrementally using partial loading (improved RAM usage)."""
    log.info("Detecting silences (low RAM mode - partial loading)...")
    try:
        # --- Get duration using ffprobe via pydub.utils (lighter than loading) ---
        log.debug(f"Getting audio duration for {audio_file} using mediainfo...")
        info = mediainfo(str(audio_file)) # mediainfo expects string path
        total_duration_s = float(info['duration'])
        total_length_ms = int(total_duration_s * 1000)
        log.info(f"Audio duration: {total_duration_s:.2f}s ({total_length_ms}ms)")
        # --- End Duration Check ---

        if total_length_ms == 0:
            log.warning(f"Audio file {audio_file} is empty or has zero duration.")
            return []

        all_silent_ranges_s: List[Tuple[float, float]] = []
        chunk_size_ms = config.incremental_chunk_size_ms
        # Overlap chunks slightly to catch silences spanning boundaries? (Adds complexity)
        # overlap_ms = config.min_silence_duration_ms # Example overlap
        overlap_ms = 0 # Keep it simple for now

        log.info(f"Processing in {chunk_size_ms}ms chunks...")
        disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG

        for i in tqdm(range(0, total_length_ms, chunk_size_ms), desc="Analyzing chunks for silence", disable=disable_tqdm):
            chunk_start_ms = i
            # Load only the current chunk (+ overlap?)
            # Duration to load: chunk_size, but clamp at the end of the file
            load_duration_ms = min(chunk_size_ms + overlap_ms, total_length_ms - chunk_start_ms)

            if load_duration_ms <= 0: continue # Should not happen with range, but safety check

            log.debug(f"Loading chunk: Offset={chunk_start_ms}ms, Duration={load_duration_ms}ms")
            try:
                # --- Load only the necessary chunk ---
                chunk = AudioSegment.from_file(
                    audio_file,
                    format="flac",
                    start_second=chunk_start_ms / 1000.0, # from_file uses seconds for start
                    duration=load_duration_ms / 1000.0     # duration also in seconds
                )
            except EOFError:
                log.warning(f"Reached EOF unexpectedly while reading chunk starting at {chunk_start_ms}ms. Stopping silence detection.")
                break # Stop if we can't read a chunk
            except Exception as load_err:
                 log.error(f"Error loading audio chunk starting at {chunk_start_ms}ms: {load_err}. Skipping chunk.")
                 continue # Skip this chunk

            log.debug(f"Detecting silence in loaded chunk ({len(chunk)}ms)...")
            chunk_silences_ms = silence.detect_silence(
                chunk,
                min_silence_len=config.min_silence_duration_ms,
                silence_thresh=config.silence_threshold_dbfs
            )

            # Adjust timestamps relative to the whole file and convert to seconds
            for start_in_chunk_ms, end_in_chunk_ms in chunk_silences_ms:
                # Important: Add the offset of the CHUNK start time
                global_start_s = (chunk_start_ms + start_in_chunk_ms) / 1000.0
                global_end_s = (chunk_start_ms + end_in_chunk_ms) / 1000.0

                # Avoid adding silences that might be artifacts of chunk boundaries if not using overlap
                # Or if using overlap, adjust to avoid double counting (more complex)
                # Simple approach: Ensure the silence is fully within the non-overlapped part?
                # For now, we add all detected silences within the loaded chunk duration

                all_silent_ranges_s.append((global_start_s, global_end_s))

        log.info(f"Detected {len(all_silent_ranges_s)} potential silent ranges (boundary accuracy may vary).")
        # Optional: Add merging logic for adjacent/overlapping ranges detected across chunks.
        # Sort and merge overlapping/adjacent intervals
        if not all_silent_ranges_s: return []
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

    except CouldntDecodeError as e:
        log.exception(f"Could not decode audio file {audio_file}: {e}")
        raise IOError(f"Failed to decode audio file {audio_file}") from e
    except KeyError as e:
         log.exception(f"Failed to get duration from mediainfo for {audio_file} - ffprobe/ffmpeg installed and working? Error: {e}")
         raise RuntimeError(f"Could not get audio duration for {audio_file}") from e
    except Exception as e:
        log.exception(f"Error during low RAM silence detection: {e}")
        raise RuntimeError("Silence detection failed") from e

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

def split_tracks(
    speaker_files: List[Path],
    cut_points_s: List[float],
    output_dir: Path) -> None:
    """
    Split speaker tracks at the given timestamps (in seconds) and save them.

    Args:
        speaker_files: List of Path objects for the original speaker audio files.
        cut_points_s: List of cut point timestamps in seconds.
        output_dir: Path object for the directory to save chunked files.
    """
    if not cut_points_s or len(cut_points_s) < 2:
        log.error("Cannot split tracks: Need at least two cut points (start and end).")
        raise ValueError("Invalid cut_points list provided for splitting.")
    if not speaker_files:
        log.warning("No speaker files provided for splitting.")
        return

    num_segments = len(cut_points_s) - 1
    log.info(f"Splitting {len(speaker_files)} tracks into {num_segments} segments...")
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure creation again just before loop
    disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG

    for file_path in tqdm(speaker_files, desc="Splitting files", disable=disable_tqdm):
        log.debug(f"Processing file for splitting: {file_path}")
        try:
            if not file_path.is_file():
                 log.warning(f"Speaker file not found, skipping: {file_path}")
                 continue
            log.debug(f"Loading audio for {file_path.name}...")
            audio = AudioSegment.from_file(file_path, format="flac")
            log.debug(f"Audio loaded (Duration: {len(audio)/1000.0:.2f}s). Splitting into chunks...")

            for i in range(num_segments):
                start_s, end_s = cut_points_s[i], cut_points_s[i+1]
                start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
                start_ms, end_ms = max(0, start_ms), min(len(audio), end_ms)

                if start_ms >= end_ms:
                     log.warning(f"Skipping zero/negative duration chunk {i+1} for {file_path.name} ({start_ms}ms - {end_ms}ms)")
                     continue

                log.debug(f"Extracting chunk {i+1}: {start_ms}ms - {end_ms}ms from {file_path.name}")
                segment = audio[start_ms:end_ms]

                indice = f"{i+1:02d}"
                output_filename = f"{file_path.stem}-{indice}.flac"
                output_filepath = output_dir / output_filename

                # Optional: Explicit delete before writing if overwrite is crucial
                # if output_filepath.exists():
                #     log.warning(f"Output chunk exists, overwriting: {output_filepath}")
                #     try: output_filepath.unlink()
                #     except OSError as del_e: log.error(f"Could not delete existing chunk {output_filepath}: {del_e}")

                log.debug(f"Exporting chunk {i+1} to {output_filepath}")
                segment.export(output_filepath, format="flac")
                log.debug(f"Chunk {i+1} for {file_path.name} exported successfully.")

        except CouldntDecodeError as e:
            log.exception(f"Could not decode speaker file {file_path}, skipping splitting.")
            continue
        except Exception as e:
            log.exception(f"Error splitting file {file_path}, skipping.")
            continue

    log.info(f"Splitting complete. Chunked files saved in {output_dir}")


# --- Main Pipeline Function for this Module ---

def chunk_audio(
    input_folder: Path,
    output_dir: Path,
    merged_file: Path,
    cut_points_file: Path,
    use_low_ram: bool = False,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG # Use default config if none provided
    ) -> None:
    """
    Main pipeline function to chunk audio files based on silence detection.

    Args:
        input_folder: Path to the folder containing input FLAC speaker tracks.
        output_dir: Path to the folder where chunked FLAC files will be saved.
        merged_file: Path to save the merged audio file of all speakers.
        cut_points_file: Path to save the text file listing cut points.
        use_low_ram: If True, use incremental methods to save RAM.
        config: Configuration settings for chunking.
    """
    log.info("Starting audio chunking pipeline...")
    log.info(f"Input folder: {input_folder}")
    log.info(f"Output chunk folder: {output_dir}")
    log.info(f"Merged audio output: {merged_file}")
    log.info(f"Cut points file: {cut_points_file}")
    log.info(f"Using low RAM mode: {use_low_ram}")
    log.info(f"Chunking configuration: {config}")

    try:
        # 1. Get input files
        speaker_files = get_audio_files_from_folder(input_folder)
        if not speaker_files:
            log.warning(f"No FLAC files found in {input_folder}. Audio chunking cannot proceed.")
            return # Exit gracefully if no input

        # 2. Merge tracks
        log.info("Merging speaker tracks...")
        if use_low_ram:
            _merge_tracks_low_ram(speaker_files, merged_file, config.incremental_chunk_size_ms)
        else:
            _merge_tracks_high_ram(speaker_files, merged_file)
        log.info("Track merging complete.")

        # Ensure merged file exists before proceeding
        if not merged_file.is_file():
             raise IOError(f"Merged audio file was not created successfully at {merged_file}")

        # Get duration of merged file for cut point determination
        try:
             merged_audio = AudioSegment.from_file(merged_file, format="flac")
             audio_duration_s = len(merged_audio) / 1000.0
             log.info(f"Merged audio duration: {audio_duration_s:.2f} seconds.")
        except Exception as e:
             log.exception(f"Failed to get duration of merged file {merged_file}")
             raise IOError("Could not read merged audio file duration.") from e

        # 3. Detect Silence
        log.info("Detecting silence in merged track...")
        if use_low_ram:
            silence_timestamps = _detect_silence_low_ram(merged_file, config)
        else:
            silence_timestamps = _detect_silence_high_ram(merged_file, config)
        log.info("Silence detection complete.")

        # 4. Determine Cut Points
        cut_points = determine_cut_points(silence_timestamps, audio_duration_s, config)

        # 5. Write Cut Points
        write_cut_points(cut_points, cut_points_file)

        # 6. Split Tracks
        split_tracks(speaker_files, cut_points, output_dir)

        log.info("Audio chunking pipeline completed successfully.")

    except (FileNotFoundError, ValueError, IOError, RuntimeError) as e:
        # Catch specific, expected errors from steps
        log.exception(f"Audio chunking pipeline failed: {e}")
        # Re-raise or handle as needed by the calling pipeline
        raise
    except Exception as e:
        # Catch any unexpected errors
        log.exception(f"An unexpected error occurred during audio chunking: {e}")
        raise RuntimeError("Unexpected failure during audio chunking") from e


# --- Example Usage ---
def run_chunking_example():
    """Example function to demonstrate running the chunk_audio pipeline."""
    print("\n--- Running Audio Chunking Example ---")
    # Configure logging for example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.setLevel(logging.INFO) # Ensure module log is also at INFO

    # Define test paths relative to current dir
    base_test_dir = Path("./temp_chunking_test")
    input_dir = base_test_dir / "input_tracks"
    output_dir = base_test_dir / "output_chunks"
    final_dir = base_test_dir / "final_outputs"
    merged_file = final_dir / "merged_test.flac"
    cut_points_file = final_dir / "cuts_test.txt"

    # Create dummy input files (requires pydub to create silence)
    try:
        print(f"Creating dummy input files in {input_dir}...")
        input_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True) # Create final dir for outputs

        # Create 2 dummy silent FLAC files of different lengths for testing
        silence_10s = AudioSegment.silent(duration=10000) # 10 seconds
        silence_15s = AudioSegment.silent(duration=15000) # 15 seconds
        silence_10s.export(input_dir / "speakerA.flac", format="flac")
        silence_15s.export(input_dir / "speakerB.flac", format="flac")
        print("Dummy files created.")

        # Define chunking configuration (using defaults here)
        config = ChunkingConfig(
             min_chunk_duration_s=5, # Shorter durations for test
             max_chunk_duration_s=12
        )

        # Run the chunking process
        print("\nRunning chunk_audio function...")
        chunk_audio(
            input_folder=input_dir,
            output_dir=output_dir,
            merged_file=merged_file,
            cut_points_file=cut_points_file,
            use_low_ram=False, # Use high RAM for simpler test case
            config=config
        )
        print("\nChunking example finished successfully.")
        print(f"Check outputs in: {output_dir}, {final_dir}")

    except Exception as e:
        print(f"\nChunking example failed: {e}")
        log.exception("Chunking example execution failed")
    finally:
        # Clean up dummy files/dirs
        # print("\nCleaning up test directories...")
        # if base_test_dir.exists():
        #     shutil.rmtree(base_test_dir)
        # print("Cleanup complete.")
        pass # Keep files for inspection

if __name__ == "__main__":
    run_chunking_example()