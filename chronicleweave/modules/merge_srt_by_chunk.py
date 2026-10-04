# chronicleweave/modules/merge_srt_by_chunk.py

import logging
from pathlib import Path
import srt # External library: https://pypi.org/project/srt/
from datetime import timedelta
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict
import re
from dataclasses import dataclass, field
import sys

# --- Module Logger ---
log = logging.getLogger(__name__)

# --- Configuration ---
@dataclass(frozen=True)
class SRTMergeConfig:
    """Configuration for merging SRT files."""
    chunk_gap_s: float = 0.1 # Gap to add between chunks in seconds (reduced default)

    def __post_init__(self):
        if self.chunk_gap_s < 0:
             raise ValueError("chunk_gap_s cannot be negative")

DEFAULT_MERGE_CONFIG = SRTMergeConfig()

# --- Helper Functions ---

def load_srt_file(
    file_path: Path,
    speaker_name: str,
    time_offset: timedelta
    ) -> List[srt.Subtitle]:
    """
    Load an SRT file, adjust timestamps, and tag content with speaker name.

    Args:
        file_path: Path to the input SRT file.
        speaker_name: Name of the speaker for tagging content.
        time_offset: Timedelta offset to apply to subtitle start/end times.

    Returns:
        A list of adjusted srt.Subtitle objects, or an empty list if loading/parsing fails.
    """
    subtitles: List[srt.Subtitle] = []
    log.debug(f"Loading SRT file: {file_path} with offset {time_offset}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Parse SRT content; handle potential errors from the srt library
            parsed_subs = list(srt.parse(content))

        for sub in parsed_subs:
            # Adjust timestamps
            sub.start += time_offset
            sub.end += time_offset
            # Prepend speaker tag
            sub.content = f"[{speaker_name}] {sub.content}"
            subtitles.append(sub)
        log.debug(f"Loaded and processed {len(subtitles)} subtitles from {file_path.name}")
        return subtitles

    except FileNotFoundError:
        log.error(f"SRT file not found: {file_path}")
        return []
    except srt.SRTParseError as e:
        log.error(f"Failed to parse SRT file {file_path}: {e}")
        return []
    except Exception as e:
        # Catch other potential errors like decoding errors, IOErrors
        log.exception(f"Error loading or processing SRT file {file_path}:")
        return []


def parse_srt_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Parses filenames expected in 'speaker-chunkNumber.srt' format.

    Args:
        filename: The filename string.

    Returns:
        A tuple (speaker_name, chunk_number) if parsing succeeds, otherwise None.
    """
    # Basic check for suffix
    if not filename.lower().endswith(".srt"):
        return None

    base_name = filename[:-4] # Remove .srt suffix

    # Use regex for slightly more robust parsing than rsplit
    # Allows speaker names with hyphens, requires digits at the end after a hyphen.
    match = re.match(r'^(.+)-(\d+)$', base_name)
    if match:
        speaker_name = match.group(1)
        chunk_number_str = match.group(2)
        try:
            chunk_number = int(chunk_number_str)
            # Basic sanity check (e.g., avoid excessively large chunk numbers)
            if 0 < chunk_number < 10000: # Arbitrary upper limit
                 return speaker_name, chunk_number
            else:
                 log.warning(f"Parsed chunk number {chunk_number} from '{filename}' seems invalid, skipping.")
                 return None
        except ValueError:
            # Should not happen if regex matches \d+ but safety check
             log.warning(f"Could not convert chunk number '{chunk_number_str}' to int in filename '{filename}', skipping.")
             return None
    else:
        log.warning(f"Filename '{filename}' does not match expected 'speaker-chunkNumber.srt' pattern, skipping.")
        return None


# --- Main Merging Function ---

def merge_srt_by_chunk(
    srt_folder: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: str,
    config: SRTMergeConfig = DEFAULT_MERGE_CONFIG
    ) -> None:
    """
    Merge all SRT files matching 'speaker-chunkNumber.srt' in a folder,
    adjusting timestamps based on chunk order and applying a gap between chunks.

    Args:
        srt_folder: Path to the folder containing individual SRT files.
        output_dir: Path to the folder where the merged SRT will be saved.
        output_filename: Name for the final merged SRT file (e.g., "merged.srt").
        config: Configuration object for merging parameters.

    Raises:
        FileNotFoundError: If the srt_folder does not exist.
        NotADirectoryError: If srt_folder is not a directory.
        IOError: If the output file cannot be written.
        ValueError: If output_filename is invalid.
    """
    input_path = Path(srt_folder)
    output_path = Path(output_dir)
    output_filepath = output_path / output_filename

    log.info(f"Merging SRT files from '{input_path}' into '{output_filepath}'...")
    log.info(f"Using inter-chunk gap: {config.chunk_gap_s}s")

    # --- Input Validation ---
    if not input_path.exists():
        log.error(f"Input SRT folder not found: {input_path}")
        raise FileNotFoundError(f"Input SRT folder not found: {input_path}")
    if not input_path.is_dir():
        log.error(f"Input path is not a directory: {input_path}")
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")
    if not output_filename or not output_filename.lower().endswith(".srt"):
        log.error(f"Invalid output filename: '{output_filename}'. Must end with .srt")
        raise ValueError("Output filename must be a non-empty string ending with .srt")

    # --- Organize Input Files ---
    # Use defaultdict for cleaner structure: {chunk_number: {speaker_name: file_path}}
    chunk_files: Dict[int, Dict[str, Path]] = defaultdict(dict)
    file_count = 0
    skipped_count = 0
    for file_path in input_path.glob("*.srt"):
        file_count += 1
        parse_result = parse_srt_filename(file_path.name)
        if parse_result:
            speaker_name, chunk_number = parse_result
            if speaker_name in chunk_files[chunk_number]:
                 log.warning(f"Duplicate entry found for speaker '{speaker_name}' in chunk {chunk_number}. Keeping first found: {chunk_files[chunk_number][speaker_name]}. Ignoring: {file_path}")
            else:
                 chunk_files[chunk_number][speaker_name] = file_path
        else:
            skipped_count += 1
            # Log already happens in parse_srt_filename

    if not chunk_files:
        log.warning(f"No valid SRT files matching the 'speaker-chunkNumber.srt' pattern found in '{input_path}'. No merge performed.")
        if file_count > 0:
             log.warning(f"({skipped_count} files were skipped due to naming pattern mismatch).")
        # Create an empty output file? Or just return? Let's create empty file for consistency.
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            output_filepath.write_text("", encoding="utf-8")
            log.info(f"Empty SRT file created at {output_filepath} as no valid input files were found.")
        except IOError as e:
            log.error(f"Failed to write empty output SRT file: {e}")
            raise IOError(f"Failed to write empty output SRT file {output_filepath}") from e
        return

    log.info(f"Found {len(chunk_files)} chunks to process from {file_count - skipped_count} valid files ({skipped_count} skipped).")

    # --- Merge Process ---
    all_subtitles: List[srt.Subtitle] = []
    current_offset = timedelta(0) # Start at 0 offset for the first chunk
    chunk_gap_delta = timedelta(seconds=config.chunk_gap_s)

    # Process chunks in numerical order
    sorted_chunk_numbers = sorted(chunk_files.keys())
    disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG

    for chunk_number in tqdm(sorted_chunk_numbers, desc="Merging SRT chunks", disable=disable_tqdm):
        log.debug(f"Processing Chunk {chunk_number} with offset {current_offset}...")
        current_chunk_subs: List[srt.Subtitle] = []
        speakers_in_chunk = chunk_files[chunk_number]

        for speaker_name, file_path in speakers_in_chunk.items():
            # Load SRT, apply current offset, and tag speaker
            subtitles = load_srt_file(file_path, speaker_name, current_offset)
            current_chunk_subs.extend(subtitles)

        if not current_chunk_subs:
            log.warning(f"Chunk {chunk_number} yielded no valid subtitles. Skipping.")
            continue # Move to the next chunk, offset remains unchanged for now

        # Sort subtitles *within* the current chunk based on their *adjusted* start times
        current_chunk_subs.sort(key=lambda sub: sub.start)

        # Add this chunk's processed subtitles to the main list
        all_subtitles.extend(current_chunk_subs)

        # Determine the maximum end time *within this processed chunk*
        # to calculate the offset for the *next* chunk
        try:
            # Ensure we don't try to get max of an empty sequence if loading failed for all files in chunk
            max_end_time_in_chunk = max(sub.end for sub in current_chunk_subs)
            # Update offset for the *next* chunk: max end time + gap
            current_offset = max_end_time_in_chunk + chunk_gap_delta
            log.debug(f"Chunk {chunk_number} processed. Max end time: {max_end_time_in_chunk}. Next offset: {current_offset}")
        except ValueError:
             # This should not happen due to the 'if not current_chunk_subs' check above, but safety first
             log.warning(f"Could not determine max end time for chunk {chunk_number} (empty?). Offset not advanced.")
             # Offset remains the same as before this chunk

    # --- Write Merged Output ---
    if not all_subtitles:
        log.warning("No subtitles were collected after processing all chunks. Output file will be empty.")

    log.info(f"Composing final SRT file with {len(all_subtitles)} subtitles...")
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        composed_srt = srt.compose(all_subtitles) # Compose might raise error on empty list depending on library version
        output_filepath.write_text(composed_srt, encoding='utf-8')
        log.info(f"Merged SRT file saved successfully to '{output_filepath}'.")
    except IOError as e:
        log.exception(f"Failed to write merged SRT file: {output_filepath}")
        raise IOError(f"Failed to write merged SRT file {output_filepath}") from e
    except Exception as e:
        log.exception(f"An unexpected error occurred during final SRT composition or writing:")
        raise RuntimeError("Failed to compose or write final SRT") from e


# --- Example Usage ---
def run_merge_srt_example():
    """Example function to demonstrate merging SRT files."""
    print("\n--- Running SRT Merge Example ---")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.setLevel(logging.INFO)

    # Define test paths
    base_test_dir = Path("./temp_mergesrt_test")
    input_dir = base_test_dir / "srt_input_chunks"
    output_dir = base_test_dir / "srt_output_merged"
    output_filename = "merged_transcript.srt"

    # Create dummy input files
    try:
        print(f"Creating dummy input SRT files in {input_dir}...")
        input_dir.mkdir(parents=True, exist_ok=True)

        # Chunk 1
        srt1_content = "1\n00:00:01,000 --> 00:00:03,000\nSpeaker A, chunk 1, line 1\n\n2\n00:00:04,000 --> 00:00:05,500\nSpeaker A, chunk 1, line 2\n"
        (input_dir / "SpeakerA-01.srt").write_text(srt1_content, encoding='utf-8')
        srt2_content = "1\n00:00:02,500 --> 00:00:04,500\nSpeaker B, chunk 1, line 1\n"
        (input_dir / "SpeakerB-01.srt").write_text(srt2_content, encoding='utf-8')

        # Chunk 2 (Starts relative to 0 again internally)
        srt3_content = "1\n00:00:00,500 --> 00:00:02,000\nSpeaker A, chunk 2, line 1\n"
        (input_dir / "SpeakerA-02.srt").write_text(srt3_content, encoding='utf-8')

        # Malformed filename
        (input_dir / "SpeakerC-Chunk3.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nBad name\n", encoding='utf-8')

        print("Dummy files created.")

        # Define config (use smaller gap for testing)
        config = SRTMergeConfig(chunk_gap_s=0.5)

        # Run the merge function
        print("\nRunning merge_srt_by_chunk function...")
        merge_srt_by_chunk(
            srt_folder=input_dir,
            output_dir=output_dir,
            output_filename=output_filename,
            config=config
        )
        print("\nSRT merge example finished.")
        print(f"Check output in: {output_dir / output_filename}")

        # Verify output (basic check)
        output_file = output_dir / output_filename
        if output_file.exists() and output_file.stat().st_size > 0:
             print("Merged SRT file created successfully.")
             # Optional: print content
             # print("\n--- Merged SRT Content ---")
             # print(output_file.read_text(encoding='utf-8'))
             # print("--- End Merged SRT ---")
        else:
             print("ERROR: Merged SRT file not found or is empty.")

    except Exception as e:
        print(f"\nSRT merge example failed: {e}")
        log.exception("SRT merge example execution failed")
    finally:
        # Clean up dummy files/dirs
        # print("\nCleaning up test directories...")
        # if base_test_dir.exists():
        #     shutil.rmtree(base_test_dir)
        # print("Cleanup complete.")
        pass # Keep files

if __name__ == "__main__":
    run_merge_srt_example()