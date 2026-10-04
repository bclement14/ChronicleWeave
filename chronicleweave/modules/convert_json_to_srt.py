# chronicleweave/modules/convert_json_to_srt.py

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from tqdm import tqdm
import sys
from numbers import Number # For type checking

# --- Module Logger ---
log = logging.getLogger(__name__)

# --- Helper Function ---

def format_srt_time(seconds: float) -> str:
    """
    Convert seconds (float) to SRT time format "HH:MM:SS,ms".

    Handles potential rounding issues and negative inputs.

    Args:
        seconds: Time in seconds.

    Returns:
        The formatted SRT time string.

    Raises:
        ValueError: If input is not a valid number.
    """
    if not isinstance(seconds, Number):
        raise ValueError(f"Invalid input: seconds must be a number, got {type(seconds)}")
    if seconds < 0:
        log.warning(f"Negative timestamp encountered ({seconds:.3f}s). Clamping to 00:00:00,000.")
        seconds = 0.0

    # Calculate hours, minutes, seconds, milliseconds
    total_seconds = int(seconds)
    milliseconds = round((seconds - total_seconds) * 1000)

    # Handle milliseconds rounding up to 1000
    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    sec = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"


def json_to_srt(json_path: Path, srt_path: Path) -> bool:
    """
    Convert a single WhisperX JSON file to an SRT file.

    Args:
        json_path: Path to the input JSON file.
        srt_path: Path to save the output SRT file.

    Returns:
        True if conversion was successful, False otherwise.
    """
    log.debug(f"Converting '{json_path.name}' to SRT...")
    try:
        # Load JSON data
        with open(json_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        # --- Validation ---
        if not isinstance(data, dict):
            log.error(f"Invalid JSON structure in {json_path}: Root is not a dictionary.")
            return False
        if "segments" not in data or not isinstance(data["segments"], list):
            log.error(f"Invalid JSON structure in {json_path}: Missing or invalid 'segments' list.")
            return False

        segments = data["segments"]
        srt_blocks: List[str] = []
        index = 1

        # --- Process Segments ---
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                log.warning(f"Skipping invalid segment at index {i} in {json_path}: Not a dictionary.")
                continue

            start_time_sec = segment.get("start")
            end_time_sec = segment.get("end")
            text = segment.get("text", "").strip()

            # Validate essential keys and types
            if not isinstance(start_time_sec, Number) or not isinstance(end_time_sec, Number):
                log.warning(f"Skipping segment {i} in {json_path}: Invalid or missing 'start'/'end' time (must be numeric). Found start={start_time_sec}, end={end_time_sec}")
                continue
            if end_time_sec < start_time_sec:
                 log.warning(f"Segment {i} in {json_path} has end time ({end_time_sec:.3f}) before start time ({start_time_sec:.3f}). Clamping start to end.")
                 start_time_sec = end_time_sec # Or skip? For now, clamp to make a zero-duration entry.

            # Fallback to joining words if 'text' is empty but 'words' exist
            # (As done in corrector module, good practice here too)
            if not text and "words" in segment and isinstance(segment["words"], list):
                words_list = segment["words"]
                if all(isinstance(w, dict) and "word" in w for w in words_list):
                     text = " ".join(str(w.get("word", "")).strip() for w in words_list if str(w.get("word", "")).strip()).strip()
                     if text:
                         log.debug(f"Segment {i} in {json_path}: Reconstructed text from 'words' list.")
            if not text:
                log.warning(f"Skipping segment {i} in {json_path}: No valid text found in 'text' or 'words'.")
                continue

            # Format times and create block
            try:
                start_time_str = format_srt_time(float(start_time_sec))
                end_time_str = format_srt_time(float(end_time_sec))
            except ValueError as time_err:
                 log.warning(f"Skipping segment {i} in {json_path} due to invalid timestamp value: {time_err}")
                 continue

            block = f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n"
            srt_blocks.append(block)
            index += 1

        # --- Write SRT File ---
        log.debug(f"Writing {len(srt_blocks)} blocks to {srt_path}...")
        srt_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        with open(srt_path, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(srt_blocks)) # Join blocks with one newline

        return True

    except FileNotFoundError:
        log.error(f"Input JSON file not found: {json_path}")
        return False
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON file {json_path}: {e}")
        return False
    except IOError as e:
        log.error(f"I/O error processing file {json_path} or writing {srt_path}: {e}")
        return False
    except Exception as e:
        log.exception(f"An unexpected error occurred converting {json_path.name} to SRT:")
        return False


def convert_json_folder_to_srt(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Convert all JSON files in an input directory to SRT subtitle files
    in the output directory.

    Args:
        input_dir: Path to the directory containing corrected JSON files.
        output_dir: Path to the directory where generated SRT files will be saved.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        NotADirectoryError: If the input path is not a directory.
        IOError: If the output directory cannot be created.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    log.info(f"Converting JSON files from '{input_path}' to SRT format in '{output_path}'...")

    # --- Input Validation ---
    if not input_path.exists():
        log.error(f"Input directory not found: {input_path}")
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        log.error(f"Input path is not a directory: {input_path}")
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    # --- Output Directory Setup ---
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.exception(f"Failed to create output directory: {output_path}")
        raise IOError(f"Could not create output directory {output_path}") from e

    # --- File Processing ---
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        log.warning(f"No JSON files found in '{input_path}'. Nothing to convert.")
        return

    log.info(f"Found {len(json_files)} JSON files to convert.")
    success_count = 0
    error_count = 0

    # Disable tqdm if logging level is DEBUG or lower
    disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG
    for json_file_path in tqdm(json_files, desc="Converting JSON to SRT", disable=disable_tqdm):
        srt_filename = json_file_path.stem + ".srt"
        srt_file_path = output_path / srt_filename

        if json_to_srt(json_file_path, srt_file_path):
            success_count += 1
        else:
            error_count += 1
            log.warning(f"Failed to convert {json_file_path.name}. Check previous error messages.")

    # --- Summary Logging ---
    log.info("JSON to SRT conversion complete.")
    log.info(f"Successfully converted: {success_count} files.")
    if error_count > 0:
        log.warning(f"Failed to convert: {error_count} files. Please review logs.")


# --- Example Usage ---
def run_conversion_example():
    """Example function to demonstrate JSON to SRT conversion."""
    print("\n--- Running JSON to SRT Conversion Example ---")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.setLevel(logging.INFO) # Ensure module logger level

    # Define test paths
    base_test_dir = Path("./temp_jsonsrt_test")
    input_dir = base_test_dir / "corrected_json"
    output_dir = base_test_dir / "srt_output"

    # Create dummy input file
    try:
        print(f"Creating dummy input file in {input_dir}...")
        input_dir.mkdir(parents=True, exist_ok=True)
        dummy_json_data = {
            "segments": [
                {"start": 1.234, "end": 3.456, "text": "Hello world."},
                {"start": 4.000, "end": 5.500, "text": "This is a test.", "words":[]}, # Text exists
                {"start": 6.1, "end": 7.9, "words": [{"word":"Another"}, {"word":"test."}]}, # Only words exist
                {"start": -1.0, "end": 8.5, "text": "Negative start test."}, # Invalid time test
                {"start": 9.0, "end": 8.5, "text": "End before start test."}, # Invalid time test
                {"start": 10.0, "end": 11.0} # Missing text/words test
            ]
        }
        input_file = input_dir / "test_transcript.json"
        input_file.write_text(json.dumps(dummy_json_data, indent=2), encoding='utf-8')
        print("Dummy file created.")

        # Run the conversion
        print("\nRunning convert_json_folder_to_srt function...")
        convert_json_folder_to_srt(input_dir=input_dir, output_dir=output_dir)
        print("\nConversion example finished.")
        print(f"Check outputs in: {output_dir}")

        # Verify output (basic check)
        output_file = output_dir / "test_transcript.srt"
        if output_file.exists() and output_file.stat().st_size > 0:
             print("Output SRT file created successfully.")
             # You can optionally read and print the content here for verification
             # print("\n--- Output SRT Content ---")
             # print(output_file.read_text(encoding='utf-8'))
             # print("--- End Output SRT ---")
        else:
             print("ERROR: Output SRT file not found or is empty.")

    except Exception as e:
        print(f"\nConversion example failed: {e}")
        log.exception("Conversion example execution failed")
    finally:
        # Clean up dummy files/dirs
        # print("\nCleaning up test directories...")
        # if base_test_dir.exists():
        #     shutil.rmtree(base_test_dir)
        # print("Cleanup complete.")
        pass # Keep files for inspection

if __name__ == "__main__":
    run_conversion_example()