# chronicleweave/modules/convert_srt_to_script.py

import re
import logging
from pathlib import Path
from typing import Union

# --- Module Logger ---
log = logging.getLogger(__name__)

# --- Constants ---
SRT_BLOCK_SPLIT_REGEX = re.compile(r'\n\s*\n')

# --- Main Function ---
def srt_to_script(
    input_srt_path: Union[str, Path],
    output_script_path: Union[str, Path]
    ) -> None:
    """
    Convert an SRT file (potentially with speaker tags like [Speaker Name])
    into a plain text script format, preserving the full content lines.

    Args:
        input_srt_path: Path to the input cleaned SRT file (with tags).
        output_script_path: Path to save the resulting plain text script.

    Raises:
        FileNotFoundError, IOError, OSError, RuntimeError as before.
    """
    input_path = Path(input_srt_path)
    output_path = Path(output_script_path)

    log.info(f"Converting SRT '{input_path.name}' into plain text script (preserving tags)...")
    log.info(f"Output will be saved to '{output_path}'")

    if not input_path.is_file():
        log.error(f"Input SRT file not found: {input_path}")
        raise FileNotFoundError(f"Input SRT file not found: {input_path}")

    try:
        content = input_path.read_text(encoding="utf-8")
        log.debug(f"Read {len(content)} characters from {input_path.name}")
    except Exception as e:
        log.exception(f"Failed to read input SRT file {input_path}:")
        raise IOError(f"Failed to read input SRT file {input_path}") from e

    # --- Processing ---
    blocks = SRT_BLOCK_SPLIT_REGEX.split(content.strip())
    output_lines = []
    processed_block_count = 0
    skipped_malformed = 0

    log.debug(f"Processing {len(blocks)} potential subtitle blocks...")
    for block in blocks:
        if not block.strip(): continue

        lines = block.splitlines()

        # Check for standard structure (index, time, at least one content line)
        if len(lines) >= 3:
            # --- CORRECTED: Keep the full content lines (including tags) ---
            full_content_line = " ".join(lines[2:]).strip() # Join multi-lines with space
            # --- END CORRECTION ---

            if full_content_line: # Only add if there's actual text content
                output_lines.append(full_content_line)
                processed_block_count += 1
            else:
                 log.debug("Skipping block with no text content.")
        else:
            log.warning(f"Skipping malformed SRT block (less than 3 lines): '{lines[0] if lines else 'Empty Block'}...'")
            skipped_malformed += 1

    log.info(f"Extracted text from {processed_block_count} subtitle blocks.")
    if skipped_malformed > 0:
         log.warning(f"Skipped {skipped_malformed} potentially malformed blocks.")

    # --- Writing Output ---
    try:
        log.info(f"Writing plain text script to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Join the extracted dialogue lines with double newlines
        final_script_content = "\n\n".join(output_lines)
        output_path.write_text(final_script_content, encoding="utf-8")
        log.info("Plain text script saved successfully.")
    except Exception as e:
        log.exception(f"Failed to write output script file: {output_path}")
        raise IOError(f"Failed to write output script file {output_path}") from e

# --- Example Usage ---
def run_script_conversion_example():
    """Example function to demonstrate SRT to script conversion."""
    print("\n--- Running SRT to Script Conversion Example (Corrected) ---")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log.setLevel(logging.INFO)

    base_test_dir = Path("./temp_scriptconv_test_v2")
    input_file = base_test_dir / "cleaned_with_tags.srt"
    output_file = base_test_dir / "final_script_with_tags.txt"

    try:
        print(f"Creating dummy input SRT file: {input_file}...")
        base_test_dir.mkdir(parents=True, exist_ok=True)
        # Use output similar to corrected merge_speaker_entries output
        dummy_srt_content = """1
00:00:01,000 --> 00:00:05,000
[Alice] Hello Bob. How are you?

2
00:00:06,000 --> 00:00:08,000
This is an untagged line.

3
00:00:08,500 --> 00:00:12,000
[Bob] I am Bob. Yes, Bob.
"""
        input_file.write_text(dummy_srt_content, encoding='utf-8')
        print("Dummy file created.")

        print("\nRunning srt_to_script function...")
        srt_to_script(input_srt_path=input_file, output_script_path=output_file)
        print("\nScript conversion example finished.")
        print(f"Check output file: {output_file}")

        if output_file.exists():
             print("Script file created successfully.")
             print("\n--- Script Content (Should Retain Tags) ---")
             print(output_file.read_text(encoding='utf-8'))
             print("--- End Script Content ---")
             # Check expected content
             expected_content = "[Alice] Hello Bob. How are you?\n\nThis is an untagged line.\n\n[Bob] I am Bob. Yes, Bob."
             assert output_file.read_text(encoding='utf-8') == expected_content
             print("Content verified.")
        else: print("ERROR: Script file not found or is empty.")

    except Exception as e: print(f"\nScript conversion example failed: {e}"); log.exception("Exec failed")
    finally: pass # Keep files

if __name__ == "__main__":
    run_script_conversion_example()