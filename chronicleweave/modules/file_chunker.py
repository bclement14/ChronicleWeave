# chronicleweave/modules/file_chunker.py

import os
import logging
from pathlib import Path
from typing import Union, List

log = logging.getLogger(__name__)

def chunk_text_file_by_lines(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    chunk_size: int = 500,
    output_filename_template: str = "chunk_{:02d}.txt",
    compact_mode: bool = False
) -> List[Path]:
    """
    Splits a large text file into smaller chunks of a specified number of lines.

    Args:
        input_file: Path to the large text file to be chunked.
        output_dir: Directory where the chunk files will be saved.
        chunk_size: The number of lines each chunk should contain.
        output_filename_template: A format string for the output chunk filenames.
                                  It must contain one format specifier for the chunk number.
        compact_mode: If True, removes extra blank lines to make chunks more compact.

    Returns:
        A list of Path objects for the created chunk files.
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If the output directory cannot be created or files cannot be written.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.is_file():
        log.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.exception(f"Failed to create output directory: {output_path}")
        raise IOError(f"Could not create output directory {output_path}") from e

    created_chunks = []
    try:
        # Read the entire file first
        with open(input_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        original_line_count = len(all_lines)
        
        # Apply compact mode to the entire file if enabled
        if compact_mode:
            all_lines = compact_lines(all_lines)
            log.info(f"Applied compact mode: reduced from {original_line_count} to {len(all_lines)} lines")
        
        # Now chunk the processed lines
        chunk_number = 0
        for i in range(0, len(all_lines), chunk_size):
            chunk_lines = all_lines[i:i + chunk_size]
            
            if not chunk_lines:
                break  # End of file

            chunk_file_path = output_path / output_filename_template.format(chunk_number)
            with open(chunk_file_path, 'w', encoding='utf-8') as chunk_f:
                chunk_f.writelines(chunk_lines)
            
            log.info(f"Successfully created chunk: {chunk_file_path}")
            created_chunks.append(chunk_file_path)
            chunk_number += 1
            
        log.info(f"File chunking complete. Created {len(created_chunks)} chunks.")
        return created_chunks

    except IOError as e:
        log.exception(f"An I/O error occurred during chunking of {input_path}")
        raise
    except Exception as e:
        log.exception(f"An unexpected error occurred during chunking of {input_path}")
        raise RuntimeError(f"An unexpected error occurred during file chunking: {e}") from e

def compact_lines(lines: List[str]) -> List[str]:
    """
    Remove extra blank lines to make text more compact.
    
    Args:
        lines: List of lines to compact.
        
    Returns:
        List of lines with extra blank lines removed.
    """
    if not lines:
        return lines
    
    compacted = []
    in_blank_sequence = False
    
    for line in lines:
        is_blank = line.strip() == ""
        
        if is_blank:
            # Only add blank line if we're not already in a blank sequence
            if not in_blank_sequence:
                compacted.append(line)
                in_blank_sequence = True
        else:
            # Add non-blank line and reset blank sequence flag
            compacted.append(line)
            in_blank_sequence = False
    
    # Remove trailing blank lines
    while compacted and compacted[-1].strip() == "":
        compacted.pop()
    
    return compacted
