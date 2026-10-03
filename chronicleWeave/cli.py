#!/usr/bin/env python3
"""
ChronicleWeave CLI - Command Line Interface for the Audio Processing Pipeline

Usage:
    chronicleweave [options]
    python -m chronicleweave.cli [options]
"""

import sys
import argparse
from pathlib import Path
from typing import Union, List, Tuple

# Add the parent directory to the path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronicleweave.pipeline import run_pipeline


def parse_steps_argument(steps_arg: str) -> Union[slice, List[int], Tuple[int, ...]]:
    """Parse the --steps CLI argument into a form acceptable to run_pipeline."""
    if not steps_arg:
        return slice(1, 10)  # default: steps 1-9

    if "-" in steps_arg:
        try:
            start_str, end_str = steps_arg.split("-", 1)
            return slice(int(start_str.strip()), int(end_str.strip()) + 1)
        except ValueError:
            raise ValueError(f"Invalid range format: {steps_arg}. Use format like '1-5'")

    try:
        return [int(s.strip()) for s in steps_arg.split(",")]
    except ValueError:
        raise ValueError(f"Invalid steps format: {steps_arg}. Use comma-separated numbers or range")

def create_cli_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="chronicleweave",
        description="ChronicleWeave Audio Processing Pipeline - Transcribe and process audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (steps 1-9)
  chronicleweave

  # Run with custom base path
  chronicleweave --base-path ./my_session

  # Run only specific steps
  chronicleweave --steps 1,2,3

  # Run with diarization enabled
  chronicleweave --diarize

  # Custom file chunking settings
  chronicleweave --chunk-size 750 --chunk-template "part_{:03d}.txt"

  # Put chunks at base level instead of inside final_outputs
  chronicleweave --chunks-outside-final

  # Disable compact mode (keep extra blank lines)
  chronicleweave --no-compact-chunks

  # Debug mode with verbose logging
  chronicleweave --log-level DEBUG --verbose

  # Test only the file chunking step
  chronicleweave --steps 9 --base-path ./test_session
        """
    )

    # Basic pipeline options
    parser.add_argument(
        "--base-path", "-b",
        default=".",
        help="Base directory for the session (default: current directory)"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=str,
        help="Steps to run (comma-separated list, e.g., '1,2,3' or range '1-5')"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    # WhisperX options
    parser.add_argument(
        "--no-whisperx",
        action="store_true",
        help="Skip WhisperX transcription step (manual transcription required)"
    )
    
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization in WhisperX (requires HF_TOKEN)"
    )

    parser.add_argument(
        "--whisperx-model",
        help="Whisper checkpoint name (default: large-v3-turbo). Examples: large-v3, "
             "large-v3-turbo, bofenghuang/whisper-large-v3-distil-fr-v0.2"
    )

    parser.add_argument(
        "--whisperx-language",
        help="Language code passed to whisperx --language (default: fr). Use 'auto' for detection."
    )

    # Audio processing options
    parser.add_argument(
        "--no-low-ram",
        action="store_true",
        help="Disable low RAM mode for audio chunking"
    )

    # File chunking options (Step 9)
    chunking_group = parser.add_argument_group("File Chunking Options (Step 9)")
    chunking_group.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of lines per chunk (default: 1000)"
    )
    
    chunking_group.add_argument(
        "--chunk-template",
        default="script_chunk_{:02d}.txt",
        help="Filename template for chunks (default: script_chunk_{:02d}.txt)"
    )
    
    chunking_group.add_argument(
        "--chunks-outside-final",
        action="store_true",
        help="Put script chunks at base level instead of inside final_outputs/"
    )
    
    chunking_group.add_argument(
        "--chunks-folder",
        default="script_chunks",
        help="Custom folder name for script chunks (default: script_chunks)"
    )
    
    chunking_group.add_argument(
        "--no-compact-chunks",
        action="store_true",
        help="Disable compact mode (keep extra blank lines in chunks)"
    )

    # Input/output options
    parser.add_argument(
        "--input-folder",
        help="Override default input audio folder name (default: tracks)"
    )

    # Track preparation
    parser.add_argument(
        "--no-prepare-tracks",
        action="store_true",
        help="Skip auto-unzip of Craig archive and speaker renaming (Step 0)"
    )
    parser.add_argument(
        "--mapping-file",
        help="Path to a custom speaker mapping JSON (default: packaged speaker_mapping.json)"
    )

    # Verbose output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (sets log level to DEBUG)"
    )

    # Version and help
    parser.add_argument(
        "--version",
        action="version",
        version="ChronicleWeave 1.0.0"
    )

    return parser

def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()

    # Handle verbose flag
    if args.verbose:
        args.log_level = "DEBUG"

    # Parse steps argument
    try:
        steps_to_run = parse_steps_argument(args.steps)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare keyword arguments for run_pipeline
    pipeline_kwargs = {
        "base_path": args.base_path,
        "steps_to_run": steps_to_run,
        "log_level": args.log_level,
        "run_whisperx": not args.no_whisperx,
        "diarize": args.diarize,
        "use_low_ram": not args.no_low_ram,
        "script_chunk_size": args.chunk_size,
        "script_chunk_filename_template": args.chunk_template,
        "script_chunks_in_final_folder": not args.chunks_outside_final,
        "script_chunks_folderName": args.chunks_folder,
        "script_chunks_compact_mode": not args.no_compact_chunks,
    }

    # Add optional arguments if provided
    if args.input_folder:
        pipeline_kwargs["input_audio_folderName"] = args.input_folder

    if args.no_prepare_tracks:
        pipeline_kwargs["auto_prepare_tracks"] = False
    if args.mapping_file:
        pipeline_kwargs["speaker_mapping_path"] = Path(args.mapping_file)
    if args.whisperx_model:
        pipeline_kwargs["whisperx_model"] = args.whisperx_model
    if args.whisperx_language:
        pipeline_kwargs["whisperx_language"] = args.whisperx_language

    try:
        # Run the pipeline
        run_pipeline(**pipeline_kwargs)
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 