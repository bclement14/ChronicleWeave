#!/usr/bin/env python3
"""ChronicleWeave command line: `chronicleweave` (one session) and `chronicleweave-batch` (many)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from chronicleweave.batch import format_summary, parse_only, run_batch
from chronicleweave.envfile import load_env
from chronicleweave.pipeline import parse_steps, run_pipeline


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", "-s", help="One step or one range within 1-9, e.g. '1-8' (default) or '5'.")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--verbose", "-v", action="store_true", help="Same as --log-level DEBUG.")
    parser.add_argument("--env-file", type=Path, help="Load this .env file instead of searching for one.")
    parser.add_argument("--no-whisperx", action="store_true",
                        help="Do not run WhisperX; use JSON files already placed in wx_output/.")
    parser.add_argument("--diarize", action="store_true", help="Enable WhisperX diarization (needs HF_TOKEN).")
    parser.add_argument("--whisperx-model", help="Whisper model name or a local CTranslate2 model folder (default: large-v3).")
    parser.add_argument("--whisperx-language", help="Language code for WhisperX (default: fr).")
    parser.add_argument("--input-folder", help="Input tracks folder name (default: tracks).")
    parser.add_argument("--no-prepare-tracks", action="store_true",
                        help="Do not unzip or rename; only validate that tracks/ holds speaker Tags.")
    group = parser.add_argument_group("Script chunking (step 8)")
    group.add_argument("--chunk-size", type=int, default=1000)
    group.add_argument("--chunk-template", default="script_chunk_{:02d}.txt")
    group.add_argument("--chunks-outside-final", action="store_true")
    group.add_argument("--chunks-folder", default="script_chunks")
    group.add_argument("--no-compact-chunks", action="store_true")
    parser.add_argument("--version", action="version", version="ChronicleWeave 1.1.0")


def pipeline_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "steps_to_run": parse_steps(args.steps),
        "log_level": "DEBUG" if args.verbose else args.log_level,
        "run_whisperx": not args.no_whisperx,
        "diarize": args.diarize,
        "script_chunk_size": args.chunk_size,
        "script_chunk_filename_template": args.chunk_template,
        "script_chunks_in_final_folder": not args.chunks_outside_final,
        "script_chunks_folderName": args.chunks_folder,
        "script_chunks_compact_mode": not args.no_compact_chunks,
    }
    if args.input_folder:
        kwargs["input_audio_folderName"] = args.input_folder
    if args.no_prepare_tracks:
        kwargs["auto_prepare_tracks"] = False
    if args.whisperx_model:
        kwargs["whisperx_model"] = args.whisperx_model
    if args.whisperx_language:
        kwargs["whisperx_language"] = args.whisperx_language
    return kwargs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chronicleweave", description="Process one session folder.")
    parser.add_argument("--base-path", "-b", default=".", help="Session folder (default: current directory).")
    add_pipeline_arguments(parser)
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file)
        kwargs = pipeline_kwargs_from_args(args)
        run_pipeline(base_path=args.base_path, **kwargs)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        return 1
    print("✅ Pipeline completed successfully.")
    return 0


def batch_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chronicleweave-batch", description="Process every session folder under ROOT.")
    parser.add_argument("root", type=Path, help="Folder containing session<N> folders.")
    parser.add_argument("--only", help="Session numbers to consider, e.g. '13,15-19'.")
    parser.add_argument("--force", action="store_true", help="Re-process sessions that are already done.")
    add_pipeline_arguments(parser)
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file)
        kwargs = pipeline_kwargs_from_args(args)
        only = parse_only(args.only)
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    results = run_batch(args.root, only=only, force=args.force, pipeline_kwargs=kwargs)
    print(format_summary(results))
    return 1 if any(r.status == "FAILED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
