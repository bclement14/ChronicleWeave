# chronicleweave/pipeline.py

import os
import subprocess
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field, replace, fields
from typing import Optional, Union, List, Tuple, Sequence, Dict, Any
import select

# Attempt to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Determine the script's directory or a common project root to look for .env
    # For robustness, you might want to specify a path or use find_dotenv()
    # load_dotenv(find_dotenv(usecwd=True)) # More robust way to find .env
    load_dotenv() # Loads .env from current working directory or parents
    # Logger is not set up yet, so can't log here directly without custom print
except ImportError:
    # python-dotenv not installed, which is fine.
    pass


# --- Imports ---
from .modules.audio_chunker import chunk_audio, ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from .modules.prepare_tracks import prepare_tracks, DEFAULT_MAPPING_PATH
from .modules.whisperx_corrector_core import correct_whisperx_outputs, WhisperXCorrectorConfig, DEFAULT_CONFIG as DEFAULT_CORRECTOR_CONFIG
from .modules.convert_json_to_srt import convert_json_folder_to_srt
from .modules.merge_srt_by_chunk import merge_srt_by_chunk, SRTMergeConfig, DEFAULT_MERGE_CONFIG
from .modules.merge_speaker_entries import merge_speaker_entries
from .modules.convert_srt_to_script import srt_to_script
from .modules.llm_processor import process_with_llm, LLMConfig, DEFAULT_LLM_CONFIG
from .modules.file_chunker import chunk_text_file_by_lines

# --- Configuration Dataclasses ---

# Forward declare LLMConfig for PipelineConfig's type hint if LLMConfig needs its own __str__
# For simplicity, if LLMConfig is simple or its __str__ is also basic, this might not be strictly needed.
# However, it's good practice if they reference each other in complex ways.
# class LLMConfig: ... # This would be needed if LLMConfig also had a complex __str__ referencing PipelineConfig

@dataclass(frozen=True)
class PipelineConfig:
    """Internal configuration settings for the audio processing pipeline."""
    # Paths and Names
    base_path: Path = field(default_factory=Path.cwd)
    input_audio_folderName: str = "tracks"
    chunks_audio_folderName: str = "chunked_tracks"
    whisperx_output_folderName: str = "wx_output"
    corrected_json_folderName: str = "json_files"
    srt_folderName: str = "srt_files"
    final_folderName: str = "final_outputs"
    script_chunks_folderName: str = "script_chunks"  # Will be created inside final_outputs by default
    merged_audio_filename: str = "merged_audio.flac"
    cut_points_filename: str = "cut_points.txt"
    merged_transcript_filename: str = "merged_transcript.srt"
    cleaned_transcript_filename: str = "cleaned_transcript.srt"
    final_script_filename: str = "final_script.txt"

    # Core Flags
    use_low_ram: bool = True
    run_whisperx: bool = True
    diarize: bool = False
    huggingface_token: Optional[str] = None # Sourced from env or this config

    # Track preparation (auto-unzip + speaker rename)
    auto_prepare_tracks: bool = True
    speaker_mapping_path: Optional[Path] = None  # None = use packaged default

    # Execution Control
    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, 10) # Default: run steps 1-9
    log_level: str = "INFO"

    # File Chunking Configuration
    script_chunk_size: int = 1000  # Number of lines per chunk
    script_chunk_filename_template: str = "script_chunk_{:02d}.txt"
    script_chunks_in_final_folder: bool = True  # If True, chunks go inside final_outputs/, else at base level
    script_chunks_compact_mode: bool = True  # Remove extra blank lines for more compact chunks

    # External Tools Config
    whisperx_docker_image: str = "chronicleweave-whisperx"
    whisperx_model: str = "large-v3-turbo"  # Whisper checkpoint passed to `whisperx --model`.
    whisperx_language: str = "fr"           # ISO code; "auto" lets WhisperX detect.

    # Module Specific Configs
    chunking_config: ChunkingConfig = field(default_factory=lambda: DEFAULT_CHUNKING_CONFIG)
    corrector_config: WhisperXCorrectorConfig = field(default_factory=lambda: DEFAULT_CORRECTOR_CONFIG)
    srtmerge_config: SRTMergeConfig = field(default_factory=lambda: DEFAULT_MERGE_CONFIG)
    llm_config: LLMConfig = field(default_factory=lambda: DEFAULT_LLM_CONFIG)

    def get_full_path(self, folder_or_file_attr: str) -> Path:
        """Helper to get the full path relative to the base_path."""
        return self.base_path / getattr(self, folder_or_file_attr)

    def __str__(self) -> str:
        # Creates a string representation suitable for logging, redacting sensitive info.
        cfg_dict = vars(self).copy()
        
        if 'huggingface_token' in cfg_dict and cfg_dict['huggingface_token']:
            cfg_dict['huggingface_token'] = "****REDACTED****"
        
        # Redact or summarize nested LLMConfig
        if 'llm_config' in cfg_dict and isinstance(cfg_dict['llm_config'], LLMConfig):
             # Assuming LLMConfig has its own __str__ or a similar safe representation
             cfg_dict['llm_config'] = str(cfg_dict['llm_config'])
        
        # Shorten long path representations for clarity in logs
        for key, value in cfg_dict.items():
            if isinstance(value, Path):
                cfg_dict[key] = f".../{value.name}" if len(str(value)) > 60 else str(value)
            elif isinstance(value, (ChunkingConfig, WhisperXCorrectorConfig, SRTMergeConfig)):
                # For other nested configs, you might want a simple class name or their own __str__
                cfg_dict[key] = f"<{value.__class__.__name__} Instance>"


        # Format into a string
        parts = []
        for k, v in cfg_dict.items():
            if isinstance(v, str) and len(v) > 100: # Shorten very long strings
                parts.append(f"{k}='{v[:100]}...'")
            else:
                parts.append(f"{k}={v!r}") # Use repr for most things

        return f"{self.__class__.__name__}({', '.join(parts)})"

# --- Logger Setup ---
log = logging.getLogger("chronicleweave.pipeline") # Specific logger for this module

# --- Helper Functions ---

def _build_pipeline_config(
    config_obj: Optional["PipelineConfig"],
    base_path: Union[str, Path],
    explicit_kwargs: Dict[str, Any],
    llm_config_overrides: Optional[Union[Dict[str, Any], "LLMConfig"]],
    extra_kwargs: Dict[str, Any],
) -> "PipelineConfig":
    """Merge defaults / config_obj / explicit kwargs / extra kwargs into one PipelineConfig.

    Precedence (low → high): defaults → config_obj → explicit non-None args → extra_kwargs.
    None values are ignored at every layer (so callers can pass ``None`` to mean "don't
    override"). Unknown keys are warned about and dropped.
    """
    pipeline_fields = {f.name for f in fields(PipelineConfig)}

    base = config_obj if isinstance(config_obj, PipelineConfig) else PipelineConfig()
    if config_obj is not None and not isinstance(config_obj, PipelineConfig):
        log.warning(f"Ignoring invalid config_obj type: {type(config_obj)}")

    overrides: Dict[str, Any] = {}
    for source in (explicit_kwargs, extra_kwargs):
        for key, value in source.items():
            if value is None or key == "base_path":
                continue
            if key in pipeline_fields:
                overrides[key] = value
            else:
                log.warning(f"Ignoring unknown PipelineConfig field: {key!r}")

    # base_path: explicit/extra wins over config_obj; resolve to absolute Path.
    bp_source = extra_kwargs.get("base_path") or base_path or base.base_path
    overrides["base_path"] = Path(bp_source).resolve()

    # Nested LLMConfig override: dict-merge or full replace.
    base_llm = overrides.get("llm_config") or base.llm_config
    if isinstance(llm_config_overrides, LLMConfig):
        overrides["llm_config"] = llm_config_overrides
    elif isinstance(llm_config_overrides, dict):
        llm_fields = {f.name for f in fields(LLMConfig)}
        clean = {k: v for k, v in llm_config_overrides.items() if k in llm_fields and v is not None}
        unknown = set(llm_config_overrides) - llm_fields
        if unknown:
            log.warning(f"Ignoring unknown LLMConfig fields: {unknown}")
        if clean:
            overrides["llm_config"] = replace(base_llm, **clean)
    elif llm_config_overrides is not None:
        log.warning(f"Ignoring invalid llm_config_overrides type: {type(llm_config_overrides)}")

    return replace(base, **overrides)


def should_run_step(step_number: int, steps_to_run: Union[slice, List[int], Tuple[int, ...]]) -> bool:
    """Determine if a 1-based step number should run."""
    max_steps = 10 # Steps 1-9 exist, +1 for slice range validity
    if isinstance(steps_to_run, slice):
        start, stop, step = steps_to_run.indices(max_steps)
        # Ensure start is at least 1 for 1-based indexing of steps
        start = max(1, start if start is not None else 1)
        # Ensure stop is at least start, and not more than max_steps
        stop = max(start, stop if stop is not None else max_steps)
        return step_number in range(start, stop, step)
    if isinstance(steps_to_run, (list, tuple)):
        return step_number in steps_to_run
    log.warning(f"Invalid steps_to_run type: {type(steps_to_run)}. Running step {step_number} by default as a fallback.")
    return True # Fallback, should ideally not be reached with prior checks

def _setup_logger(level: str) -> None:
    """Configure the root logger for the application."""
    log_level_upper = level.upper()
    if log_level_upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log.warning(f"Invalid log level '{level}', defaulting to INFO.")
        log_level_upper = "INFO"
    numeric_level = getattr(logging, log_level_upper)

    root_logger = logging.getLogger() # Get the root logger
    
    # Set level for the root logger. Handlers can have their own more restrictive levels.
    root_logger.setLevel(min(root_logger.level, numeric_level) if root_logger.handlers else numeric_level)


    # Check if a suitable handler (streaming to sys.stdout) already exists on the root logger
    # This prevents adding duplicate handlers if _setup_logger is called multiple times
    # or if other parts of a larger application also configure the root logger.
    handler_exists = any(
        isinstance(h, logging.StreamHandler) and getattr(h.stream, 'name', None) == sys.stdout.name
        for h in root_logger.handlers
    )

    if not handler_exists:
        handler = logging.StreamHandler(sys.stdout) # Log to STDOUT
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        # The handler's level should also be set.
        # If not set, it defaults to NOTSET (0), meaning it processes all messages
        # that the logger it's attached to passes to it.
        # Setting it explicitly can be useful if you want this handler to be more restrictive.
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
        # Use the module-specific logger for this initial message for consistency
        log.info(f"Root logger configured with a new StreamHandler (to STDOUT) at level {log_level_upper}.")
    else:
        # If a handler exists, ensure its level is at least as verbose as requested, if desired.
        # This part can be tricky: you might not want to override levels of pre-existing handlers.
        # For now, we'll just log that handlers exist.
        log.debug(f"Root logger already has handlers. Requested level: {log_level_upper}. Current root logger level: {logging.getLevelName(root_logger.level)}")
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h.stream, 'name', None) == sys.stdout.name:
                # Optionally, update the level of the existing relevant handler
                # h.setLevel(min(h.level, numeric_level) if h.level != 0 else numeric_level)
                pass # For now, don't change existing handler levels

    if 'load_dotenv' in globals() and load_dotenv(): # Check if .env was actually loaded
        log.debug(".env file loaded successfully.")
    elif 'load_dotenv' in globals():
        log.debug(".env file not found or already loaded by another mechanism.")


def run_whisperx_docker(
    session_path: Path,
    chunks_folder_name: str, # Changed to be just the name, not the full path
    config: PipelineConfig # Pass the full config for access to other settings if needed
    ) -> None:
    """Run WhisperX Docker, streaming output in real-time."""
    log.info("Attempting WhisperX transcription via Docker (streaming output)...")
    
    # Construct paths using session_path and names from config
    chunks_dir = session_path / chunks_folder_name # Chunks folder relative to session_path
    whisperx_output_dir_on_host = session_path / config.whisperx_output_folderName

    # Ensure host output directory for WhisperX exists (Docker will mount into it)
    try:
        whisperx_output_dir_on_host.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create WhisperX host output directory: {whisperx_output_dir_on_host}. Error: {e}")
        raise

    if not chunks_dir.is_dir():
        log.error(f"Chunk directory not found: {chunks_dir}")
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")

    flac_files = list(chunks_dir.glob("*.flac"))
    if not flac_files:
        log.warning(f"No .flac files found in {chunks_dir}. WhisperX will not run.")
        # Depending on requirements, this could be an error or just a skip.
        # For now, let it proceed, WhisperX CLI might handle empty input gracefully or error out.
        # raise FileNotFoundError(f"No .flac files found in {chunks_dir}")
        return # Or raise if it's critical

    log.info(f"Found {len(flac_files)} .flac files for transcription in {chunks_dir}.")

    # Docker paths (relative to the mount point inside the container)
    # session_path.resolve() is mounted to /app in the container
    container_app_path = Path("/app")
    container_chunk_dir = container_app_path / chunks_folder_name
    container_whisperx_output_dir = container_app_path / config.whisperx_output_folderName

    input_args_for_container = [str(container_chunk_dir / f.name) for f in flac_files]
    
    whisperx_args = [
        "whisperx", *input_args_for_container,
        "--model", config.whisperx_model,
        "--language", config.whisperx_language,
        "--output_dir", str(container_whisperx_output_dir),
        "--output_format", "json",
    ]

    if config.diarize:
        hf_token = config.huggingface_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            log.error("Diarization enabled, but Hugging Face token (HF_TOKEN) missing in config or environment.")
            raise ValueError("Diarization enabled, but Hugging Face token missing.")
        log.info("Diarization enabled for WhisperX.")
        whisperx_args.extend(["--diarize", "--hf_token", hf_token])
    
    # Add low VRAM options if specified, though WhisperX handles this internally to some extent
    # if config.use_low_ram: # This flag is for audio_chunker, WhisperX might have its own
    #    whisperx_args.extend(["--compute_type", "int8"]) # Example, check WhisperX docs

    volume_mount = f"{session_path.resolve()}:/app" # Mount the whole session path
    
    docker_command: List[str] = [
        "docker", "run",
        "--gpus", "all", # Assumes NVIDIA GPU and nvidia-docker toolkit
        "--rm",          # Automatically remove the container when it exits
        "-v", volume_mount,
        "--ipc=host",    # Often recommended for PyTorch/shared memory
        config.whisperx_docker_image,
        *whisperx_args
    ]

    log.info(f"Executing Docker command: {' '.join(docker_command)}")

    process = None
    try:
        # bufsize=1 for line buffering, text=True for string streams
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', # Explicit encoding
            errors='replace', # Handle potential decoding errors in output
            bufsize=1 
        )

        # Non-blocking read using select
        streams = {process.stdout: sys.stdout, process.stderr: sys.stderr}
        while process.poll() is None: # While process is running
            # select.select waits until one of the file descriptors is "ready"
            # The last argument is a timeout in seconds.
            readable_streams, _, _ = select.select(streams.keys(), [], [], 0.1) 
            
            for stream_obj in readable_streams:
                line = stream_obj.readline() # Read one line
                if line:
                    # Determine if it's STDOUT or STDERR from Docker
                    prefix = "[Docker STDOUT] " if stream_obj is process.stdout else "[Docker STDERR] "
                    # Write to corresponding system stream (sys.stdout or sys.stderr)
                    streams[stream_obj].write(prefix + line)
                    streams[stream_obj].flush() # Ensure it's written immediately

        # After process finishes, read any remaining output
        stdout_remaining, stderr_remaining = process.communicate()
        if stdout_remaining:
            for line in stdout_remaining.splitlines(): # Split into lines
                 sys.stdout.write("[Docker STDOUT] " + line + "\n")
            sys.stdout.flush()
        if stderr_remaining:
             for line in stderr_remaining.splitlines():
                 sys.stderr.write("[Docker STDERR] " + line + "\n")
             sys.stderr.flush()

        retcode = process.returncode
        if retcode != 0:
            log.error(f"WhisperX Docker command failed with exit code {retcode}.")
            raise RuntimeError(f"WhisperX Docker execution failed (exit code {retcode}).")
        else:
            log.info("WhisperX Docker command completed successfully.")

    except FileNotFoundError:
        log.exception("Docker command not found. Is Docker installed and in PATH?")
        raise
    except Exception as e:
        log.exception(f"An unexpected error occurred while running WhisperX Docker: {e}")
        raise
    finally:
        # This block executes regardless of exceptions in the try block.
        if process:
            # Ensure stdout and stderr pipes are closed
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
            if process.stderr and not process.stderr.closed:
                process.stderr.close()
            # Wait for the process to terminate to release system resources
            try:
                process.wait(timeout=5) # Wait for a short period
            except subprocess.TimeoutExpired:
                log.warning("Docker process did not terminate gracefully, attempting to kill.")
                process.kill() # Force kill if it doesn't respond
                try:
                    process.wait(timeout=5) # Wait again after kill
                except subprocess.TimeoutExpired:
                    log.error("Failed to kill Docker process.")


# --- Main Pipeline Function ---

def run_pipeline(
    base_path: str = ".", # Can be relative or absolute string
    steps_to_run: Union[slice, List[int], Tuple[int, ...]] = slice(1, 10), # Default runs steps 1-9
    log_level: str = "INFO",
    run_whisperx: bool = True,
    diarize: bool = False,
    use_low_ram: bool = True,
    input_audio_folderName: Optional[str] = None, # Override default "tracks"
    # File chunking parameters
    script_chunk_size: Optional[int] = None,  # Number of lines per chunk
    script_chunk_filename_template: Optional[str] = None,  # Filename template for chunks
    script_chunks_in_final_folder: Optional[bool] = None,  # Whether to put chunks inside final_outputs/
    script_chunks_folderName: Optional[str] = None,  # Custom folder name for chunks
    script_chunks_compact_mode: Optional[bool] = None,  # Remove extra blank lines for compact chunks
    # Allow passing a dict to override specific LLMConfig fields, or a full LLMConfig object
    llm_config_overrides: Optional[Union[Dict[str, Any], LLMConfig]] = None,
    config_obj: Optional[PipelineConfig] = None, # Pass a full pre-configured PipelineConfig
    **kwargs: Any # For other PipelineConfig overrides by keyword
    ) -> None:
    """
    Run the audio transcription and processing pipeline.

    Allows setting key parameters directly or passing a full config object.
    Direct arguments override values in config_obj or defaults. Kwargs override direct args.

    Args:
        base_path: Base directory for the session. Default: current directory.
        steps_to_run: 1-based steps to run (slice, list, or tuple). Default runs 1-9.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        run_whisperx: Whether to run the WhisperX Docker step automatically.
        diarize: Enable speaker diarization in WhisperX (requires Hugging Face token).
        use_low_ram: Use low RAM mode for audio chunking (pydub).
        input_audio_folderName: Optional override for the input tracks folder name.
        
        # File chunking parameters (Step 9):
        script_chunk_size: Number of lines per chunk. Default: 1000.
        script_chunk_filename_template: Filename template for chunks. Default: "script_chunk_{:02d}.txt".
        script_chunks_in_final_folder: If True, chunks go inside final_outputs/. Default: True.
        script_chunks_folderName: Custom folder name for chunks. Default: "script_chunks".
        script_chunks_compact_mode: Remove extra blank lines for compact chunks. Default: True.
        
        llm_config_overrides: Optional dict or LLMConfig object to override LLM settings.
        config_obj: An optional pre-configured PipelineConfig object.
        **kwargs: Additional configuration overrides matching PipelineConfig attributes.
    """
    _setup_logger(log_level) # Configure logging as the first step

    # __version__ might not be globally defined if not installed as a package
    try: version_info = __version__
    except NameError: version_info = "unknown"
    log.info(f"Initializing chronicleweave pipeline (v{version_info})...")

    explicit_kwargs = {
        "steps_to_run": steps_to_run, "log_level": log_level,
        "run_whisperx": run_whisperx, "diarize": diarize, "use_low_ram": use_low_ram,
        "input_audio_folderName": input_audio_folderName,
        "script_chunk_size": script_chunk_size,
        "script_chunk_filename_template": script_chunk_filename_template,
        "script_chunks_in_final_folder": script_chunks_in_final_folder,
        "script_chunks_folderName": script_chunks_folderName,
        "script_chunks_compact_mode": script_chunks_compact_mode,
    }
    config = _build_pipeline_config(
        config_obj=config_obj,
        base_path=base_path,
        explicit_kwargs=explicit_kwargs,
        llm_config_overrides=llm_config_overrides,
        extra_kwargs=kwargs,
    )
    config.base_path.mkdir(parents=True, exist_ok=True)

    log.info("Effective configuration applied.")
    if config.log_level == "DEBUG":
        # Use the __str__ method for a safer representation in logs
        log.debug(f"Full effective configuration: {config}")


    # Define paths using the final config
    input_audio_folder = config.get_full_path("input_audio_folderName")
    chunked_audio_folder = config.get_full_path("chunks_audio_folderName")
    whisperx_output_folder = config.get_full_path("whisperx_output_folderName")
    corrected_json_folder = config.get_full_path("corrected_json_folderName")
    srt_folder = config.get_full_path("srt_folderName")
    final_folder = config.get_full_path("final_folderName") # Main output for final user-facing files
    
    # Script chunks folder placement is conditional
    if config.script_chunks_in_final_folder:
        script_chunks_folder = final_folder / config.script_chunks_folderName
    else:
        script_chunks_folder = config.base_path / config.script_chunks_folderName

    # Specific file paths within the final_folder
    merged_file_path = final_folder / config.merged_audio_filename
    cut_points_file_path = final_folder / config.cut_points_filename
    merged_srt_path = final_folder / config.merged_transcript_filename
    cleaned_srt_path = final_folder / config.cleaned_transcript_filename
    final_script_path = final_folder / config.final_script_filename

    # Create final output folder early (if it's different from base_path)
    try:
        final_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.exception(f"Failed creating final output directory: {final_folder}")
        raise IOError(f"Could not create final output directory {final_folder}") from e

    # --- Input/Output Path Collision Check ---
    if input_audio_folder.resolve() == final_folder.resolve():
         log.warning(f"Input audio folder ('{input_audio_folder.name}') and final output folder ('{final_folder.name}') "
                     f"resolve to the same path: {input_audio_folder.resolve()}. "
                     "This might lead to data overwrites or unexpected behavior if not intended.")
    # More specific checks can be added if intermediate folders might collide dangerously.

    steps_executed: List[int] = []
    pipeline_successful = True # Assume success until a step fails critically

    # --- Step Execution ---
    # Step 0: Prepare tracks (auto-unzip Craig archive + apply speaker mapping)
    # Idempotent: skipped silently when already done. Tied to Step 1 so it only
    # runs when chunking would also run.
    if config.auto_prepare_tracks and should_run_step(1, config.steps_to_run):
        log.info("=== Step 0: Preparing tracks (unzip + speaker rename) ===")
        try:
            extracted, renamed = prepare_tracks(
                session_path=config.base_path,
                tracks_folder_name=config.input_audio_folderName,
                mapping_path=config.speaker_mapping_path,
            )
            if extracted:
                log.info("Craig archive extracted into tracks/.")
            if renamed:
                log.info(f"Speaker mapping applied: {renamed}")
            elif not extracted:
                log.debug("Step 0: nothing to do (already prepared).")
        except Exception:
            # Non-critical: if this fails, the user can still run the manual flow.
            log.exception("!!! Step 0 (Prepare Tracks) FAILED (Non-Critical) !!!")

    # Step 1: Chunking Audio
    if should_run_step(1, config.steps_to_run):
        log.info("=== Step 1: Chunking audio tracks ===")
        try:
            if not input_audio_folder.is_dir():
                raise FileNotFoundError(f"Input audio directory '{input_audio_folder}' not found.")
            chunk_audio(
                input_folder=input_audio_folder, output_dir=chunked_audio_folder,
                merged_file=merged_file_path, cut_points_file=cut_points_file_path,
                use_low_ram=config.use_low_ram, config=config.chunking_config
            )
            steps_executed.append(1); log.info("--- Step 1 completed ---")
        except Exception as e:
            log.exception("!!! Step 1 (Chunking Audio) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return # Halt pipeline on critical failure

    # Step 2: WhisperX Transcription
    if should_run_step(2, config.steps_to_run):
        log.info("=== Step 2: WhisperX transcription ===")
        try:
            if config.run_whisperx:
                # Pass session_path (which is config.base_path) and the name of the chunks folder
                run_whisperx_docker(config.base_path, config.chunks_audio_folderName, config)
            else:
                log.warning("run_whisperx is False. Manual transcription step required.")
                log.warning(f"Ensure JSON outputs from WhisperX (for chunks in '{config.chunks_audio_folderName}/') "
                            f"are placed in '{config.whisperx_output_folderName}/' relative to {config.base_path}.")
                # Basic check for outputs after manual step
                if not whisperx_output_folder.is_dir() or not any(whisperx_output_folder.iterdir()):
                    log.error(f"WhisperX output directory '{whisperx_output_folder}' is empty or missing after expected manual step.")
                    raise FileNotFoundError(f"WhisperX output dir '{whisperx_output_folder}' empty/missing after manual step.")
            steps_executed.append(2); log.info("--- Step 2 completed ---")
        except Exception as e:
            log.exception("!!! Step 2 (WhisperX Transcription) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

    # Step 3: Correcting WhisperX Output
    if should_run_step(3, config.steps_to_run):
        log.info("=== Step 3: Correcting WhisperX output ===")
        try:
            if not whisperx_output_folder.is_dir():
                raise FileNotFoundError(f"WhisperX output directory '{whisperx_output_folder}' for correction not found.")
            correct_whisperx_outputs(
                input_dir=whisperx_output_folder, output_dir=corrected_json_folder,
                config=config.corrector_config
            )
            steps_executed.append(3); log.info("--- Step 3 completed ---")
        except Exception as e:
            log.exception("!!! Step 3 (Correcting WhisperX) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

    # Step 4: Converting Corrected JSON to SRT
    if should_run_step(4, config.steps_to_run):
        log.info("=== Step 4: Converting corrected JSON to SRT ===")
        try:
            if not corrected_json_folder.is_dir():
                raise FileNotFoundError(f"Corrected JSON directory '{corrected_json_folder}' for SRT conversion not found.")
            convert_json_folder_to_srt(input_dir=corrected_json_folder, output_dir=srt_folder)
            steps_executed.append(4); log.info("--- Step 4 completed ---")
        except Exception as e:
            log.exception("!!! Step 4 (JSON to SRT) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

    # Step 5: Merging SRT files by chunk
    if should_run_step(5, config.steps_to_run):
        log.info("=== Step 5: Merging SRT files by chunk ===")
        try:
            if not srt_folder.is_dir():
                raise FileNotFoundError(f"SRT directory '{srt_folder}' for merging not found.")
            merge_srt_by_chunk(
                srt_folder=srt_folder, output_dir=final_folder, # Merged SRT goes to final_outputs
                output_filename=config.merged_transcript_filename,
                config=config.srtmerge_config
            )
            steps_executed.append(5); log.info("--- Step 5 completed ---")
        except Exception as e:
            log.exception("!!! Step 5 (Merging SRTs) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

    # Step 6: Merging successive speaker entries
    if should_run_step(6, config.steps_to_run):
        log.info("=== Step 6: Merging successive speaker entries ===")
        try:
            if not merged_srt_path.is_file():
                raise FileNotFoundError(f"Merged SRT file '{merged_srt_path}' for speaker merging not found.")
            merge_speaker_entries(input_srt_file=merged_srt_path, output_srt_file=cleaned_srt_path)
            steps_executed.append(6); log.info("--- Step 6 completed ---")
        except Exception as e:
            log.exception("!!! Step 6 (Merging Speaker Entries) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

    # Step 7: Creating final readable script
    if should_run_step(7, config.steps_to_run):
        log.info("=== Step 7: Creating final readable script ===")
        try:
            if not cleaned_srt_path.is_file():
                raise FileNotFoundError(f"Cleaned SRT file '{cleaned_srt_path}' for script generation not found.")
            srt_to_script(input_srt_path=cleaned_srt_path, output_script_path=final_script_path)
            steps_executed.append(7); log.info("--- Step 7 completed ---")
        except Exception as e:
            log.exception("!!! Step 7 (SRT to Script) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

   

    # Step 8: Chunking Final Script
    if should_run_step(8, config.steps_to_run):
        log.info("=== Step 8: Chunking final script ===")
        try:
            if not final_script_path.is_file():
                raise FileNotFoundError(f"Final script file '{final_script_path}' for chunking not found.")
            
            # Create script chunks directory
            try:
                script_chunks_folder.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.exception(f"Failed creating script chunks directory: {script_chunks_folder}")
                raise IOError(f"Could not create script chunks directory {script_chunks_folder}") from e
            
            # Chunk the final script
            chunked_files = chunk_text_file_by_lines(
                input_file=final_script_path,
                output_dir=script_chunks_folder,
                chunk_size=config.script_chunk_size,
                output_filename_template=config.script_chunk_filename_template,
                compact_mode=config.script_chunks_compact_mode
            )
            
            log.info(f"Successfully created {len(chunked_files)} script chunks in {script_chunks_folder}")
            steps_executed.append(8)
            log.info("--- Step 8 completed ---")
        except Exception as e:
            log.exception("!!! Step 8 (Script Chunking) FAILED CRITICALLY !!!")
            pipeline_successful = False
        if not pipeline_successful: return

     # Step 9: LLM Processing
    if should_run_step(9, config.steps_to_run):
        log.info("=== Step 9: LLM Processing (Summary/Narrative Generation) ===")
        try:
            if not final_script_path.is_file():
                raise FileNotFoundError(f"Final script file '{final_script_path}' for LLM processing not found.")
            
            process_with_llm(
                final_script_file=final_script_path,
                output_dir=final_folder, # LLM outputs also go to final_folder
                llm_config=config.llm_config,
                pipeline_config=config # Pass the full pipeline_config for context
            )
            steps_executed.append(9)
            log.info("--- Step 9 completed ---")
        except Exception as e:
            # LLM processing failure is logged as an error but considered non-critical for the overall pipeline result
            # if other primary transcription steps succeeded.
            log.exception("!!! Step 9 (LLM Processing) FAILED (Non-Critical) !!!")
            log.warning("LLM processing step encountered an error. Pipeline will continue if prior steps were successful.")
            # pipeline_successful is not set to False here to allow main pipeline to be "successful"
            # if only the LLM part fails. This behavior can be changed if LLM is critical.
        # No 'if not pipeline_successful: return' here for LLM step by design (non-critical)

    # --- Final Summary ---
    log.info("========================================")
    if pipeline_successful:
        # Determine what was requested vs. what ran
        all_possible_steps = list(range(1, 10)) # Steps 1 through 9
        requested_steps_list: List[int] = []
        if isinstance(config.steps_to_run, slice):
            start, stop, step_val = config.steps_to_run.indices(10) # Max steps is 10 (for 1-9)
            requested_steps_list = list(range(max(1, start), stop, step_val))
        elif isinstance(config.steps_to_run, (list, tuple)):
            requested_steps_list = [s for s in config.steps_to_run if 1 <= s <= 9]
        
        if set(steps_executed) == set(requested_steps_list):
            if set(steps_executed) == set(all_possible_steps):
                log.info(f"✅ Full pipeline (steps {steps_executed}) finished successfully!")
            else:
                log.info(f"✅ Requested pipeline steps {steps_executed} finished successfully!")
        else:
            # This might happen if LLM step failed (but was non-critical) or if some requested steps were skipped internally.
            log.warning(f"Pipeline finished. Executed steps: {steps_executed}. Requested steps: {requested_steps_list}. "
                        "Some requested steps may not have run or an optional step (like LLM) may have failed.")
        log.info(f"Find outputs in: {final_folder.resolve()}")
        log.info(f"Find script chunks in: {script_chunks_folder.resolve()}")
    else:
        log.error(f"☠️ Pipeline FAILED after executing steps: {steps_executed}. Check logs for critical errors.")
        # Re-raise a generic error to signal failure to the caller if used programmatically
        raise RuntimeError("chronicleweave pipeline failed during execution.")
