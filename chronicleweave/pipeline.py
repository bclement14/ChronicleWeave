# chronicleweave/pipeline.py

import json
import os
import re
import shutil
import subprocess
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, replace, fields
from typing import Optional, Union, List, Sequence, Dict, Any, Mapping, NamedTuple, Iterator, Set, Tuple

# --- Imports ---
from .modules.audio_chunker import chunk_audio, ChunkingConfig, DEFAULT_CHUNKING_CONFIG
from .modules.prepare_tracks import prepare_tracks, speaker_config_from_env, SpeakerConfig
from .modules.whisperx_corrector_core import correct_whisperx_outputs, WhisperXCorrectorConfig, DEFAULT_CONFIG as DEFAULT_CORRECTOR_CONFIG
from .modules.convert_json_to_srt import convert_json_folder_to_srt
from .modules.merge_srt_by_chunk import merge_srt_by_chunk
from .modules.merge_speaker_entries import merge_speaker_entries
from .modules.convert_srt_to_script import srt_to_script
from .modules.llm_processor import process_with_llm, LLMConfig, DEFAULT_LLM_CONFIG
from .modules.file_chunker import chunk_text_file_by_lines, validate_filename_template

# --- Configuration Dataclasses ---

# Forward declare LLMConfig for PipelineConfig's type hint if LLMConfig needs its own __str__
# For simplicity, if LLMConfig is simple or its __str__ is also basic, this might not be strictly needed.
# However, it's good practice if they reference each other in complex ways.
# class LLMConfig: ... # This would be needed if LLMConfig also had a complex __str__ referencing PipelineConfig

class StepRange(NamedTuple):
    first: int
    last: int

    def includes(self, step: int) -> bool:
        return self.first <= step <= self.last


DEFAULT_STEPS = StepRange(1, 8)
LAST_CLEANUP_STEP = 8
DONE_MARKER_NAME = ".chronicleweave_done.json"
RUNNING_MARKER_NAME = ".chronicleweave_running"
LOG_FILE_NAME = "pipeline.log"


class PipelineError(RuntimeError):
    def __init__(self, step: int, message: str):
        super().__init__(f"Step {step} failed: {message}")
        self.step = step


def parse_steps(spec) -> StepRange:
    if spec is None:
        return DEFAULT_STEPS
    if isinstance(spec, StepRange):
        first, last = spec
    elif isinstance(spec, str):
        text = spec.strip()
        try:
            if "-" in text:
                a, b = text.split("-", 1)
                first, last = int(a), int(b)
            else:
                first = last = int(text)
        except ValueError as e:
            raise ValueError(f"Invalid --steps value {spec!r}; use e.g. '1-8' or '5'") from e
    elif isinstance(spec, int):
        first = last = spec
    elif isinstance(spec, slice):
        first = spec.start if spec.start is not None else 1
        last = (spec.stop - 1) if spec.stop is not None else 9
    elif isinstance(spec, (list, tuple)):
        values = sorted({int(v) for v in spec})
        if not values:
            raise ValueError("Empty step selection")
        if values != list(range(values[0], values[-1] + 1)):
            raise ValueError(f"Steps must be one contiguous range, got {list(spec)}")
        first, last = values[0], values[-1]
    else:
        raise ValueError(f"Unsupported steps value {spec!r}")
    if not (1 <= first <= last <= 9):
        raise ValueError(f"Steps must satisfy 1 <= first <= last <= 9, got {first}-{last}")
    return StepRange(first, last)


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
    run_whisperx: bool = True
    diarize: bool = False
    huggingface_token: Optional[str] = None # Sourced from env or this config

    # Track preparation (auto-unzip + speaker rename)
    auto_prepare_tracks: bool = True
    speakers: Optional[SpeakerConfig] = None  # None -> read CW_SPEAKERS/CW_IGNORE_TRACKS from the environment

    # Execution Control
    steps_to_run: StepRange = DEFAULT_STEPS
    log_level: str = "INFO"

    # File Chunking Configuration
    script_chunk_size: int = 1000  # Number of lines per chunk
    script_chunk_filename_template: str = "script_chunk_{:02d}.txt"
    script_chunks_in_final_folder: bool = True  # If True, chunks go inside final_outputs/, else at base level
    script_chunks_compact_mode: bool = True  # Remove extra blank lines for more compact chunks

    # External Tools Config
    whisperx_docker_image: str = "chronicleweave-whisperx"
    whisperx_model: str = "large-v3"        # Whisper checkpoint, or a local model folder, passed to `whisperx --model`.
    whisperx_language: str = "fr"           # ISO code; "auto" lets WhisperX detect.
    whisperx_cache_dir: Optional[Path] = None  # None -> DEFAULT_CACHE_DIR; mounted at /root/.cache.

    # Module Specific Configs
    chunking_config: ChunkingConfig = field(default_factory=lambda: DEFAULT_CHUNKING_CONFIG)
    corrector_config: WhisperXCorrectorConfig = field(default_factory=lambda: DEFAULT_CORRECTOR_CONFIG)
    llm_config: LLMConfig = field(default_factory=lambda: DEFAULT_LLM_CONFIG)

    def __post_init__(self) -> None:
        validate_filename_template(self.script_chunk_filename_template)  # step 8 writes only inside its folder

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
            elif isinstance(value, (ChunkingConfig, WhisperXCorrectorConfig)):
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
docker_log = logging.getLogger("chronicleweave.docker")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "chronicleweave"


def redact_command(cmd: Sequence[str]) -> str:
    parts = list(cmd)
    for i, part in enumerate(parts[:-1]):
        if part == "--hf_token":
            parts[i + 1] = "****"
    return " ".join(parts)


def build_whisperx_command(
    session_path: Path,
    chunk_files: Sequence[Path],
    config: PipelineConfig,
    environ: Mapping[str, str] = os.environ,
) -> List[str]:
    cache_dir = (config.whisperx_cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)  # created by us, not root-owned by docker
    mounts = ["-v", f"{session_path.resolve()}:/app", "-v", f"{cache_dir}:/root/.cache"]
    model_arg = config.whisperx_model
    model_path = Path(config.whisperx_model).expanduser()
    if model_path.is_dir():
        container_model = f"/models/{model_path.name}"
        mounts += ["-v", f"{model_path.resolve()}:{container_model}:ro"]
        model_arg = container_model
    env_args: List[str] = []
    if environ.get("HF_HUB_OFFLINE"):
        env_args += ["-e", f"HF_HUB_OFFLINE={environ['HF_HUB_OFFLINE']}"]
    container_chunks = Path("/app") / config.chunks_audio_folderName
    whisperx_args = [
        "whisperx", *[str(container_chunks / f.name) for f in chunk_files],
        "--model", model_arg,
        "--language", config.whisperx_language,
        "--output_dir", str(Path("/app") / config.whisperx_output_folderName),
        "--output_format", "json",
    ]
    if config.diarize:
        hf_token = config.huggingface_token or environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Diarization enabled, but Hugging Face token (HF_TOKEN) missing.")
        whisperx_args += ["--diarize", "--hf_token", hf_token]
    return ["docker", "run", "--gpus", "all", "--rm", "--ipc=host", *mounts, *env_args,
            config.whisperx_docker_image, *whisperx_args]

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


def should_run_step(step_number: int, steps_to_run) -> bool:
    return parse_steps(steps_to_run).includes(step_number)


def _script_chunks_folder(config: PipelineConfig) -> Path:
    if config.script_chunks_in_final_folder:
        return config.get_full_path("final_folderName") / config.script_chunks_folderName
    return config.base_path / config.script_chunks_folderName


def step_output_paths(config: PipelineConfig, step: int) -> List[Path]:
    final = config.get_full_path("final_folderName")
    table = {
        1: [config.get_full_path("chunks_audio_folderName"), final / config.merged_audio_filename,
            final / config.cut_points_filename],
        2: [config.get_full_path("whisperx_output_folderName")] if config.run_whisperx else [],
        3: [config.get_full_path("corrected_json_folderName")],
        4: [config.get_full_path("srt_folderName")],
        5: [final / config.merged_transcript_filename],
        6: [final / config.cleaned_transcript_filename],
        7: [final / config.final_script_filename],
        8: [_script_chunks_folder(config)],
    }
    return table.get(step, [])


def _folded(path: Path) -> Tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def _same_or_inside(path: Path, folder: Path) -> bool:
    """`path` is `folder` or lies below it, ignoring letter case: session folders live on
    case-insensitive NTFS, where `Tracks` and `tracks` are the same folder."""
    p, f = _folded(path), _folded(folder)
    return p[:len(f)] == f


def _strictly_inside(path: Path, folder: Path) -> bool:
    """`path` lies below `folder` (exact match) and is not `folder` itself in any letter case."""
    return folder in path.parents and _folded(path) != _folded(folder)


def _is_archive_name(name: str) -> bool:
    folded = name.casefold()
    return folded.startswith("craig-") and folded.endswith(".zip")


def _archive_below(folder: Path) -> Optional[Path]:
    """The first Craig archive found anywhere below `folder` (symlinks are not followed, like rmtree)."""
    if not folder.is_dir() or folder.is_symlink():
        return None
    for dirpath, _dirnames, filenames in os.walk(folder, followlinks=False):
        for name in filenames:
            if _is_archive_name(name):
                return Path(dirpath) / name
    return None


def check_tracks_folder(config: PipelineConfig) -> None:
    """The input tracks folder must be a folder inside the session (step 0 renames and deletes there)."""
    session = config.base_path.resolve()
    tracks = config.get_full_path("input_audio_folderName").resolve()
    if not _strictly_inside(tracks, session):
        raise PipelineError(0, f"The tracks folder {tracks} must be a folder inside the session folder {session}")


def _guard_deletable(config: PipelineConfig, paths: Sequence[Path]) -> None:
    """Refuse the whole cleanup, before anything is deleted, if any path is unsafe to delete."""
    session = config.base_path.resolve()
    tracks = config.get_full_path("input_audio_folderName").resolve()
    final = config.get_full_path("final_folderName").resolve()
    for path in paths:
        resolved = path.resolve()
        if not _strictly_inside(resolved, session):
            raise PipelineError(0, f"Refusing to delete {path}: outside the session folder {session}")
        if _same_or_inside(resolved, tracks) or _same_or_inside(tracks, resolved):
            raise PipelineError(0, f"Refusing to delete {path}: overlaps the input tracks folder {tracks}")
        if _same_or_inside(final, resolved):
            raise PipelineError(0, f"Refusing to delete {path}: it is or contains the final outputs folder {final}")
        if _is_archive_name(resolved.name):
            raise PipelineError(0, f"Refusing to delete Craig archive {path}")
        archive = _archive_below(resolved)
        if archive is not None:
            raise PipelineError(0, f"Refusing to delete {path}: it contains the Craig archive {archive}")


def _cleanup_paths(config: PipelineConfig, first_step: int) -> List[Path]:
    return [p for s in range(first_step, LAST_CLEANUP_STEP + 1) for p in step_output_paths(config, s)]


def cleanup_outputs(config: PipelineConfig, first_step: int) -> List[Path]:
    paths = _cleanup_paths(config, first_step)
    _guard_deletable(config, paths)
    removed: List[Path] = []
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
        elif path.exists():
            path.unlink()
            removed.append(path)
    if removed:
        log.info("Removed previous outputs: " + ", ".join(str(p.relative_to(config.base_path)) for p in removed))
    return removed


class _StepInput(NamedTuple):
    producer: int           # step that writes it; 0 = made outside steps 1-9 (manual-mode WhisperX JSON)
    path: Path
    pattern: Optional[str]  # glob that must match inside `path`; None = `path` is a file
    what: str


def _step_inputs(config: PipelineConfig, step: int) -> List[_StepInput]:
    """What step `step` reads. The expected set (coverage, final gate) comes from chunked_tracks/,
    or from the hand-placed wx_output/ in manual mode (see expected_chunk_stems)."""
    final = config.get_full_path("final_folderName")
    wx_json = _StepInput(2 if config.run_whisperx else 0, config.get_full_path("whisperx_output_folderName"),
                         "*.json", "WhisperX JSON files")
    expected = (_StepInput(1, config.get_full_path("chunks_audio_folderName"), "*.flac", "chunk FLAC files")
                if config.run_whisperx else wx_json)
    corrected = _StepInput(3, config.get_full_path("corrected_json_folderName"), "*.json", "corrected JSON files")
    cut_points = _StepInput(1, final / config.cut_points_filename, None, "cut points")
    script = _StepInput(7, final / config.final_script_filename, None, "final script")
    table = {
        2: [expected],
        3: [wx_json, expected],
        4: [corrected, expected],
        5: [_StepInput(4, config.get_full_path("srt_folderName"), "*.srt", "SRT files"), cut_points],
        6: [_StepInput(5, final / config.merged_transcript_filename, None, "merged transcript")],
        7: [_StepInput(6, final / config.cleaned_transcript_filename, None, "cleaned transcript"),
            expected, corrected],
        8: [script],
        9: [script],
    }
    return table.get(step, [])


def check_step_inputs(config: PipelineConfig, first_step: int, last_step: Optional[int] = None) -> None:
    """Fail before anything is deleted if a step in first..last needs an input that no step in the range makes.

    `last_step` defaults to the default range end (8), or `first_step` when that is later."""
    last = max(first_step, DEFAULT_STEPS.last) if last_step is None else last_step
    if first_step == 1:
        tracks = config.get_full_path("input_audio_folderName")
        has_archive = any(p.is_file() for p in config.base_path.glob("craig-*.flac.zip"))
        if not has_archive and not (tracks.is_dir() and any(tracks.glob("*.flac"))):
            raise PipelineError(1, f"missing input for step 1: no Craig archive and no FLAC in {tracks}")
    checked: Set[Path] = set()
    for step in range(first_step, last + 1):
        for need in _step_inputs(config, step):
            if need.producer >= first_step or need.path in checked:
                continue  # made by a step of this run, or already checked
            checked.add(need.path)
            if need.pattern is None:
                if not need.path.is_file():
                    raise PipelineError(first_step, f"missing input for step {step}: {need.what} {need.path}")
            elif not need.path.is_dir() or not any(need.path.glob(need.pattern)):
                raise PipelineError(first_step, f"missing input for step {step}: no {need.what} in {need.path}")


def expected_chunk_stems(config: PipelineConfig, step: int = 2) -> List[str]:
    """Every chunk item step 1 produced (or, in manual mode, every hand-placed WhisperX JSON).
    `step` is the step reported if none is found."""
    if config.run_whisperx:
        folder, pattern = config.get_full_path("chunks_audio_folderName"), "*.flac"
    else:
        folder, pattern = config.get_full_path("whisperx_output_folderName"), "*.json"
    stems = sorted(p.stem for p in folder.glob(pattern)) if folder.is_dir() else []
    if not stems:
        raise PipelineError(step, f"no chunk items found in {folder}")
    return stems


def check_coverage(step: int, folder: Path, expected: Sequence[str], suffix: str) -> None:
    found = {p.stem for p in folder.glob(f"*{suffix}")} if folder.is_dir() else set()
    missing = sorted(set(expected) - found)
    extra = sorted(found - set(expected))
    if missing or extra:
        raise PipelineError(step, f"{folder.name}: missing {missing[:10]}; unexpected {extra[:10]}")


_TAG_RE = re.compile(r"^\[([^\]]+)\]")


def _speakers_with_speech(json_folder: Path) -> Set[str]:
    speaking: Set[str] = set()
    for path in json_folder.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            text = str(seg.get("text") or "").strip()
            if not text and isinstance(seg.get("words"), list):
                text = " ".join(str(w.get("word", "")).strip() for w in seg["words"] if isinstance(w, dict)).strip()
            if text:
                speaking.add(path.stem.rsplit("-", 1)[0])
                break
    return speaking


def check_final_script(config: PipelineConfig, expected: Sequence[str], speakers: Optional[SpeakerConfig]) -> None:
    script = config.get_full_path("final_folderName") / config.final_script_filename
    text = script.read_text(encoding="utf-8") if script.is_file() else ""
    if not text.strip():
        raise PipelineError(7, f"final script is empty: {script}")
    allowed = {stem.rsplit("-", 1)[0] for stem in expected}
    if speakers is not None and not allowed <= speakers.tags:
        raise PipelineError(7, f"chunk speakers {sorted(allowed - speakers.tags)} are not mapped Tags")
    found = {m.group(1) for line in text.splitlines() if (m := _TAG_RE.match(line))}
    unexpected = found - allowed
    if unexpected:
        raise PipelineError(7, f"unexpected tags in final script: {sorted(unexpected)}")
    missing = _speakers_with_speech(config.get_full_path("corrected_json_folderName")) - found
    if missing:
        raise PipelineError(7, f"speakers with transcribed speech missing from the script: {sorted(missing)}")


def _git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() or None
    except Exception:
        return None


def write_completion_marker(config: PipelineConfig, steps: StepRange, started: datetime, finished: datetime) -> Path:
    marker = config.get_full_path("final_folderName") / DONE_MARKER_NAME
    payload = {
        "steps": [steps.first, steps.last],
        "whisperx_model": config.whisperx_model,
        "whisperx_language": config.whisperx_language,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
    }
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, marker)
    return marker


@contextmanager
def session_log_file(session_path: Path, level: int) -> Iterator[Path]:
    """Write the session's log to pipeline.log at INFO or more verbose, whatever the console level."""
    path = session_path / LOG_FILE_NAME
    file_level = min(level, logging.INFO)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(file_level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(min(previous_level, file_level) if previous_level else file_level)
    root.addHandler(handler)  # the console handler keeps its own (possibly higher) level
    try:
        yield path
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()

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


def run_whisperx_docker(
    session_path: Path,
    chunks_folder_name: str, # Changed to be just the name, not the full path
    config: PipelineConfig # Pass the full config for access to other settings if needed
    ) -> None:
    chunks_dir = session_path / chunks_folder_name
    out_dir = session_path / config.whisperx_output_folderName
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")
    flac_files = sorted(chunks_dir.glob("*.flac"))
    if not flac_files:
        raise FileNotFoundError(f"No .flac chunk files in {chunks_dir}; nothing to transcribe.")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_whisperx_command(session_path, flac_files, config)
    log.info(f"Transcribing {len(flac_files)} chunk files with WhisperX ({config.whisperx_model}).")
    log.info(f"Executing Docker command: {redact_command(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        docker_log.info(line.rstrip())
    retcode = process.wait()
    if retcode != 0:
        raise RuntimeError(f"WhisperX Docker execution failed (exit code {retcode}).")
    log.info("WhisperX Docker command completed successfully.")


# --- Main Pipeline Function ---

def _run_step(number: int, title: str, fn):
    log.info(f"=== Step {number}: {title} ===")
    try:
        result = fn()
    except PipelineError:
        raise
    except Exception as e:
        log.exception(f"Step {number} ({title}) failed")
        raise PipelineError(number, f"{title}: {e}") from e
    log.info(f"--- Step {number} completed ---")
    return result


def run_pipeline(
    base_path: str = ".",
    steps_to_run=None,
    log_level: str = "INFO",
    run_whisperx: bool = True,
    diarize: bool = False,
    input_audio_folderName: Optional[str] = None,
    script_chunk_size: Optional[int] = None,
    script_chunk_filename_template: Optional[str] = None,
    script_chunks_in_final_folder: Optional[bool] = None,
    script_chunks_folderName: Optional[str] = None,
    script_chunks_compact_mode: Optional[bool] = None,
    llm_config_overrides: Optional[Union[Dict[str, Any], LLMConfig]] = None,
    config_obj: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> None:
    """Run the pipeline for one session. Raises PipelineError on any failure."""
    _setup_logger(log_level)
    explicit_kwargs = {
        "steps_to_run": parse_steps(steps_to_run) if steps_to_run is not None else None,
        "log_level": log_level, "run_whisperx": run_whisperx, "diarize": diarize,
        "input_audio_folderName": input_audio_folderName,
        "script_chunk_size": script_chunk_size,
        "script_chunk_filename_template": script_chunk_filename_template,
        "script_chunks_in_final_folder": script_chunks_in_final_folder,
        "script_chunks_folderName": script_chunks_folderName,
        "script_chunks_compact_mode": script_chunks_compact_mode,
    }
    config = _build_pipeline_config(config_obj, base_path, explicit_kwargs, llm_config_overrides, kwargs)
    steps = parse_steps(config.steps_to_run)
    config.base_path.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(config.log_level).upper(), logging.INFO)
    with session_log_file(config.base_path, level):
        try:
            _run_steps(config, steps)
        except PipelineError as e:
            log.error(f"Pipeline FAILED for {config.base_path.name}: {e}")
            raise


def _run_steps(config: PipelineConfig, steps: StepRange) -> None:
    started = datetime.now()
    final = config.get_full_path("final_folderName")
    tracks = config.get_full_path("input_audio_folderName")
    chunks = config.get_full_path("chunks_audio_folderName")
    wx = config.get_full_path("whisperx_output_folderName")
    jsons = config.get_full_path("corrected_json_folderName")
    srts = config.get_full_path("srt_folderName")
    cut_points = final / config.cut_points_filename
    merged_srt = final / config.merged_transcript_filename
    cleaned_srt = final / config.cleaned_transcript_filename
    script = final / config.final_script_filename
    done_marker = final / DONE_MARKER_NAME
    running_marker = final / RUNNING_MARKER_NAME
    rebuilds_script = steps.first <= 7
    log.info(f"Session {config.base_path} — steps {steps.first}-{steps.last}, model {config.whisperx_model}")
    check_tracks_folder(config)  # before step 0, which renames and deletes inside it

    # Start of run, in this order (spec 4.7): tracks-folder check -> speaker config -> step 0 -> input
    # checks -> delete-path guard -> done-marker removal -> run-in-progress marker -> cleanup. Nothing is
    # deleted before every check has passed; step 0 only touches tracks/.
    speakers = config.speakers
    if speakers is None and (steps.includes(1) or steps.includes(7)):
        # an input of step 0 and of the step-7 gate; a missing CW_SPEAKERS fails here, logged
        speakers = _run_step(0, "Reading speaker configuration", speaker_config_from_env)
    if steps.includes(1):
        _run_step(0, "Preparing tracks", lambda: prepare_tracks(
            config.base_path, speakers, config.input_audio_folderName, rename=config.auto_prepare_tracks))
    check_step_inputs(config, steps.first, steps.last)
    if steps.first <= LAST_CLEANUP_STEP:
        _guard_deletable(config, _cleanup_paths(config, steps.first))
    final.mkdir(parents=True, exist_ok=True)
    if rebuilds_script:
        done_marker.unlink(missing_ok=True)  # the final script is about to be rebuilt
        running_marker.write_text(started.isoformat(timespec="seconds"), encoding="utf-8")
    if steps.first <= LAST_CLEANUP_STEP:
        _run_step(0, "Removing previous outputs", lambda: cleanup_outputs(config, steps.first))

    if steps.includes(1):
        _run_step(1, "Chunking audio", lambda: chunk_audio(
            tracks, chunks, final / config.merged_audio_filename, cut_points, config.chunking_config))

    def transcribe():
        if config.run_whisperx:
            run_whisperx_docker(config.base_path, config.chunks_audio_folderName, config)
        check_coverage(2, wx, expected_chunk_stems(config, 2), ".json")

    if steps.includes(2):
        _run_step(2, "WhisperX transcription", transcribe)
    if steps.includes(3):
        def correct():
            correct_whisperx_outputs(wx, jsons, overwrite=True, config=config.corrector_config)
            check_coverage(3, jsons, expected_chunk_stems(config, 3), ".json")
        _run_step(3, "Correcting WhisperX output", correct)
    if steps.includes(4):
        def to_srt():
            convert_json_folder_to_srt(input_dir=jsons, output_dir=srts)
            check_coverage(4, srts, expected_chunk_stems(config, 4), ".srt")
        _run_step(4, "Converting JSON to SRT", to_srt)
    if steps.includes(5):
        _run_step(5, "Merging SRT files by chunk", lambda: merge_srt_by_chunk(
            srt_folder=srts, output_dir=final, output_filename=config.merged_transcript_filename,
            cut_points_file=cut_points))
    if steps.includes(6):
        _run_step(6, "Merging successive speaker entries", lambda: merge_speaker_entries(
            input_srt_file=merged_srt, output_srt_file=cleaned_srt))
    if steps.includes(7):
        def write_script():
            srt_to_script(input_srt_path=cleaned_srt, output_script_path=script)
            check_final_script(config, expected_chunk_stems(config, 7), speakers)
        _run_step(7, "Creating final script", write_script)
    if steps.includes(8):
        def chunk_script():
            folder = _script_chunks_folder(config)
            folder.mkdir(parents=True, exist_ok=True)
            chunk_text_file_by_lines(input_file=script, output_dir=folder,
                                     chunk_size=config.script_chunk_size,
                                     output_filename_template=config.script_chunk_filename_template,
                                     compact_mode=config.script_chunks_compact_mode)
        _run_step(8, "Chunking final script", chunk_script)
    if steps.includes(9):
        _run_step(9, "LLM processing", lambda: process_with_llm(
            final_script_file=script, output_dir=final, llm_config=config.llm_config, pipeline_config=config))

    if steps.includes(7):  # the final gate passed
        write_completion_marker(config, steps, started, datetime.now())
        running_marker.unlink(missing_ok=True)
    log.info(f"Steps {steps.first}-{steps.last} finished successfully for {config.base_path.name}.")
