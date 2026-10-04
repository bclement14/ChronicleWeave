# chronicleweave/batch.py
"""Run the pipeline over every session folder under a root, one after another."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from .pipeline import DONE_MARKER_NAME, LOG_FILE_NAME, RUNNING_MARKER_NAME, run_pipeline

log = logging.getLogger(__name__)
SESSION_DIR_RE = re.compile(r"^session(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SessionDir:
    number: int
    path: Path


@dataclass
class BatchResult:
    session: SessionDir
    status: str
    duration_s: float
    log_path: Optional[Path]
    error: Optional[str]


def discover_sessions(root: Path) -> List[SessionDir]:
    found: List[SessionDir] = []
    for child in Path(root).iterdir():
        m = SESSION_DIR_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        has_archive = any(p.is_file() for p in child.glob("craig-*.flac.zip"))
        tracks = child / "tracks"
        has_tracks = tracks.is_dir() and any(tracks.glob("*.flac"))
        if has_archive or has_tracks:
            found.append(SessionDir(int(m.group(1)), child))
    return sorted(found, key=lambda s: s.number)


def is_session_done(path: Path, final_folder: str = "final_outputs", script_name: str = "final_script.txt") -> bool:
    final = Path(path) / final_folder
    if (final / DONE_MARKER_NAME).is_file():
        return True
    # Legacy sessions have a script and no run-in-progress marker. pipeline.log is not used:
    # every run writes it, including step-9-only runs on old sessions.
    return (final / script_name).is_file() and not (final / RUNNING_MARKER_NAME).exists()


def parse_only(spec: Optional[str]) -> Optional[Set[int]]:
    if spec is None:
        return None
    numbers: Set[int] = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            numbers.update(range(int(a), int(b) + 1))
        else:
            numbers.add(int(part))
    return numbers


def run_batch(
    root: Path,
    only: Optional[Set[int]],
    force: bool,
    pipeline_kwargs: Dict[str, Any],
    runner: Callable[..., None] = run_pipeline,
) -> List[BatchResult]:
    results: List[BatchResult] = []
    for session in discover_sessions(root):
        if only is not None and session.number not in only:
            continue
        log_path = session.path / LOG_FILE_NAME
        if not force and is_session_done(session.path):
            log.info(f"{session.path.name}: already done, skipping.")
            results.append(BatchResult(session, "SKIPPED", 0.0, None, None))
            continue
        log.info(f"===== {session.path.name}: starting =====")
        start = time.monotonic()
        try:
            runner(base_path=str(session.path), **pipeline_kwargs)
            results.append(BatchResult(session, "OK", time.monotonic() - start, log_path, None))
        except Exception as e:  # one failed session must not stop the batch
            log.error(f"{session.path.name}: FAILED: {e}")
            results.append(BatchResult(session, "FAILED", time.monotonic() - start, log_path, str(e)))
    return results


def format_summary(results: Sequence[BatchResult]) -> str:
    lines = [f"{'session':<14}{'status':<9}{'minutes':>8}  log / error"]
    for r in results:
        detail = r.error if r.error else (str(r.log_path) if r.log_path else "")
        lines.append(f"{r.session.path.name:<14}{r.status:<9}{r.duration_s / 60:>8.1f}  {detail}")
    return "\n".join(lines)
