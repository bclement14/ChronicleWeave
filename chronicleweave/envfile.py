# chronicleweave/envfile.py
"""Locate and load exactly one .env file (called by the CLI after argument parsing)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

log = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def find_env_file(
    env_file: Optional[Path], cwd: Path, repo_root: Optional[Path] = REPO_ROOT
) -> Optional[Path]:
    """Return the .env file to load: explicit file, else first .env walking up from cwd,
    else <repo_root>/.env. Raises FileNotFoundError for a missing explicit file."""
    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--env-file not found or not a file: {path}")
        return path
    current = Path(cwd).resolve()
    for folder in (current, *current.parents):
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    if repo_root is not None:
        candidate = Path(repo_root) / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(
    env_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
    repo_root: Optional[Path] = REPO_ROOT,
) -> Optional[Path]:
    """Load one .env file into os.environ without overriding existing variables."""
    path = find_env_file(env_file, cwd or Path.cwd(), repo_root)
    if path is None:
        log.info("No .env file found; using the shell environment only.")
        return None
    load_dotenv(path, override=False)
    log.info(f"Loaded environment from {path}")
    return path
