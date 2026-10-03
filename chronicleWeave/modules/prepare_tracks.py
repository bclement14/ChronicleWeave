# chronicleweave/modules/prepare_tracks.py
"""Prepare a session's `tracks/` folder before chunking.

Two responsibilities:

1. **Unzip**: if a Craig recording archive (``craig-*.flac.zip``) is sitting at
   the session root and ``tracks/`` is missing or empty, extract the archive's
   ``*.flac`` files into ``tracks/``.
2. **Rename**: apply a project-level speaker mapping (loaded from JSON) so that
   raw Discord usernames like ``1-player_one.flac`` become friendly display
   names like ``Titar.flac``. Downstream stages pick up the display name
   automatically because the speaker label is derived from the filename stem.

Both operations are idempotent: running twice on the same session does nothing
the second time.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Default location for the mapping file: shipped inside the chronicleweave package.
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "speaker_mapping.json"

# Filenames inside Craig archives that we don't want to extract as audio tracks.
_CRAIG_NON_TRACK_FILES = {"info.txt", "raw.dat"}


@dataclass(frozen=True)
class SpeakerRule:
    """A single mapping rule. ``match`` is a case-insensitive substring."""
    match: str
    name: str


def load_speaker_mapping(path: Optional[Path] = None) -> List[SpeakerRule]:
    """Load mapping rules from JSON. Returns [] if the file is missing."""
    mapping_path = Path(path) if path else DEFAULT_MAPPING_PATH
    if not mapping_path.is_file():
        log.info(f"No speaker mapping file at {mapping_path}; raw track names will be kept.")
        return []
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.error(f"Could not read speaker mapping {mapping_path}: {e}. Skipping rename.")
        return []

    raw_rules = data.get("mappings", []) if isinstance(data, dict) else data
    rules: List[SpeakerRule] = []
    for entry in raw_rules:
        if not isinstance(entry, dict):
            continue
        match = entry.get("match")
        name = entry.get("name")
        if isinstance(match, str) and isinstance(name, str) and match and name:
            rules.append(SpeakerRule(match=match.lower(), name=name))
        else:
            log.warning(f"Ignoring malformed mapping entry: {entry!r}")
    log.debug(f"Loaded {len(rules)} speaker mapping rule(s) from {mapping_path}.")
    return rules


def _username_from_track(stem: str) -> str:
    """Strip leading ``N-`` index and leading underscores from a Craig track stem.

    e.g. ``"1-player_one"`` -> ``"player_one"``,
         ``"4-_playerfour"``  -> ``"playerfour"``.
    """
    stripped = re.sub(r"^\d+-", "", stem)
    return stripped.lstrip("_")


def resolve_display_name(stem: str, rules: List[SpeakerRule]) -> Optional[str]:
    """Return the display name for a track stem, or None if no rule matches."""
    username = _username_from_track(stem).lower()
    for rule in rules:
        if rule.match in username:
            return rule.name
    return None


def _find_craig_zip(session_path: Path) -> Optional[Path]:
    """Return the first ``craig-*.flac.zip`` directly under the session, if any."""
    candidates = sorted(session_path.glob("craig-*.flac.zip"))
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(
            f"Multiple Craig archives found in {session_path}; using {candidates[0].name}."
        )
    return candidates[0]


def unzip_craig_archive(session_path: Path, tracks_folder: Path) -> bool:
    """Extract ``*.flac`` from a Craig archive into ``tracks_folder``.

    Returns True if extraction happened, False if it was skipped (no archive or
    tracks already extracted). Non-flac archive members (info.txt, raw.dat) are
    deliberately not extracted.
    """
    archive = _find_craig_zip(session_path)
    if archive is None:
        return False

    if tracks_folder.is_dir() and any(tracks_folder.glob("*.flac")):
        log.info(
            f"tracks/ already populated ({tracks_folder}); skipping extraction of {archive.name}."
        )
        return False

    tracks_folder.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {archive.name} -> {tracks_folder}")
    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_name = Path(member.filename).name
            if not member_name or member.is_dir():
                continue
            if member_name in _CRAIG_NON_TRACK_FILES:
                continue
            if not member_name.lower().endswith(".flac"):
                continue
            target = tracks_folder / member_name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    log.info(f"Extracted {extracted} flac track(s) from {archive.name}.")
    return extracted > 0


def apply_speaker_mapping(
    tracks_folder: Path,
    rules: List[SpeakerRule],
) -> Dict[str, str]:
    """Rename ``tracks_folder/*.flac`` according to the mapping rules.

    Tracks whose username matches a rule are renamed to ``<DisplayName>.flac``.
    Tracks already matching a display name (e.g. on a re-run) are left alone.
    Unknown tracks are kept under their original name and logged.

    Returns a dict ``{old_stem: new_stem}`` for the tracks that were renamed.
    """
    if not rules:
        return {}
    if not tracks_folder.is_dir():
        return {}

    display_names = {rule.name for rule in rules}
    renamed: Dict[str, str] = {}

    for track in sorted(tracks_folder.glob("*.flac")):
        old_stem = track.stem
        if old_stem in display_names:
            continue  # already mapped on a previous run
        display = resolve_display_name(old_stem, rules)
        if display is None:
            log.warning(f"No mapping rule matches track '{track.name}'; leaving unchanged.")
            continue
        new_path = tracks_folder / f"{display}.flac"
        if new_path.exists() and new_path != track:
            log.warning(
                f"Cannot rename {track.name} -> {new_path.name}: target already exists."
            )
            continue
        track.rename(new_path)
        renamed[old_stem] = display
        log.info(f"Renamed {track.name} -> {new_path.name}")
    return renamed


def prepare_tracks(
    session_path: Path,
    tracks_folder_name: str = "tracks",
    mapping_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, str]]:
    """Run unzip + rename for a session. Idempotent.

    Returns ``(extracted, renamed)`` where ``extracted`` is True iff a Craig
    archive was unpacked this call, and ``renamed`` is the per-track rename
    map.
    """
    session_path = Path(session_path)
    tracks_folder = session_path / tracks_folder_name

    extracted = unzip_craig_archive(session_path, tracks_folder)
    rules = load_speaker_mapping(mapping_path)
    renamed = apply_speaker_mapping(tracks_folder, rules)
    return extracted, renamed
