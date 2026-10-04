# chronicleweave/modules/prepare_tracks.py
"""Step 0: prepare a session's tracks/ folder from a Craig archive.

Speaker names come from CW_SPEAKERS ("username:Tag,...") and dropped tracks from
CW_IGNORE_TRACKS ("token,..."). Archives are never modified.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

log = logging.getLogger(__name__)

SPEAKERS_VAR = "CW_SPEAKERS"
IGNORE_VAR = "CW_IGNORE_TRACKS"
_EXTRACT_TMP = ".extract-tmp"


class TrackPreparationError(ValueError):
    """Raised when tracks cannot be prepared safely."""


@dataclass(frozen=True)
class SpeakerConfig:
    speakers: Mapping[str, str]          # lowercase username -> Tag
    ignore: Tuple[str, ...] = ()         # lowercase substrings

    @property
    def tags(self) -> FrozenSet[str]:
        return frozenset(self.speakers.values())


def _tokens(value: Optional[str]) -> List[str]:
    return [t.strip() for t in (value or "").split(",") if t.strip()]


def parse_speaker_config(speakers_value: Optional[str], ignore_value: Optional[str]) -> SpeakerConfig:
    entries = _tokens(speakers_value)
    if not entries:
        raise TrackPreparationError(f"{SPEAKERS_VAR} is missing or empty; set it in .env (see .env.example).")
    speakers: Dict[str, str] = {}
    seen_tags: Dict[str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise TrackPreparationError(f"{SPEAKERS_VAR} entry {entry!r} must look like username:Tag.")
        username, tag = (part.strip() for part in entry.split(":", 1))
        if not username or not tag:
            raise TrackPreparationError(f"{SPEAKERS_VAR} entry {entry!r} has an empty username or Tag.")
        if "/" in tag or "\\" in tag or ".." in tag or tag.startswith("."):
            raise TrackPreparationError(
                f"{SPEAKERS_VAR} entry {entry!r}: the Tag {tag!r} must be a plain file name "
                "(no '/', '\\', '..' and no leading '.')."
            )
        key = username.lower()
        if key in speakers:
            raise TrackPreparationError(f"{SPEAKERS_VAR} lists username {username!r} twice.")
        if tag.lower() in seen_tags:
            raise TrackPreparationError(
                f"{SPEAKERS_VAR} maps two usernames to the same Tag {tag!r} (case-insensitive)."
            )
        speakers[key] = tag
        seen_tags[tag.lower()] = username
    ignore = tuple(t.lower() for t in _tokens(ignore_value))
    return SpeakerConfig(speakers=speakers, ignore=ignore)


def speaker_config_from_env(environ: Mapping[str, str] = os.environ) -> SpeakerConfig:
    return parse_speaker_config(environ.get(SPEAKERS_VAR), environ.get(IGNORE_VAR))


def username_from_track(stem: str) -> str:
    return re.sub(r"^\d+-", "", stem).lstrip("_")


def classify_track(stem: str, cfg: SpeakerConfig) -> Tuple[str, Optional[str]]:
    username = username_from_track(stem).lower()
    if any(token in username for token in cfg.ignore):
        return "ignored", None
    if stem in cfg.tags:
        return "tag", stem
    if username in cfg.speakers:
        return "mapped", cfg.speakers[username]
    return "unknown", None


def _unknown_error(name: str) -> TrackPreparationError:
    return TrackPreparationError(
        f"Unknown speaker track {name!r}: add its username to CW_SPEAKERS or CW_IGNORE_TRACKS."
    )


def find_craig_archive(session_path: Path) -> Optional[Path]:
    candidates = sorted(p for p in Path(session_path).glob("craig-*.flac.zip") if p.is_file())
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(f"Multiple Craig archives found in {session_path}; using {candidates[0].name}.")
    return candidates[0]


@dataclass
class PrepareResult:
    extracted: List[str] = field(default_factory=list)
    renamed: Dict[str, str] = field(default_factory=dict)
    deleted: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def _normalise_existing(tracks: Path, cfg: SpeakerConfig, rename: bool, result: PrepareResult) -> None:
    for track in sorted(tracks.glob("*.flac")):
        kind, tag = classify_track(track.stem, cfg)
        if kind == "unknown":
            raise _unknown_error(track.name)
        if kind == "tag":
            continue
        if not rename:
            raise TrackPreparationError(
                f"Track {track.name!r} is not renamed to a speaker Tag and --no-prepare-tracks is set."
            )
        if kind == "ignored":
            track.unlink()
            result.deleted.append(track.name)
            log.info(f"Deleted ignored track {track.name}")
            continue
        target = tracks / f"{tag}.flac"
        if target.exists():
            raise TrackPreparationError(f"Cannot rename {track.name} -> {target.name}: target already exists.")
        track.rename(target)
        result.renamed[track.stem] = tag
        log.info(f"Renamed {track.name} -> {target.name}")


def _extract_missing(archive: Path, tracks: Path, cfg: SpeakerConfig, result: PrepareResult) -> None:
    tmp = tracks / _EXTRACT_TMP
    with zipfile.ZipFile(archive) as zf:
        wanted: List[Tuple[zipfile.ZipInfo, str]] = []
        claimed: Dict[str, str] = {}
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".flac"):
                continue
            kind, tag = classify_track(Path(name).stem, cfg)
            if kind == "unknown":
                raise _unknown_error(name)
            if kind == "ignored":
                continue
            final_name = f"{tag}.flac"
            if final_name in claimed:
                raise TrackPreparationError(
                    f"Archive entries {claimed[final_name]!r} and {member.filename!r} both map to Tag {tag!r}."
                )
            claimed[final_name] = member.filename
            final_path = tracks / final_name
            if final_path.exists():
                if final_path.stat().st_size != member.file_size:
                    log.warning(f"{final_name} differs in size from {name} in {archive.name}; keeping the existing file.")
                continue
            wanted.append((member, final_name))
        if not wanted:
            return
        tmp.mkdir(parents=True, exist_ok=True)
        for member, final_name in wanted:
            staged = tmp / Path(member.filename).name
            with zf.open(member) as src, open(staged, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(staged, tracks / final_name)
            result.extracted.append(final_name)
            log.info(f"Extracted {member.filename} -> {final_name}")
    shutil.rmtree(tmp, ignore_errors=True)


def prepare_tracks(
    session_path: Path,
    speakers: SpeakerConfig,
    tracks_folder_name: str = "tracks",
    rename: bool = True,
) -> PrepareResult:
    session_path = Path(session_path)
    tracks = session_path / tracks_folder_name
    session_resolved, tracks_resolved = session_path.resolve(), tracks.resolve()
    if session_resolved not in tracks_resolved.parents or (
        str(tracks_resolved).casefold() == str(session_resolved).casefold()
    ):
        raise TrackPreparationError(f"The tracks folder {tracks} must be a folder inside the session folder {session_path}.")
    result = PrepareResult()
    leftover = tracks / _EXTRACT_TMP
    if rename and (leftover.exists() or leftover.is_symlink()):
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)
        else:
            leftover.unlink()
        log.warning(f"Removed leftover {leftover} from an interrupted extraction.")
    if tracks.is_dir():
        _normalise_existing(tracks, speakers, rename, result)
    if rename:
        archive = find_craig_archive(session_path)
        if archive is not None:
            tracks.mkdir(parents=True, exist_ok=True)
            _extract_missing(archive, tracks, speakers, result)
    flacs = sorted(tracks.glob("*.flac")) if tracks.is_dir() else []
    result.tags = [p.stem for p in flacs]
    if not result.tags:
        raise TrackPreparationError(f"Step 0: no speaker track in {tracks} after preparation.")
    bad = [p.name for p in flacs if p.stem not in speakers.tags]
    if bad:
        raise TrackPreparationError(f"Step 0: tracks not matching a speaker Tag remain: {bad}")
    return result
