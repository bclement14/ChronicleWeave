# chronicleweave/modules/merge_srt_by_chunk.py

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Union

import srt  # External library: https://pypi.org/project/srt/

# --- Module Logger ---
log = logging.getLogger(__name__)

_CUT_LINE = re.compile(r"^Chunk\s+(\d+)\s*,\s*([0-9.+\-eE]+)\s*$")


def read_cut_points(path: Path) -> List[float]:
    boundaries: List[float] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _CUT_LINE.match(line)
        if not m:
            raise ValueError(f"{path}: unreadable cut point line {line!r}")
        index, value = int(m.group(1)), float(m.group(2))
        if index != len(boundaries):
            raise ValueError(f"{path}: cut point indices must be contiguous from 0, got {index}")
        boundaries.append(value)
    if len(boundaries) < 2:
        raise ValueError(f"{path}: need at least 2 boundaries, got {len(boundaries)}")
    if boundaries[0] != 0.0:
        raise ValueError(f"{path}: first boundary must be 0.0, got {boundaries[0]}")
    if any(not (b == b and abs(b) != float("inf")) for b in boundaries):
        raise ValueError(f"{path}: boundaries must be finite numbers")
    if any(b2 <= b1 for b1, b2 in zip(boundaries, boundaries[1:])):
        raise ValueError(f"{path}: boundaries must be strictly increasing")
    return boundaries


def load_srt_file(file_path: Path, speaker_name: str, time_offset: timedelta) -> List[srt.Subtitle]:
    content = Path(file_path).read_text(encoding="utf-8")
    subtitles = list(srt.parse(content))  # raises srt.SRTParseError on bad content
    for sub in subtitles:
        sub.start += time_offset
        sub.end += time_offset
        sub.content = f"[{speaker_name}] {sub.content}"
    return subtitles


def parse_srt_filename(filename: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)-(\d+)\.srt$", filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"SRT file name {filename!r} does not match '<speaker>-<NN>.srt'")
    return m.group(1), int(m.group(2))


def merge_srt_by_chunk(
    srt_folder: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: str,
    cut_points_file: Union[str, Path],
) -> int:
    srt_folder, output_dir = Path(srt_folder), Path(output_dir)
    if not output_filename.lower().endswith(".srt"):
        raise ValueError("Output filename must end with .srt")
    boundaries = read_cut_points(Path(cut_points_file))
    n_chunks = len(boundaries) - 1
    files = sorted(srt_folder.glob("*.srt"))
    if not files:
        raise FileNotFoundError(f"No SRT files in {srt_folder}")
    all_subs: List[srt.Subtitle] = []
    for path in files:
        speaker, chunk = parse_srt_filename(path.name)
        if not 1 <= chunk <= n_chunks:
            raise ValueError(f"{path.name}: chunk {chunk} is outside the {n_chunks} chunks in {cut_points_file}")
        offset = timedelta(seconds=boundaries[chunk - 1])
        all_subs.extend(load_srt_file(path, speaker, offset))
    all_subs.sort(key=lambda s: (s.start, s.end))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / output_filename).write_text(srt.compose(all_subs), encoding="utf-8")
    log.info(f"Merged {len(all_subs)} subtitles from {len(files)} files into {output_dir / output_filename}")
    return len(all_subs)
