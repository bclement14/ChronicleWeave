# tools/asr_ab/ab.py
"""Blind A/B comparison of two transcriptions of the same chunks (spec 4.9). Local only."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Segment = Tuple[float, float, str]


def load_chunk_segments(json_dir: Path) -> Dict[str, List[Segment]]:
    out: Dict[str, List[Segment]] = {}
    for path in sorted(Path(json_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = []
        for s in data.get("segments", []):
            text = str(s.get("text") or "").strip()
            if text:
                segs.append((float(s["start"]), float(s["end"]), text))
        out[path.stem] = segs
    return out


def _safe_gaps(segs: List[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    """Gaps strictly inside (lo, hi) that no segment of either model crosses."""
    gaps: List[Tuple[float, float]] = []
    reach = None
    for s, e in sorted(segs):
        if reach is not None and s >= reach and lo < reach and s < hi:
            gaps.append((reach, s))
        reach = e if reach is None else max(reach, e)
    return gaps


def _split_span(span: Tuple[float, float], segs: List[Tuple[float, float]], max_len: float, hard_max: float,
                windows: List[Tuple[float, float]], left_out: List[Tuple[float, float]]) -> None:
    lo, hi = span
    if hi - lo <= max_len:
        windows.append(span)
        return
    inner = [(s, e) for s, e in segs if lo <= s and e <= hi]
    gaps = _safe_gaps(inner, lo, hi)
    if not gaps:  # cutting would split a segment and fake a difference: keep it whole, or leave it out
        (windows if hi - lo <= hard_max else left_out).append(span)
        return
    mid = (lo + hi) / 2
    g0, g1 = max(gaps, key=lambda g: (g[1] - g[0], -abs((g[0] + g[1]) / 2 - mid)))  # largest, then most central
    _split_span((lo, g0), inner, max_len, hard_max, windows, left_out)
    _split_span((g1, hi), inner, max_len, hard_max, windows, left_out)


def build_windows(a: List[Segment], b: List[Segment], max_len: float = 25.0, gap: float = 1.0,
                  hard_max: float = 60.0, left_out: Optional[List[Tuple[float, float]]] = None,
                  ) -> List[Tuple[float, float]]:
    """Windows of speech bounded by pauses in both transcripts. A span longer than `max_len` is split
    only at a gap that no segment of either model crosses (the largest one), so every segment lies in
    exactly one window and different segmentation of the same speech gives the same text. A span with
    no such gap is kept whole up to `hard_max` seconds; a longer one is not judged (added to `left_out`)."""
    segs = sorted((s, e) for s, e, _ in a + b)
    merged: List[List[float]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    windows: List[Tuple[float, float]] = []
    dropped = left_out if left_out is not None else []
    for s, e in merged:
        _split_span((s, e), segs, max_len, hard_max, windows, dropped)
    return windows


def window_text(segments: List[Segment], window: Tuple[float, float]) -> str:
    lo, hi = window
    return " ".join(t for s, e, t in segments if lo <= (s + e) / 2 < hi or (s >= lo and e <= hi)).strip()


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", text.lower()).split())


def select_items(a_dir: Path, b_dir: Path, n: int = 30, seed: int = 7,
                 left_out: Optional[List[Tuple[str, float, float]]] = None) -> List[dict]:
    a_all, b_all = load_chunk_segments(a_dir), load_chunk_segments(b_dir)
    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for stem in sorted(set(a_all) | set(b_all)):
        a, b = a_all.get(stem, []), b_all.get(stem, [])
        too_long: List[Tuple[float, float]] = []
        windows = build_windows(a, b, left_out=too_long)
        if left_out is not None:
            left_out.extend((stem, s, e) for s, e in too_long)
        for start, end in windows:
            at, bt = window_text(a, (start, end)), window_text(b, (start, end))
            if normalise(at) != normalise(bt):
                by_speaker[stem.rsplit("-", 1)[0]].append(
                    {"stem": stem, "start": start, "end": end, "a_text": at, "b_text": bt})
    rng = random.Random(seed)
    for items in by_speaker.values():
        rng.shuffle(items)
    picked: List[dict] = []
    speakers = sorted(by_speaker)
    while len(picked) < n and any(by_speaker[s] for s in speakers):
        for s in speakers:
            if by_speaker[s] and len(picked) < n:
                picked.append(by_speaker[s].pop())
    for i, item in enumerate(picked, start=1):
        item["id"] = f"i{i}"
    return picked


_TAG_PREFIX = re.compile(r"^\s*\[[^\]]*\]", re.MULTILINE)


def repetition_loops(text: str, min_words: int = 3, min_repeats: int = 3, max_words: int = 30) -> int:
    """Count runs of the same phrase (min_words..max_words words) repeated min_repeats+ times in a row.
    The `[Tag]` speaker prefixes of the script are removed first, so they cannot form loops themselves."""
    words = normalise(_TAG_PREFIX.sub(" ", text)).split()
    loops, i = 0, 0
    while i < len(words):
        found = False
        for size in range(min_words, max_words + 1):
            phrase = words[i:i + size]
            if len(phrase) < size:
                break
            reps = 1
            while words[i + reps * size:i + (reps + 1) * size] == phrase:
                reps += 1
            if reps >= min_repeats:
                loops += 1
                i += reps * size
                found = True
                break
        if not found:
            i += 1
    return loops


def _cut_clip(chunk: Path, start: float, end: float, out: Path) -> None:
    s = max(0.0, start - 1.0)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{s:.3f}", "-t", f"{end + 1.0 - s:.3f}",
           "-i", str(chunk), "-c:a", "libopus", "-b:a", "48k", str(out)]
    subprocess.run(cmd, check=True)


def build_page(items: List[dict], a_name: str, b_name: str, chunks_dir: Path, out_dir: Path, seed: int = 7) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    clips = out_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    mapping: Dict[str, Dict[str, str]] = {}
    rows = []
    for item in items:
        _cut_clip(Path(chunks_dir) / f"{item['stem']}.flac", item["start"], item["end"], clips / f"{item['id']}.ogg")
        if rng.random() < 0.5:
            x_text, y_text, mapping[item["id"]] = item["a_text"], item["b_text"], {"X": a_name, "Y": b_name}
        else:
            x_text, y_text, mapping[item["id"]] = item["b_text"], item["a_text"], {"X": b_name, "Y": a_name}
        rows.append(
            f"<section><h3>{html.escape(item['id'])} — {html.escape(item['stem'].rsplit('-', 1)[0])}</h3>"
            f"<audio controls preload='none' src='clips/{item['id']}.ogg'></audio>"
            f"<p><b>X:</b> {html.escape(x_text) or '<i>(rien)</i>'}</p>"
            f"<p><b>Y:</b> {html.escape(y_text) or '<i>(rien)</i>'}</p>"
            + "".join(f"<label><input type='radio' name='{item['id']}' value='{v}'> {v}</label> " for v in ("X", "Y", "same"))
            + "</section>")
    page = out_dir / "index.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><title>Transcription A/B</title>"
        "<style>body{font-family:sans-serif;max-width:860px;margin:auto;padding:16px}section{border-bottom:1px solid #ccc;padding:8px 0}</style>"
        "<h1>Transcription A/B</h1><p>Listen, then pick the more accurate text (X, Y or same). Export at the end.</p>"
        + "".join(rows)
        + "<button onclick=\"const a={};document.querySelectorAll('input:checked').forEach(i=>a[i.name]=i.value);"
          "const l=document.createElement('a');l.href=URL.createObjectURL(new Blob([JSON.stringify(a,null,1)],{type:'application/json'}));"
          "l.download='answers.json';l.click();\">Export answers</button>",
        encoding="utf-8")
    mapping_path = out_dir / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    return page, mapping_path


def score(answers_path: Path, mapping_path: Path, scripts: Dict[str, Path]) -> dict:
    """scripts maps model name -> that model's final_script.txt."""
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    models = sorted({m for v in mapping.values() for m in v.values()})
    wins = {m: 0 for m in models}
    ties = 0
    ignored = 0
    for item_id, choice in answers.items():
        if item_id not in mapping or choice not in ("X", "Y", "same"):
            ignored += 1
        elif choice == "same":
            ties += 1
        else:
            wins[mapping[item_id][choice]] += 1
    non_tied = sum(wins.values())
    loops = {name: repetition_loops(Path(path).read_text(encoding="utf-8")) for name, path in scripts.items()}
    decision = "inconclusive" if non_tied < 20 else "pending"  # final rule applied in main()
    return {"wins": wins, "ties": ties, "non_tied": non_tied, "ignored": ignored, "loops": loops,
            "decision": decision, "models": models}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="asr_ab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--a", type=Path, required=True); b.add_argument("--b", type=Path, required=True)
    b.add_argument("--a-name", required=True); b.add_argument("--b-name", required=True)
    b.add_argument("--chunks", type=Path, required=True); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=30)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True); s.add_argument("--mapping", type=Path, required=True)
    s.add_argument("--a-name", required=True); s.add_argument("--a-script", type=Path, required=True)
    s.add_argument("--b-name", required=True); s.add_argument("--b-script", type=Path, required=True)
    s.add_argument("--candidate", required=True, help="Model name that would replace the baseline.")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        left_out: List[Tuple[str, float, float]] = []
        items = select_items(args.a, args.b, n=args.n, left_out=left_out)
        page, mapping = build_page(items, args.a_name, args.b_name, args.chunks, args.out)
        print(f"{len(items)} items. Open {page} in a browser; mapping kept in {mapping}.")
        if left_out:
            print(f"{len(left_out)} span(s) longer than 60 s with no common pause were not judged.")
        return 0
    cand = args.candidate
    mapping_models = sorted({m for v in json.loads(args.mapping.read_text(encoding="utf-8")).values() for m in v.values()})
    if sorted({args.a_name, args.b_name}) != mapping_models:
        print(f"error: --a-name/--b-name ({args.a_name}, {args.b_name}) do not match the models in the mapping {mapping_models}", file=sys.stderr)
        return 2
    if cand not in mapping_models:
        print(f"error: --candidate {cand!r} is not one of {mapping_models}", file=sys.stderr)
        return 2
    result = score(args.answers, args.mapping, {args.a_name: args.a_script, args.b_name: args.b_script})
    base = [m for m in result["models"] if m != cand][0]
    if result["non_tied"] < 20:
        result["decision"] = f"inconclusive -> keep {base}"
    elif result["wins"][cand] * 3 >= result["non_tied"] * 2 and result["loops"][cand] <= result["loops"][base]:
        result["decision"] = f"switch to {cand}"
    else:
        result["decision"] = f"keep {base}"
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
