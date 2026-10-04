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
from typing import Dict, List, Tuple

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


def build_windows(a: List[Segment], b: List[Segment], max_len: float = 25.0, gap: float = 1.0) -> List[Tuple[float, float]]:
    spans = sorted((s, e) for s, e, _ in a + b)
    merged: List[List[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    windows: List[Tuple[float, float]] = []
    for s, e in merged:
        while e - s > max_len:
            windows.append((s, s + max_len))
            s += max_len
        windows.append((s, e))
    return windows


def window_text(segments: List[Segment], window: Tuple[float, float]) -> str:
    lo, hi = window
    return " ".join(t for s, e, t in segments if lo <= (s + e) / 2 < hi or (s >= lo and e <= hi)).strip()


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", text.lower()).split())


def select_items(a_dir: Path, b_dir: Path, n: int = 30, seed: int = 7) -> List[dict]:
    a_all, b_all = load_chunk_segments(a_dir), load_chunk_segments(b_dir)
    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for stem in sorted(set(a_all) | set(b_all)):
        a, b = a_all.get(stem, []), b_all.get(stem, [])
        for start, end in build_windows(a, b):
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


def repetition_loops(text: str, min_words: int = 3, min_repeats: int = 3) -> int:
    words = normalise(text).split()
    loops, i = 0, 0
    while i < len(words):
        found = False
        for size in range(min_words, 13):
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


def score(answers_path: Path, mapping_path: Path, a_script: Path, b_script: Path) -> dict:
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    models = sorted({m for v in mapping.values() for m in v.values()})
    wins = {m: 0 for m in models}
    ties = 0
    for item_id, choice in answers.items():
        if choice == "same":
            ties += 1
        elif item_id in mapping and choice in ("X", "Y"):
            wins[mapping[item_id][choice]] += 1
    non_tied = sum(wins.values())
    loops = {
        "a_script": repetition_loops(Path(a_script).read_text(encoding="utf-8")),
        "b_script": repetition_loops(Path(b_script).read_text(encoding="utf-8")),
    }
    decision = "inconclusive" if non_tied < 20 else "pending"  # final rule applied in main()
    return {"wins": wins, "ties": ties, "non_tied": non_tied, "loops": loops,
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
    s.add_argument("--a-script", type=Path, required=True); s.add_argument("--b-script", type=Path, required=True)
    s.add_argument("--candidate", required=True, help="Model name that would replace the baseline.")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        items = select_items(args.a, args.b, n=args.n)
        page, mapping = build_page(items, args.a_name, args.b_name, args.chunks, args.out)
        print(f"{len(items)} items. Open {page} in a browser; mapping kept in {mapping}.")
        return 0
    result = score(args.answers, args.mapping, args.a_script, args.b_script)
    cand = args.candidate
    other = [m for m in result["models"] if m != cand]
    if result["non_tied"] < 20:
        result["decision"] = "inconclusive -> keep baseline"
    elif result["wins"].get(cand, 0) * 3 >= result["non_tied"] * 2 and result["loops"]["b_script"] <= result["loops"]["a_script"]:
        result["decision"] = f"switch to {cand}"
    else:
        result["decision"] = f"keep {other[0] if other else 'baseline'}"
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
