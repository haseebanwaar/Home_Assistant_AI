"""Step 5 — offline boundary scoring over a recorded capture.

Reads a capture folder's observations.jsonl, computes a boundary_score between
each consecutive batch (no VLM needed), and writes data/debug/boundaries.jsonl
plus a readable table. Deterministic, so thresholds can be eyeballed:

    python -m tools.boundaries data/captures/run_20260723_131738
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from memory.boundaries.boundary_detector import (
    compute_visual_change,
    normalize_inactivity,
    score_boundary,
)
from memory.debug import DEBUG_DIR, write_jsonl


def load_observations(capture_dir):
    manifest = os.path.join(capture_dir, "observations.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"no observations.jsonl found in {capture_dir}")
    raw = open(manifest, encoding="utf-8").read()
    decoder = json.JSONDecoder()
    obs, i, n = [], 0, len(raw)
    while i < n:
        while i < n and raw[i] in " \r\n\t":
            i += 1
        if i >= n:
            break
        rec, end = decoder.raw_decode(raw, i)
        obs.append(rec)
        i = end
    return obs


def _last_frame(capture_dir, obs):
    frames = obs.get("frames") or []
    if not frames:
        return None
    img = Image.open(os.path.join(capture_dir, frames[-1])).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _app_of(obs):
    """Representative app for a batch: last process name, else derived from title."""
    procs = obs.get("process_names") or []
    if procs:
        return procs[-1].lower()
    titles = obs.get("window_titles") or []
    if titles:
        t = titles[-1]
        return (t.rsplit(" - ", 1)[-1] if " - " in t else t).strip().lower()
    return ""


def _title_of(obs):
    titles = obs.get("window_titles") or []
    return (titles[-1] if titles else "").strip()


def main():
    parser = argparse.ArgumentParser(description="Score boundaries over a recorded capture.")
    parser.add_argument("capture_dir", help="path to data/captures/<run_id>")
    parser.add_argument("--expected", type=float, default=60.0,
                        help="expected seconds between batches (default 60)")
    args = parser.parse_args()

    observations = load_observations(args.capture_dir)
    print(f"Loaded {len(observations)} observation(s) from {args.capture_dir}")

    # Fresh per-run output (deterministic regeneration from the capture).
    out_path = os.path.join(DEBUG_DIR, "boundaries.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    print(f"\n{'batch':>5} {'score':>6} {'label':>10}  {'app':>4} {'ttl':>4} "
          f"{'vis':>5} {'idle':>5}  title")
    prev = None
    prev_frame = None
    for obs in observations:
        frame = _last_frame(args.capture_dir, obs)
        if prev is None:
            # First observation: no transition to score; seed as append.
            result = score_boundary(False, False, 0.0, 0.0)
        else:
            app_changed = _app_of(obs) != _app_of(prev)
            title_changed = _title_of(obs) != _title_of(prev)
            visual_change = compute_visual_change(prev_frame, frame)
            inactivity = normalize_inactivity(
                obs["timestamp"] - prev["timestamp"], expected_seconds=args.expected)
            result = score_boundary(app_changed, title_changed, visual_change, inactivity)

        record = {
            "batch": obs["batch"],
            "timestamp": obs["timestamp"],
            "score": round(result.score, 3),
            "label": result.label,
            "app_changed": result.app_changed,
            "title_changed": result.title_changed,
            "visual_change": round(result.visual_change, 3),
            "inactivity": round(result.inactivity, 3),
            "app": _app_of(obs),
            "window_title": _title_of(obs),
        }
        write_jsonl("boundaries", record)
        print(f"{obs['batch']:>5} {result.score:>6.3f} {result.label:>10}  "
              f"{'Y' if result.app_changed else '.':>4} "
              f"{'Y' if result.title_changed else '.':>4} "
              f"{result.visual_change:>5.2f} {result.inactivity:>5.2f}  "
              f"{_title_of(obs)[:60]}")
        prev, prev_frame = obs, frame

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
