"""Step 0 — offline replay harness.

Feeds a recorded capture folder back through the SAME VLM describe path the live
screen loop uses (RealtimeScreenCapture._describe_frames), with deterministic
frames and no live screen. This is the debugging foundation for every later
pipeline feature.

Record a run first by starting the app with RECORD_CAPTURE=1, then:

    python -m tools.replay data/captures/run_20260722_120000
    python -m tools.replay data/captures/run_... --limit 3
    python -m tools.replay data/captures/run_... --log   # also write to Qdrant

Descriptions are printed and written to <capture_dir>/replay_descriptions.jsonl.
"""
import argparse
import asyncio
import json
import os

import numpy as np
from PIL import Image

from providers.local_openAI import get_model_name_vlm
from sources.screen import RealtimeScreenCapture


def load_observations(capture_dir):
    manifest = os.path.join(capture_dir, "observations.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"no observations.jsonl found in {capture_dir}")
    # Robust to both proper JSONL (one object per line) and a pretty-printed
    # file (e.g. an editor's format-on-save with indentation/CRLF). We decode a
    # stream of concatenated JSON objects instead of assuming one per line.
    raw = open(manifest, encoding="utf-8").read()
    decoder = json.JSONDecoder()
    observations = []
    i, n = 0, len(raw)
    while i < n:
        while i < n and raw[i] in " \r\n\t":
            i += 1
        if i >= n:
            break
        obj, end = decoder.raw_decode(raw, i)
        observations.append(obj)
        i = end
    return observations


def load_frames(capture_dir, rel_paths):
    frames = []
    for rel in rel_paths:
        img = Image.open(os.path.join(capture_dir, rel)).convert("RGB")
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


async def main():
    parser = argparse.ArgumentParser(description="Replay a recorded capture through the VLM describe path.")
    parser.add_argument("capture_dir", help="path to data/captures/<run_id>")
    parser.add_argument("--limit", type=int, default=None, help="replay only the first N observations")
    parser.add_argument("--log", action="store_true", help="also write descriptions to the real activity logger (Qdrant)")
    args = parser.parse_args()

    observations = load_observations(args.capture_dir)
    if args.limit:
        observations = observations[: args.limit]
    print(f"Loaded {len(observations)} observation(s) from {args.capture_dir}")

    model = await get_model_name_vlm()
    print(f"VLM model: {model}")

    activity_logger = None
    if args.log:
        from qdrant_client import QdrantClient
        from vector_store.activity_logger import ActivityLogger
        activity_logger = ActivityLogger(client=QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_db")))

    # start_capture=False: reuse the exact describe logic without touching the screen.
    cap = RealtimeScreenCapture(
        video_source="", model_name_vlm=model,
        fps=1.0, activity_logger=activity_logger, start_capture=False,
    )

    structured = cap.structured_extraction
    if structured:
        print("STRUCTURED_EXTRACTION=1 — replaying through the JSON extraction path.")
        from memory.debug import write_jsonl

    out_path = os.path.join(args.capture_dir, "replay_descriptions.jsonl")
    stats = {"ok": 0, "retry": 0, "fallback": 0, "empty": 0, "total": 0}
    with open(out_path, "w", encoding="utf-8") as out:
        for obs in observations:
            frames = load_frames(args.capture_dir, obs["frames"])
            cap.current_minute_apps = list(obs.get("window_titles", []))

            if structured:
                # Route the domain profile from the recorded process names (Step 3),
                # falling back to window titles for captures recorded before that.
                cap.current_minute_processes = list(obs.get("process_names", []))
                result, status, profile_name = await cap._extract_structured(frames)
                description = result.summary
                stats[status] = stats.get(status, 0) + 1
                stats["total"] += 1
                record = result.model_dump()
                record.update({
                    "batch": obs["batch"], "timestamp": obs["timestamp"],
                    "status": status, "selected_profile": profile_name,
                    "process_names": obs.get("process_names", []),
                    "window_titles": obs.get("window_titles", []),
                })
                write_jsonl("extractions", record)
                print(f"\n=== batch {obs['batch']} | {len(frames)} frames | {status} "
                      f"| profile={profile_name} | {result.activity_type} | {len(result.entities)} entities ===")
                print(description)
            else:
                description = await cap._describe_frames(frames)
                print(f"\n=== batch {obs['batch']} | {len(frames)} frames | {obs.get('window_titles')} ===")
                print(description)

            out.write(json.dumps({
                "batch": obs["batch"],
                "timestamp": obs["timestamp"],
                "window_titles": obs.get("window_titles", []),
                "frames": len(frames),
                "description": description,
            }) + "\n")

            if activity_logger is not None:
                activity_logger.log_activity(description, obs["timestamp"], "screen", cap.current_minute_apps)

    print(f"\nWrote {out_path}")
    if structured and stats["total"]:
        validated = stats["ok"] + stats["retry"]
        pct = 100.0 * validated / stats["total"]
        print(f"Extraction validation: {validated}/{stats['total']} ({pct:.0f}%) "
              f"[ok={stats['ok']} retry={stats['retry']} fallback={stats['fallback']}]")
        print("Wrote full objects to data/debug/extractions.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
