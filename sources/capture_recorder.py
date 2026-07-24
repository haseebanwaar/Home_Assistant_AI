"""Step 0 — capture recorder.

When RECORD_CAPTURE=1, the live screen loop dumps every processed minute-batch
to disk: each frame as a JPEG plus one line in observations.jsonl describing the
batch. tools/replay.py reads that folder back through the same VLM describe path,
so every later pipeline feature can be tested deterministically with no screen.
"""
import json
import logging
import os
import time

import numpy as np
from PIL import Image

logger = logging.getLogger("home_assistant")


class CaptureRecorder:
    """Records processed frame batches to data/captures/<run_id>/."""

    def __init__(self, base_dir="data/captures", run_id=None):
        run_id = run_id or os.getenv("CAPTURE_RUN_ID") or time.strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, run_id)
        self.frames_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, "observations.jsonl")
        self.batch = 0
        self.frame_seq = 0
        logger.info("Capture recording to %s", self.run_dir)

    def record_batch(self, frames, timestamp, window_titles, process_names=None):
        """Save one minute-batch: its frames plus a manifest line.

        Frames are RGB numpy arrays (as held by the screen buffer). Overlap
        frames carried across batches are re-saved so each observation is
        self-contained for replay. process_names (Step 3) lets replay route the
        domain profile the same way the live loop does.
        """
        if not frames:
            return
        rel_paths = []
        for frame in frames:
            self.frame_seq += 1
            name = f"{self.frame_seq:06d}.jpg"
            path = os.path.join(self.frames_dir, name)
            try:
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path, quality=85)
            except Exception:
                logger.exception("Failed to save frame %s", name)
                continue
            rel_paths.append(f"frames/{name}")

        record = {
            "batch": self.batch,
            "timestamp": timestamp,
            "window_titles": list(window_titles or []),
            "process_names": list(process_names or []),
            "frames": rel_paths,
        }
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Recorded batch %d (%d frames) -> %s",
                    self.batch, len(rel_paths), self.manifest_path)
        self.batch += 1
