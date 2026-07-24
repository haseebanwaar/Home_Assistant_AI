"""Step 5 — Stage-1 boundary detector (plan §6.3), scored, observe-only.

Computes a boundary_score in [0,1] between two consecutive observations from a
weighted mix of four signals, then classifies it. No sessions are created yet —
this stage only scores and logs so the thresholds can be eyeballed.

Weights are chosen so that app/title changes dominate: a pure visual change
(e.g. scrolling within the same app, same title) can contribute at most the
visual weight (0.25) and therefore always stays below the new_event threshold
(0.40) — satisfying the Step 5 "small visual changes stay < 0.40" criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Signal weights (sum to 1.0).
WEIGHTS = {
    "app_changed": 0.40,
    "title_changed": 0.20,
    "visual_change": 0.25,
    "inactivity": 0.15,
}

# Score thresholds.
THRESHOLD_BOUNDARY = 0.60   # >= => "boundary"
THRESHOLD_NEW_EVENT = 0.40  # >= => "new_event", else "append"


@dataclass
class BoundaryResult:
    score: float
    label: str
    app_changed: bool
    title_changed: bool
    visual_change: float
    inactivity: float


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_visual_change(frame_a, frame_b):
    """Mean absolute pixel difference between two frames, normalized to [0,1].
    Frames are RGB uint8 arrays. Resized-independent as long as shapes match;
    mismatched shapes are treated as a full change.
    """
    if frame_a is None or frame_b is None:
        return 0.0
    a = np.asarray(frame_a, dtype=np.float32)
    b = np.asarray(frame_b, dtype=np.float32)
    if a.shape != b.shape:
        return 1.0
    return _clamp01(float(np.abs(a - b).mean()) / 255.0)


def normalize_inactivity(gap_seconds, expected_seconds=60.0, idle_scale=300.0):
    """Map an inter-observation time gap to [0,1].

    Gaps up to `expected_seconds` are normal (0). Beyond that, inactivity ramps
    linearly, reaching 1.0 at `expected_seconds + idle_scale`.
    """
    excess = max(0.0, float(gap_seconds) - float(expected_seconds))
    return _clamp01(excess / max(idle_scale, 1e-6))


def classify(score):
    if score >= THRESHOLD_BOUNDARY:
        return "boundary"
    if score >= THRESHOLD_NEW_EVENT:
        return "new_event"
    return "append"


def score_boundary(app_changed, title_changed, visual_change, inactivity,
                   weights=WEIGHTS):
    """Weighted boundary score + label from the four signals."""
    vc = _clamp01(visual_change)
    ia = _clamp01(inactivity)
    score = (
        weights["app_changed"] * (1.0 if app_changed else 0.0)
        + weights["title_changed"] * (1.0 if title_changed else 0.0)
        + weights["visual_change"] * vc
        + weights["inactivity"] * ia
    )
    score = _clamp01(score)
    # Rule-based label on the categorical signals (the numeric score is still
    # reported): an app switch is a hard boundary; a title change within the same
    # app is a distinct new action (new_event); otherwise fall back to the score
    # so a big idle gap can still promote, but scrolling stays an append.
    if app_changed:
        label = "boundary"
    elif title_changed:
        label = "new_event"
    else:
        label = classify(score)
    return BoundaryResult(
        score=score,
        label=label,
        app_changed=bool(app_changed),
        title_changed=bool(title_changed),
        visual_change=vc,
        inactivity=ia,
    )
