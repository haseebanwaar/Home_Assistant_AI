"""Step 4 — gap-spanning frame sampler (plan §6.2).

When a batch/gap covers a long interval, we don't want to send only the last
minute of frames to the VLM — we want a temporal spread across the WHOLE gap,
weighted toward the present. sample_gap_frames picks `count` frames: ~30% spread
uniformly across the full span, ~70% drawn from the recent tail near the present.

Frames are assumed roughly evenly spaced across [span_start, span_end], so we
assign each a synthetic timestamp by linear interpolation. The chosen timestamps
are returned for logging (the Step 4 debug handle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GapSample:
    frames: list
    timestamps: List[float]
    indices: List[int]


def _linspace_idx(lo, hi, k):
    """k evenly spaced integer indices in [lo, hi] inclusive."""
    if k <= 0 or hi < lo:
        return []
    if k == 1:
        return [hi]
    return [round(lo + (hi - lo) * j / (k - 1)) for j in range(k)]


def sample_gap_frames(frames, span_start, span_end, count=10,
                      recent_fraction=0.3, recent_weight=0.7):
    """Select a temporal spread of `count` frames across a gap.

    Args:
        frames: ordered frames (oldest -> newest).
        span_start, span_end: epoch seconds bounding the gap.
        count: how many frames to return.
        recent_fraction: the recent tail is the last `recent_fraction` of the span.
        recent_weight: fraction of the samples drawn from that recent tail.

    Returns a GapSample (frames, per-frame timestamps, source indices).
    """
    n = len(frames)
    if n == 0:
        return GapSample([], [], [])

    if n == 1:
        ts_all = [span_end]
    else:
        step = (span_end - span_start) / (n - 1)
        ts_all = [span_start + step * i for i in range(n)]

    if n <= count:
        return GapSample(list(frames), ts_all, list(range(n)))

    n_recent = min(count, round(count * recent_weight))
    n_uniform = count - n_recent

    uniform = _linspace_idx(0, n - 1, n_uniform)

    recent_start = min(int(n * (1.0 - recent_fraction)), n - 1)
    recent = _linspace_idx(recent_start, n - 1, n_recent)

    idx = sorted(set(uniform) | set(recent))

    # Top up if de-duplication dropped below count (uniform/recent overlapped).
    i = 0
    while len(idx) < count and i < n:
        if i not in idx:
            idx.append(i)
        i += 1
    idx = sorted(idx)[:count]

    return GapSample([frames[i] for i in idx], [ts_all[i] for i in idx], idx)
