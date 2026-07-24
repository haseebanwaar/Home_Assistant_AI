"""Step 1 — structured observation record (plan §6.1).

A typed metadata record for a single capture / active-window sample. This is the
raw signal every later stage (extraction, profiles, boundary detection) keys off
of, so it is captured even when the VLM never runs.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Observation(BaseModel):
    timestamp: float
    frame_path: Optional[str] = None
    process_name: Optional[str] = None  # executable, e.g. "opera.exe"
    application: Optional[str] = None    # friendly name, e.g. "opera"
    window_title: Optional[str] = None
    url: Optional[str] = None
    screen_id: Optional[int] = None
