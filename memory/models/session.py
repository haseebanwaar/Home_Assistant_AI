"""Step 6 — session + event data shapes (plan §4.1, §7).

A Session groups continuous work on the same (activity_type, application,
project_id). It can be paused when the user switches away and resumed (same
session_id) when they return — it is never destroyed by a switch. An Event is a
contiguous time span within a session; events tile the global timeline with no
gaps or overlaps.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    event_id: str
    session_id: str
    activity_type: str
    application: str
    project_id: Optional[str] = None
    span_start: float
    span_end: float
    span_seconds: float = 0.0
    boundary_label: str = "append"
    summary: str = ""
    # The extractor's automatic signal. A user-selected priority is stored
    # separately in Neo4j so later observations cannot overwrite it.
    importance: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class Session(BaseModel):
    session_id: str
    activity_type: str
    application: str
    project_id: Optional[str] = None
    start: float
    end: float
    state: str = "active"          # "active" | "paused"
    active_seconds: float = 0.0
    resume_count: int = 0
    event_ids: List[str] = Field(default_factory=list)

    @property
    def session_key(self):
        return (self.activity_type, self.application, self.project_id)
