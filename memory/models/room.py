"""Rooms — persistent activity/topic channels (product vision).

A Room is a durable feed the user organizes their life around (Reading, Scripture,
a show's discussion, a coding project). Captured Events are auto-routed into rooms
by a RoomMatcher; user notes and room-scoped chat land here too (later phases).

Kinds:
- "activity" — auto, one per activity_type (Reading/Watching/Browsing/...)
- "project"  — auto, one per coding project_id
- "topic"    — user-defined (e.g. Scripture) with custom keyword/app matchers
- "daily"    — the single catch-all room every event also flows into
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RoomMatcher(BaseModel):
    activity_types: List[str] = Field(default_factory=list)
    apps: List[str] = Field(default_factory=list)            # substrings of application
    title_keywords: List[str] = Field(default_factory=list)  # lowercase substrings of summary
    project_ids: List[str] = Field(default_factory=list)
    entity_types: List[str] = Field(default_factory=list)


class Room(BaseModel):
    room_id: str
    name: str
    kind: str = "topic"          # activity | project | topic | daily
    auto: bool = True            # auto-created vs user-defined
    matcher: RoomMatcher = Field(default_factory=RoomMatcher)
    description: str = ""
    color: str = "#8B7CF6"
    icon: str = "forum"
    archived: bool = False
    pinned: bool = False
    position: int = 0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
