"""Rooms — persistent activity/topic channels (product vision).

A Room is a durable feed the user organizes their life around (Reading, Scripture,
a show's discussion, a coding project). Captured Events are auto-routed into rooms
by a RoomMatcher; user notes and room-scoped chat land here too (later phases).

Kinds:
- "screen"   — auto, the single room for everything captured from the PC screen
- "camera"   — auto, the single room for everything the home cameras see
- "topic"    — user-defined (e.g. Scripture) with custom keyword/app matchers
- "daily"    — the single catch-all room every event also flows into
- "activity" / "project" — legacy auto kinds, one per activity_type / project_id.
  No longer created; `Neo4jStore.consolidate_source_rooms()` folds them into the
  two source rooms. Which app or camera an event came from is a tag on the event,
  not a room of its own.

Note that "camera" also marks the `home` memory domain in the graph queries.
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
    instructions: str = ""
    color: str = "#8B7CF6"
    icon: str = "forum"
    archived: bool = False
    pinned: bool = False
    position: int = 0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
