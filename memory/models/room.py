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

from typing import List, Literal, Optional

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
    kind: str = "topic"          # screen | camera | topic | daily | agent
    auto: bool = True            # auto-created vs user-defined
    matcher: RoomMatcher = Field(default_factory=RoomMatcher)
    description: str = ""
    instructions: str = ""
    # Legacy compatibility field.  All rooms use Claude Code while the runtime
    # is enabled; disabling it globally is the direct-chat emergency fallback.
    assistant_mode: Literal["chat", "agent"] = "agent"
    # quick caps the loop at 3/3; investigate uses normal room budgets; act is
    # intended for rooms with explicitly granted writable workspace tools.
    execution_profile: Literal["quick", "investigate", "act"] = "investigate"
    agent_tools: List[str] = Field(default_factory=lambda: ["graph"])
    # Blank means a stable directory derived from room_id under
    # AGENT_WORKSPACE_ROOT. Relative paths stay under that root; absolute paths
    # deliberately grant this room access to a different directory.
    agent_workspace: str = ""
    # Zero inherits the runtime default (or Research's larger default).
    agent_request_limit: int = Field(default=0, ge=0, le=256)
    agent_tool_calls_limit: int = Field(default=0, ge=0, le=1024)
    color: str = "#8B7CF6"
    icon: str = "forum"
    archived: bool = False
    pinned: bool = False
    position: int = 0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
