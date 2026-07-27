"""Room routing — one room per capture source.

Rooms used to be created per activity type, per coding project and per camera,
which meant a dozen near-empty channels after a day of use: the room list became
the noise it was supposed to organize. Instead there are two auto rooms, one per
place observations come from:

    Screen   — everything seen on the PC screen
    Cameras  — everything seen by the home cameras

Which *app* or *which camera* an event came from is not a room; it is a tag on
the event (`application`), shown on the bubble in the feed. So Opera, PyCharm and
a terminal all live in Screen, tagged; each camera's events live in Cameras,
tagged with the camera's name.

User-defined "topic" rooms still win over the source rooms when their
keyword/app/project matcher fires — those are explicit intent, not clutter.

Scoring weights (higher = more specific / stronger user intent):
    title keyword 4 · project 3 · app 2 · entity type 1 · activity 1
"""
from __future__ import annotations

import re

from memory.models.room import Room, RoomMatcher

W_KEYWORD = 4
W_PROJECT = 3
W_APP = 2
W_ENTITY = 1
W_ACTIVITY = 1

DAILY_ROOM_ID = "daily"
CAMERA_ROOM_ID = "camera"
SCREEN_ROOM_ID = "screen"

# room_id -> (name, kind, description, color, icon). `kind` is load-bearing:
# 'camera' marks the home memory domain everywhere in the graph queries.
SOURCE_ROOMS = {
    CAMERA_ROOM_ID: (
        "Cameras", "camera",
        "Everything the home cameras see. Each event is tagged with its camera.",
        "#F59E0B", "videocam",
    ),
    SCREEN_ROOM_ID: (
        "Screen", "screen",
        "Everything captured from the PC screen. Each event is tagged with its app.",
        "#8B7CF6", "desktop_windows",
    ),
}


def _slug(text):
    """Room-id fragment for a user-supplied name ("My Reading" -> "my-reading")."""
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-") or "room"


def source_room_id(source):
    """Map a capture source (`log_context`) onto its room.

    Anything camera-shaped ('camera', 'mobile_camera', 'camera:front-door') is
    home; everything else is screen, which is also the safe default for older
    events that carry no source at all.
    """
    return CAMERA_ROOM_ID if "camera" in str(source or "").lower() else SCREEN_ROOM_ID


class RoomRegistry:
    def __init__(self, rooms=None):
        self.rooms = {r.room_id: r for r in (rooms or [])}

    def add(self, room):
        self.rooms[room.room_id] = room
        return room

    def ensure_daily(self):
        rid = DAILY_ROOM_ID
        if rid not in self.rooms:
            self.add(Room(room_id=rid, name="Daily", kind="daily", auto=True,
                          color="#6EE7D8", icon="calendar_today", pinned=True))
        return self.rooms[rid]

    def ensure_source_room(self, source):
        """Idempotently return the Screen or Cameras room for a capture source."""
        rid = source_room_id(source)
        if rid not in self.rooms:
            name, kind, description, color, icon = SOURCE_ROOMS[rid]
            self.add(Room(room_id=rid, name=name, kind=kind, auto=True,
                          description=description, color=color, icon=icon,
                          pinned=True, matcher=RoomMatcher()))
        return self.rooms[rid]

    @staticmethod
    def _score(matcher, event, entity_types):
        s = 0
        if event.get("activity_type") and event["activity_type"] in matcher.activity_types:
            s += W_ACTIVITY
        app = (event.get("application") or "").lower()
        if any(a.lower() in app for a in matcher.apps):
            s += W_APP
        summary = (event.get("summary") or "").lower()
        if any(k.lower() in summary for k in matcher.title_keywords):
            s += W_KEYWORD
        pid = event.get("project_id")
        if pid and pid in matcher.project_ids:
            s += W_PROJECT
        if any(t in (entity_types or []) for t in matcher.entity_types):
            s += W_ENTITY
        return s

    def route(self, event, entity_types=None):
        """Return the best room for an event: a matching topic room, else source.

        The source room is ensured either way, so both channels exist from the
        first observation rather than appearing halfway through a session.
        """
        source_room = self.ensure_source_room(event.get("source"))

        best, best_score = None, 0
        for room in self.rooms.values():
            # Only user intent outranks the source room. Every auto room is
            # skipped, not just today's three kinds: legacy 'activity'/'project'
            # rooms left in a graph from before source rooms existed still carry
            # live matchers, and `activity_types: ['watching']` — what the camera
            # prompt hardcodes — silently captured every camera event away from
            # Cameras. An auto room must never be able to win this contest.
            if room.auto or room.kind in ("daily", "camera", "screen") or room.archived:
                continue
            score = self._score(room.matcher, event, entity_types)
            if score > best_score:
                best, best_score = room, score
        return best or source_room
