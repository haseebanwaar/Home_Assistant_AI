"""Room routing — auto-first (product vision, Phase 1).

Given an event and its entity types, pick the best-matching room. Auto-creates a
per-activity room and a per-project room on demand, so rooms appear as the user
works. User "topic" rooms with keyword/app matchers outscore the generic auto
rooms (most-specific-wins, same spirit as the Step-3 profile routing).

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


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-") or "unknown"


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

    def ensure_activity_room(self, activity_type):
        activity_type = activity_type or "other"
        rid = f"activity:{_slug(activity_type)}"
        if rid not in self.rooms:
            self.add(Room(room_id=rid, name=activity_type.capitalize(), kind="activity",
                          auto=True, matcher=RoomMatcher(activity_types=[activity_type])))
        return self.rooms[rid]

    def ensure_project_room(self, project_id):
        rid = f"project:{_slug(project_id)}"
        if rid not in self.rooms:
            self.add(Room(room_id=rid, name=f"Coding: {project_id}", kind="project",
                          auto=True, matcher=RoomMatcher(project_ids=[project_id])))
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
        """Return the best room for an event, auto-ensuring activity/project rooms."""
        self.ensure_activity_room(event.get("activity_type"))
        if event.get("project_id"):
            self.ensure_project_room(event["project_id"])

        best, best_score = None, 0
        for room in self.rooms.values():
            if room.kind == "daily" or room.archived:
                continue
            score = self._score(room.matcher, event, entity_types)
            if score > best_score:
                best, best_score = room, score
        # Fallback should never trigger (activity room always scores >=1), but be safe.
        return best or self.ensure_activity_room(event.get("activity_type"))
