"""Keeping the Rooms screen worth looking at.

Auto rooms are now one per capture source (Screen, Cameras) and both are pinned,
so the pile-up this module was written for — a room per activity type and per
project name ever seen, a folder opened once becoming a room forever — no longer
accumulates. What is left to tidy is user topic rooms that went quiet and any
legacy auto rooms still around from before the consolidation.

This module decides what is junk. It only ever *proposes*: nothing here deletes,
and archiving is reversible, because a room the user cares about can easily look
thin (a project they think about often but touch rarely).

Rules, all deliberately conservative:
- never touch a pinned room, a user-made (non-auto) room, or the Daily room;
- never touch a room with notes or chat in it — the user invested there;
- "stale" = no activity for `stale_days` AND less than `thin_minutes` of total
  activity, so a long-running project that went quiet is left alone.
"""
from __future__ import annotations

import time

DEFAULT_STALE_DAYS = 21
DEFAULT_THIN_MINUTES = 30


def _protected(room):
    """Rooms hygiene must never act on."""
    return (room.get("pinned")
            or not room.get("auto", True)
            or room.get("kind") == "daily"
            # Any note or chat message means the user engaged with this room.
            or (room.get("notes") or 0) > 0
            or (room.get("messages") or 0) > 0)


def stale_rooms(stats, stale_days=DEFAULT_STALE_DAYS,
                thin_minutes=DEFAULT_THIN_MINUTES, now=None):
    """Auto rooms that are both cold and thin — safe archive candidates."""
    now = now if now is not None else time.time()
    cutoff = now - stale_days * 86400
    candidates = []
    for room in stats:
        if _protected(room):
            continue
        last = room.get("last_event_at")
        minutes = room.get("active_minutes") or 0
        if last is None:
            # Never had an event at all: an empty auto room.
            candidates.append({**room, "reason": "no activity recorded",
                               "idle_days": None})
            continue
        if last < cutoff and minutes < thin_minutes:
            candidates.append({
                **room,
                "reason": (f"{minutes:.0f} min total, last active "
                           f"{(now - last) / 86400:.0f} days ago"),
                "idle_days": round((now - last) / 86400),
            })
    candidates.sort(key=lambda r: (r.get("active_minutes") or 0))
    return candidates


def merge_suggestions(overlaps, stats=None):
    """Rank room pairs that look like one topic, richest room first as target."""
    minutes = {row["room_id"]: (row.get("active_minutes") or 0)
               for row in (stats or [])}
    suggestions = []
    for pair in overlaps:
        a, b = pair["room_a"], pair["room_b"]
        # Merge the smaller room into the larger one — keeps the established feed.
        if minutes.get(b, 0) > minutes.get(a, 0):
            source, target = a, b
            source_name, target_name = pair["name_a"], pair["name_b"]
        else:
            source, target = b, a
            source_name, target_name = pair["name_b"], pair["name_a"]
        suggestions.append({
            "source_room_id": source, "source_name": source_name,
            "target_room_id": target, "target_name": target_name,
            "shared_entities": pair["shared"],
            "overlap_pct": pair["overlap_pct"],
            "reason": (f"{pair['shared']} shared entities "
                       f"({pair['overlap_pct']:.0f}% overlap)"),
        })
    return suggestions
