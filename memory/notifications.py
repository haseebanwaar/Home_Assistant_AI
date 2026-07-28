"""Durable, deduplicated notifications derived from memory events."""
from __future__ import annotations

from collections import deque
import json
import os
import re
from threading import RLock
import time
import uuid


CRITICAL_TERMS = {
    "fire": "safety",
    "smoke": "safety",
    "gas leak": "safety",
    "water leak": "safety",
    "flooding": "safety",
    "fell down": "safety",
    "unconscious": "safety",
    "medical emergency": "safety",
    "intruder": "security",
    "break-in": "security",
    "forced entry": "security",
    "weapon": "security",
}

IMPORTANT_TERMS = {
    "person at the door": "home",
    "unknown person": "security",
    "suspicious": "security",
    "package delivered": "delivery",
    "delivery arrived": "delivery",
    "door left open": "home",
    "vehicle arrived": "home",
    "offline": "system",
    "disconnected": "system",
    "failed": "system",
    "failure": "system",
    "deadline": "schedule",
    "appointment": "schedule",
    "meeting": "schedule",
    "completed": "progress",
    "finished": "progress",
}

_NEGATIONS = ("no ", "not ", "without ", "did not ")


def _matched_category(text, terms):
    lowered = text.casefold()
    for term, category in terms.items():
        index = lowered.find(term)
        if index < 0:
            continue
        prefix = lowered[max(0, index - 16):index]
        if any(prefix.rstrip().endswith(neg.rstrip()) for neg in _NEGATIONS):
            continue
        return category
    return None


def classify_event(event):
    """Return (severity, category) or (None, None) for ordinary activity."""
    summary = str(event.get("summary") or "").strip()
    importance = float(event.get("importance") or 0.0)
    critical_category = _matched_category(summary, CRITICAL_TERMS)
    if critical_category or importance >= 0.95:
        return "critical", critical_category or "critical"
    important_category = _matched_category(summary, IMPORTANT_TERMS)
    if important_category or importance >= 0.75:
        return "important", important_category or "activity"
    return None, None


def _signature(text):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return " ".join(words[:24])


class NotificationCenter:
    """Small JSON-backed alert inbox safe to call from capture threads."""

    def __init__(self, path="data/notifications.json", max_items=300,
                 important_cooldown_seconds=600):
        self.path = path
        self.max_items = max_items
        self.important_cooldown_seconds = important_cooldown_seconds
        self._lock = RLock()
        self._items = deque(maxlen=max_items)
        self._sequence = 0
        self._load()

    def _load(self):
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("items") if isinstance(payload, dict) else payload
            for item in (items or [])[-self.max_items:]:
                self._items.append(dict(item))
                self._sequence = max(
                    self._sequence, int(item.get("sequence") or 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A damaged optional inbox must never stop capture.
            self._items.clear()
            self._sequence = 0

    def _save(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"items": list(self._items)}, handle,
                      ensure_ascii=False, indent=2)
        os.replace(temp_path, self.path)

    def consider_event(self, event):
        """Create one alert for a notable event, or return None when suppressed."""
        severity, category = classify_event(event)
        if severity is None:
            return None
        now = float(event.get("timestamp") or event.get("span_end") or time.time())
        event_id = str(event.get("event_id") or "").strip()
        summary = str(event.get("summary") or "").strip()
        if not summary:
            return None
        source = str(event.get("source") or "screen")
        signature = _signature(summary)

        with self._lock:
            same_event = next(
                (item for item in reversed(self._items)
                 if event_id and item.get("event_id") == event_id),
                None)
            if same_event:
                already_at_least_as_severe = (
                    same_event.get("severity") == "critical"
                    or same_event.get("severity") == severity)
                if already_at_least_as_severe:
                    return None

            for item in reversed(self._items):
                age = now - float(item.get("timestamp") or 0)
                if age > 1800:
                    break
                if item.get("signature") == signature:
                    return None
                if (severity == "important"
                        and age < self.important_cooldown_seconds
                        and item.get("source") == source
                        and item.get("category") == category):
                    return None

            self._sequence += 1
            application = str(event.get("application") or "").strip()
            if severity == "critical":
                title = f"Critical {category} alert"
            elif application:
                title = f"Important · {application}"
            else:
                title = "Important activity"
            item = {
                "id": uuid.uuid4().hex,
                "sequence": self._sequence,
                "event_id": event_id or None,
                "severity": severity,
                "category": category,
                "title": title,
                "body": summary,
                "source": source,
                "application": application or None,
                "room_id": event.get("room_id"),
                # The footage this alert was raised from. An alert the user can
                # watch is checkable; one they can only read has to be believed.
                "clip_id": (str(event.get("clip_id")).strip() or None
                            if event.get("clip_id") else None),
                "timestamp": now,
                "read": False,
                "signature": signature,
            }
            self._items.append(item)
            self._save()
            return dict(item)

    def list(self, since=0, limit=100, unread_only=False):
        with self._lock:
            items = [
                dict(item) for item in reversed(self._items)
                if int(item.get("sequence") or 0) > int(since)
                and (not unread_only or not item.get("read"))
            ][:max(1, min(int(limit), 300))]
            return {
                "latest_sequence": self._sequence,
                "unread_count": sum(
                    1 for item in self._items if not item.get("read")),
                "notifications": items,
            }

    def mark_read(self, notification_id):
        with self._lock:
            for item in self._items:
                if item.get("id") == notification_id:
                    item["read"] = True
                    self._save()
                    return dict(item)
        return None

    def mark_all_read(self):
        with self._lock:
            changed = 0
            for item in self._items:
                if not item.get("read"):
                    item["read"] = True
                    changed += 1
            if changed:
                self._save()
            return changed
