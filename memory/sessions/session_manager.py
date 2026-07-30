"""Step 6 — turn the boundary stream into sessions + events (plan §4.1, §7).

Feed observations in chronological order via `observe(...)`. The manager keys
sessions on (activity_type, application, project_id):

- a new key creates a session;
- switching away pauses the previous session (it is kept, not destroyed);
- returning to a prior key resumes that same session (same session_id).

Events are opened on a switch or a boundary/new_event label and extended while
observations keep appending. Event spans tile the timeline: an event's span_end
is fixed to the next event's span_start the moment that next event opens
(incremental close), so consecutive events share edges with no gaps/overlaps.
`finalize()` extends the last still-open event by a tail so it has real duration.

Tiling is capped by `idle_grace_seconds`. Tiling assumes the gap between two
observations was spent doing the outgoing thing, which is true for a switch a
minute later and false for a machine left alone overnight — that is how a window
nobody touched collects hours. When a grace is set, an event ends no later than
`last known user activity + grace`, so an unattended gap is simply not counted
and the timeline has a hole where the user was away. Callers that have no input
signal (cameras: nobody is at the keyboard) leave it None and keep full tiling.

Sessions key on (application, project_id) — NOT the VLM's per-minute
activity_type, which jitters and would shatter one continuous window into several
sessions. activity_type is instead a session property set to the dominant label
across its events (each event keeps its own per-minute activity).

ID strategy (Step-a / live):
- "counter" (default): readable sess-N / evt-N, stable within a single run — used
  by the offline tool.
- "deterministic": session_id = "date|app|project",
  event_id = "session_id@span_start" — stable ACROSS restarts, so live idempotent
  MERGE upserts resume the same session instead of duplicating it.
"""
from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from memory.models.session import Event, Session

NEW_EVENT_LABELS = {"boundary", "new_event"}


@dataclass
class ObserveResult:
    """What one observe() call changed — consumed by the live pipeline."""
    session: Session
    current_event: Event            # the now-open event (provisional span_end)
    closed_event: Optional[Event]   # the event that just got its final span_end


def _det_session_id(timestamp, application, project_id):
    date = datetime.date.fromtimestamp(timestamp).isoformat()
    return f"{date}|{application}|{project_id or 'none'}"


def _det_event_id(session_id, span_start):
    return f"{session_id}@{span_start:.3f}"


class SessionManager:
    def __init__(self, id_strategy="counter", idle_grace_seconds=None):
        if id_strategy not in ("counter", "deterministic"):
            raise ValueError(f"unknown id_strategy: {id_strategy}")
        self.id_strategy = id_strategy
        # None disables the cap and restores pure tiling (see module docstring).
        self.idle_grace_seconds = (None if idle_grace_seconds is None
                                   else max(0.0, float(idle_grace_seconds)))
        self.sessions = {}          # session_key -> Session
        self.events: List[Event] = []  # chronological
        self._order = []            # session_key insertion order
        self._current_key = None
        self._current_event: Optional[Event] = None
        # event_id -> the latest moment the user was known to be present during
        # it. Kept here rather than on Event so the stored graph shape is
        # unchanged; only span_end (which it caps) is persisted.
        self._active_until = {}
        self._sid = 0
        self._eid = 0
        self._finalized = False

    @staticmethod
    def _key(application, project_id):
        return (application, project_id)

    def _new_session_id(self, timestamp, application, project_id):
        if self.id_strategy == "deterministic":
            return _det_session_id(timestamp, application, project_id)
        self._sid += 1
        return f"sess-{self._sid}"

    def _new_event_id(self, session_id, span_start):
        if self.id_strategy == "deterministic":
            return _det_event_id(session_id, span_start)
        self._eid += 1
        return f"evt-{self._eid}"

    def _close_event(self, event, at):
        """End `event` at `at`, capped by the idle grace, and refresh its span.

        Must be called before `event.span_end` is overwritten: the current value
        is the last observation this event saw, which is the fallback for when the
        user was last known to be present.
        """
        limit = None
        if self.idle_grace_seconds is not None:
            active_until = self._active_until.get(event.event_id, event.span_end)
            limit = active_until + self.idle_grace_seconds
        end = at if limit is None else min(at, limit)
        # An event never runs backwards, however the caps land.
        event.span_end = max(end, event.span_start)
        event.span_seconds = round(event.span_end - event.span_start, 3)

    def observe(self, timestamp, activity_type, application, project_id=None,
                boundary_label="append", summary="",
                active_until=None) -> ObserveResult:
        """Ingest one chronological observation; returns what changed.

        `active_until` is the wall-clock time the user was last known to be at the
        keyboard (sources.idle). It defaults to `timestamp` — i.e. "present now" —
        which is the right assumption for any caller without an input signal.
        """
        key = self._key(application, project_id)
        switching = key != self._current_key

        # Pause the session we're leaving — keep it, don't destroy it.
        if switching and self._current_key is not None:
            self.sessions[self._current_key].state = "paused"

        session = self.sessions.get(key)
        if session is None:
            session = Session(
                session_id=self._new_session_id(timestamp, application, project_id),
                activity_type=activity_type, application=application,
                project_id=project_id, start=timestamp, end=timestamp,
                state="active",
            )
            self.sessions[key] = session
            self._order.append(key)
        else:
            if session.state == "paused":
                session.resume_count += 1
            session.state = "active"

        start_new_event = (
            switching or boundary_label in NEW_EVENT_LABELS or self._current_event is None
        )
        closed_event = None
        if start_new_event:
            # Incremental close: the outgoing event ends exactly where this one
            # begins, giving it a final, non-overlapping span.
            if self._current_event is not None:
                closed_event = self._current_event
                self._close_event(closed_event, timestamp)

            event = Event(
                event_id=self._new_event_id(session.session_id, timestamp),
                session_id=session.session_id,
                activity_type=activity_type, application=application,
                project_id=project_id, span_start=timestamp, span_end=timestamp,
                boundary_label=boundary_label, summary=summary,
            )
            self.events.append(event)
            session.event_ids.append(event.event_id)
            self._current_event = event
        else:
            self._current_event.span_end = timestamp
            self._current_event.span_seconds = round(
                self._current_event.span_end - self._current_event.span_start, 3)
            if summary and not self._current_event.summary:
                self._current_event.summary = summary

        # Remember the newest evidence that the user was present during this
        # event; it is what caps the event's end if a long gap follows.
        presence = timestamp if active_until is None else float(active_until)
        eid = self._current_event.event_id
        self._active_until[eid] = max(self._active_until.get(eid, presence), presence)

        session.end = timestamp
        self._current_key = key

        # Roll spans up into the affected session(s).
        self._rollup(session)
        if closed_event is not None:
            self._rollup(self.sessions_by_id.get(closed_event.session_id, session))

        return ObserveResult(session=session, current_event=self._current_event,
                             closed_event=closed_event)

    @property
    def sessions_by_id(self):
        return {s.session_id: s for s in self.sessions.values()}

    def _rollup(self, session):
        mine = [e for e in self.events if e.session_id == session.session_id]
        if mine:
            session.active_seconds = round(sum(e.span_seconds for e in mine), 3)
            session.start = min(e.span_start for e in mine)
            session.end = max(e.span_end for e in mine)
            # Dominant activity across the session's events (robust to VLM jitter).
            session.activity_type = Counter(e.activity_type for e in mine).most_common(1)[0][0]

    def finalize(self, tail_seconds=None):
        """Extend the last open event by a tail and roll up totals. Idempotent.

        Event spans are already tiled incrementally by observe(); this only gives
        the final still-open event a real duration and refreshes rollups.
        """
        if self._finalized or not self.events:
            self._finalized = True
            return

        evs = self.events
        starts = [e.span_start for e in evs]
        if tail_seconds is None:
            diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
            tail_seconds = sorted(diffs)[len(diffs) // 2] if diffs else 60.0

        # Re-assert tiling (no-op if already tiled) then extend the last event
        # (whose span_end is currently the last observation timestamp) by a tail.
        # Both go through _close_event so the idle cap is applied here too —
        # re-tiling blindly would hand every unattended gap straight back.
        for k in range(len(evs) - 1):
            self._close_event(evs[k], evs[k + 1].span_start)
        self._close_event(evs[-1], evs[-1].span_end + tail_seconds)

        for s in self.sessions.values():
            self._rollup(s)

        self._finalized = True

    def ordered_sessions(self) -> List[Session]:
        return [self.sessions[k] for k in self._order]
