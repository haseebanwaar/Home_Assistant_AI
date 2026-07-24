"""Graph-backed assistant tools (frontend-plan Step 1).

Exposes the Neo4j memory graph (sessions/events/entities/co-occurrence) to the
voice assistant as function-calling tools, so it can answer questions like
"what did I work on yesterday?", "when did I last use PyInstaller?", and "what
was on screen alongside the API logs?" — none of which the vector-only
retrieve_memory tool can do.

Handlers are sync (the Neo4j driver is sync); the ToolRegistry runs them
off-thread. They resolve the graph store lazily via `get_store` so registration
can happen at startup once the store exists, and they degrade to a plain message
when the graph is unavailable.
"""
from __future__ import annotations

import datetime
import logging

logger = logging.getLogger("home_assistant")


def _hm(ts):
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except (OverflowError, OSError, ValueError, TypeError):
        return "--:--"


def _resolve_date(date):
    """Accept 'today'/'yesterday'/None/ISO 'YYYY-MM-DD' -> ISO date string."""
    if not date:
        return datetime.date.today().isoformat()
    d = str(date).strip().lower()
    if d in ("today", "now"):
        return datetime.date.today().isoformat()
    if d == "yesterday":
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    return str(date).strip()


def make_graph_tools(get_store):
    def recall_timeline(date=None):
        store = get_store()
        if store is None:
            return "Graph memory is not available."
        ds = _resolve_date(date)
        try:
            sessions = store.sessions_with_events(ds)
        except Exception as exc:
            logger.warning("recall_timeline failed: %s", exc)
            return "Could not read the timeline."
        if not sessions:
            return f"No recorded activity for {ds}."
        out = []
        for s in sessions:
            events = [e for e in (s.get("events") or []) if e]
            ev_txt = "; ".join(
                f"{_hm(e.get('span_start'))} {(e.get('summary') or '').strip()[:80]}"
                for e in events[:8])
            mins = (s.get("active_seconds") or 0) / 60
            out.append(f"{ds} — {s.get('application', '?')} "
                       f"({s.get('activity', '?')}, {mins:.0f} min): {ev_txt}")
        return out

    def entity_history(name):
        store = get_store()
        if store is None:
            return "Graph memory is not available."
        try:
            rows = store.events_for_entity(name)
        except Exception as exc:
            logger.warning("entity_history failed: %s", exc)
            return f"Could not look up '{name}'."
        if not rows:
            return f"No memory of '{name}'."
        return [f"{_hm(r.get('span_start'))}-{_hm(r.get('span_end'))}: "
                f"{(r.get('summary') or '').strip()[:100]}" for r in rows[:10]]

    def find_related(entity):
        store = get_store()
        if store is None:
            return "Graph memory is not available."
        try:
            rows = store.co_occurring_entities(entity)
        except Exception as exc:
            logger.warning("find_related failed: %s", exc)
            return f"Could not look up '{entity}'."
        if not rows:
            return f"Nothing was seen on screen together with '{entity}'."
        items = ", ".join(f"{r['name']} (x{r['shared_frames']})" for r in rows[:12])
        return f"Seen on screen together with '{entity}': {items}"

    return recall_timeline, entity_history, find_related


RECALL_TIMELINE_SCHEMA = {
    "name": "recall_timeline",
    "description": "Recall what the user did on a given day from their screen-activity "
                   "graph: the sessions (per app/project) and the events within them, "
                   "with times. Use for 'what did I do today/yesterday', 'what was I "
                   "working on', 'how long was I in <app>'.",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "'today', 'yesterday', or an ISO date 'YYYY-MM-DD'. "
                               "Defaults to today.",
            },
        },
    },
}

ENTITY_HISTORY_SCHEMA = {
    "name": "entity_history",
    "description": "Find when the user encountered a specific named thing on screen "
                   "(a file, library, tool, person, project, URL, topic) and what was "
                   "happening then. Use for 'when did I last look at <X>', "
                   "'have I seen <X> before'.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The entity name, e.g. 'PyInstaller', 'pandas', 'loggers.py'."},
        },
        "required": ["name"],
    },
}

FIND_RELATED_SCHEMA = {
    "name": "find_related",
    "description": "Find entities that appeared on screen in the SAME FRAME as a given "
                   "entity (true co-occurrence). Use for 'what was on screen alongside "
                   "<X>', 'what came up with <X>'.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "The entity to find co-occurrences for."},
        },
        "required": ["entity"],
    },
}


def register_graph_tools(registry, get_store):
    """Register the three graph tools against a lazy store getter."""
    recall_timeline, entity_history, find_related = make_graph_tools(get_store)
    registry.register("recall_timeline", RECALL_TIMELINE_SCHEMA, recall_timeline)
    registry.register("entity_history", ENTITY_HISTORY_SCHEMA, entity_history)
    registry.register("find_related", FIND_RELATED_SCHEMA, find_related)
