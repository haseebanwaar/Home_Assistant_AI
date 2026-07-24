"""Step 7 — Neo4j store: connect + apply schema (plan §19).

Thin wrapper over the official driver. This step only stands the infra up:
connectivity, constraints, and indexes. Real Day/Session/Event writes land in
Step 8. Constraints use IS UNIQUE so the idempotent MERGE writes in later steps
can't create duplicates on replay.
"""
from __future__ import annotations

import datetime
import logging
import os
import re

from neo4j import GraphDatabase

logger = logging.getLogger("home_assistant")

# Uniqueness constraints (also create backing indexes). Keyed by the stable ids
# our file-backed model already produces.
CONSTRAINTS = [
    "CREATE CONSTRAINT day_date IF NOT EXISTS FOR (d:Day) REQUIRE d.date IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
    "CREATE CONSTRAINT room_id IF NOT EXISTS FOR (r:Room) REQUIRE r.room_id IS UNIQUE",
    "CREATE CONSTRAINT room_note_id IF NOT EXISTS FOR (n:RoomNote) REQUIRE n.note_id IS UNIQUE",
    "CREATE CONSTRAINT room_message_id IF NOT EXISTS FOR (m:RoomMessage) REQUIRE m.message_id IS UNIQUE",
]

# Secondary indexes for span-aware and lookup queries.
INDEXES = [
    "CREATE INDEX event_span_start IF NOT EXISTS FOR (e:Event) ON (e.span_start)",
    "CREATE INDEX session_activity IF NOT EXISTS FOR (s:Session) ON (s.activity_type)",
    "CREATE INDEX session_project IF NOT EXISTS FOR (s:Session) ON (s.project_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
]


class Neo4jStore:
    def __init__(self, uri=None, username=None, password=None, database=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    # -- lifecycle ---------------------------------------------------------
    def verify(self):
        self._driver.verify_connectivity()
        return True

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers -----------------------------------------------------------
    def run(self, query, **params):
        """Execute a write/read query; returns the list of records."""
        return self._driver.execute_query(
            query, database_=self.database, **params
        ).records

    # -- schema ------------------------------------------------------------
    def apply_schema(self):
        """Create all constraints + indexes (idempotent)."""
        for stmt in CONSTRAINTS + INDEXES:
            self._driver.execute_query(stmt, database_=self.database)
        logger.info("Applied %d constraints + %d indexes",
                    len(CONSTRAINTS), len(INDEXES))

    def list_constraints(self):
        return [dict(r) for r in self.run("SHOW CONSTRAINTS")]

    def list_indexes(self):
        return [dict(r) for r in self.run("SHOW INDEXES")]

    # -- smoke -------------------------------------------------------------
    def merge_day(self, date_str):
        """MERGE a :Day node and return it (idempotent on date)."""
        records = self.run(
            "MERGE (d:Day {date: $date}) "
            "ON CREATE SET d.created_at = timestamp() "
            "RETURN d.date AS date, d.created_at AS created_at",
            date=date_str,
        )
        return dict(records[0]) if records else None

    # -- Step 8: dual-write Day/Session/Event ------------------------------
    def write_timeline(self, sessions, events):
        """Persist sessions + events in ONE transaction (plan §4, §9).

        Idempotent: MERGE on session_id/event_id, so replaying a capture updates
        rather than duplicates. Atomic: a failure mid-write rolls the whole thing
        back, so the graph is never left half-populated.

        Accepts pydantic Session/Event objects or plain dicts.
        """
        sess_rows = [_as_dict(s) for s in sessions]
        evt_rows = [_as_dict(e) for e in events]

        def _tx(tx):
            for s in sess_rows:
                date = datetime.date.fromtimestamp(s["start"]).isoformat()
                tx.run(_SESSION_CYPHER, date=date, **_session_params(s))
            for e in evt_rows:
                tx.run(_EVENT_CYPHER, **_event_params(e))
            return len(sess_rows), len(evt_rows)

        with self._driver.session(database=self.database) as session:
            return session.execute_write(_tx)

    def events_today(self, date_str=None):
        """The plan §14 span-aware read: today's events under their sessions."""
        date_str = date_str or datetime.date.today().isoformat()
        return [dict(r) for r in self.run(_EVENTS_TODAY_CYPHER, date=date_str)]

    def write_event_knowledge(self, items):
        """Ingest per-event entities + claims in one transaction (plan §10 basic).

        `items`: list of {event_id, entities:[{entity_id,name,type,confidence,
        role,co_presence}], claims:[{claim_id,text,confidence}]}.

        Entities use exact/normalized-name matching only (fuzzy POSSIBLY_SAME_AS
        is Step 12). MENTIONS carry {confidence, role, co_presence}; each claim
        gets a SUPPORTS evidence edge from its event. Idempotent via MERGE.
        """
        def _tx(tx):
            n_ent = n_claim = 0
            for it in items:
                eid = it["event_id"]
                for en in it.get("entities", []):
                    tx.run(_MENTION_CYPHER, eid=eid, **en)
                    n_ent += 1
                for cl in it.get("claims", []):
                    tx.run(_CLAIM_CYPHER, eid=eid, **cl)
                    n_claim += 1
            return n_ent, n_claim

        with self._driver.session(database=self.database) as session:
            return session.execute_write(_tx)

    def entities_for_session(self, session_id):
        """Debug handle: entities mentioned in a session, with co_presence."""
        return [dict(r) for r in self.run(_ENTITIES_FOR_SESSION_CYPHER, sid=session_id)]

    def orphan_claims(self):
        """Claims with no SUPPORTS evidence edge (should be zero)."""
        rows = self.run(
            "MATCH (c:Claim) WHERE NOT (c)<-[:SUPPORTS]-(:Event) RETURN count(c) AS n"
        )
        return rows[0]["n"] if rows else 0

    def knowledge_counts(self):
        rows = self.run(
            "MATCH (n:Entity) WITH count(n) AS entities "
            "MATCH ()-[m:MENTIONS]->() WITH entities, count(m) AS mentions "
            "MATCH (c:Claim) WITH entities, mentions, count(c) AS claims "
            "MATCH ()-[sp:SUPPORTS]->() RETURN entities, mentions, claims, count(sp) AS supports"
        )
        return dict(rows[0]) if rows else {}

    # -- Step 11: graph + hybrid retrieval ---------------------------------
    @staticmethod
    def _norm(name):
        return re.sub(r"\s+", " ", (name or "").strip().lower())

    def co_occurring_entities(self, entity_name, limit=25):
        """Entities seen in the SAME FRAME as `entity_name` (plan §13).

        Filters MENTIONS on co_presence='same_frame', so this answers true
        co-occurrence ("what appeared together"), not merely same-session.
        """
        return [dict(r) for r in self.run(
            _CO_OCCURRENCE_CYPHER, nid=self._norm(entity_name), limit=limit)]

    def events_for_entity(self, entity_name, limit=50):
        """Span-aware timeline of events mentioning an entity."""
        return [dict(r) for r in self.run(
            _EVENTS_FOR_ENTITY_CYPHER, nid=self._norm(entity_name), limit=limit)]

    def entities_for_events(self, event_ids):
        """Entities mentioned by a set of events (for hybrid enrichment)."""
        if not event_ids:
            return {}
        rows = self.run(_ENTITIES_FOR_EVENTS_CYPHER, ids=list(event_ids))
        out = {}
        for r in rows:
            out.setdefault(r["event"], []).append(
                {"name": r["name"], "type": r["type"],
                 "role": r["role"], "co_presence": r["co_presence"]})
        return out

    def top_entities(self, limit=25):
        return [dict(r) for r in self.run(
            "MATCH (n:Entity)<-[m:MENTIONS]-(:Event) "
            "RETURN n.name AS name, n.type AS type, count(m) AS mentions "
            "ORDER BY mentions DESC LIMIT $limit", limit=limit)]

    # -- Rooms (Phase 1) ---------------------------------------------------
    def _load_rooms(self):
        import json as _json
        from memory.models.room import Room, RoomMatcher
        rooms = []
        for r in self.run(
                "MATCH (r:Room) RETURN r.room_id AS room_id, r.name AS name, "
                "r.kind AS kind, r.auto AS auto, r.matcher_json AS matcher_json, "
                "r.description AS description, r.color AS color, r.icon AS icon, "
                "r.archived AS archived, r.pinned AS pinned, r.position AS position, "
                "r.created_at AS created_at, r.updated_at AS updated_at"):
            matcher = RoomMatcher()
            if r.get("matcher_json"):
                try:
                    matcher = RoomMatcher(**_json.loads(r["matcher_json"]))
                except Exception:
                    pass
            rooms.append(Room(room_id=r["room_id"], name=r.get("name") or r["room_id"],
                               kind=r.get("kind") or "topic",
                               auto=bool(r.get("auto")), matcher=matcher,
                               description=r.get("description") or "",
                               color=r.get("color") or "#8B7CF6",
                               icon=r.get("icon") or "forum",
                               archived=bool(r.get("archived")),
                               pinned=bool(r.get("pinned")),
                               position=int(r.get("position") or 0),
                               created_at=r.get("created_at"),
                               updated_at=r.get("updated_at")))
        return rooms

    def create_room(self, room):
        """Create a user-managed topic room. Raises ValueError on duplicate id."""
        import json as _json
        from memory.models.room import Room
        room = room if isinstance(room, Room) else Room(**room)
        if room.kind != "topic":
            raise ValueError("user-created rooms must have kind 'topic'")
        if self.get_room(room.room_id):
            raise ValueError("room_id already exists")
        rows = self.run(_CREATE_ROOM_CYPHER, **_room_params(room, _json))
        return dict(rows[0]) if rows else None

    def update_room(self, room_id, changes):
        """Update room metadata/matcher. The Daily identity and kind are protected."""
        import json as _json
        room = next((r for r in self._load_rooms() if r.room_id == room_id), None)
        if room is None:
            return None
        allowed = {"name", "description", "color", "icon", "archived", "pinned",
                   "position", "matcher"}
        for key, value in changes.items():
            if key in allowed:
                setattr(room, key, value)
        if room_id == "daily":
            room.name, room.kind, room.archived = "Daily", "daily", False
        rows = self.run(_UPDATE_ROOM_CYPHER, **_room_params(room, _json))
        return dict(rows[0]) if rows else None

    def delete_room(self, room_id):
        """Delete a room and its private notes/messages; events remain intact."""
        if room_id == "daily":
            raise ValueError("the Daily room cannot be deleted")
        rows = self.run(_DELETE_ROOM_CYPHER, room_id=room_id)
        return bool(rows and rows[0]["deleted"])

    def assign_rooms(self, events):
        """Route events to rooms and link them (idempotent). Auto-creates rooms.

        `events`: list of {event_id, activity_type, application, project_id,
        summary, entity_types}. Every event is also linked to the Daily room.
        """
        import json as _json
        from memory.rooms.registry import RoomRegistry

        reg = RoomRegistry(self._load_rooms())
        daily = reg.ensure_daily()
        rooms_to_merge, links = {}, []
        for ev in events:
            room = reg.route(ev, ev.get("entity_types") or [])
            rooms_to_merge[room.room_id] = room
            links.append((room.room_id, ev["event_id"], "primary"))
            links.append((daily.room_id, ev["event_id"], "daily"))
        rooms_to_merge[daily.room_id] = daily

        def _tx(tx):
            for r in rooms_to_merge.values():
                tx.run(_MERGE_ROOM_CYPHER, **_room_params(r, _json))
            for room_id, event_id, assignment in links:
                if assignment == "primary":
                    manual = tx.run(_HAS_MANUAL_PRIMARY_CYPHER, event_id=event_id).single()
                    if manual and manual["n"]:
                        continue
                    tx.run(_REMOVE_AUTO_PRIMARY_CYPHER, event_id=event_id)
                tx.run(_LINK_ROOM_EVENT_CYPHER, room_id=room_id, event_id=event_id,
                       assignment=assignment, manual=False)

        with self._driver.session(database=self.database) as session:
            session.execute_write(_tx)
        return {"rooms": len(rooms_to_merge), "links": len(links)}

    def list_rooms(self, include_archived=False):
        return [dict(r) for r in self.run(
            _LIST_ROOMS_CYPHER, include_archived=include_archived)]

    def get_room(self, room_id):
        rows = self.run("MATCH (r:Room {room_id: $room_id}) "
                        "RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind, "
                        "r.auto AS auto, r.description AS description, r.color AS color, "
                        "r.icon AS icon, r.archived AS archived, r.pinned AS pinned, "
                        "r.position AS position, r.matcher_json AS matcher_json",
                        room_id=room_id)
        if not rows:
            return None
        out = dict(rows[0])
        import json as _json
        try:
            out["matcher"] = _json.loads(out.pop("matcher_json") or "{}")
        except (TypeError, ValueError):
            out["matcher"] = {}
        return out

    def set_event_room(self, event_id, room_id, mode="primary"):
        """Manually move an event to a primary room or add it as secondary."""
        if mode not in {"primary", "secondary"}:
            raise ValueError("mode must be 'primary' or 'secondary'")
        if not self.get_room(room_id):
            raise ValueError("room does not exist")
        if mode == "primary":
            self.run(_REMOVE_ALL_PRIMARY_CYPHER, event_id=event_id)
        rows = self.run(_LINK_ROOM_EVENT_CYPHER, room_id=room_id, event_id=event_id,
                        assignment=mode, manual=True)
        return bool(rows)

    def remove_event_room(self, event_id, room_id):
        if room_id == "daily":
            raise ValueError("Daily membership cannot be removed")
        rows = self.run(
            "MATCH (r:Room {room_id: $room_id})-[rel:CONTAINS]->"
            "(e:Event {event_id: $event_id}) DELETE rel RETURN count(rel) AS n",
            room_id=room_id, event_id=event_id)
        return bool(rows and rows[0]["n"])

    def reroute_events(self, room_id=None):
        """Re-evaluate automatic routing for historical events.

        Manual primary assignments are deliberately preserved.
        """
        query = (
            "MATCH (e:Event) "
            "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
            "RETURN e.event_id AS event_id, e.activity_type AS activity_type, "
            "e.application AS application, e.project_id AS project_id, "
            "e.summary AS summary, collect(DISTINCT n.type) AS entity_types"
        )
        events = [dict(r) for r in self.run(query)]
        result = self.assign_rooms(events)
        return {**result, "events": len(events), "trigger_room_id": room_id}

    # -- Phase 2: notes + room-scoped chat ---------------------------------
    def add_note(self, room_id, text, ts=None):
        """User thought written into a room. Creates the room if missing (topic)."""
        import time as _t
        import uuid as _uuid
        note_id = _uuid.uuid4().hex
        ts = ts if ts is not None else _t.time()
        self.run(_ADD_NOTE_CYPHER, room_id=room_id, note_id=note_id, text=text, ts=ts)
        return {"note_id": note_id, "room_id": room_id, "text": text, "ts": ts}

    def update_note(self, room_id, note_id, text):
        rows = self.run(_UPDATE_NOTE_CYPHER, room_id=room_id, note_id=note_id, text=text)
        return dict(rows[0]) if rows else None

    def delete_note(self, room_id, note_id):
        rows = self.run(_DELETE_NOTE_CYPHER, room_id=room_id, note_id=note_id)
        return bool(rows and rows[0]["deleted"])

    def add_message(self, room_id, role, text, ts=None):
        """A room-scoped chat message (role: 'user' | 'assistant')."""
        import time as _t
        import uuid as _uuid
        message_id = _uuid.uuid4().hex
        ts = ts if ts is not None else _t.time()
        self.run(_ADD_MESSAGE_CYPHER, room_id=room_id, message_id=message_id,
                 role=role, text=text, ts=ts)
        return {"message_id": message_id, "room_id": room_id, "role": role, "text": text, "ts": ts}

    def room_messages(self, room_id, limit=20):
        """Recent chat messages in chronological order (for chat history)."""
        rows = [dict(r) for r in self.run(_ROOM_MESSAGES_CYPHER, room_id=room_id, limit=limit)]
        return list(reversed(rows))

    def room_context(self, room_id, event_limit=8, note_limit=8, entity_limit=15):
        """Grounding for room-scoped chat: recent events, notes, and top entities."""
        events = [r["summary"] for r in self.run(
            _ROOM_FEED_CYPHER, room_id=room_id, limit=event_limit) if r.get("summary")]
        notes = [r["text"] for r in self.run(
            _ROOM_NOTES_CYPHER, room_id=room_id, limit=note_limit)]
        entities = [r["name"] for r in self.run(
            _ROOM_ENTITIES_CYPHER, room_id=room_id, limit=entity_limit)]
        return {"events": events, "notes": notes, "entities": entities}

    def room_feed_full(self, room_id, date_str=None, limit=200, offset=0,
                       kinds=None, query=None):
        """Merged, time-ordered feed: events + notes + chat messages (newest first)."""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        fetch_limit = min(2000, limit + offset)
        params = {"room_id": room_id, "limit": fetch_limit}
        events = self.run(_ROOM_FEED_DATED_CYPHER if date_str else _ROOM_FEED_CYPHER,
                          **(_with_day(params, date_str) if date_str else params))
        items = [{"kind": "event", "event_id": e.get("event_id"),
                  "assignment": e.get("assignment"), "manual": e.get("manual"),
                  "ts": e["span_start"], "text": e.get("summary"),
                  "span_end": e.get("span_end"), "activity_type": e.get("activity_type"),
                  "application": e.get("application")} for e in events]
        for n in self.run(_ROOM_NOTES_CYPHER, room_id=room_id, limit=fetch_limit):
            items.append({"kind": "note", "note_id": n["note_id"],
                          "ts": n["ts"], "text": n["text"]})
        for m in self.run(_ROOM_MESSAGES_CYPHER, room_id=room_id, limit=fetch_limit):
            items.append({"kind": "message", "ts": m["ts"], "text": m["text"], "role": m["role"]})
        if date_str:
            start = datetime.datetime.fromisoformat(date_str).timestamp()
            end = start + 86400
            items = [item for item in items if start <= (item.get("ts") or 0) < end]
        items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
        if kinds:
            allowed = set(kinds)
            items = [item for item in items if item["kind"] in allowed]
        if query:
            needle = query.casefold()
            items = [item for item in items if needle in (item.get("text") or "").casefold()]
        return items[offset:offset + limit]

    def room_feed(self, room_id, date_str=None, limit=200):
        params = {"room_id": room_id, "limit": limit}
        if date_str:
            start = datetime.datetime.fromisoformat(date_str).timestamp()
            params["start"], params["end"] = start, start + 86400
            return [dict(r) for r in self.run(_ROOM_FEED_DATED_CYPHER, **params)]
        return [dict(r) for r in self.run(_ROOM_FEED_CYPHER, **params)]

    # -- Step 13: daily rollup ---------------------------------------------
    def sessions_with_events(self, date_str):
        """Day's sessions, each with its events (ordered)."""
        return [dict(r) for r in self.run(_SESSIONS_WITH_EVENTS_CYPHER, date=date_str)]

    def day_entities(self, date_str, limit=30):
        return [dict(r) for r in self.run(_DAY_ENTITIES_CYPHER, date=date_str, limit=limit)]

    def day_claims(self, date_str, limit=20):
        return [dict(r) for r in self.run(_DAY_CLAIMS_CYPHER, date=date_str, limit=limit)]

    def daily_metrics(self, date_str):
        """Deterministic productivity metrics for a day (Phase 3 Coach input)."""
        totals = self.run(_DAILY_TOTALS_CYPHER, date=date_str)
        t = dict(totals[0]) if totals else {}
        by_activity = [dict(r) for r in self.run(_DAILY_BY_ACTIVITY_CYPHER, date=date_str)]
        by_app = [dict(r) for r in self.run(_DAILY_BY_APP_CYPHER, date=date_str)]
        by_project = [dict(r) for r in self.run(_DAILY_BY_PROJECT_CYPHER, date=date_str)]

        active = t.get("active_seconds") or 0.0
        events = t.get("events") or 0
        switches = t.get("switches") or 0
        active_hours = active / 3600.0
        avg_event = (active / events) if events else 0.0
        focus_score = round(min(100.0, (avg_event / 300.0) * 100.0))
        return {
            "date": date_str,
            "active_seconds": round(active, 1),
            "active_minutes": round(active / 60.0, 1),
            "events": events,
            "sessions": t.get("sessions") or 0,
            "switches": switches,
            "longest_block_seconds": round(t.get("longest_block") or 0.0, 1),
            "avg_event_seconds": round(avg_event, 1),
            "switches_per_hour": round(switches / active_hours, 1) if active_hours else 0.0,
            "focus_score": focus_score,
            "by_activity": by_activity,
            "by_app": by_app,
            "by_project": by_project,
        }

    def resolve_entities(self, fuzzy_threshold=0.9):
        """Step 12: propose POSSIBLY_SAME_AS links over all entities (idempotent).

        Reads every :Entity, finds alias candidates, and MERGEs soft
        POSSIBLY_SAME_AS edges. Never merges nodes. Returns the candidate count.
        """
        from memory.resolution.entity_resolver import find_same_as_candidates
        ents = [dict(r) for r in self.run(
            "MATCH (n:Entity) RETURN n.entity_id AS entity_id, n.name AS name, n.type AS type")]
        pairs = find_same_as_candidates(ents, fuzzy_threshold=fuzzy_threshold)

        def _tx(tx):
            for p in pairs:
                tx.run(_POSSIBLY_SAME_AS_CYPHER, a=p["a"], b=p["b"],
                       score=p["score"], method=p["method"])
        with self._driver.session(database=self.database) as session:
            session.execute_write(_tx)
        return len(pairs)

    def possibly_same_as(self, limit=100):
        return [dict(r) for r in self.run(_POSSIBLY_SAME_AS_LIST_CYPHER, limit=limit)]

    # -- Step 14: quarantine / consolidation / shortcut edges --------------
    def consolidate(self, min_events=2):
        """Promote entities with corroborating evidence; quarantine the rest.

        An entity is promoted to 'active' only when it's mentioned across
        >= min_events DISTINCT events — so nothing is promoted from a single
        (weak) observation. Idempotent; recomputes on every call.
        """
        rows = self.run(_CONSOLIDATE_CYPHER, min_events=min_events)
        return dict(rows[0]) if rows else {"active": 0, "quarantined": 0}

    def rebuild_shortcuts(self):
        """Drop and rebuild derived shortcut edges from the base MENTIONS data.

        Shortcut edges are a query cache, always reconstructable:
        - (:Session)-[:INVOLVES {mentions}]->(:Entity)
        - (:Entity)-[:CO_OCCURS {count}]->(:Entity)   (same-frame, a<b)
        """
        self.run("MATCH ()-[r:INVOLVES]->() DELETE r")
        self.run("MATCH ()-[r:CO_OCCURS]->() DELETE r")
        self.run(_REBUILD_INVOLVES_CYPHER)
        self.run(_REBUILD_CO_OCCURS_CYPHER)
        rows = self.run(
            "MATCH ()-[i:INVOLVES]->() WITH count(i) AS involves "
            "MATCH ()-[c:CO_OCCURS]->() RETURN involves, count(c) AS co_occurs")
        return dict(rows[0]) if rows else {"involves": 0, "co_occurs": 0}

    def status_counts(self):
        rows = self.run(
            "MATCH (n:Entity) RETURN n.memory_status AS status, count(n) AS n "
            "ORDER BY status")
        return {r["status"]: r["n"] for r in rows}

    def wipe(self):
        """Delete ALL graph data (Day/Session/Event/Entity/Claim + rels)."""
        self.run("MATCH (n) DETACH DELETE n")
        return self.counts()

    def counts(self):
        rows = self.run(
            "MATCH (d:Day) WITH count(d) AS days "
            "MATCH (s:Session) WITH days, count(s) AS sessions "
            "MATCH (e:Event) RETURN days, sessions, count(e) AS events"
        )
        return dict(rows[0]) if rows else {"days": 0, "sessions": 0, "events": 0}


def _as_dict(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def _with_day(params, date_str):
    start = datetime.datetime.fromisoformat(date_str).timestamp()
    return {**params, "start": start, "end": start + 86400}


def _session_params(s):
    return {
        "sid": s["session_id"],
        "activity_type": s.get("activity_type"),
        "application": s.get("application"),
        "project_id": s.get("project_id"),
        "start": s.get("start"),
        "end": s.get("end"),
        "state": s.get("state"),
        "active_seconds": s.get("active_seconds"),
        "resume_count": s.get("resume_count", 0),
    }


def _event_params(e):
    return {
        "eid": e["event_id"],
        "sid": e["session_id"],
        "activity_type": e.get("activity_type"),
        "application": e.get("application"),
        "project_id": e.get("project_id"),
        "span_start": e.get("span_start"),
        "span_end": e.get("span_end"),
        "span_seconds": e.get("span_seconds"),
        "boundary_label": e.get("boundary_label"),
        "summary": e.get("summary"),
    }


def _room_params(room, json_module):
    matcher = room.matcher.model_dump() if hasattr(room.matcher, "model_dump") else dict(room.matcher)
    return {
        "room_id": room.room_id,
        "name": room.name,
        "kind": room.kind,
        "auto": room.auto,
        "matcher_json": json_module.dumps(matcher),
        "description": room.description,
        "color": room.color,
        "icon": room.icon,
        "archived": room.archived,
        "pinned": room.pinned,
        "position": room.position,
    }


_SESSION_CYPHER = """
MERGE (s:Session {session_id: $sid})
SET s.activity_type = $activity_type,
    s.application = $application,
    s.project_id = $project_id,
    s.start = $start,
    s.end = $end,
    s.state = $state,
    s.active_seconds = $active_seconds,
    s.resume_count = $resume_count
WITH s
MERGE (d:Day {date: $date})
  ON CREATE SET d.created_at = timestamp()
MERGE (d)-[:HAS_SESSION]->(s)
"""

_EVENT_CYPHER = """
MERGE (e:Event {event_id: $eid})
SET e.activity_type = $activity_type,
    e.application = $application,
    e.project_id = $project_id,
    e.span_start = $span_start,
    e.span_end = $span_end,
    e.span_seconds = $span_seconds,
    e.boundary_label = $boundary_label,
    e.summary = $summary
WITH e
MATCH (s:Session {session_id: $sid})
MERGE (s)-[:HAS_EVENT]->(e)
"""

_MENTION_CYPHER = """
MATCH (e:Event {event_id: $eid})
MERGE (n:Entity {entity_id: $entity_id})
  ON CREATE SET n.name = $name, n.type = $type, n.memory_status = 'quarantined'
  ON MATCH SET n.name = coalesce(n.name, $name), n.type = coalesce(n.type, $type)
MERGE (e)-[m:MENTIONS]->(n)
SET m.confidence = $confidence, m.role = $role, m.co_presence = $co_presence
"""

_CLAIM_CYPHER = """
MATCH (e:Event {event_id: $eid})
MERGE (c:Claim {claim_id: $claim_id})
  ON CREATE SET c.text = $text, c.confidence = $confidence
MERGE (e)-[:SUPPORTS]->(c)
"""

_CO_OCCURRENCE_CYPHER = """
MATCH (n:Entity {entity_id: $nid})<-[:MENTIONS {co_presence: 'same_frame'}]-(e:Event)
      -[:MENTIONS {co_presence: 'same_frame'}]->(other:Entity)
WHERE other.entity_id <> n.entity_id
RETURN other.name AS name, other.type AS type,
       count(DISTINCT e) AS shared_frames
ORDER BY shared_frames DESC, name
LIMIT $limit
"""

_EVENTS_FOR_ENTITY_CYPHER = """
MATCH (n:Entity {entity_id: $nid})<-[m:MENTIONS]-(e:Event)<-[:HAS_EVENT]-(s:Session)
RETURN e.event_id AS event, e.span_start AS span_start, e.span_end AS span_end,
       e.summary AS summary, s.session_id AS session,
       m.role AS role, m.co_presence AS co_presence
ORDER BY e.span_start
LIMIT $limit
"""

_ENTITIES_FOR_EVENTS_CYPHER = """
MATCH (e:Event)-[m:MENTIONS]->(n:Entity)
WHERE e.event_id IN $ids
RETURN e.event_id AS event, n.name AS name, n.type AS type,
       m.role AS role, m.co_presence AS co_presence
"""

_CONSOLIDATE_CYPHER = """
MATCH (n:Entity)
OPTIONAL MATCH (n)<-[m:MENTIONS]-(e:Event)
WITH n, count(DISTINCT e) AS events, coalesce(max(m.confidence), 0.0) AS max_conf
SET n.evidence_events = events,
    n.max_confidence = max_conf,
    n.memory_status = CASE WHEN events >= $min_events THEN 'active' ELSE 'quarantined' END
WITH collect(n.memory_status) AS statuses
RETURN size([s IN statuses WHERE s = 'active']) AS active,
       size([s IN statuses WHERE s = 'quarantined']) AS quarantined
"""

_REBUILD_INVOLVES_CYPHER = """
MATCH (s:Session)-[:HAS_EVENT]->(:Event)-[m:MENTIONS]->(n:Entity)
WITH s, n, count(m) AS mentions
MERGE (s)-[i:INVOLVES]->(n)
SET i.mentions = mentions
"""

_REBUILD_CO_OCCURS_CYPHER = """
MATCH (a:Entity)<-[:MENTIONS {co_presence: 'same_frame'}]-(e:Event)
      -[:MENTIONS {co_presence: 'same_frame'}]->(b:Entity)
WHERE a.entity_id < b.entity_id
WITH a, b, count(DISTINCT e) AS c
MERGE (a)-[r:CO_OCCURS]->(b)
SET r.count = c
"""

_MERGE_ROOM_CYPHER = """
MERGE (r:Room {room_id: $room_id})
  ON CREATE SET r.name = $name, r.kind = $kind, r.auto = $auto,
                r.matcher_json = $matcher_json, r.description = $description,
                r.color = $color, r.icon = $icon, r.archived = $archived,
                r.pinned = $pinned, r.position = $position,
                r.created_at = timestamp(), r.updated_at = timestamp()
  ON MATCH SET r.name = coalesce(r.name, $name), r.kind = coalesce(r.kind, $kind),
               r.description = coalesce(r.description, $description),
               r.color = coalesce(r.color, $color), r.icon = coalesce(r.icon, $icon),
               r.archived = coalesce(r.archived, false),
               r.pinned = coalesce(r.pinned, $pinned),
               r.position = coalesce(r.position, $position)
"""

_LINK_ROOM_EVENT_CYPHER = """
MATCH (r:Room {room_id: $room_id}), (e:Event {event_id: $event_id})
MERGE (r)-[rel:CONTAINS]->(e)
SET rel.assignment = $assignment, rel.manual = $manual, rel.updated_at = timestamp()
RETURN r.room_id AS room_id
"""

_LIST_ROOMS_CYPHER = """
MATCH (r:Room)
WHERE $include_archived OR NOT coalesce(r.archived, false)
OPTIONAL MATCH (r)-[:CONTAINS]->(e:Event)
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.auto AS auto, r.description AS description, r.color AS color,
       r.icon AS icon, coalesce(r.archived, false) AS archived,
       coalesce(r.pinned, false) AS pinned, coalesce(r.position, 0) AS position,
       count(e) AS events, max(e.span_end) AS last_active
ORDER BY pinned DESC, position, events DESC, r.name
"""

_ROOM_FEED_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[rel:CONTAINS]->(e:Event)
RETURN e.event_id AS event_id, e.span_start AS span_start, e.span_end AS span_end,
       e.summary AS summary, e.activity_type AS activity_type,
       e.application AS application, rel.assignment AS assignment, rel.manual AS manual
ORDER BY e.span_start DESC
LIMIT $limit
"""

_ADD_NOTE_CYPHER = """
MERGE (r:Room {room_id: $room_id})
  ON CREATE SET r.name = $room_id, r.kind = 'topic', r.auto = false, r.created_at = timestamp()
CREATE (n:RoomNote {note_id: $note_id, text: $text, ts: $ts})
MERGE (r)-[:HAS_NOTE]->(n)
"""

_ADD_MESSAGE_CYPHER = """
MERGE (r:Room {room_id: $room_id})
  ON CREATE SET r.name = $room_id, r.kind = 'topic', r.auto = false, r.created_at = timestamp()
CREATE (m:RoomMessage {message_id: $message_id, role: $role, text: $text, ts: $ts})
MERGE (r)-[:HAS_MESSAGE]->(m)
"""

_ROOM_NOTES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:HAS_NOTE]->(n:RoomNote)
RETURN n.note_id AS note_id, n.text AS text, n.ts AS ts
ORDER BY n.ts DESC
LIMIT $limit
"""

_ROOM_MESSAGES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:HAS_MESSAGE]->(m:RoomMessage)
RETURN m.message_id AS message_id, m.role AS role, m.text AS text, m.ts AS ts
ORDER BY m.ts DESC
LIMIT $limit
"""

_ROOM_ENTITIES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(:Event)-[:MENTIONS]->(n:Entity)
RETURN n.name AS name, count(*) AS c
ORDER BY c DESC, name
LIMIT $limit
"""

_ROOM_FEED_DATED_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[rel:CONTAINS]->(e:Event)
WHERE e.span_start >= $start AND e.span_start < $end
RETURN e.event_id AS event_id, e.span_start AS span_start, e.span_end AS span_end,
       e.summary AS summary, e.activity_type AS activity_type,
       e.application AS application, rel.assignment AS assignment, rel.manual AS manual
ORDER BY e.span_start DESC
LIMIT $limit
"""

_CREATE_ROOM_CYPHER = """
CREATE (r:Room {
  room_id: $room_id, name: $name, kind: $kind, auto: $auto,
  matcher_json: $matcher_json, description: $description, color: $color,
  icon: $icon, archived: $archived, pinned: $pinned, position: $position,
  created_at: timestamp(), updated_at: timestamp()
})
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.description AS description, r.color AS color, r.icon AS icon,
       r.archived AS archived, r.pinned AS pinned, r.position AS position
"""

_UPDATE_ROOM_CYPHER = """
MATCH (r:Room {room_id: $room_id})
SET r.name = $name, r.description = $description, r.color = $color,
    r.icon = $icon, r.archived = $archived, r.pinned = $pinned,
    r.position = $position, r.matcher_json = $matcher_json,
    r.updated_at = timestamp()
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.description AS description, r.color AS color, r.icon AS icon,
       r.archived AS archived, r.pinned AS pinned, r.position AS position
"""

_DELETE_ROOM_CYPHER = """
OPTIONAL MATCH (r:Room {room_id: $room_id})
OPTIONAL MATCH (r)-[:HAS_NOTE|HAS_MESSAGE]->(owned)
WITH r, collect(owned) AS owned, r IS NOT NULL AS existed
FOREACH (node IN owned | DETACH DELETE node)
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE r)
RETURN existed AS deleted
"""

_HAS_MANUAL_PRIMARY_CYPHER = """
MATCH (:Room)-[rel:CONTAINS]->(e:Event {event_id: $event_id})
WHERE rel.assignment = 'primary' AND coalesce(rel.manual, false)
RETURN count(rel) AS n
"""

_REMOVE_AUTO_PRIMARY_CYPHER = """
MATCH (:Room)-[rel:CONTAINS]->(e:Event {event_id: $event_id})
WHERE coalesce(rel.assignment, 'primary') = 'primary'
  AND NOT coalesce(rel.manual, false)
DELETE rel
"""

_REMOVE_ALL_PRIMARY_CYPHER = """
MATCH (:Room)-[rel:CONTAINS]->(e:Event {event_id: $event_id})
WHERE coalesce(rel.assignment, 'primary') = 'primary'
DELETE rel
"""

_UPDATE_NOTE_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:HAS_NOTE]->(n:RoomNote {note_id: $note_id})
SET n.text = $text, n.updated_at = timestamp()
RETURN n.note_id AS note_id, n.text AS text, n.ts AS ts
"""

_DELETE_NOTE_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:HAS_NOTE]->(n:RoomNote {note_id: $note_id})
DETACH DELETE n
RETURN true AS deleted
"""

_SESSIONS_WITH_EVENTS_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(s:Session)
OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:Event)
WITH s, e ORDER BY e.span_start
RETURN s.session_id AS session_id, s.activity_type AS activity,
       s.application AS application, s.project_id AS project_id,
       s.active_seconds AS active_seconds, s.resume_count AS resume_count,
       s.state AS state, s.start AS start,
       collect(CASE WHEN e IS NULL THEN NULL ELSE {
         event_id: e.event_id, span_start: e.span_start, span_end: e.span_end,
         span_seconds: e.span_seconds, summary: e.summary,
         activity: e.activity_type, boundary: e.boundary_label
       } END) AS events
ORDER BY s.start
"""

_DAY_ENTITIES_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(:Event)-[m:MENTIONS]->(n:Entity)
RETURN n.name AS name, n.type AS type, count(m) AS mentions
ORDER BY mentions DESC, name
LIMIT $limit
"""

_DAILY_TOTALS_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(s:Session)-[:HAS_EVENT]->(e:Event)
RETURN sum(e.span_seconds) AS active_seconds, count(e) AS events,
       count(DISTINCT s) AS sessions, max(e.span_seconds) AS longest_block,
       sum(CASE WHEN e.boundary_label <> 'append' THEN 1 ELSE 0 END) AS switches
"""

_DAILY_BY_ACTIVITY_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
RETURN e.activity_type AS activity, round(sum(e.span_seconds) / 60.0, 1) AS minutes
ORDER BY minutes DESC
"""

_DAILY_BY_APP_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
RETURN e.application AS app, round(sum(e.span_seconds) / 60.0, 1) AS minutes
ORDER BY minutes DESC
LIMIT 10
"""

_DAILY_BY_PROJECT_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(s:Session)-[:HAS_EVENT]->(e:Event)
WHERE s.project_id IS NOT NULL
RETURN s.project_id AS project, round(sum(e.span_seconds) / 60.0, 1) AS minutes
ORDER BY minutes DESC
LIMIT 10
"""

_DAY_CLAIMS_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)-[:SUPPORTS]->(c:Claim)
RETURN c.text AS text, c.confidence AS confidence
ORDER BY c.confidence DESC
LIMIT $limit
"""

_POSSIBLY_SAME_AS_CYPHER = """
MATCH (a:Entity {entity_id: $a}), (b:Entity {entity_id: $b})
MERGE (a)-[r:POSSIBLY_SAME_AS]->(b)
SET r.score = $score, r.method = $method
"""

_POSSIBLY_SAME_AS_LIST_CYPHER = """
MATCH (a:Entity)-[r:POSSIBLY_SAME_AS]->(b:Entity)
RETURN a.name AS a, a.type AS a_type, b.name AS b, b.type AS b_type,
       r.score AS score, r.method AS method
ORDER BY r.score DESC, a
LIMIT $limit
"""

_ENTITIES_FOR_SESSION_CYPHER = """
MATCH (s:Session {session_id: $sid})-[:HAS_EVENT]->(e:Event)-[m:MENTIONS]->(n:Entity)
RETURN n.name AS name, n.type AS type,
       m.role AS role, m.co_presence AS co_presence, m.confidence AS confidence,
       e.event_id AS event
ORDER BY m.confidence DESC
"""

_EVENTS_TODAY_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(s:Session)-[:HAS_EVENT]->(e:Event)
RETURN s.session_id AS session,
       s.activity_type AS activity,
       s.application AS application,
       s.project_id AS project_id,
       e.event_id AS event,
       e.span_start AS span_start,
       e.span_end AS span_end,
       e.span_seconds AS span_seconds
ORDER BY e.span_start
"""
