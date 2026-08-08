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
import time

from neo4j import GraphDatabase

from memory.retrieval.terms import tokenize
from memory.summary.naming import fold_apps, fold_projects
from memory.summary.reports import PRODUCTIVITY_DOMAIN

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
    "CREATE CONSTRAINT memory_correction_id IF NOT EXISTS FOR (c:MemoryCorrection) REQUIRE c.correction_id IS UNIQUE",
    "CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE",
    "CREATE CONSTRAINT assistant_message_id IF NOT EXISTS FOR (m:AssistantMessage) REQUIRE m.message_id IS UNIQUE",
    "CREATE CONSTRAINT focus_session_id IF NOT EXISTS FOR (f:FocusSession) REQUIRE f.focus_id IS UNIQUE",
    "CREATE CONSTRAINT nudge_id IF NOT EXISTS FOR (n:Nudge) REQUIRE n.nudge_id IS UNIQUE",
    # Long-term tier (memory/consolidation.py): compressed periods, plus the
    # projects and goals that outlive any single day.
    "CREATE CONSTRAINT rollup_id IF NOT EXISTS FOR (r:Rollup) REQUIRE r.rollup_id IS UNIQUE",
    "CREATE CONSTRAINT project_key IF NOT EXISTS FOR (p:Project) REQUIRE p.project_key IS UNIQUE",
    "CREATE CONSTRAINT goal_key IF NOT EXISTS FOR (g:Goal) REQUIRE g.goal_key IS UNIQUE",
    # Written reports, one per (period, end date), so today's writer can read
    # what the last fortnight's reports said and scored.
    "CREATE CONSTRAINT written_report_id IF NOT EXISTS FOR (r:WrittenReport) REQUIRE r.report_id IS UNIQUE",
]

# Secondary indexes for span-aware and lookup queries.
INDEXES = [
    "CREATE INDEX event_span_start IF NOT EXISTS FOR (e:Event) ON (e.span_start)",
    "CREATE INDEX session_activity IF NOT EXISTS FOR (s:Session) ON (s.activity_type)",
    "CREATE INDEX session_project IF NOT EXISTS FOR (s:Session) ON (s.project_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX nudge_ts IF NOT EXISTS FOR (n:Nudge) ON (n.ts)",
    # Rollups are read by period and by recency, which is what the coarse
    # retrieval tier scans instead of the event table.
    "CREATE INDEX rollup_kind IF NOT EXISTS FOR (r:Rollup) ON (r.kind)",
    "CREATE INDEX rollup_end IF NOT EXISTS FOR (r:Rollup) ON (r.end_date)",
    "CREATE INDEX project_status IF NOT EXISTS FOR (p:Project) ON (p.status)",
    "CREATE INDEX written_report_end IF NOT EXISTS FOR (r:WrittenReport) ON (r.end_date)",
]


class Neo4jStore:
    def __init__(self, uri=None, username=None, password=None, database=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        # alias entity_id -> canonical entity_id; read on every knowledge write,
        # invalidated whenever an alias is recorded or removed.
        self._alias_cache = None

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
        aliases = self.entity_aliases()

        def _tx(tx):
            n_ent = n_claim = 0
            for it in items:
                eid = it["event_id"]
                for en in it.get("entities", []):
                    canonical = aliases.get(en.get("entity_id"))
                    if canonical:
                        # Apply the user's past merge instead of re-creating the
                        # entity they already told us was a duplicate.
                        en = {**en, "entity_id": canonical}
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

    def co_occurring_entities(self, entity_name, limit=25, domain=None):
        """Entities seen in the SAME FRAME as `entity_name` (plan §13).

        Filters MENTIONS on co_presence='same_frame', so this answers true
        co-occurrence ("what appeared together"), not merely same-session.
        """
        return [dict(r) for r in self.run(
            _CO_OCCURRENCE_CYPHER, nid=self._norm(entity_name), limit=limit,
            domain=domain)]

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

    # -- Phase 2: Memory Explorer -----------------------------------------
    def memory_search(self, query, limit=40, kinds=None, start=None, end=None,
                      room_id=None, domain=None):
        """Term-based keyword search across graph-backed memory types.

        Results share a stable shape so the API can merge them with semantic
        vector hits without exposing database-specific records to the client.

        Returns [] for a query with no content terms ("what did I do today");
        that is a scope question, and EvidenceRetriever answers it chronologically.
        """
        needle = self._norm(query)
        terms = tokenize(query)
        if not needle or not terms:
            return []
        allowed = set(kinds or (
            "event", "note", "message", "entity", "claim", "room", "insight"))
        params = {
            "needle": needle, "terms": terms,
            "limit": max(1, min(int(limit), 200)),
            "start": start, "end": end, "room_id": room_id, "domain": domain,
        }
        results = []
        searches = {
            "event": _SEARCH_EVENTS_CYPHER,
            "note": _SEARCH_NOTES_CYPHER,
            "message": _SEARCH_MESSAGES_CYPHER,
            "entity": _SEARCH_ENTITIES_CYPHER,
            "claim": _SEARCH_CLAIMS_CYPHER,
            "room": _SEARCH_ROOMS_CYPHER,
            "insight": _SEARCH_NUDGES_CYPHER,
        }
        for kind, cypher in searches.items():
            if kind not in allowed:
                continue
            results.extend(dict(row) for row in self.run(cypher, **params))
        results.sort(
            key=lambda item: (item.get("score") or 0, item.get("ts") or 0),
            reverse=True,
        )
        return results[:params["limit"]]

    def recent_events(self, start=None, end=None, room_id=None, domain=None,
                      limit=20):
        """Newest events inside a scope, ignoring keywords (scope-only questions)."""
        return [dict(row) for row in self.run(
            _RECENT_EVENTS_CYPHER, start=start, end=end, room_id=room_id,
            domain=domain,
            limit=max(1, min(int(limit), 200)))]

    def recent_nudges(self, start=None, end=None, limit=10):
        """Newest durable proactive insights inside an optional time window."""
        return [dict(row) for row in self.run(
            _RECENT_NUDGES_CYPHER, start=start, end=end,
            limit=max(1, min(int(limit), 50)))]

    def events_in_room(self, event_ids, room_id):
        """Subset of `event_ids` that the room contains — post-filter for vector hits.

        Room membership lives in the graph, so semantic hits are filtered here
        rather than by a Qdrant payload field (which older points would lack).
        """
        ids = [i for i in (event_ids or []) if i]
        if not ids or not room_id:
            return set()
        return {row["event_id"] for row in self.run(
            _EVENTS_IN_ROOM_CYPHER, ids=ids, room_id=room_id)}

    def entity_detail(self, entity_id, domain=None):
        params = {"entity_id": self._norm(entity_id), "domain": domain}
        rows = self.run(_ENTITY_DETAIL_CYPHER, **params)
        if not rows:
            return None
        entity = dict(rows[0])
        entity["events"] = [dict(r) for r in self.run(
            _ENTITY_DETAIL_EVENTS_CYPHER, **params, limit=100)]
        entity["claims"] = [dict(r) for r in self.run(
            _ENTITY_DETAIL_CLAIMS_CYPHER, **params, limit=50)]
        entity["rooms"] = [dict(r) for r in self.run(
            _ENTITY_DETAIL_ROOMS_CYPHER, **params, limit=30)]
        entity["co_occurring"] = self.co_occurring_entities(
            entity_id, limit=30, domain=domain)
        return entity

    def event_detail(self, event_id):
        rows = self.run(_EVENT_DETAIL_CYPHER, event_id=event_id)
        if not rows:
            return None
        out = dict(rows[0])
        out["entities"] = [dict(r) for r in self.run(
            _EVENT_DETAIL_ENTITIES_CYPHER, event_id=event_id)]
        out["claims"] = [dict(r) for r in self.run(
            _EVENT_DETAIL_CLAIMS_CYPHER, event_id=event_id)]
        out["rooms"] = [dict(r) for r in self.run(
            _EVENT_DETAIL_ROOMS_CYPHER, event_id=event_id)]
        return out

    def update_event_summary(self, event_id, summary):
        import uuid as _uuid
        rows = self.run(
            _UPDATE_EVENT_SUMMARY_CYPHER, event_id=event_id, summary=summary,
            correction_id=_uuid.uuid4().hex)
        return dict(rows[0]) if rows else None

    def update_event_metadata(self, event_id, priority=None, flagged=None,
                              flag_reason=None):
        """Apply user triage without changing or deleting the event.

        Priority is a durable override of the extractor-derived importance.
        Flagging is independent so an important event can still be set aside
        for later review.
        """
        allowed = {"high", "normal", "low"}
        if priority is not None and priority not in allowed:
            raise ValueError("priority must be high, normal, or low")
        set_flag_reason = flag_reason is not None
        rows = self.run(
            _UPDATE_EVENT_METADATA_CYPHER,
            event_id=event_id,
            set_priority=priority is not None,
            priority=priority,
            set_flagged=flagged is not None,
            flagged=bool(flagged) if flagged is not None else False,
            set_flag_reason=set_flag_reason,
            flag_reason=(str(flag_reason).strip() if set_flag_reason else None),
        )
        return dict(rows[0]) if rows else None

    # -- Entity aliases (corrections that persist) -------------------------
    def entity_aliases(self, limit=1000):
        """alias entity_id -> canonical entity_id, from user merges and renames.

        Cached because this is read on every knowledge write; the cache is
        cleared whenever an alias is recorded or removed.
        """
        if self._alias_cache is None:
            try:
                self._alias_cache = {
                    row["alias_id"]: row["canonical_id"]
                    for row in self.run(_LIST_ALIASES_CYPHER, limit=limit)}
            except Exception as exc:
                logger.warning("loading entity aliases failed: %s", exc)
                return {}
        return self._alias_cache

    def list_entity_aliases(self, limit=200):
        return [dict(row) for row in self.run(_LIST_ALIASES_CYPHER, limit=limit)]

    def record_entity_alias(self, alias_id, canonical_id, name=None, source="manual"):
        alias_id, canonical_id = self._norm(alias_id), self._norm(canonical_id)
        if not alias_id or not canonical_id or alias_id == canonical_id:
            return None
        self.run(_RECORD_ALIAS_CYPHER, alias_id=alias_id,
                 canonical_id=canonical_id, name=name or alias_id, source=source)
        self._alias_cache = None
        return {"alias_id": alias_id, "canonical_id": canonical_id}

    def delete_entity_alias(self, alias_id):
        rows = self.run(_DELETE_ALIAS_CYPHER, alias_id=self._norm(alias_id))
        self._alias_cache = None
        return bool(rows)

    def canonical_name_hints(self, limit=20):
        """[(wrong_name, canonical_name)] pairs to steer the extractor's naming."""
        return [(row["wrong_name"], row["canonical_name"])
                for row in self.run(_CANONICAL_NAMES_CYPHER, limit=limit)]

    def update_entity(self, entity_id, name=None, entity_type=None):
        import uuid as _uuid
        entity_id = self._norm(entity_id)
        rows = self.run(
            _UPDATE_ENTITY_CYPHER, entity_id=entity_id,
            name=name, entity_type=entity_type,
            correction_id=_uuid.uuid4().hex)
        if not rows:
            return None
        # A rename leaves entity_id (the normalized ORIGINAL name) untouched, so
        # the next capture emitting the corrected name would normalize to a new
        # id and silently fork the entity. Alias the new spelling back to it.
        if name:
            self.record_entity_alias(self._norm(name), entity_id,
                                     name=name, source="rename")
        return dict(rows[0])

    def merge_entities(self, source_id, target_id):
        """Merge one entity into another while preserving mention evidence."""
        source_id, target_id = self._norm(source_id), self._norm(target_id)
        if source_id == target_id:
            raise ValueError("source and target must differ")

        def _tx(tx):
            nodes = list(tx.run(
                "MATCH (source:Entity {entity_id: $source_id}), "
                "(target:Entity {entity_id: $target_id}) "
                "RETURN source.entity_id AS source, target.entity_id AS target, "
                "source.name AS source_name",
                source_id=source_id, target_id=target_id))
            if not nodes:
                return None
            mentions = list(tx.run(_ENTITY_MENTIONS_FOR_MERGE_CYPHER,
                                   source_id=source_id))
            for mention in mentions:
                tx.run(_MERGE_ENTITY_MENTION_CYPHER, target_id=target_id,
                       event_id=mention["event_id"],
                       confidence=mention["confidence"], role=mention["role"],
                       co_presence=mention["co_presence"])
            source_name = nodes[0].get("source_name") or source_id
            # Remember the merge BEFORE deleting, so future captures that emit
            # the old name are folded into the target instead of resurrecting it.
            tx.run(_RECORD_ALIAS_CYPHER, alias_id=source_id,
                   canonical_id=target_id, name=source_name, source="merge")
            tx.run(_REPOINT_ALIASES_CYPHER, old_canonical_id=source_id,
                   canonical_id=target_id)
            tx.run("MATCH (source:Entity {entity_id: $source_id}) DETACH DELETE source",
                   source_id=source_id)
            return {"source_id": source_id, "target_id": target_id,
                    "moved_mentions": len(mentions), "alias_recorded": True}

        with self._driver.session(database=self.database) as session:
            result = session.execute_write(_tx)
        self._alias_cache = None  # a merge changed the mapping
        return result

    def split_entity(self, source_id, new_entity_id, name, entity_type, event_ids):
        """Move selected event mentions from an entity into a new entity."""
        source_id, new_entity_id = self._norm(source_id), self._norm(new_entity_id)
        if not event_ids:
            raise ValueError("at least one event_id is required")
        if self.entity_detail(new_entity_id) is not None:
            raise ValueError("new entity_id already exists")

        def _tx(tx):
            source = list(tx.run(
                "MATCH (n:Entity {entity_id: $source_id}) RETURN n.entity_id AS id",
                source_id=source_id))
            if not source:
                return None
            tx.run(_CREATE_SPLIT_ENTITY_CYPHER, entity_id=new_entity_id,
                   name=name, entity_type=entity_type)
            # A split is the inverse of a merge: drop any alias that would fold
            # this name straight back into the entity it was just separated from.
            tx.run(_DELETE_ALIAS_CYPHER, alias_id=new_entity_id)
            moved = 0
            for event_id in event_ids:
                rows = list(tx.run(
                    _MOVE_ENTITY_MENTION_CYPHER, source_id=source_id,
                    target_id=new_entity_id, event_id=event_id))
                moved += 1 if rows else 0
            return {"source_id": source_id, "entity_id": new_entity_id,
                    "moved_mentions": moved}

        with self._driver.session(database=self.database) as session:
            result = session.execute_write(_tx)
        self._alias_cache = None  # a split may have removed an alias
        return result

    def update_claim(self, claim_id, text):
        import uuid as _uuid
        rows = self.run(
            _UPDATE_CLAIM_CYPHER, claim_id=claim_id, text=text,
            correction_id=_uuid.uuid4().hex)
        return dict(rows[0]) if rows else None

    def delete_claim(self, claim_id):
        rows = self.run(_DELETE_CLAIM_CYPHER, claim_id=claim_id)
        return bool(rows and rows[0]["deleted"])

    def forget_event(self, event_id):
        rows = self.run(_FORGET_EVENT_CYPHER, event_id=event_id)
        if rows and rows[0]["deleted"]:
            self.run("MATCH (n:Entity) WHERE NOT (n)<-[:MENTIONS]-(:Event) DETACH DELETE n")
        return dict(rows[0]) if rows else None

    def forget_session(self, session_id):
        rows = self.run(_SESSION_EVENT_IDS_CYPHER, session_id=session_id)
        event_ids = [row["event_id"] for row in rows]
        for event_id in event_ids:
            self.forget_event(event_id)
        removed = self.run(_DELETE_EMPTY_SESSION_CYPHER, session_id=session_id)
        return {"session_id": session_id, "event_ids": event_ids,
                "deleted": bool(removed and removed[0]["deleted"])}

    def forget_day(self, date_str, domain=None):
        rows = self.run(
            _DAY_EVENT_IDS_CYPHER, date=date_str, domain=domain)
        event_ids = [row["event_id"] for row in rows]
        for event_id in event_ids:
            self.forget_event(event_id)
        self.run(_DELETE_EMPTY_DAY_CYPHER, date=date_str)
        return {"date": date_str, "event_ids": event_ids, "deleted": True}

    def forget_entity(self, entity_id):
        rows = self.run(_FORGET_ENTITY_CYPHER, entity_id=self._norm(entity_id))
        return bool(rows and rows[0]["deleted"])

    # -- Phase 3: grounded assistant + focus -------------------------------
    def create_conversation(self, title="New conversation", scope="all",
                            room_id=None, from_ts=None, to_ts=None):
        import uuid as _uuid
        conversation_id = _uuid.uuid4().hex
        rows = self.run(
            _CREATE_CONVERSATION_CYPHER, conversation_id=conversation_id,
            title=title, scope=scope, room_id=room_id,
            from_ts=from_ts, to_ts=to_ts)
        return dict(rows[0]) if rows else None

    def list_conversations(self, limit=100):
        return [dict(row) for row in self.run(
            _LIST_CONVERSATIONS_CYPHER, limit=max(1, min(int(limit), 200)))]

    def get_conversation(self, conversation_id, message_limit=200):
        import json as _json
        rows = self.run(_GET_CONVERSATION_CYPHER,
                        conversation_id=conversation_id)
        if not rows:
            return None
        result = dict(rows[0])
        result["messages"] = [dict(row) for row in self.run(
            _CONVERSATION_MESSAGES_CYPHER,
            conversation_id=conversation_id,
            limit=max(1, min(int(message_limit), 500)))]
        for message in result["messages"]:
            try:
                message["citations"] = _json.loads(
                    message.pop("citations_json") or "[]")
            except (TypeError, ValueError):
                message["citations"] = []
        return result

    def update_conversation(self, conversation_id, changes):
        current = self.get_conversation(conversation_id, message_limit=1)
        if current is None:
            return None
        values = {
            "conversation_id": conversation_id,
            "title": changes.get("title", current.get("title")),
            "scope": changes.get("scope", current.get("scope") or "all"),
            "room_id": changes.get("room_id", current.get("room_id")),
            "from_ts": changes.get("from_ts", current.get("from_ts")),
            "to_ts": changes.get("to_ts", current.get("to_ts")),
        }
        rows = self.run(_UPDATE_CONVERSATION_CYPHER, **values)
        return dict(rows[0]) if rows else None

    def add_conversation_message(self, conversation_id, role, text,
                                 citations=None, ts=None):
        import json as _json
        import time as _time
        import uuid as _uuid
        rows = self.run(
            _ADD_CONVERSATION_MESSAGE_CYPHER,
            conversation_id=conversation_id,
            message_id=_uuid.uuid4().hex, role=role, text=text,
            citations_json=_json.dumps(citations or []),
            ts=ts if ts is not None else _time.time())
        return dict(rows[0]) if rows else None

    def delete_conversation(self, conversation_id):
        rows = self.run(_DELETE_CONVERSATION_CYPHER,
                        conversation_id=conversation_id)
        return bool(rows and rows[0]["deleted"])

    def start_focus_session(self, goal, room_id=None, planned_minutes=25):
        import time as _time
        import uuid as _uuid
        focus_id = _uuid.uuid4().hex
        rows = self.run(
            _START_FOCUS_CYPHER, focus_id=focus_id, goal=goal,
            room_id=room_id, planned_minutes=planned_minutes,
            started_at=_time.time())
        return dict(rows[0]) if rows else None

    def active_focus_session(self):
        rows = self.run(_ACTIVE_FOCUS_CYPHER)
        return dict(rows[0]) if rows else None

    def list_focus_sessions(self, limit=50):
        return [dict(row) for row in self.run(
            _LIST_FOCUS_CYPHER, limit=max(1, min(int(limit), 200)))]

    def stop_focus_session(self, focus_id):
        import time as _time
        ended_at = _time.time()
        rows = self.run(_STOP_FOCUS_CYPHER, focus_id=focus_id,
                        ended_at=ended_at)
        if not rows:
            return None
        result = dict(rows[0])
        metrics = self.run(
            _FOCUS_METRICS_CYPHER, start=result["started_at"],
            end=ended_at, room_id=result.get("room_id"))
        result["metrics"] = dict(metrics[0]) if metrics else {
            "events": 0, "active_seconds": 0, "applications": []}
        self.run(_SAVE_FOCUS_SUMMARY_CYPHER, focus_id=focus_id,
                 ended_at=ended_at,
                 events=result["metrics"].get("events") or 0,
                 active_seconds=result["metrics"].get("active_seconds") or 0)
        return result

    # -- Project arc -------------------------------------------------------
    def room_weekly(self, room_id, start, end):
        """Per-week activity buckets for a room (weeks start Monday)."""
        return [dict(row) for row in self.run(
            _ROOM_WEEKLY_CYPHER, room_id=room_id, start=start, end=end)]

    def room_week_highlights(self, room_id, start, end, limit=5):
        rows = self.run(_ROOM_WEEK_HIGHLIGHTS_CYPHER, room_id=room_id,
                        start=start, end=end, limit=int(limit))
        if not rows:
            return {"claims": [], "summaries": []}
        row = dict(rows[0])
        return {"claims": [c for c in (row.get("claims") or []) if c],
                "summaries": [s for s in (row.get("summaries") or []) if s]}

    def room_week_entities(self, room_id, start, end, limit=8):
        return [dict(row) for row in self.run(
            _ROOM_WEEK_ENTITIES_CYPHER, room_id=room_id, start=start,
            end=end, limit=int(limit))]

    # -- Room hygiene ------------------------------------------------------
    def room_stats(self):
        """Per-room activity counts, used to spot stale and thin rooms."""
        return [dict(row) for row in self.run(_ROOM_STATS_CYPHER)]

    def room_overlap(self, min_shared=3, min_overlap=0.5, limit=20):
        """Room pairs that look like the same topic, by shared entities."""
        return [dict(row) for row in self.run(
            _ROOM_OVERLAP_CYPHER, min_shared=int(min_shared),
            min_overlap=float(min_overlap), limit=max(1, min(int(limit), 100)))]

    def merge_rooms(self, source_id, target_id):
        """Move a room's events/notes/chat into another, then archive the source."""
        import time as _time
        if source_id == target_id:
            raise ValueError("source and target must differ")
        rows = self.run(_MERGE_ROOMS_CYPHER, source_id=source_id,
                        target_id=target_id, now=_time.time())
        if not rows:
            return None
        result = dict(rows[0])
        result["source_room_id"] = source_id
        return result

    def promote_room(self, room_id, name=None, pinned=None):
        """Turn an auto room into a user-owned topic room (hygiene leaves it alone)."""
        import time as _time
        rows = self.run(_PROMOTE_ROOM_CYPHER, room_id=room_id, name=name,
                        pinned=pinned, now=_time.time())
        return dict(rows[0]) if rows else None

    def archive_rooms(self, room_ids):
        import time as _time
        ids = [r for r in (room_ids or []) if r]
        if not ids:
            return []
        return [row["room_id"] for row in self.run(
            _ARCHIVE_ROOMS_CYPHER, room_ids=ids, now=_time.time())]

    # -- Proactive nudges --------------------------------------------------
    def record_nudge(self, text, kind="insight", focus_id=None, evidence=None):
        """Persist a spoken nudge so the user's reaction to it can be learned from."""
        import json as _json
        import time as _time
        import uuid as _uuid
        nudge_id = _uuid.uuid4().hex
        self.run(_RECORD_NUDGE_CYPHER, nudge_id=nudge_id, text=text, kind=kind,
                 focus_id=focus_id, evidence_json=_json.dumps(evidence or []),
                 ts=_time.time())
        return nudge_id

    def set_nudge_feedback(self, nudge_id, feedback):
        import time as _time
        rows = self.run(_NUDGE_FEEDBACK_CYPHER, nudge_id=nudge_id,
                        feedback=feedback, ts=_time.time())
        return dict(rows[0]) if rows else None

    def recent_nudge_feedback(self, limit=8):
        """Nudges the user reacted to — the narrator's restraint signal."""
        return [dict(row) for row in self.run(
            _NUDGE_FEEDBACK_HISTORY_CYPHER, limit=max(1, min(int(limit), 50)))]

    def list_nudges(self, limit=50):
        return [dict(row) for row in self.run(
            _LIST_NUDGES_CYPHER, limit=max(1, min(int(limit), 200)))]

    def get_focus_session(self, focus_id):
        rows = self.run(_GET_FOCUS_CYPHER, focus_id=focus_id)
        return dict(rows[0]) if rows else None

    def focus_events(self, start, end, room_id=None, limit=200):
        """Events overlapping a focus window (the recap's raw material)."""
        return [dict(row) for row in self.run(
            _FOCUS_EVENTS_CYPHER, start=start, end=end, room_id=room_id,
            limit=max(1, min(int(limit), 500)))]

    def save_focus_recap(self, focus_id, recap, breakdown):
        rows = self.run(
            _SAVE_FOCUS_RECAP_CYPHER, focus_id=focus_id, recap=recap,
            on_task_pct=breakdown.get("on_task_pct"),
            on_task_minutes=breakdown.get("on_task_minutes"),
            off_task_minutes=breakdown.get("off_task_minutes"))
        return bool(rows)

    # -- Rooms (Phase 1) ---------------------------------------------------
    def _load_rooms(self):
        import json as _json
        from memory.models.room import Room, RoomMatcher
        rooms = []
        for r in self.run(
                "MATCH (r:Room) RETURN r.room_id AS room_id, r.name AS name, "
                "r.kind AS kind, r.auto AS auto, r.matcher_json AS matcher_json, "
                "r.description AS description, r.color AS color, r.icon AS icon, "
                "r.instructions AS instructions, r.assistant_mode AS assistant_mode, "
                "r.execution_profile AS execution_profile, "
                "r.agent_tools_json AS agent_tools_json, "
                "r.agent_workspace AS agent_workspace, "
                "r.agent_request_limit AS agent_request_limit, "
                "r.agent_tool_calls_limit AS agent_tool_calls_limit, "
                "r.archived AS archived, r.pinned AS pinned, r.position AS position, "
                "r.created_at AS created_at, r.updated_at AS updated_at"):
            matcher = RoomMatcher()
            if r.get("matcher_json"):
                try:
                    matcher = RoomMatcher(**_json.loads(r["matcher_json"]))
                except Exception:
                    pass
            try:
                agent_tools = _json.loads(r.get("agent_tools_json") or "[]")
            except (TypeError, ValueError):
                agent_tools = []
            agent_tools = list(dict.fromkeys(["graph", *agent_tools]))
            rooms.append(Room(room_id=r["room_id"], name=r.get("name") or r["room_id"],
                               kind=r.get("kind") or "topic",
                               auto=bool(r.get("auto")), matcher=matcher,
                               description=r.get("description") or "",
                               instructions=r.get("instructions") or "",
                               assistant_mode="agent",
                               execution_profile=r.get("execution_profile") or "investigate",
                               agent_tools=agent_tools,
                               agent_workspace=r.get("agent_workspace") or "",
                               agent_request_limit=int(
                                   r.get("agent_request_limit") or 0),
                               agent_tool_calls_limit=int(
                                   r.get("agent_tool_calls_limit") or 0),
                               color=r.get("color") or "#8B7CF6",
                               icon=r.get("icon") or "forum",
                               archived=bool(r.get("archived")),
                               pinned=bool(r.get("pinned")),
                               position=int(r.get("position") or 0),
                               created_at=r.get("created_at"),
                               updated_at=r.get("updated_at")))
        return rooms

    def ensure_source_room(self, source):
        """Idempotently create the Screen or Cameras room for a capture source.

        One room per source, not per camera or per app — the specific camera/app
        is a tag on each event. Safe to call every startup: ON MATCH in the merge
        preserves any name/colour/instructions the user has since edited.
        """
        import json as _json
        from memory.rooms.registry import RoomRegistry

        room = RoomRegistry().ensure_source_room(source)
        self.run(_MERGE_ROOM_CYPHER, **_room_params(room, _json))
        return self.get_room(room.room_id)

    def ensure_agent_rooms(self, agents):
        """Idempotently create/update the built-in personal-agent rooms."""
        import json as _json
        from memory.models.room import Room

        rooms = []
        for position, agent in enumerate(agents, start=10):
            room = Room(
                room_id=agent.room_id, name=agent.name, kind="agent",
                auto=True, description=agent.description,
                instructions=agent.instructions, color=agent.color,
                icon=agent.icon, pinned=True, position=position,
                assistant_mode=agent.assistant_mode,
                execution_profile=agent.execution_profile,
                agent_tools=list(agent.agent_tools),
                agent_workspace=agent.workspace,
            )
            self.run(_MERGE_ROOM_CYPHER, **_room_params(room, _json))
            rooms.append(self.get_room(room.room_id))
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
        return self.get_room(room.room_id) if rows else None

    def update_room(self, room_id, changes):
        """Update room metadata/matcher. The Daily identity and kind are protected."""
        import json as _json
        from memory.models.room import Room

        room = next((r for r in self._load_rooms() if r.room_id == room_id), None)
        if room is None:
            return None
        allowed = {
            "name", "description", "instructions", "execution_profile",
            "agent_tools", "agent_workspace", "color", "icon", "archived", "pinned",
            "agent_request_limit", "agent_tool_calls_limit",
            "position", "matcher",
        }
        for key, value in changes.items():
            if key in allowed:
                setattr(room, key, value)
        # Revalidate after applying a partial patch: `setattr` alone bypasses
        # Pydantic's Literal/list validation.
        room = Room.model_validate(room.model_dump())
        room.agent_tools = list(dict.fromkeys(["graph", *room.agent_tools]))
        if room_id == "daily":
            room.name, room.kind, room.archived = "Daily", "daily", False
        rows = self.run(_UPDATE_ROOM_CYPHER, **_room_params(room, _json))
        return self.get_room(room_id) if rows else None

    def delete_room(self, room_id):
        """Delete a room and its private notes/messages; events remain intact."""
        if room_id == "daily":
            raise ValueError("the Daily room cannot be deleted")
        rows = self.run(_DELETE_ROOM_CYPHER, room_id=room_id)
        return bool(rows and rows[0]["deleted"])

    def assign_rooms(self, events):
        """Route events to rooms and link them (idempotent). Auto-creates rooms.

        `events`: list of {event_id, source, activity_type, application,
        project_id, summary, entity_types}. `source` picks the Screen/Cameras
        room; a matching user topic room still wins. Every event is also linked
        to the Daily room.
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

    def consolidate_source_rooms(self, purge_empty=True):
        """Fold legacy per-activity/per-project/per-camera rooms into Screen and
        Cameras. One-shot migration for graphs written before source rooms.

        Events are relinked, never deleted: a legacy camera room's events become
        Cameras events, everything else becomes Screen events. Manual primaries
        the user set by hand are left exactly where they are. Legacy rooms are
        archived rather than deleted when they still hold notes or chat, so
        nothing the user wrote can vanish in a migration.
        """
        from memory.rooms.registry import CAMERA_ROOM_ID, SCREEN_ROOM_ID

        for source in ("camera", "screen"):
            self.ensure_source_room(source)

        legacy = self.run(_LEGACY_AUTO_ROOMS_CYPHER,
                          keep=[CAMERA_ROOM_ID, SCREEN_ROOM_ID])
        moved, archived, deleted = 0, 0, 0
        for row in legacy:
            room_id = row["room_id"]
            target = CAMERA_ROOM_ID if row["kind"] == "camera" else SCREEN_ROOM_ID
            result = self.run(_RELINK_ROOM_EVENTS_CYPHER,
                              room_id=room_id, target_room_id=target)
            moved += (result[0]["moved"] if result else 0)
            if row["notes"] or row["messages"]:
                self.run("MATCH (r:Room {room_id: $room_id}) SET r.archived = true",
                         room_id=room_id)
                archived += 1
            elif purge_empty:
                self.run(_DELETE_ROOM_CYPHER, room_id=room_id)
                deleted += 1
            else:
                self.run("MATCH (r:Room {room_id: $room_id}) SET r.archived = true",
                         room_id=room_id)
                archived += 1

        # Events that only ever reached the Daily room (logged before rooms
        # existed, or when a routing write failed) would stay invisible in both
        # channels, so adopt them too.
        adopted = self.run(_ADOPT_UNROUTED_EVENTS_CYPHER,
                           camera_room=CAMERA_ROOM_ID, screen_room=SCREEN_ROOM_ID)
        # The room an event used to sit in is a weaker signal than the event
        # itself: a camera event that historically landed in 'activity:other'
        # would arrive in Screen. Fix any such disagreement, but only where the
        # event states its domain outright.
        normalized = self.run(_NORMALIZE_SOURCE_LINKS_CYPHER,
                              camera_room=CAMERA_ROOM_ID, screen_room=SCREEN_ROOM_ID)
        return {"rooms_processed": len(legacy), "events_moved": moved,
                "events_adopted": (adopted[0]["adopted"] if adopted else 0),
                "events_corrected": (normalized[0]["normalized"] if normalized else 0),
                "rooms_archived": archived, "rooms_deleted": deleted}

    def list_rooms(self, include_archived=False):
        import json as _json
        rooms = [dict(r) for r in self.run(
            _LIST_ROOMS_CYPHER, include_archived=include_archived)]
        for room in rooms:
            room["assistant_mode"] = "agent"
            room["execution_profile"] = room.get("execution_profile") or "investigate"
            raw_tools = room.pop("agent_tools_json", None)
            room["agent_workspace"] = room.get("agent_workspace") or ""
            room["agent_request_limit"] = int(
                room.get("agent_request_limit") or 0)
            room["agent_tool_calls_limit"] = int(
                room.get("agent_tool_calls_limit") or 0)
            if raw_tools is None:
                room["agent_tools"] = ["graph"]
                continue
            try:
                room["agent_tools"] = _json.loads(raw_tools)
            except (TypeError, ValueError):
                room["agent_tools"] = []
            room["agent_tools"] = list(dict.fromkeys(
                ["graph", *room["agent_tools"]]))
        return rooms

    def get_room(self, room_id):
        rows = self.run("MATCH (r:Room {room_id: $room_id}) "
                        "RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind, "
                        "r.auto AS auto, r.description AS description, r.color AS color, "
                        "r.instructions AS instructions, "
                        "r.assistant_mode AS assistant_mode, "
                        "r.execution_profile AS execution_profile, "
                        "r.agent_tools_json AS agent_tools_json, "
                        "r.agent_workspace AS agent_workspace, "
                        "r.agent_request_limit AS agent_request_limit, "
                        "r.agent_tool_calls_limit AS agent_tool_calls_limit, "
                        "r.icon AS icon, r.archived AS archived, r.pinned AS pinned, "
                        "r.position AS position, r.matcher_json AS matcher_json",
                        room_id=room_id)
        if not rows:
            return None
        out = dict(rows[0])
        out["assistant_mode"] = "agent"
        out["execution_profile"] = out.get("execution_profile") or "investigate"
        import json as _json
        try:
            out["matcher"] = _json.loads(out.pop("matcher_json") or "{}")
        except (TypeError, ValueError):
            out["matcher"] = {}
        out["agent_workspace"] = out.get("agent_workspace") or ""
        out["agent_request_limit"] = int(
            out.get("agent_request_limit") or 0)
        out["agent_tool_calls_limit"] = int(
            out.get("agent_tool_calls_limit") or 0)
        raw_tools = out.pop("agent_tools_json", None)
        if raw_tools is not None:
            try:
                out["agent_tools"] = _json.loads(raw_tools)
            except (TypeError, ValueError):
                out["agent_tools"] = []
        out["agent_tools"] = list(dict.fromkeys(
            ["graph", *(out.get("agent_tools") or [])]))
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

        Manual primary assignments are deliberately preserved. Historical events
        predate the `source` field, so their capture source is recovered from
        `memory_domain` — 'home' is what camera capture writes.
        """
        query = (
            "MATCH (e:Event) "
            "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
            "RETURN e.event_id AS event_id, e.activity_type AS activity_type, "
            "e.application AS application, e.project_id AS project_id, "
            "e.summary AS summary, e.memory_domain AS memory_domain, "
            "collect(DISTINCT n.type) AS entity_types"
        )
        events = [dict(r) for r in self.run(query)]
        for event in events:
            event["source"] = (
                "camera" if event.get("memory_domain") == "home" else "screen")
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

    def add_message(self, room_id, role, text, ts=None, citations=None):
        """A room-scoped chat message (role: 'user' | 'assistant')."""
        import json as _json
        import time as _t
        import uuid as _uuid
        message_id = _uuid.uuid4().hex
        ts = ts if ts is not None else _t.time()
        self.run(_ADD_MESSAGE_CYPHER, room_id=room_id, message_id=message_id,
                 role=role, text=text, ts=ts,
                 citations_json=_json.dumps(citations or []))
        return {"message_id": message_id, "room_id": room_id, "role": role,
                "text": text, "ts": ts, "citations": citations or []}

    def room_messages(self, room_id, limit=20):
        """Recent chat messages in chronological order (for chat history)."""
        rows = [dict(r) for r in self.run(_ROOM_MESSAGES_CYPHER, room_id=room_id, limit=limit)]
        import json as _json
        for row in rows:
            try:
                row["citations"] = _json.loads(row.pop("citations_json") or "[]")
            except (TypeError, ValueError):
                row["citations"] = []
        return list(reversed(rows))

    def room_applications(self, room_id, start=None, end=None):
        """The apps/cameras that have events in this room — the feed's source chips.

        Ordered by how much of the room each one accounts for, so the busiest
        source is the first chip.
        """
        return [dict(r) for r in self.run(
            _ROOM_APPLICATIONS_CYPHER, room_id=room_id,
            start=start, end=end, limit=40)]

    def room_event_ids(self, room_id, start=None, end=None, applications=None,
                       limit=2000):
        """Every event id in scope — the whole slice, not just the prompt window.

        Question-driven retrieval is held to this set, so a filter cannot be
        defeated by an older-but-relevant event, nor accidentally drop one just
        because it fell outside the handful of lines the prompt quotes.
        """
        apps = ([a.strip().lower() for a in applications if a and a.strip()]
                if applications else None)
        return [r["event_id"] for r in self.run(
            _ROOM_EVENT_IDS_CYPHER, room_id=room_id, start=start, end=end,
            applications=apps or None, limit=limit) if r.get("event_id")]

    def room_context(self, room_id, event_limit=8, note_limit=8, entity_limit=15,
                     start=None, end=None, applications=None):
        """Grounding for room-scoped chat: recent events, notes, and top entities.

        `start`/`end` (epoch seconds) and `applications` narrow the context to
        exactly what the user has filtered the feed down to — asking about one
        camera should not pull in what the other three saw. Each event line
        carries its source so the model can attribute what it is told.
        """
        apps = ([a.strip().lower() for a in applications if a and a.strip()]
                if applications else None)
        event_rows = self.run(
            _ROOM_CONTEXT_EVENTS_CYPHER, room_id=room_id, limit=event_limit * 4,
            start=start, end=end, applications=apps or None)
        # User triage should improve the assistant too: low-priority noise and
        # items set aside for review do not consume the small grounding window.
        kept = [
            row for row in event_rows
            if row.get("summary")
            and row.get("priority", "normal") != "low"
            and not row.get("flagged")
        ][:event_limit]
        events = [
            f"[{row['application']}] {row['summary']}" if row.get("application")
            else row["summary"]
            for row in kept
        ]
        notes = [r["text"] for r in self.run(
            _ROOM_CONTEXT_NOTES_CYPHER, room_id=room_id, limit=note_limit,
            start=start, end=end)]
        entities = [r["name"] for r in self.run(
            _ROOM_CONTEXT_ENTITIES_CYPHER, room_id=room_id, limit=entity_limit,
            start=start, end=end, applications=apps or None)]
        return {"events": events, "notes": notes, "entities": entities}

    def room_feed_full(self, room_id, date_str=None, limit=200, offset=0,
                       kinds=None, query=None, priorities=None, flagged=None,
                       start=None, end=None, applications=None):
        """Merged, time-ordered feed: events + notes + chat messages (newest first).

        `start`/`end` and `applications` are the same scope the chat context uses,
        so what the user sees in the feed is what the assistant is given.
        """
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
                  "application": e.get("application"),
                  "importance": e.get("importance"),
                  "confidence": e.get("confidence"),
                  "priority": e.get("priority") or "normal",
                  "priority_source": e.get("priority_source") or "automatic",
                  "flagged": bool(e.get("flagged")),
                  "flag_reason": e.get("flag_reason")} for e in events]
        for n in self.run(_ROOM_NOTES_CYPHER, room_id=room_id, limit=fetch_limit):
            items.append({"kind": "note", "note_id": n["note_id"],
                          "ts": n["ts"], "text": n["text"]})
        import json as _json
        for m in self.run(_ROOM_MESSAGES_CYPHER, room_id=room_id, limit=fetch_limit):
            try:
                message_citations = _json.loads(m.get("citations_json") or "[]")
            except (TypeError, ValueError):
                message_citations = []
            items.append({"kind": "message", "ts": m["ts"], "text": m["text"],
                          "role": m["role"], "citations": message_citations})
        if date_str:
            day_start = datetime.datetime.fromisoformat(date_str).timestamp()
            day_end = day_start + 86400
            items = [item for item in items
                     if day_start <= (item.get("ts") or 0) < day_end]
        if start is not None:
            items = [item for item in items if (item.get("ts") or 0) >= start]
        if end is not None:
            items = [item for item in items if (item.get("ts") or 0) < end]
        if applications:
            # Notes and chat have no source of their own — they belong to the room
            # as a whole, so a source filter only narrows the activity.
            allowed_apps = {a.strip().lower() for a in applications if a and a.strip()}
            items = [item for item in items
                     if item["kind"] != "event"
                     or (item.get("application") or "").lower() in allowed_apps]
        items.sort(key=lambda x: x.get("ts") or 0, reverse=True)
        if kinds:
            allowed = set(kinds)
            items = [item for item in items if item["kind"] in allowed]
        if query:
            needle = query.casefold()
            items = [item for item in items if needle in (item.get("text") or "").casefold()]
        if priorities:
            allowed_priorities = set(priorities)
            items = [item for item in items
                     if item["kind"] != "event"
                     or item.get("priority", "normal") in allowed_priorities]
        if flagged is not None:
            items = [item for item in items
                     if item["kind"] != "event"
                     or bool(item.get("flagged")) == bool(flagged)]
        return items[offset:offset + limit]

    def room_feed(self, room_id, date_str=None, limit=200):
        params = {"room_id": room_id, "limit": limit}
        if date_str:
            start = datetime.datetime.fromisoformat(date_str).timestamp()
            params["start"], params["end"] = start, start + 86400
            return [dict(r) for r in self.run(_ROOM_FEED_DATED_CYPHER, **params)]
        return [dict(r) for r in self.run(_ROOM_FEED_CYPHER, **params)]

    # -- Step 13: daily rollup ---------------------------------------------
    def sessions_with_events(self, date_str, domain=None):
        """Day's sessions, each with its events (ordered)."""
        return [dict(r) for r in self.run(
            _SESSIONS_WITH_EVENTS_CYPHER, date=date_str, domain=domain)]

    def day_entities(self, date_str, limit=30, domain=None):
        return [dict(r) for r in self.run(
            _DAY_ENTITIES_CYPHER, date=date_str, limit=limit, domain=domain)]

    def day_claims(self, date_str, limit=20, domain=PRODUCTIVITY_DOMAIN):
        return [dict(r) for r in self.run(
            _DAY_CLAIMS_CYPHER, date=date_str, limit=limit, domain=domain)]

    def range_metrics(self, start_date, end_date, domain=PRODUCTIVITY_DOMAIN):
        """Deterministic productivity metrics over an inclusive date range.

        Screen-only by default — see `_EVENT_DOMAIN_EXPR`. Pass domain=None for
        every source, or 'home' for the camera side.
        """
        params = {"start": start_date, "end": end_date, "domain": domain}
        totals = self.run(_RANGE_TOTALS_CYPHER, **params)
        t = dict(totals[0]) if totals else {}
        by_activity = [dict(r) for r in self.run(_RANGE_BY_ACTIVITY_CYPHER, **params)]
        # Folded before truncation, not after: one project reaches the graph
        # under several spellings, and each spelling is individually small
        # enough to fall outside a top-10 cut. The queries therefore return a
        # wide slice and the canonical rows are trimmed here instead.
        by_app = fold_apps([dict(r) for r in self.run(_RANGE_BY_APP_CYPHER, **params)])[:10]
        by_project = fold_projects(
            [dict(r) for r in self.run(_RANGE_BY_PROJECT_CYPHER, **params)])[:10]

        active = t.get("active_seconds") or 0.0
        events = t.get("events") or 0
        switches = t.get("switches") or 0
        active_hours = active / 3600.0
        avg_event = (active / events) if events else 0.0
        focus_score = round(min(100.0, (avg_event / 300.0) * 100.0))
        return {
            "start_date": start_date,
            "end_date": end_date,
            "domain": domain,
            "active_seconds": round(active, 1),
            "active_minutes": round(active / 60.0, 1),
            "events": events,
            "sessions": t.get("sessions") or 0,
            "active_days": t.get("active_days") or 0,
            "switches": switches,
            "longest_block_seconds": round(t.get("longest_block") or 0.0, 1),
            "avg_event_seconds": round(avg_event, 1),
            "switches_per_hour": round(switches / active_hours, 1) if active_hours else 0.0,
            "focus_score": focus_score,
            "by_activity": by_activity,
            "by_app": by_app,
            "by_project": by_project,
        }

    def daily_metrics(self, date_str, domain=PRODUCTIVITY_DOMAIN):
        """One day's productivity metrics (Phase 3 Coach input)."""
        metrics = self.range_metrics(date_str, date_str, domain=domain)
        metrics["date"] = date_str
        return metrics

    def activity_series(self, start_date, end_date, domain=PRODUCTIVITY_DOMAIN):
        """Minutes per activity per day over an inclusive range.

        Returns the flat (date, activity, minutes, events) rows; the caller
        pivots them into whichever chart shape it needs.
        """
        return [dict(r) for r in self.run(
            _ACTIVITY_SERIES_CYPHER, start=start_date, end=end_date, domain=domain)]

    def event_spans(self, start_date, end_date, domain=PRODUCTIVITY_DOMAIN,
                    limit=20000):
        """Raw (span_start, span_seconds) rows — the hour-of-day view's input."""
        return [dict(r) for r in self.run(
            _EVENT_SPANS_CYPHER, start=start_date, end=end_date, domain=domain,
            limit=max(1, min(int(limit), 50000)))]

    # -- Written reports ---------------------------------------------------
    #
    # The deterministic report is recomputed from the graph on every read, so it
    # never needed storing. A written one does: it is a model call at high
    # effort, it carries scores that only mean something as a series, and the
    # next day's writer is given the last fortnight of them so today's report is
    # calibrated against its own history rather than against an imagined day.

    def save_written_report(self, end_date, period, report, model=None,
                            effort=None, start_date=None):
        """Store (or replace) the written report for one period.

        Keyed on period + end date, so re-writing a day overwrites it rather
        than accumulating drafts. `report` is the model's structured output as a
        plain dict; it is stored whole as JSON, with the score lifted out as a
        property so a fortnight of them can be read without parsing every body.
        """
        import json as _json
        scores = report.get("scores") or []
        overall = report.get("overall_score")
        rows = self.run(
            _SAVE_WRITTEN_REPORT_CYPHER,
            report_id=f"{period}:{end_date}",
            date=end_date, period=period,
            start_date=start_date or end_date,
            headline=str(report.get("headline") or ""),
            overall_score=int(overall) if overall is not None else None,
            score_names=[str(s.get("name") or "") for s in scores],
            body=_json.dumps(report, ensure_ascii=False),
            model=model, effort=effort, ts=time.time())
        return dict(rows[0]) if rows else None

    def written_reports(self, start_date, end_date, period="daily", limit=30):
        """Stored reports whose end date falls in the window, newest first."""
        import json as _json
        out = []
        for row in self.run(_WRITTEN_REPORTS_CYPHER, start=start_date,
                            end=end_date, period=period,
                            limit=max(1, min(int(limit), 120))):
            item = dict(row)
            try:
                item["report"] = _json.loads(item.pop("body") or "{}")
            except (TypeError, ValueError):
                # A body we cannot parse must not take the whole history down —
                # the date and score are still worth returning.
                item.pop("body", None)
                item["report"] = {}
            out.append(item)
        return out

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

    # -- Long-term tier: rollups, projects, goals, decay --------------------
    def save_rollup(self, payload, dates=()):
        """Upsert one :Rollup and connect it to the days it summarizes.

        Also links the tier below it (days into a week, weeks into a month), so
        a coarse answer can always be expanded into the finer summaries it was
        built from, and from there into the events themselves.
        """
        from memory.consolidation import DAY, MONTH, WEEK

        rollup_id = payload["rollup_id"]
        self.run(_SAVE_ROLLUP_CYPHER, rollup_id=rollup_id, props=dict(payload),
                 dates=[d for d in (dates or ()) if d])
        child_kind = {WEEK: DAY, MONTH: WEEK}.get(payload.get("kind"))
        if child_kind:
            self.run(_LINK_ROLLUP_TIERS_CYPHER, rollup_id=rollup_id,
                     child_kind=child_kind)
        return payload

    def get_rollup(self, kind, key):
        from memory.consolidation import rollup_id as _rid
        rows = self.run(_GET_ROLLUP_CYPHER, rollup_id=_rid(kind, key))
        return dict(rows[0]["props"]) if rows else None

    def list_rollups(self, kind=None, start=None, end=None, limit=30):
        return [dict(row["props"]) for row in self.run(
            _LIST_ROLLUPS_CYPHER, kind=kind, start=start, end=end,
            limit=max(1, min(int(limit), 200)))]

    def set_rollup_narrative(self, rollup_id, narrative):
        rows = self.run(_SET_ROLLUP_NARRATIVE_CYPHER, rollup_id=rollup_id,
                        narrative=narrative)
        return bool(rows)

    def sync_projects(self, dormant_before=0.0):
        """Promote `Session.project_id` strings into :Project nodes with a lifespan.

        A project is dormant, never deleted: months later it is still the answer
        to "what was I working on then", it just isn't current work.
        """
        self.run(_SYNC_PROJECTS_CYPHER, dormant_before=float(dormant_before))
        self.run(_LINK_PROJECT_SESSIONS_CYPHER)
        rows = self.run(
            "MATCH (p:Project) RETURN coalesce(p.status, 'active') AS status, "
            "count(p) AS n")
        return {row["status"]: row["n"] for row in rows}

    def list_projects(self, status=None, limit=50):
        return [dict(row) for row in self.run(
            _LIST_PROJECTS_CYPHER, status=status,
            limit=max(1, min(int(limit), 200)))]

    def sync_goals(self):
        """Promote focus-session goals into :Goal nodes with their pursuit history."""
        rows = self.run(_SYNC_GOALS_CYPHER)
        self.run(_LINK_GOAL_SESSIONS_CYPHER)
        return {"goals": rows[0]["goals"] if rows else 0}

    def list_goals(self, limit=50):
        return [dict(row) for row in self.run(
            _LIST_GOALS_CYPHER, limit=max(1, min(int(limit), 200)))]

    def prune_stale_entities(self, before, min_events=2):
        """Delete quarantined entities that were never corroborated and are old.

        An entity the user merged or renamed into is protected regardless of
        age: that is a correction they made by hand, not a stale guess.
        """
        ids = [row["entity_id"] for row in self.run(
            _STALE_ENTITY_IDS_CYPHER, before=float(before),
            min_events=int(min_events))]
        if ids:
            self.run(_DELETE_ENTITIES_CYPHER, ids=ids)
        return len(ids)

    def prune_orphan_claims(self):
        ids = [row["claim_id"] for row in self.run(_ORPHAN_CLAIM_IDS_CYPHER)]
        if ids:
            self.run(_DELETE_CLAIMS_CYPHER, ids=ids)
        return len(ids)

    def long_term_context(self, end_date=None, months=3, weeks=6, limit=12):
        """The coarse retrieval tier: compressed periods instead of raw events.

        Answering "how has this project gone since spring" from the event table
        means scanning every minute of it. This returns the stored summaries and
        the project lifespans instead, which is a fixed, small read whatever the
        history length.
        """
        from memory.consolidation import MONTH, WEEK
        end_date = end_date or datetime.date.today().isoformat()
        return {
            "end_date": end_date,
            "months": self.list_rollups(kind=MONTH, end=end_date,
                                        limit=max(1, int(months))),
            "weeks": self.list_rollups(kind=WEEK, end=end_date,
                                       limit=max(1, int(weeks))),
            "projects": self.list_projects(limit=limit),
            "goals": self.list_goals(limit=limit),
        }

    def earliest_day(self):
        """The first date the graph holds anything for, or None on an empty one.

        The lifelong horizon needs a real starting point rather than an
        arbitrary lookback: "since you started" is a different window on a
        two-week-old install than on a three-year-old one.
        """
        rows = self.run(
            "MATCH (d:Day) RETURN min(d.date) AS earliest")
        return rows[0]["earliest"] if rows else None

    def rollup_counts(self):
        rows = self.run(
            "MATCH (r:Rollup) RETURN r.kind AS kind, count(r) AS n ORDER BY kind")
        return {row["kind"]: row["n"] for row in rows}

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
        "memory_domain": e.get("memory_domain", "personal"),
        "importance": e.get("importance", 0.5),
        "confidence": e.get("confidence", 0.5),
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
        "instructions": room.instructions,
        "assistant_mode": room.assistant_mode,
        "execution_profile": room.execution_profile,
        "agent_tools_json": json_module.dumps(room.agent_tools),
        "agent_workspace": room.agent_workspace,
        "agent_request_limit": room.agent_request_limit,
        "agent_tool_calls_limit": room.agent_tool_calls_limit,
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
    e.summary = $summary,
    e.memory_domain = $memory_domain,
    e.importance = $importance,
    e.confidence = $confidence
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
  AND ($domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain)
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

# Keyword search scores every record by how many query TERMS its text contains
# (`hits`), plus a bonus when the full query phrase appears verbatim. Matching
# the whole question as one substring — the previous behaviour — meant a
# natural-language query never matched anything. See memory/retrieval/terms.py.
_SEARCH_EVENTS_CYPHER = """
MATCH (e:Event)
OPTIONAL MATCH (r:Room)-[:CONTAINS]->(e)
WITH e, collect(DISTINCT {room_id: r.room_id, name: r.name}) AS rooms
WHERE ($start IS NULL OR e.span_start >= $start)
  AND ($end IS NULL OR e.span_start < $end)
  AND ($room_id IS NULL OR any(room IN rooms WHERE room.room_id = $room_id))
  AND ($domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain)
WITH e, rooms, toLower(coalesce(e.summary, '')) AS hay
WITH e, rooms, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'event' AS kind, e.event_id AS id,
       coalesce(e.application, e.activity_type, 'Activity') AS title,
       e.summary AS text, e.span_start AS ts,
       e.span_start AS span_start, e.span_end AS span_end, rooms,
       100 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, ts DESC
LIMIT $limit
"""

_SEARCH_NOTES_CYPHER = """
MATCH (r:Room)-[:HAS_NOTE]->(n:RoomNote)
WHERE ($start IS NULL OR n.ts >= $start)
  AND ($end IS NULL OR n.ts < $end)
  AND ($room_id IS NULL OR r.room_id = $room_id)
  AND ($domain IS NULL OR
       CASE WHEN r.kind = 'camera' THEN 'home' ELSE 'personal' END = $domain)
WITH r, n, toLower(coalesce(n.text, '')) AS hay
WITH r, n, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'note' AS kind, n.note_id AS id, r.name AS title, n.text AS text,
       n.ts AS ts, n.ts AS span_start, n.ts AS span_end,
       [{room_id: r.room_id, name: r.name}] AS rooms,
       90 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, ts DESC
LIMIT $limit
"""

_SEARCH_MESSAGES_CYPHER = """
MATCH (r:Room)-[:HAS_MESSAGE]->(m:RoomMessage)
WHERE ($start IS NULL OR m.ts >= $start)
  AND ($end IS NULL OR m.ts < $end)
  AND ($room_id IS NULL OR r.room_id = $room_id)
  AND ($domain IS NULL OR
       CASE WHEN r.kind = 'camera' THEN 'home' ELSE 'personal' END = $domain)
WITH r, m, toLower(coalesce(r.name, '') + ' ' + coalesce(m.role, '') + ' ' +
     coalesce(m.text, '') +
     CASE WHEN m.role = 'insight'
          THEN ' proactive insight notification notify notified alert nudge'
          ELSE '' END) AS hay
WITH r, m, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'message' AS kind, m.message_id AS id,
       r.name + ' · ' + coalesce(m.role, 'message') AS title,
       m.text AS text, m.ts AS ts, m.ts AS span_start, m.ts AS span_end,
       [{room_id: r.room_id, name: r.name}] AS rooms,
       80 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, ts DESC
LIMIT $limit
"""

_SEARCH_ENTITIES_CYPHER = """
MATCH (n:Entity)<-[:MENTIONS]-(e:Event)
WHERE ($start IS NULL OR e.span_start >= $start)
  AND ($end IS NULL OR e.span_start < $end)
  AND ($room_id IS NULL OR EXISTS {
        MATCH (:Room {room_id: $room_id})-[:CONTAINS]->(e)
      })
  AND ($domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain)
WITH DISTINCT n
WITH n, toLower(coalesce(n.name, '')) AS hay
WITH n, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'entity' AS kind, n.entity_id AS id, n.name AS title,
       coalesce(n.type, 'entity') AS text, null AS ts, [] AS rooms,
       110 + 8 * hits + CASE WHEN hay = $needle THEN 40
                             WHEN hay CONTAINS $needle THEN 20 ELSE 0 END AS score
ORDER BY score DESC, title
LIMIT $limit
"""

_SEARCH_CLAIMS_CYPHER = """
MATCH (c:Claim)<-[:SUPPORTS]-(e:Event)
OPTIONAL MATCH (r:Room)-[:CONTAINS]->(e)
WITH c, e, collect(DISTINCT {room_id: r.room_id, name: r.name}) AS rooms
WHERE ($start IS NULL OR e.span_start >= $start)
  AND ($end IS NULL OR e.span_start < $end)
  AND ($room_id IS NULL OR any(room IN rooms WHERE room.room_id = $room_id))
  AND ($domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain)
WITH c, e, rooms, toLower(coalesce(c.text, '')) AS hay
WITH c, e, rooms, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'claim' AS kind, c.claim_id AS id, 'Claim' AS title, c.text AS text,
       e.span_start AS ts, e.span_start AS span_start, e.span_end AS span_end, rooms,
       95 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, ts DESC
LIMIT $limit
"""

_SEARCH_ROOMS_CYPHER = """
MATCH (r:Room)
WHERE ($domain IS NULL OR
       CASE WHEN r.kind = 'camera' THEN 'home' ELSE 'personal' END = $domain)
WITH r, toLower(coalesce(r.name, '') + ' ' + coalesce(r.description, '')) AS hay
WITH r, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'room' AS kind, r.room_id AS id, r.name AS title,
       coalesce(r.description, r.kind) AS text, null AS ts,
       [{room_id: r.room_id, name: r.name}] AS rooms,
       105 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, title
LIMIT $limit
"""

_SEARCH_NUDGES_CYPHER = """
MATCH (n:Nudge)
WHERE ($start IS NULL OR n.ts >= $start)
  AND ($end IS NULL OR n.ts < $end)
  AND $room_id IS NULL
WITH n, toLower(coalesce(n.text, '') + ' ' + coalesce(n.kind, '') +
     ' proactive insight notification notify notified alert nudge') AS hay
WITH n, hay, size([t IN $terms WHERE hay CONTAINS t]) AS hits
WHERE hits > 0
RETURN 'insight' AS kind, n.nudge_id AS id,
       'Proactive insight' AS title, n.text AS text,
       n.ts AS ts, n.ts AS span_start, n.ts AS span_end, [] AS rooms,
       115 + 8 * hits + CASE WHEN hay CONTAINS $needle THEN 25 ELSE 0 END AS score
ORDER BY score DESC, ts DESC
LIMIT $limit
"""

# Chronological fetch used when a question carries no content terms at all
# ("what did I do today") — the answer comes from the scope, not from keywords.
_RECENT_EVENTS_CYPHER = """
MATCH (e:Event)
OPTIONAL MATCH (r:Room)-[:CONTAINS]->(e)
WITH e, collect(DISTINCT {room_id: r.room_id, name: r.name}) AS rooms
WHERE ($start IS NULL OR e.span_start >= $start)
  AND ($end IS NULL OR e.span_start < $end)
  AND ($room_id IS NULL OR any(room IN rooms WHERE room.room_id = $room_id))
  AND ($domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain)
  AND coalesce(e.summary, '') <> ''
RETURN 'event' AS kind, e.event_id AS id,
       coalesce(e.application, e.activity_type, 'Activity') AS title,
       e.summary AS text, e.span_start AS ts,
       e.span_start AS span_start, e.span_end AS span_end, rooms, 60 AS score
ORDER BY ts DESC
LIMIT $limit
"""

_RECENT_NUDGES_CYPHER = """
MATCH (n:Nudge)
WHERE ($start IS NULL OR n.ts >= $start)
  AND ($end IS NULL OR n.ts < $end)
RETURN 'insight' AS kind, n.nudge_id AS id,
       'Proactive insight' AS title, n.text AS text,
       n.ts AS ts, n.ts AS span_start, n.ts AS span_end,
       [] AS rooms, 65 AS score
ORDER BY n.ts DESC
LIMIT $limit
"""

_EVENTS_IN_ROOM_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
WHERE e.event_id IN $ids
RETURN e.event_id AS event_id
"""

_ENTITY_DETAIL_CYPHER = """
MATCH (n:Entity {entity_id: $entity_id})
OPTIONAL MATCH (n)<-[:MENTIONS]-(e:Event)
WITH n, e
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN n.entity_id AS entity_id, n.name AS name, n.type AS type,
       n.memory_status AS memory_status, n.max_confidence AS max_confidence,
       count(DISTINCT e) AS mentions, min(e.span_start) AS first_seen,
       max(e.span_end) AS last_seen
"""

_ENTITY_DETAIL_EVENTS_CYPHER = """
MATCH (n:Entity {entity_id: $entity_id})<-[m:MENTIONS]-(e:Event)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN e.event_id AS event_id, e.summary AS summary,
       e.application AS application, e.activity_type AS activity_type,
       e.span_start AS span_start, e.span_end AS span_end,
       m.confidence AS confidence, m.role AS role
ORDER BY e.span_start DESC
LIMIT $limit
"""

_ENTITY_DETAIL_CLAIMS_CYPHER = """
MATCH (n:Entity {entity_id: $entity_id})<-[:MENTIONS]-(e:Event)-[:SUPPORTS]->(c:Claim)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN DISTINCT c.claim_id AS claim_id, c.text AS text,
       c.confidence AS confidence, max(e.span_start) AS last_seen
ORDER BY last_seen DESC
LIMIT $limit
"""

_ENTITY_DETAIL_ROOMS_CYPHER = """
MATCH (n:Entity {entity_id: $entity_id})<-[:MENTIONS]-(e:Event)<-[:CONTAINS]-(r:Room)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN r.room_id AS room_id, r.name AS name, count(DISTINCT e) AS events
ORDER BY events DESC
LIMIT $limit
"""

_EVENT_DETAIL_CYPHER = """
MATCH (e:Event {event_id: $event_id})<-[:HAS_EVENT]-(s:Session)
RETURN e.event_id AS event_id, e.summary AS summary,
       e.original_summary AS original_summary, e.corrected_at AS corrected_at,
       e.application AS application, e.activity_type AS activity_type,
       e.project_id AS project_id, e.span_start AS span_start,
       e.span_end AS span_end, e.span_seconds AS span_seconds,
       e.boundary_label AS boundary_label, s.session_id AS session_id,
       coalesce(e.importance, 0.5) AS importance,
       coalesce(e.confidence, 0.5) AS confidence,
       coalesce(e.user_priority,
         CASE WHEN coalesce(e.importance, 0.5) >= 0.75 THEN 'high'
              WHEN coalesce(e.importance, 0.5) < 0.3 THEN 'low'
              ELSE 'normal' END) AS priority,
       CASE WHEN e.user_priority IS NULL THEN 'automatic' ELSE 'user' END AS priority_source,
       coalesce(e.flagged, false) AS flagged, e.flag_reason AS flag_reason
"""

_EVENT_DETAIL_ENTITIES_CYPHER = """
MATCH (e:Event {event_id: $event_id})-[m:MENTIONS]->(n:Entity)
RETURN n.entity_id AS entity_id, n.name AS name, n.type AS type,
       m.confidence AS confidence, m.role AS role
ORDER BY m.confidence DESC
"""

_EVENT_DETAIL_CLAIMS_CYPHER = """
MATCH (e:Event {event_id: $event_id})-[:SUPPORTS]->(c:Claim)
RETURN c.claim_id AS claim_id, c.text AS text, c.confidence AS confidence
ORDER BY c.confidence DESC
"""

_EVENT_DETAIL_ROOMS_CYPHER = """
MATCH (r:Room)-[rel:CONTAINS]->(e:Event {event_id: $event_id})
RETURN r.room_id AS room_id, r.name AS name, rel.assignment AS assignment,
       rel.manual AS manual
ORDER BY assignment
"""

_UPDATE_EVENT_SUMMARY_CYPHER = """
MATCH (e:Event {event_id: $event_id})
CREATE (c:MemoryCorrection {
  correction_id: $correction_id, target_type: 'event', target_id: $event_id,
  field: 'summary', old_value: e.summary, new_value: $summary,
  created_at: timestamp()
})
SET e.original_summary = coalesce(e.original_summary, e.summary),
    e.summary = $summary, e.corrected_at = timestamp()
RETURN e.event_id AS event_id, e.summary AS summary,
       e.original_summary AS original_summary, e.corrected_at AS corrected_at
"""

_UPDATE_EVENT_METADATA_CYPHER = """
MATCH (e:Event {event_id: $event_id})
SET e.user_priority =
      CASE WHEN $set_priority THEN $priority ELSE e.user_priority END,
    e.flagged =
      CASE WHEN $set_flagged THEN $flagged ELSE coalesce(e.flagged, false) END,
    e.flag_reason =
      CASE WHEN $set_flag_reason THEN $flag_reason ELSE e.flag_reason END,
    e.reviewed_at = timestamp()
RETURN e.event_id AS event_id,
       coalesce(e.user_priority,
         CASE WHEN coalesce(e.importance, 0.5) >= 0.75 THEN 'high'
              WHEN coalesce(e.importance, 0.5) < 0.3 THEN 'low'
              ELSE 'normal' END) AS priority,
       CASE WHEN e.user_priority IS NULL THEN 'automatic' ELSE 'user' END AS priority_source,
       coalesce(e.flagged, false) AS flagged, e.flag_reason AS flag_reason,
       e.reviewed_at AS reviewed_at
"""

_UPDATE_ENTITY_CYPHER = """
MATCH (n:Entity {entity_id: $entity_id})
CREATE (c:MemoryCorrection {
  correction_id: $correction_id, target_type: 'entity', target_id: $entity_id,
  field: 'name/type', old_value: coalesce(n.name, '') + '|' + coalesce(n.type, ''),
  new_value: coalesce($name, n.name) + '|' + coalesce($entity_type, n.type),
  created_at: timestamp()
})
SET n.name = coalesce($name, n.name), n.type = coalesce($entity_type, n.type),
    n.corrected_at = timestamp()
RETURN n.entity_id AS entity_id, n.name AS name, n.type AS type
"""

_ENTITY_MENTIONS_FOR_MERGE_CYPHER = """
MATCH (e:Event)-[m:MENTIONS]->(:Entity {entity_id: $source_id})
RETURN e.event_id AS event_id, m.confidence AS confidence,
       m.role AS role, m.co_presence AS co_presence
"""

# A merge deletes the source entity, so without this the next capture simply
# recreates it and the user's correction is undone. The alias survives the node
# and is applied to every later write — that is what makes curation stick.
_RECORD_ALIAS_CYPHER = """
MERGE (a:EntityAlias {alias_id: $alias_id})
SET a.canonical_id = $canonical_id, a.name = $name,
    a.source = $source, a.created_at = timestamp()
"""

# Aliases that pointed at the merged-away entity must follow it to the new
# canonical, or a two-step merge (A->B, B->C) would strand A at a dead node.
_REPOINT_ALIASES_CYPHER = """
MATCH (a:EntityAlias {canonical_id: $old_canonical_id})
SET a.canonical_id = $canonical_id
"""

_LIST_ALIASES_CYPHER = """
MATCH (a:EntityAlias)
WHERE a.alias_id <> a.canonical_id
RETURN a.alias_id AS alias_id, a.canonical_id AS canonical_id,
       a.name AS name, a.source AS source
ORDER BY a.created_at DESC
LIMIT $limit
"""

_CANONICAL_NAMES_CYPHER = """
MATCH (a:EntityAlias)
WHERE a.alias_id <> a.canonical_id
MATCH (n:Entity {entity_id: a.canonical_id})
RETURN DISTINCT a.name AS wrong_name, n.name AS canonical_name
ORDER BY canonical_name
LIMIT $limit
"""

_DELETE_ALIAS_CYPHER = """
MATCH (a:EntityAlias {alias_id: $alias_id})
DELETE a
RETURN $alias_id AS alias_id
"""

_MERGE_ENTITY_MENTION_CYPHER = """
MATCH (e:Event {event_id: $event_id}), (target:Entity {entity_id: $target_id})
MERGE (e)-[m:MENTIONS]->(target)
SET m.confidence = CASE
      WHEN m.confidence IS NULL OR coalesce($confidence, 0) > m.confidence
      THEN $confidence ELSE m.confidence END,
    m.role = coalesce(m.role, $role),
    m.co_presence = coalesce(m.co_presence, $co_presence)
"""

_CREATE_SPLIT_ENTITY_CYPHER = """
MERGE (n:Entity {entity_id: $entity_id})
ON CREATE SET n.name = $name, n.type = $entity_type,
              n.memory_status = 'quarantined', n.created_at = timestamp()
"""

_MOVE_ENTITY_MENTION_CYPHER = """
MATCH (e:Event {event_id: $event_id})-[old:MENTIONS]->
      (source:Entity {entity_id: $source_id}),
      (target:Entity {entity_id: $target_id})
MERGE (e)-[m:MENTIONS]->(target)
SET m.confidence = old.confidence, m.role = old.role,
    m.co_presence = old.co_presence
DELETE old
RETURN e.event_id AS event_id
"""

_UPDATE_CLAIM_CYPHER = """
MATCH (cl:Claim {claim_id: $claim_id})
CREATE (c:MemoryCorrection {
  correction_id: $correction_id, target_type: 'claim', target_id: $claim_id,
  field: 'text', old_value: cl.text, new_value: $text, created_at: timestamp()
})
SET cl.original_text = coalesce(cl.original_text, cl.text),
    cl.text = $text, cl.corrected_at = timestamp()
RETURN cl.claim_id AS claim_id, cl.text AS text
"""

_DELETE_CLAIM_CYPHER = """
OPTIONAL MATCH (c:Claim {claim_id: $claim_id})
WITH c, c IS NOT NULL AS existed
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE c)
RETURN existed AS deleted
"""

_FORGET_EVENT_CYPHER = """
OPTIONAL MATCH (e:Event {event_id: $event_id})
OPTIONAL MATCH (e)-[:SUPPORTS]->(claim:Claim)
WITH e, collect(DISTINCT claim) AS claims, e IS NOT NULL AS existed
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE e)
WITH existed,
     [claim IN claims WHERE NOT (claim)<-[:SUPPORTS]-(:Event)] AS orphan_claims
FOREACH (claim IN orphan_claims | DETACH DELETE claim)
RETURN existed AS deleted, $event_id AS event_id
"""

_SESSION_EVENT_IDS_CYPHER = """
MATCH (s:Session {session_id: $session_id})-[:HAS_EVENT]->(e:Event)
RETURN e.event_id AS event_id
"""

_DELETE_EMPTY_SESSION_CYPHER = """
OPTIONAL MATCH (s:Session {session_id: $session_id})
WITH s, s IS NOT NULL AS existed
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE s)
RETURN existed AS deleted
"""

_DAY_EVENT_IDS_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN DISTINCT e.event_id AS event_id
"""

_DELETE_EMPTY_DAY_CYPHER = """
OPTIONAL MATCH (d:Day {date: $date})
OPTIONAL MATCH (d)-[:HAS_SESSION]->(s:Session)
WITH d, [s IN collect(s) WHERE NOT (s)-[:HAS_EVENT]->()] AS empty_sessions
FOREACH (s IN empty_sessions | DETACH DELETE s)
WITH d
OPTIONAL MATCH (d)-[:HAS_SESSION]->(remaining:Session)
WITH d, count(remaining) AS remaining_count
FOREACH (_ IN CASE WHEN d IS NOT NULL AND remaining_count = 0
                   THEN [1] ELSE [] END | DETACH DELETE d)
RETURN true AS deleted
"""

_FORGET_ENTITY_CYPHER = """
OPTIONAL MATCH (n:Entity {entity_id: $entity_id})
WITH n, n IS NOT NULL AS existed
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE n)
RETURN existed AS deleted
"""

_CREATE_CONVERSATION_CYPHER = """
CREATE (c:Conversation {
  conversation_id: $conversation_id, title: $title, scope: $scope,
  room_id: $room_id, from_ts: $from_ts, to_ts: $to_ts,
  created_at: timestamp(), updated_at: timestamp()
})
RETURN c.conversation_id AS conversation_id, c.title AS title,
       c.scope AS scope, c.room_id AS room_id,
       c.from_ts AS from_ts, c.to_ts AS to_ts,
       c.created_at AS created_at
"""

_LIST_CONVERSATIONS_CYPHER = """
MATCH (c:Conversation)
OPTIONAL MATCH (c)-[:HAS_ASSISTANT_MESSAGE]->(m:AssistantMessage)
RETURN c.conversation_id AS conversation_id, c.title AS title,
       c.scope AS scope, c.room_id AS room_id,
       c.from_ts AS from_ts, c.to_ts AS to_ts,
       count(m) AS messages, max(m.ts) AS last_message,
       c.created_at AS created_at
ORDER BY coalesce(last_message, created_at) DESC
LIMIT $limit
"""

_GET_CONVERSATION_CYPHER = """
MATCH (c:Conversation {conversation_id: $conversation_id})
RETURN c.conversation_id AS conversation_id, c.title AS title,
       c.scope AS scope, c.room_id AS room_id,
       c.from_ts AS from_ts, c.to_ts AS to_ts,
       c.created_at AS created_at, c.updated_at AS updated_at
"""

_UPDATE_CONVERSATION_CYPHER = """
MATCH (c:Conversation {conversation_id: $conversation_id})
SET c.title = $title, c.scope = $scope, c.room_id = $room_id,
    c.from_ts = $from_ts, c.to_ts = $to_ts, c.updated_at = timestamp()
RETURN c.conversation_id AS conversation_id, c.title AS title,
       c.scope AS scope, c.room_id AS room_id,
       c.from_ts AS from_ts, c.to_ts AS to_ts
"""

_ADD_CONVERSATION_MESSAGE_CYPHER = """
MATCH (c:Conversation {conversation_id: $conversation_id})
CREATE (m:AssistantMessage {
  message_id: $message_id, role: $role, text: $text,
  citations_json: $citations_json, ts: $ts
})
CREATE (c)-[:HAS_ASSISTANT_MESSAGE]->(m)
SET c.updated_at = timestamp(),
    c.title = CASE
      WHEN c.title = 'New conversation' AND $role = 'user'
      THEN substring($text, 0, 64) ELSE c.title END
RETURN m.message_id AS message_id, m.role AS role, m.text AS text,
       m.citations_json AS citations_json, m.ts AS ts
"""

_CONVERSATION_MESSAGES_CYPHER = """
MATCH (:Conversation {conversation_id: $conversation_id})
      -[:HAS_ASSISTANT_MESSAGE]->(m:AssistantMessage)
RETURN m.message_id AS message_id, m.role AS role, m.text AS text,
       m.citations_json AS citations_json, m.ts AS ts
ORDER BY m.ts
LIMIT $limit
"""

_DELETE_CONVERSATION_CYPHER = """
OPTIONAL MATCH (c:Conversation {conversation_id: $conversation_id})
OPTIONAL MATCH (c)-[:HAS_ASSISTANT_MESSAGE]->(m:AssistantMessage)
WITH c, collect(m) AS messages, c IS NOT NULL AS existed
FOREACH (message IN messages | DETACH DELETE message)
FOREACH (_ IN CASE WHEN existed THEN [1] ELSE [] END | DETACH DELETE c)
RETURN existed AS deleted
"""

_START_FOCUS_CYPHER = """
CREATE (f:FocusSession {
  focus_id: $focus_id, goal: $goal, room_id: $room_id,
  planned_minutes: $planned_minutes, started_at: $started_at,
  state: 'active', created_at: timestamp()
})
RETURN f.focus_id AS focus_id, f.goal AS goal, f.room_id AS room_id,
       f.planned_minutes AS planned_minutes, f.started_at AS started_at,
       f.state AS state
"""

_ACTIVE_FOCUS_CYPHER = """
MATCH (f:FocusSession {state: 'active'})
RETURN f.focus_id AS focus_id, f.goal AS goal, f.room_id AS room_id,
       f.planned_minutes AS planned_minutes, f.started_at AS started_at,
       f.state AS state
ORDER BY f.started_at DESC
LIMIT 1
"""

_LIST_FOCUS_CYPHER = """
MATCH (f:FocusSession)
RETURN f.focus_id AS focus_id, f.goal AS goal, f.room_id AS room_id,
       f.planned_minutes AS planned_minutes, f.started_at AS started_at,
       f.ended_at AS ended_at, f.state AS state, f.events AS events,
       f.active_seconds AS active_seconds
ORDER BY f.started_at DESC
LIMIT $limit
"""

_STOP_FOCUS_CYPHER = """
MATCH (f:FocusSession {focus_id: $focus_id, state: 'active'})
SET f.state = 'completed', f.ended_at = $ended_at
RETURN f.focus_id AS focus_id, f.goal AS goal, f.room_id AS room_id,
       f.planned_minutes AS planned_minutes, f.started_at AS started_at,
       f.ended_at AS ended_at, f.state AS state
"""

_FOCUS_METRICS_CYPHER = """
MATCH (e:Event)
WHERE e.span_start < $end AND e.span_end > $start
OPTIONAL MATCH (r:Room)-[:CONTAINS]->(e)
WITH e, collect(DISTINCT r.room_id) AS room_ids
WHERE $room_id IS NULL OR $room_id IN room_ids
RETURN count(DISTINCT e) AS events,
       sum(CASE
         WHEN e.span_end > $end THEN $end ELSE e.span_end END -
           CASE WHEN e.span_start < $start THEN $start ELSE e.span_start END
       ) AS active_seconds,
       collect(DISTINCT e.application) AS applications
"""

_SAVE_FOCUS_SUMMARY_CYPHER = """
MATCH (f:FocusSession {focus_id: $focus_id})
SET f.ended_at = $ended_at, f.events = $events,
    f.active_seconds = $active_seconds
"""

# -- Project arc ----------------------------------------------------------
# Everything else here is day-scoped, so the payoff of long-term memory is
# invisible: you can see today in detail but not how a project actually went
# over a month. These bucket a room's activity into weeks.

_ROOM_WEEKLY_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
WHERE e.span_start >= $start AND e.span_start < $end
WITH e, date(datetime({epochSeconds: toInteger(e.span_start)})) AS day
WITH e, day, day - duration({days: day.dayOfWeek - 1}) AS week_start
RETURN toString(week_start) AS week_start,
       count(DISTINCT e) AS events,
       round(sum(coalesce(e.span_end, 0) - coalesce(e.span_start, 0)) / 60.0) AS active_minutes,
       count(DISTINCT date(datetime({epochSeconds: toInteger(e.span_start)}))) AS active_days,
       collect(DISTINCT e.application)[0..5] AS applications
ORDER BY week_start
"""

_ROOM_WEEK_HIGHLIGHTS_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
WHERE e.span_start >= $start AND e.span_start < $end
OPTIONAL MATCH (e)-[:SUPPORTS]->(c:Claim)
WITH e, c
ORDER BY c.confidence DESC
RETURN collect(DISTINCT c.text)[0..$limit] AS claims,
       collect(DISTINCT e.summary)[0..$limit] AS summaries
"""

_ROOM_WEEK_ENTITIES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)-[:MENTIONS]->(n:Entity)
WHERE e.span_start >= $start AND e.span_start < $end
RETURN n.name AS name, n.type AS type, count(DISTINCT e) AS mentions
ORDER BY mentions DESC
LIMIT $limit
"""

# -- Room hygiene ---------------------------------------------------------
# Auto-created project/activity rooms accumulate forever (one per project name
# ever seen), so the Rooms screen — the primary UX — silently fills with one-off
# folders. These queries surface the junk instead of hiding it.

_ROOM_STATS_CYPHER = """
MATCH (r:Room)
WHERE NOT coalesce(r.archived, false) AND r.kind <> 'daily'
OPTIONAL MATCH (r)-[:CONTAINS]->(e:Event)
OPTIONAL MATCH (r)-[:HAS_NOTE]->(n:RoomNote)
OPTIONAL MATCH (r)-[:HAS_MESSAGE]->(m:RoomMessage)
WITH r,
     count(DISTINCT e) AS events,
     count(DISTINCT n) AS notes,
     count(DISTINCT m) AS messages,
     max(e.span_end) AS last_event_at,
     sum(coalesce(e.span_end, 0) - coalesce(e.span_start, 0)) AS active_seconds
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       coalesce(r.auto, true) AS auto, coalesce(r.pinned, false) AS pinned,
       events, notes, messages, last_event_at,
       round(coalesce(active_seconds, 0) / 60.0) AS active_minutes
ORDER BY events DESC
"""

# Overlap is measured on entities rather than names: two rooms about the same
# work share what was seen in them even when the folders were named differently.
_ROOM_OVERLAP_CYPHER = """
MATCH (a:Room)-[:CONTAINS]->(:Event)-[:MENTIONS]->(n:Entity)<-[:MENTIONS]-(:Event)<-[:CONTAINS]-(b:Room)
WHERE a.room_id < b.room_id
  AND NOT coalesce(a.archived, false) AND NOT coalesce(b.archived, false)
  AND a.kind <> 'daily' AND b.kind <> 'daily'
WITH a, b, count(DISTINCT n) AS shared
WHERE shared >= $min_shared
MATCH (a)-[:CONTAINS]->(:Event)-[:MENTIONS]->(na:Entity)
WITH a, b, shared, count(DISTINCT na) AS a_total
MATCH (b)-[:CONTAINS]->(:Event)-[:MENTIONS]->(nb:Entity)
WITH a, b, shared, a_total, count(DISTINCT nb) AS b_total
WITH a, b, shared, a_total, b_total,
     toFloat(shared) / CASE WHEN a_total < b_total THEN a_total ELSE b_total END AS overlap
WHERE overlap >= $min_overlap
RETURN a.room_id AS room_a, a.name AS name_a, b.room_id AS room_b,
       b.name AS name_b, shared, round(overlap * 100) AS overlap_pct
ORDER BY overlap DESC, shared DESC
LIMIT $limit
"""

_MERGE_ROOMS_CYPHER = """
MATCH (source:Room {room_id: $source_id}), (target:Room {room_id: $target_id})
OPTIONAL MATCH (source)-[c:CONTAINS]->(e:Event)
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
  MERGE (target)-[:CONTAINS]->(e))
WITH source, target, count(DISTINCT e) AS moved_events
OPTIONAL MATCH (source)-[:HAS_NOTE]->(n:RoomNote)
FOREACH (_ IN CASE WHEN n IS NULL THEN [] ELSE [1] END |
  MERGE (target)-[:HAS_NOTE]->(n))
WITH source, target, moved_events, count(DISTINCT n) AS moved_notes
OPTIONAL MATCH (source)-[:HAS_MESSAGE]->(m:RoomMessage)
FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
  MERGE (target)-[:HAS_MESSAGE]->(m))
WITH source, target, moved_events, moved_notes, count(DISTINCT m) AS moved_messages
SET source.archived = true, source.merged_into = target.room_id,
    source.updated_at = $now
RETURN target.room_id AS room_id, moved_events, moved_notes, moved_messages
"""

# Promotion makes an auto room permanent: auto rooms are hygiene's to archive,
# user rooms are not, so this is how the user says "keep this one".
_PROMOTE_ROOM_CYPHER = """
MATCH (r:Room {room_id: $room_id})
SET r.auto = false, r.kind = 'topic', r.pinned = coalesce($pinned, r.pinned),
    r.name = coalesce($name, r.name), r.updated_at = $now
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.auto AS auto, r.pinned AS pinned
"""

_ARCHIVE_ROOMS_CYPHER = """
MATCH (r:Room)
WHERE r.room_id IN $room_ids
SET r.archived = true, r.updated_at = $now
RETURN r.room_id AS room_id
"""

_RECORD_NUDGE_CYPHER = """
CREATE (n:Nudge {
  nudge_id: $nudge_id, text: $text, kind: $kind, focus_id: $focus_id,
  evidence_json: $evidence_json, ts: $ts, feedback: null
})
RETURN n.nudge_id AS nudge_id
"""

_NUDGE_FEEDBACK_CYPHER = """
MATCH (n:Nudge {nudge_id: $nudge_id})
SET n.feedback = $feedback, n.feedback_ts = $ts
RETURN n.nudge_id AS nudge_id, n.text AS text, n.feedback AS feedback
"""

# Only nudges the user actually reacted to are worth showing the model — an
# ignored nudge is ambiguous (unseen? tolerated?), so it teaches nothing.
_NUDGE_FEEDBACK_HISTORY_CYPHER = """
MATCH (n:Nudge)
WHERE n.feedback IS NOT NULL
RETURN n.nudge_id AS nudge_id, n.text AS text, n.feedback AS feedback, n.ts AS ts
ORDER BY n.ts DESC
LIMIT $limit
"""

_LIST_NUDGES_CYPHER = """
MATCH (n:Nudge)
RETURN n.nudge_id AS nudge_id, n.text AS text, n.kind AS kind,
       n.focus_id AS focus_id, n.feedback AS feedback, n.ts AS ts
ORDER BY n.ts DESC
LIMIT $limit
"""

_GET_FOCUS_CYPHER = """
MATCH (f:FocusSession {focus_id: $focus_id})
RETURN f.focus_id AS focus_id, f.goal AS goal, f.room_id AS room_id,
       f.planned_minutes AS planned_minutes, f.started_at AS started_at,
       f.ended_at AS ended_at, f.state AS state, f.events AS events,
       f.active_seconds AS active_seconds, f.on_task_pct AS on_task_pct
"""

# Events attributed to a focus window by time overlap, so a recap can be built
# for sessions that predate this feature and re-run as late events land.
_FOCUS_EVENTS_CYPHER = """
MATCH (e:Event)
WHERE e.span_start < $end AND e.span_end > $start
  AND coalesce(e.summary, '') <> ''
OPTIONAL MATCH (r:Room)-[:CONTAINS]->(e)
WITH e, collect(DISTINCT r.room_id) AS room_ids
WHERE $room_id IS NULL OR $room_id IN room_ids
RETURN e.event_id AS event_id, e.summary AS summary,
       e.application AS application, e.activity_type AS activity_type,
       e.project_id AS project_id,
       e.span_start AS span_start, e.span_end AS span_end
ORDER BY e.span_start
LIMIT $limit
"""

_SAVE_FOCUS_RECAP_CYPHER = """
MATCH (f:FocusSession {focus_id: $focus_id})
SET f.recap = $recap, f.on_task_pct = $on_task_pct,
    f.on_task_minutes = $on_task_minutes, f.off_task_minutes = $off_task_minutes
RETURN f.focus_id AS focus_id
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

# -- Long-term tier ---------------------------------------------------------
#
# `r += $props` rather than a listed SET: the rollup property bag is built in
# memory/consolidation.py, so a new metric there must not require a schema edit
# here. The Day flag records that a compressed copy exists; the events stay put.
_SAVE_ROLLUP_CYPHER = """
MERGE (r:Rollup {rollup_id: $rollup_id})
SET r += $props, r.updated_at = timestamp()
WITH r
UNWIND $dates AS day_date
MATCH (d:Day {date: day_date})
MERGE (r)-[:SUMMARIZES]->(d)
SET d.consolidated = true
"""

# Tier links are derived from the period bounds, so they self-heal: a day
# consolidated after its week was built is picked up the next time the week is.
_LINK_ROLLUP_TIERS_CYPHER = """
MATCH (parent:Rollup {rollup_id: $rollup_id})
MATCH (child:Rollup {kind: $child_kind})
WHERE child.start_date >= parent.start_date
  AND child.end_date <= parent.end_date
MERGE (child)-[:ROLLS_UP_INTO]->(parent)
RETURN count(child) AS linked
"""

_GET_ROLLUP_CYPHER = """
MATCH (r:Rollup {rollup_id: $rollup_id})
RETURN properties(r) AS props
"""

_LIST_ROLLUPS_CYPHER = """
MATCH (r:Rollup)
WHERE ($kind IS NULL OR r.kind = $kind)
  AND ($start IS NULL OR r.end_date >= $start)
  AND ($end IS NULL OR r.start_date <= $end)
RETURN properties(r) AS props
ORDER BY r.end_date DESC, r.kind
LIMIT $limit
"""

_SET_ROLLUP_NARRATIVE_CYPHER = """
MATCH (r:Rollup {rollup_id: $rollup_id})
SET r.narrative = $narrative, r.updated_at = timestamp()
RETURN r.rollup_id AS rollup_id
"""

# Sessions carry project_id as a bare string; this promotes it to a node with a
# lifespan. Dormancy is recomputed every pass, so a revived project goes back to
# 'active' without any manual step.
_SYNC_PROJECTS_CYPHER = """
MATCH (s:Session)
WHERE coalesce(s.project_id, '') <> ''
WITH s.project_id AS project_key, count(s) AS sessions,
     min(s.start) AS first_seen, max(s.end) AS last_seen,
     sum(coalesce(s.active_seconds, 0.0)) AS active_seconds
MERGE (p:Project {project_key: project_key})
  ON CREATE SET p.name = project_key, p.created_at = timestamp()
SET p.sessions = sessions, p.first_seen = first_seen, p.last_seen = last_seen,
    p.active_seconds = active_seconds,
    p.status = CASE WHEN last_seen >= $dormant_before THEN 'active' ELSE 'dormant' END,
    p.updated_at = timestamp()
RETURN count(p) AS projects
"""

_LINK_PROJECT_SESSIONS_CYPHER = """
MATCH (s:Session) WHERE coalesce(s.project_id, '') <> ''
MATCH (p:Project {project_key: s.project_id})
MERGE (s)-[:PART_OF]->(p)
"""

_LIST_PROJECTS_CYPHER = """
MATCH (p:Project)
WHERE $status IS NULL OR p.status = $status
RETURN p.project_key AS project_key, p.name AS name,
       coalesce(p.status, 'active') AS status, p.sessions AS sessions,
       p.first_seen AS first_seen, p.last_seen AS last_seen,
       round(coalesce(p.active_seconds, 0.0) / 60.0, 1) AS active_minutes
ORDER BY p.last_seen DESC
LIMIT $limit
"""

# A goal is what the user said they were doing, which no amount of screen
# watching produces on its own — focus sessions are the only place it is stated.
_SYNC_GOALS_CYPHER = """
MATCH (f:FocusSession)
WHERE coalesce(f.goal, '') <> ''
WITH toLower(trim(f.goal)) AS goal_key, head(collect(f.goal)) AS name,
     count(f) AS sessions, min(f.started_at) AS first_seen,
     max(coalesce(f.ended_at, f.started_at)) AS last_seen,
     sum(coalesce(f.active_seconds, 0.0)) AS active_seconds
MERGE (g:Goal {goal_key: goal_key})
  ON CREATE SET g.created_at = timestamp()
SET g.name = name, g.sessions = sessions, g.first_seen = first_seen,
    g.last_seen = last_seen, g.active_seconds = active_seconds,
    g.updated_at = timestamp()
RETURN count(g) AS goals
"""

_LINK_GOAL_SESSIONS_CYPHER = """
MATCH (f:FocusSession) WHERE coalesce(f.goal, '') <> ''
MATCH (g:Goal {goal_key: toLower(trim(f.goal))})
MERGE (g)-[:PURSUED_IN]->(f)
"""

_LIST_GOALS_CYPHER = """
MATCH (g:Goal)
RETURN g.goal_key AS goal_key, g.name AS name, g.sessions AS sessions,
       g.first_seen AS first_seen, g.last_seen AS last_seen,
       round(coalesce(g.active_seconds, 0.0) / 60.0, 1) AS active_minutes
ORDER BY g.last_seen DESC
LIMIT $limit
"""

# Decay. Selection and deletion are two statements rather than one clever
# clause: the set is small, and a delete driven by an explicit id list is far
# easier to reason about — and to audit — than one fused into an aggregation.
#
# An entity the user merged or renamed into is exempt however old it is: that
# alias is a correction they made by hand, and deleting its target would quietly
# undo their edit.
_STALE_ENTITY_IDS_CYPHER = """
MATCH (n:Entity)
WHERE coalesce(n.memory_status, 'quarantined') = 'quarantined'
  AND NOT EXISTS { MATCH (:EntityAlias {canonical_id: n.entity_id}) }
OPTIONAL MATCH (n)<-[:MENTIONS]-(e:Event)
WITH n, count(e) AS mentions, coalesce(max(e.span_end), 0.0) AS last_seen
WHERE mentions < $min_events AND last_seen < $before
RETURN n.entity_id AS entity_id
"""

_DELETE_ENTITIES_CYPHER = """
MATCH (n:Entity) WHERE n.entity_id IN $ids
DETACH DELETE n
"""

_ORPHAN_CLAIM_IDS_CYPHER = """
MATCH (c:Claim)
WHERE NOT (c)<-[:SUPPORTS]-(:Event)
RETURN c.claim_id AS claim_id
"""

_DELETE_CLAIMS_CYPHER = """
MATCH (c:Claim) WHERE c.claim_id IN $ids
DETACH DELETE c
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
                 r.instructions = $instructions,
                 r.assistant_mode = $assistant_mode,
                 r.execution_profile = $execution_profile,
                 r.agent_tools_json = $agent_tools_json,
                 r.agent_workspace = $agent_workspace,
                 r.agent_request_limit = $agent_request_limit,
                 r.agent_tool_calls_limit = $agent_tool_calls_limit,
                r.color = $color, r.icon = $icon, r.archived = $archived,
                r.pinned = $pinned, r.position = $position,
                r.created_at = timestamp(), r.updated_at = timestamp()
  ON MATCH SET r.name = coalesce(r.name, $name), r.kind = coalesce(r.kind, $kind),
                r.description = coalesce(r.description, $description),
                r.instructions = coalesce(r.instructions, $instructions),
                r.assistant_mode = coalesce(r.assistant_mode, $assistant_mode),
                r.execution_profile = coalesce(r.execution_profile, $execution_profile),
                r.agent_tools_json = coalesce(r.agent_tools_json, $agent_tools_json),
                r.agent_workspace = coalesce(r.agent_workspace, $agent_workspace),
                r.agent_request_limit = coalesce(
                    r.agent_request_limit, $agent_request_limit),
                r.agent_tool_calls_limit = coalesce(
                    r.agent_tool_calls_limit, $agent_tool_calls_limit),
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
       r.instructions AS instructions,
       r.assistant_mode AS assistant_mode,
       coalesce(r.execution_profile, 'investigate') AS execution_profile,
       r.agent_tools_json AS agent_tools_json,
       r.agent_workspace AS agent_workspace,
       coalesce(r.agent_request_limit, 0) AS agent_request_limit,
       coalesce(r.agent_tool_calls_limit, 0) AS agent_tool_calls_limit,
       r.icon AS icon, coalesce(r.archived, false) AS archived,
       coalesce(r.pinned, false) AS pinned, coalesce(r.position, 0) AS position,
       count(e) AS events, max(e.span_end) AS last_active
ORDER BY pinned DESC, position, events DESC, r.name
"""

_ROOM_FEED_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[rel:CONTAINS]->(e:Event)
RETURN e.event_id AS event_id, e.span_start AS span_start, e.span_end AS span_end,
       e.summary AS summary, e.activity_type AS activity_type,
       e.application AS application, rel.assignment AS assignment, rel.manual AS manual,
       coalesce(e.importance, 0.5) AS importance,
       coalesce(e.confidence, 0.5) AS confidence,
       coalesce(e.user_priority,
         CASE WHEN coalesce(e.importance, 0.5) >= 0.75 THEN 'high'
              WHEN coalesce(e.importance, 0.5) < 0.3 THEN 'low'
              ELSE 'normal' END) AS priority,
       CASE WHEN e.user_priority IS NULL THEN 'automatic' ELSE 'user' END AS priority_source,
       coalesce(e.flagged, false) AS flagged, e.flag_reason AS flag_reason
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
CREATE (m:RoomMessage {message_id: $message_id, role: $role, text: $text,
                       citations_json: $citations_json, ts: $ts})
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
RETURN m.message_id AS message_id, m.role AS role, m.text AS text,
       m.citations_json AS citations_json, m.ts AS ts
ORDER BY m.ts DESC
LIMIT $limit
"""

# The chat-context queries below all take the same optional scope: a time window
# and a set of lowercased application values (a null means "no filter").
_ROOM_SCOPE_WHERE = """
WHERE ($start IS NULL OR coalesce(e.span_start, 0) >= $start)
  AND ($end IS NULL OR coalesce(e.span_start, 0) < $end)
  AND ($applications IS NULL
       OR toLower(coalesce(e.application, '')) IN $applications)
"""

_ROOM_CONTEXT_EVENTS_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
""" + _ROOM_SCOPE_WHERE + """
RETURN e.event_id AS event_id, e.span_start AS span_start, e.span_end AS span_end,
       e.summary AS summary, e.application AS application,
       e.activity_type AS activity_type,
       coalesce(e.user_priority,
         CASE WHEN coalesce(e.importance, 0.5) >= 0.75 THEN 'high'
              WHEN coalesce(e.importance, 0.5) < 0.3 THEN 'low'
              ELSE 'normal' END) AS priority,
       coalesce(e.flagged, false) AS flagged
ORDER BY e.span_start DESC
LIMIT $limit
"""

_ROOM_EVENT_IDS_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
""" + _ROOM_SCOPE_WHERE + """
RETURN e.event_id AS event_id
ORDER BY e.span_start DESC
LIMIT $limit
"""

_ROOM_CONTEXT_ENTITIES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
""" + _ROOM_SCOPE_WHERE + """
MATCH (e)-[:MENTIONS]->(n:Entity)
RETURN n.name AS name, count(*) AS c
ORDER BY c DESC, name
LIMIT $limit
"""

_ROOM_CONTEXT_NOTES_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:HAS_NOTE]->(n:RoomNote)
WHERE ($start IS NULL OR coalesce(n.ts, 0) >= $start)
  AND ($end IS NULL OR coalesce(n.ts, 0) < $end)
RETURN n.note_id AS note_id, n.text AS text, n.ts AS ts
ORDER BY n.ts DESC
LIMIT $limit
"""

_ROOM_APPLICATIONS_CYPHER = """
MATCH (r:Room {room_id: $room_id})-[:CONTAINS]->(e:Event)
WHERE coalesce(e.application, '') <> ''
  AND ($start IS NULL OR coalesce(e.span_start, 0) >= $start)
  AND ($end IS NULL OR coalesce(e.span_start, 0) < $end)
RETURN e.application AS application, count(*) AS events,
       max(e.span_end) AS last_active
ORDER BY events DESC, application
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
       e.application AS application, rel.assignment AS assignment, rel.manual AS manual,
       coalesce(e.importance, 0.5) AS importance,
       coalesce(e.confidence, 0.5) AS confidence,
       coalesce(e.user_priority,
         CASE WHEN coalesce(e.importance, 0.5) >= 0.75 THEN 'high'
              WHEN coalesce(e.importance, 0.5) < 0.3 THEN 'low'
              ELSE 'normal' END) AS priority,
       CASE WHEN e.user_priority IS NULL THEN 'automatic' ELSE 'user' END AS priority_source,
       coalesce(e.flagged, false) AS flagged, e.flag_reason AS flag_reason
ORDER BY e.span_start DESC
LIMIT $limit
"""

_CREATE_ROOM_CYPHER = """
CREATE (r:Room {
  room_id: $room_id, name: $name, kind: $kind, auto: $auto,
  matcher_json: $matcher_json, description: $description, color: $color,
  instructions: $instructions, assistant_mode: $assistant_mode,
  execution_profile: $execution_profile,
  agent_tools_json: $agent_tools_json,
  agent_workspace: $agent_workspace,
  agent_request_limit: $agent_request_limit,
  agent_tool_calls_limit: $agent_tool_calls_limit,
  icon: $icon, archived: $archived, pinned: $pinned, position: $position,
  created_at: timestamp(), updated_at: timestamp()
})
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.description AS description, r.color AS color, r.icon AS icon,
       r.instructions AS instructions, r.assistant_mode AS assistant_mode,
       r.execution_profile AS execution_profile,
       r.agent_tools_json AS agent_tools_json,
       r.agent_workspace AS agent_workspace,
       r.agent_request_limit AS agent_request_limit,
       r.agent_tool_calls_limit AS agent_tool_calls_limit,
       r.archived AS archived, r.pinned AS pinned, r.position AS position
"""

_UPDATE_ROOM_CYPHER = """
MATCH (r:Room {room_id: $room_id})
SET r.name = $name, r.description = $description, r.instructions = $instructions, r.color = $color,
    r.assistant_mode = $assistant_mode, r.agent_tools_json = $agent_tools_json,
    r.execution_profile = $execution_profile,
    r.agent_workspace = $agent_workspace,
    r.agent_request_limit = $agent_request_limit,
    r.agent_tool_calls_limit = $agent_tool_calls_limit,
    r.icon = $icon, r.archived = $archived, r.pinned = $pinned,
    r.position = $position, r.matcher_json = $matcher_json,
    r.updated_at = timestamp()
RETURN r.room_id AS room_id, r.name AS name, r.kind AS kind,
       r.description AS description, r.color AS color, r.icon AS icon,
       r.instructions AS instructions, r.assistant_mode AS assistant_mode,
       r.execution_profile AS execution_profile,
       r.agent_tools_json AS agent_tools_json,
       r.agent_workspace AS agent_workspace,
       r.agent_request_limit AS agent_request_limit,
       r.agent_tool_calls_limit AS agent_tool_calls_limit,
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

_LEGACY_AUTO_ROOMS_CYPHER = """
MATCH (r:Room)
WHERE coalesce(r.auto, false)
  AND NOT r.room_id IN $keep
  AND r.kind IN ['activity', 'project', 'camera']
OPTIONAL MATCH (r)-[:HAS_NOTE]->(n:RoomNote)
OPTIONAL MATCH (r)-[:HAS_MESSAGE]->(m:RoomMessage)
RETURN r.room_id AS room_id, r.kind AS kind,
       count(DISTINCT n) AS notes, count(DISTINCT m) AS messages
"""

# Auto links move to the source room; anything the user pinned by hand stays.
_RELINK_ROOM_EVENTS_CYPHER = """
MATCH (old:Room {room_id: $room_id})-[rel:CONTAINS]->(e:Event)
WHERE NOT coalesce(rel.manual, false)
WITH old, rel, e, coalesce(rel.assignment, 'primary') AS assignment
MATCH (target:Room {room_id: $target_room_id})
MERGE (target)-[link:CONTAINS]->(e)
  ON CREATE SET link.assignment = assignment, link.manual = false,
                link.updated_at = timestamp()
DELETE rel
RETURN count(DISTINCT e) AS moved
"""

# An event with no Screen/Cameras link joins one, chosen by memory domain and, for
# older events that have none, by the legacy 'camera:<id>' application token. It
# becomes the primary only when nothing else already is.
_ADOPT_UNROUTED_EVENTS_CYPHER = """
MATCH (e:Event)
WHERE NOT EXISTS {
    MATCH (r:Room)-[:CONTAINS]->(e) WHERE r.kind IN ['camera', 'screen']
}
WITH e,
     CASE WHEN coalesce(e.memory_domain, '') = 'home'
            OR toLower(coalesce(e.application, '')) STARTS WITH 'camera:'
          THEN $camera_room ELSE $screen_room END AS target_id,
     EXISTS {
        MATCH (:Room)-[rel:CONTAINS]->(e)
        WHERE coalesce(rel.assignment, 'primary') = 'primary'
     } AS has_primary
MATCH (target:Room {room_id: target_id})
MERGE (target)-[link:CONTAINS]->(e)
  ON CREATE SET link.assignment =
                    CASE WHEN has_primary THEN 'secondary' ELSE 'primary' END,
                link.manual = false, link.updated_at = timestamp()
RETURN count(DISTINCT e) AS adopted
"""

# Move an auto Screen/Cameras link to the other room when the event's own domain
# (or its legacy 'camera:<id>' app token) says it belongs there. Events that state
# neither are left alone rather than guessed at.
_NORMALIZE_SOURCE_LINKS_CYPHER = """
MATCH (r:Room)-[rel:CONTAINS]->(e:Event)
WHERE r.kind IN ['camera', 'screen'] AND NOT coalesce(rel.manual, false)
WITH e, r, rel, coalesce(e.memory_domain, '') AS domain,
     toLower(coalesce(e.application, '')) AS app
WITH e, r, rel, coalesce(rel.assignment, 'primary') AS assignment,
     CASE
       WHEN domain = 'home' OR app STARTS WITH 'camera:' THEN $camera_room
       WHEN domain = 'personal' THEN $screen_room
       ELSE null
     END AS target_id
WHERE target_id IS NOT NULL AND r.room_id <> target_id
MATCH (target:Room {room_id: target_id})
MERGE (target)-[link:CONTAINS]->(e)
  ON CREATE SET link.assignment = assignment, link.manual = false,
                link.updated_at = timestamp()
DELETE rel
RETURN count(DISTINCT e) AS normalized
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
MATCH (s)-[:HAS_EVENT]->(e:Event)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
WITH s, e ORDER BY e.span_start
RETURN s.session_id AS session_id, s.activity_type AS activity,
       s.application AS application, s.project_id AS project_id,
       s.active_seconds AS active_seconds, s.resume_count AS resume_count,
       s.state AS state, s.start AS start,
       collect(CASE WHEN e IS NULL THEN NULL ELSE {
         event_id: e.event_id, span_start: e.span_start, span_end: e.span_end,
         span_seconds: e.span_seconds, summary: e.summary,
         activity: e.activity_type, boundary: e.boundary_label,
         memory_domain: coalesce(e.memory_domain, $domain)
       } END) AS events
ORDER BY s.start
"""

_DAY_ENTITIES_CYPHER = """
MATCH (d:Day {date: $date})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)-[m:MENTIONS]->(n:Entity)
WHERE $domain IS NULL OR coalesce(
        e.memory_domain,
        CASE WHEN EXISTS {
          MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e)
        } THEN 'home' ELSE 'personal' END
      ) = $domain
RETURN n.entity_id AS entity_id, n.name AS name, n.type AS type, count(m) AS mentions
ORDER BY mentions DESC, name
LIMIT $limit
"""

# Which domain an event's *time* belongs to, for the report queries below.
#
# Productivity is a screen measurement. A camera event is an observation of the
# house — it is not work the user did — so counting its span as active minutes
# inflated every report. `memory_domain` is the primary signal, and membership of
# a camera room overrides it: the pipeline only writes 'home' when log_context is
# exactly 'camera', so a source named 'mobile_camera' or 'camera:front-door' is
# stored as 'personal' while still being routed into Cameras. That routing is the
# thing to trust, and it also covers events written before memory_domain existed.
_EVENT_DOMAIN_EXPR = """CASE
         WHEN coalesce(e.memory_domain, 'personal') = 'home'
              OR EXISTS { MATCH (:Room {kind: 'camera'})-[:CONTAINS]->(e) }
         THEN 'home' ELSE 'personal' END"""

# Appended inside an existing WHERE. $domain NULL means "every source".
_DOMAIN_CLAUSE = f"\n  AND ($domain IS NULL OR ({_EVENT_DOMAIN_EXPR}) = $domain)"

# Day.date is ISO, so a lexicographic range is a chronological one. All the
# report queries span [$start, $end] inclusive; a single day passes start == end.
_RANGE_TOTALS_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(s:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end{_DOMAIN_CLAUSE}
RETURN sum(e.span_seconds) AS active_seconds, count(e) AS events,
       count(DISTINCT s) AS sessions, max(e.span_seconds) AS longest_block,
       count(DISTINCT d.date) AS active_days,
       sum(CASE WHEN e.boundary_label <> 'append' THEN 1 ELSE 0 END) AS switches
"""

# activity_type is coalesced rather than dropped: a null key is a hole in a
# chart's category axis, and the time was still spent.
_RANGE_BY_ACTIVITY_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end{_DOMAIN_CLAUSE}
RETURN coalesce(e.activity_type, 'other') AS activity,
       round(sum(e.span_seconds) / 60.0, 1) AS minutes,
       count(e) AS events
ORDER BY minutes DESC
"""

_RANGE_BY_APP_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end
  AND e.application IS NOT NULL{_DOMAIN_CLAUSE}
RETURN e.application AS app, round(sum(e.span_seconds) / 60.0, 1) AS minutes
ORDER BY minutes DESC
LIMIT 60
"""

_RANGE_BY_PROJECT_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(s:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end
  AND s.project_id IS NOT NULL{_DOMAIN_CLAUSE}
RETURN s.project_id AS project, round(sum(e.span_seconds) / 60.0, 1) AS minutes
ORDER BY minutes DESC
LIMIT 60
"""

# One row per (day, activity) — the shape the per-activity trend charts plot.
_ACTIVITY_SERIES_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end{_DOMAIN_CLAUSE}
RETURN d.date AS date, coalesce(e.activity_type, 'other') AS activity,
       round(sum(e.span_seconds) / 60.0, 1) AS minutes,
       count(e) AS events
ORDER BY date, minutes DESC
"""

# Raw spans for the hour-of-day view. Bucketing is done in Python rather than in
# Cypher because the answer is "what hour was it where he was sitting" — the
# graph stores epoch seconds, and converting them in the database would bucket
# the day in UTC and shift every evening's work into the small hours.
_EVENT_SPANS_CYPHER = f"""
MATCH (d:Day)-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
WHERE d.date >= $start AND d.date <= $end{_DOMAIN_CLAUSE}
  AND e.span_start IS NOT NULL
RETURN d.date AS date, e.span_start AS span_start,
       coalesce(e.span_seconds, 0.0) AS span_seconds,
       coalesce(e.activity_type, 'other') AS activity
ORDER BY span_start
LIMIT $limit
"""

# Written reports: one node per (period, end date), replaced in place on rewrite.
_SAVE_WRITTEN_REPORT_CYPHER = """
MERGE (r:WrittenReport {report_id: $report_id})
SET r.end_date = $date, r.start_date = $start_date, r.period = $period,
    r.headline = $headline, r.overall_score = $overall_score,
    r.score_names = $score_names, r.body = $body,
    r.model = $model, r.effort = $effort, r.written_at = $ts
WITH r
MERGE (d:Day {date: $date})
MERGE (d)-[:HAS_REPORT]->(r)
RETURN r.report_id AS report_id, r.end_date AS end_date,
       r.overall_score AS overall_score, r.written_at AS written_at
"""

_WRITTEN_REPORTS_CYPHER = """
MATCH (r:WrittenReport)
WHERE r.period = $period AND r.end_date >= $start AND r.end_date <= $end
RETURN r.report_id AS report_id, r.end_date AS end_date,
       r.start_date AS start_date, r.period AS period,
       r.headline AS headline, r.overall_score AS overall_score,
       r.model AS model, r.effort AS effort, r.written_at AS written_at,
       r.body AS body
ORDER BY r.end_date DESC
LIMIT $limit
"""

_DAY_CLAIMS_CYPHER = f"""
MATCH (d:Day {{date: $date}})-[:HAS_SESSION]->(:Session)-[:HAS_EVENT]->(e:Event)
      -[:SUPPORTS]->(c:Claim)
WHERE true{_DOMAIN_CLAUSE}
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
