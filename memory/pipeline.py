"""Step-a — the unified memory pipeline (live + offline share this).

`MemoryPipeline.ingest(batch)` turns one processed minute-batch into timeline +
knowledge updates: it scores a boundary vs the previous batch, feeds the
SessionManager, and accumulates per-event entities/claims/text. The SAME object
is driven by the live capture loop and by the offline replay tool, so the live
path is verifiable offline.

Store writes are optional and fail-safe:
- offline: build everything, then bulk-write once (identical to the old tool);
- live: pass sinks so each ingest upserts incrementally (idempotent).

A batch is a dict:
    {timestamp, window_titles[], process_names[], repr_frame(np|None),
     extraction: {summary, activity_type, entities[], claims[], selected_profile}}
"""
from __future__ import annotations

import hashlib
import logging

from memory.boundaries.boundary_detector import (
    compute_visual_change,
    normalize_inactivity,
    score_boundary,
)
from memory.context import app_of, normalize_name, profile_name, project_of, title_of
from memory.sessions.session_manager import SessionManager

logger = logging.getLogger("home_assistant")


class MemoryPipeline:
    def __init__(self, id_strategy="counter", expected_seconds=60.0,
                 neo4j_store=None, activity_logger=None, jsonl=False,
                 log_context="screen", notification_sink=None):
        self.manager = SessionManager(id_strategy=id_strategy)
        self.expected_seconds = expected_seconds
        self.neo4j = neo4j_store
        self.activity_logger = activity_logger
        self.jsonl = jsonl  # rewrite data/debug/sessions.jsonl + events.jsonl live
        # Qdrant context label for events from this pipeline ("screen"/"camera").
        self.log_context = log_context
        self.notification_sink = notification_sink

        self._prev_ctx = None
        self._prev_frame = None
        # Naming corrections fed back into the extraction prompt. Cached with a
        # TTL so the capture loop doesn't hit the graph on every batch.
        self._naming_hints = []
        self._naming_hints_at = 0.0
        self.naming_hints_ttl = 300.0
        # Per-event accumulators (event_id -> ...).
        self._ev_entities = {}   # -> {norm: rec}
        self._ev_claims = {}     # -> {claim_id: rec}
        self._ev_text = {}       # -> {"parts": [...], "profile": str}
        self._ev_scored = set()  # events with at least one extractor score

    def naming_hints(self):
        """[(wrong_name, canonical_name)] the user has corrected before.

        Fed into the extraction prompt so a merge/rename fixes future captures
        instead of only repairing the ones already stored.
        """
        import time as _time
        if self.neo4j is None:
            return []
        now = _time.time()
        if now - self._naming_hints_at >= self.naming_hints_ttl:
            self._naming_hints_at = now
            try:
                self._naming_hints = self.neo4j.canonical_name_hints(limit=20)
            except Exception as exc:
                logger.debug("naming hints unavailable: %s", exc)
                self._naming_hints = []
        return self._naming_hints

    # -- ingest ------------------------------------------------------------
    def ingest(self, batch):
        """Process one batch; returns the ObserveResult."""
        ctx = {"window_titles": batch.get("window_titles") or [],
               "process_names": batch.get("process_names") or []}
        ext = batch.get("extraction") or {}
        ts = batch["timestamp"]
        frame = batch.get("repr_frame")

        # Boundary vs previous batch.
        if self._prev_ctx is None:
            label = "boundary"
        else:
            app_changed = app_of(ctx) != app_of(self._prev_ctx)
            title_changed = title_of(ctx) != title_of(self._prev_ctx)
            vis = compute_visual_change(self._prev_frame, frame)
            idle = normalize_inactivity(ts - self._prev_ctx["_ts"],
                                        expected_seconds=self.expected_seconds)
            label = score_boundary(app_changed, title_changed, vis, idle).label

        activity = ext.get("activity_type") or ("coding" if profile_name(ctx) == "coding" else "browsing")
        # Project is decided by the VLM (ext['project']); it returns null for
        # system tools / bare terminals, so no spurious project rooms form. Old
        # captures without the field fall back to the title heuristic.
        project = ext.get("project")
        if project is None and "project" not in ext:
            project = project_of(ctx)
        result = self.manager.observe(
            timestamp=ts, activity_type=activity,
            application=app_of(ctx), project_id=project,
            boundary_label=label, summary=title_of(ctx),
        )
        result.current_event.memory_domain = (
            "home" if self.log_context == "camera" else "personal")
        # Preserve the extractor's usefulness signal on the timeline event.
        # Multiple observations can contribute to one event, so retain the
        # strongest importance/confidence seen during that span.
        try:
            importance = float(ext.get("importance", 0.5))
            confidence = float(ext.get("confidence", 0.5))
            if result.current_event.event_id in self._ev_scored:
                result.current_event.importance = max(
                    result.current_event.importance, importance)
                result.current_event.confidence = max(
                    result.current_event.confidence, confidence)
            else:
                result.current_event.importance = importance
                result.current_event.confidence = confidence
                self._ev_scored.add(result.current_event.event_id)
        except (TypeError, ValueError):
            pass

        self._accumulate(result.current_event.event_id, ext, ctx)
        event_text = self.event_texts().get(result.current_event.event_id, {}).get("text")
        if event_text:
            result.current_event.summary = event_text[-2000:]

        # Live incremental upsert (no-op sinks -> offline bulk mode).
        self._upsert(result)
        if self.jsonl:
            self._dump_jsonl()

        ctx["_ts"] = ts
        self._prev_ctx = ctx
        self._prev_frame = frame
        return result

    def finalize(self):
        self.manager.finalize(tail_seconds=self.expected_seconds)
        # Final upsert of the last (now-closed) event.
        if self.manager.events:
            last = self.manager.events[-1]
            sess = self.manager.sessions_by_id.get(last.session_id)
            if sess is not None:
                self._upsert_event(sess, last)
        if self.jsonl:
            self._dump_jsonl()
        # Steps 12 & 14: resolve aliases, consolidate promotion status, and
        # rebuild the derived shortcut edges over the accumulated graph.
        if self.neo4j is not None:
            try:
                n = self.neo4j.resolve_entities()
                status = self.neo4j.consolidate()
                shortcuts = self.neo4j.rebuild_shortcuts()
                logger.info("Consolidation: %d alias candidate(s); entities %s; shortcuts %s.",
                            n, status, shortcuts)
            except Exception as exc:
                logger.warning("consolidation failed: %s", exc)

    def _dump_jsonl(self):
        """Rewrite sessions.jsonl + events.jsonl from current manager state."""
        import json
        import os
        from memory.debug import DEBUG_DIR
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            for name, rows in (("sessions", self.manager.ordered_sessions()),
                               ("events", self.manager.events)):
                path = os.path.join(DEBUG_DIR, f"{name}.jsonl")
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    for r in rows:
                        f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("jsonl dump failed: %s", exc)

    # -- accumulation ------------------------------------------------------
    def _accumulate(self, event_id, ext, ctx):
        ents = ext.get("entities") or []
        shares_frame = len(ents) > 1
        acc = self._ev_entities.setdefault(event_id, {})
        for en in ents:
            norm = normalize_name(en.get("name"))
            if not norm:
                continue
            conf = float(en.get("confidence", 0.5))
            rec = acc.get(norm)
            if rec is None:
                acc[norm] = {"entity_id": norm, "name": en.get("name"),
                             "type": en.get("type", "other"), "confidence": conf,
                             "same_frame": shares_frame}
            else:
                rec["confidence"] = max(rec["confidence"], conf)
                rec["same_frame"] = rec["same_frame"] or shares_frame

        claims = self._ev_claims.setdefault(event_id, {})
        for cl in ext.get("claims") or []:
            text = (cl.get("text") or "").strip()
            if not text:
                continue
            cid = hashlib.sha1(f"{event_id}|{text}".encode("utf-8")).hexdigest()[:16]
            if cid not in claims:
                claims[cid] = {"claim_id": cid, "text": text,
                               "confidence": float(cl.get("confidence", 0.5))}

        txt = self._ev_text.setdefault(event_id, {"parts": [], "profile": None})
        s = (ext.get("summary") or "").strip()
        if s and (not txt["parts"] or txt["parts"][-1] != s):
            txt["parts"].append(s)
            # Keep long-running sessions useful without growing forever.
            txt["parts"] = txt["parts"][-6:]
        txt["profile"] = txt["profile"] or ext.get("selected_profile") or profile_name(ctx)

    def _entities_for(self, event_id):
        acc = self._ev_entities.get(event_id, {})
        if not acc:
            return []
        top = max(acc.values(), key=lambda r: r["confidence"])["entity_id"]
        return [{
            "entity_id": r["entity_id"], "name": r["name"], "type": r["type"],
            "confidence": round(r["confidence"], 3),
            "role": "primary" if r["entity_id"] == top else "mention",
            "co_presence": "same_frame" if r["same_frame"] else "same_event",
        } for r in acc.values()]

    def knowledge_items(self):
        """Same shape as the old aggregate_knowledge()."""
        items = []
        for event_id in self._ev_entities.keys() | self._ev_claims.keys():
            items.append({
                "event_id": event_id,
                "entities": self._entities_for(event_id),
                "claims": list(self._ev_claims.get(event_id, {}).values()),
            })
        return items

    def event_texts(self):
        """event_id -> {text, profile} for Qdrant embedding."""
        return {eid: {"text": " ".join(v["parts"]), "profile": v["profile"]}
                for eid, v in self._ev_text.items()}

    # -- store upserts (live) ---------------------------------------------
    def _upsert(self, result):
        if self.neo4j is None and self.activity_logger is None:
            return  # offline bulk mode: caller writes at the end
        # A closed event is now final; upsert it fully.
        if result.closed_event is not None:
            closed_sess = self.manager.sessions_by_id.get(result.closed_event.session_id)
            if closed_sess is not None:
                self._upsert_event(closed_sess, result.closed_event)
        # Always upsert the current (open) session + event.
        self._upsert_event(result.session, result.current_event)

    def _upsert_event(self, session, event):
        try:
            if self.neo4j is not None:
                self.neo4j.write_timeline([session], [event])
                entities = self._entities_for(event.event_id)
                self.neo4j.write_event_knowledge([{
                    "event_id": event.event_id,
                    "entities": entities,
                    "claims": list(self._ev_claims.get(event.event_id, {}).values()),
                }])
                # Route the event into its room(s) (auto-first).
                self.neo4j.assign_rooms([{
                    "event_id": event.event_id,
                    # Decides the Screen vs Cameras room.
                    "source": self.log_context,
                    "activity_type": event.activity_type,
                    "application": event.application,
                    "project_id": event.project_id,
                    "summary": event.summary,
                    "entity_types": [e["type"] for e in entities],
                }])
            if self.activity_logger is not None:
                info = self.event_texts().get(event.event_id, {})
                if info.get("text"):
                    self.activity_logger.log_event(
                        summary=info["text"], event_id=event.event_id,
                        session_id=session.session_id, span_start=event.span_start,
                        span_end=event.span_end, profile=info.get("profile"),
                        timestamp=event.span_start, context=self.log_context,
                    )
        except Exception as exc:
            # Never let a store failure kill the capture loop.
            logger.warning("MemoryPipeline upsert failed (continuing): %s", exc)
        if self.notification_sink is not None:
            try:
                info = self.event_texts().get(event.event_id, {})
                self.notification_sink({
                    **event.model_dump(),
                    "summary": info.get("text") or event.summary,
                    "source": self.log_context,
                    "timestamp": event.span_end,
                })
            except Exception as exc:
                logger.warning("Notification classification failed (continuing): %s", exc)
