"""Step 6 — build sessions + events from a recorded capture (offline, no VLM).

Runs the Step-5 boundary detector over a capture folder, turns the labeled
stream into sessions and events via SessionManager, and writes
data/debug/sessions.jsonl + data/debug/events.jsonl plus a timeline print.

    python -m tools.sessions data/captures/run_20260723_131738

activity_type is joined from data/debug/extractions.jsonl by batch when present,
otherwise inferred from the domain profile.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from memory.debug import DEBUG_DIR, write_jsonl
from memory.pipeline import MemoryPipeline


def load_observations(capture_dir):
    manifest = os.path.join(capture_dir, "observations.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"no observations.jsonl found in {capture_dir}")
    raw = open(manifest, encoding="utf-8").read()
    decoder = json.JSONDecoder()
    obs, i, n = [], 0, len(raw)
    while i < n:
        while i < n and raw[i] in " \r\n\t":
            i += 1
        if i >= n:
            break
        rec, end = decoder.raw_decode(raw, i)
        obs.append(rec)
        i = end
    return obs


def load_extractions_by_batch():
    """Map batch -> full extraction dict (entities/claims/activity_type)."""
    path = os.path.join(DEBUG_DIR, "extractions.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "batch" in r:
            out[r["batch"]] = r
    return out


def _last_frame(capture_dir, obs):
    frames = obs.get("frames") or []
    if not frames:
        return None
    return np.asarray(Image.open(os.path.join(capture_dir, frames[-1])).convert("RGB"),
                      dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Build sessions/events from a capture.")
    parser.add_argument("capture_dir")
    parser.add_argument("--expected", type=float, default=60.0)
    parser.add_argument("--neo4j", action="store_true",
                        help="also dual-write sessions/events to Neo4j")
    parser.add_argument("--qdrant", action="store_true",
                        help="also embed event summaries into Qdrant (event-scoped)")
    args = parser.parse_args()

    observations = load_observations(args.capture_dir)
    extractions_by_batch = load_extractions_by_batch()
    print(f"Loaded {len(observations)} observation(s); "
          f"{len(extractions_by_batch)} extraction(s) from extractions.jsonl")

    # Offline bulk mode: no sinks — build everything, write once at the end.
    pipeline = MemoryPipeline(id_strategy="counter", expected_seconds=args.expected)
    for obs in observations:
        pipeline.ingest({
            "timestamp": obs["timestamp"],
            "window_titles": obs.get("window_titles", []),
            "process_names": obs.get("process_names", []),
            "repr_frame": _last_frame(args.capture_dir, obs),
            "extraction": extractions_by_batch.get(obs.get("batch"), {}),
        })
    pipeline.finalize()

    mgr = pipeline.manager

    # Fresh per-run outputs.
    for name in ("sessions", "events"):
        p = os.path.join(DEBUG_DIR, f"{name}.jsonl")
        if os.path.exists(p):
            os.remove(p)
    for s in mgr.ordered_sessions():
        write_jsonl("sessions", s)
    for e in mgr.events:
        write_jsonl("events", e)

    # Timeline print.
    print(f"\n{len(mgr.sessions)} session(s), {len(mgr.events)} event(s):\n")
    for s in mgr.ordered_sessions():
        proj = f" project={s.project_id}" if s.project_id else ""
        print(f"[{s.session_id}] {s.activity_type}/{s.application}{proj} "
              f"state={s.state} active={s.active_seconds:.0f}s resumes={s.resume_count}")
        for e in mgr.events:
            if e.session_id == s.session_id:
                print(f"    {e.event_id}  {e.span_seconds:6.0f}s  {e.boundary_label:9}  "
                      f"{e.summary[:55]}")
    print(f"\nWrote {os.path.join(DEBUG_DIR, 'sessions.jsonl')} and events.jsonl")

    if args.neo4j:
        from dotenv import load_dotenv
        from memory.stores.neo4j_store import Neo4jStore
        load_dotenv()

        knowledge = pipeline.knowledge_items()

        with Neo4jStore() as store:
            store.verify()
            store.apply_schema()
            n_s, n_e = store.write_timeline(mgr.ordered_sessions(), mgr.events)
            print(f"\nDual-write to Neo4j: {n_s} sessions, {n_e} events "
                  f"(db={store.database}). Graph totals: {store.counts()}")

            n_ent, n_claim = store.write_event_knowledge(knowledge)
            print(f"Knowledge write: {n_ent} mentions, {n_claim} claims. "
                  f"Totals: {store.knowledge_counts()}")
            print(f"Orphan claims (no evidence event): {store.orphan_claims()} (expect 0)")

            # Route events into rooms (auto-first).
            entity_types_by_event = {
                it["event_id"]: [e["type"] for e in it["entities"]] for it in knowledge}
            room_items = [{
                "event_id": e.event_id, "activity_type": e.activity_type,
                "application": e.application, "project_id": e.project_id,
                "summary": e.summary,
                "entity_types": entity_types_by_event.get(e.event_id, []),
            } for e in mgr.events]
            room_stats = store.assign_rooms(room_items)
            print(f"Rooms: {room_stats}")
            for r in store.list_rooms():
                print(f"  [{r['room_id']:<28}] {r['name'][:24]:<24} {r['kind']:<9} "
                      f"events={r['events']}")

            rows = store.events_today()
            print(f"\n§14 events-from-today (span-aware): {len(rows)} row(s)")
            for r in rows[:12]:
                print(f"  {r['session']:>7} {r['activity']:>9}/{r['application']:<18} "
                      f"{r['event']:>6} span={r['span_seconds']:.0f}s")

            # Debug handle: entities for the first session, with co_presence.
            first_sid = mgr.ordered_sessions()[0].session_id if mgr.sessions else None
            if first_sid:
                print(f"\nEntities for {first_sid} (with co_presence):")
                for r in store.entities_for_session(first_sid)[:15]:
                    print(f"  {r['name'][:34]:<34} {r['type']:<10} "
                          f"role={r['role']:<7} co={r['co_presence']:<10} "
                          f"conf={r['confidence']}")

    if args.qdrant:
        from dotenv import load_dotenv
        from qdrant_client import QdrantClient
        from vector_store.activity_logger import ActivityLogger
        load_dotenv()

        event_texts = pipeline.event_texts()
        events_by_id = {e.event_id: e for e in mgr.events}

        client = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_db"))
        try:
            logger_ = ActivityLogger(client=client)
            n = 0
            for eid, info in event_texts.items():
                ev = events_by_id.get(eid)
                if ev is None or not info["text"]:
                    continue
                logger_.log_event(
                    summary=info["text"], event_id=eid, session_id=ev.session_id,
                    span_start=ev.span_start, span_end=ev.span_end,
                    profile=info["profile"], timestamp=ev.span_start,
                )
                n += 1
            print(f"\nQdrant: embedded {n} event summary(ies) (event-scoped).")
        finally:
            client.close()


if __name__ == "__main__":
    main()
