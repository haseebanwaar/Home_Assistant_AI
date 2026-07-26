import threading
import unittest
import uuid

# neo4j, qdrant_client and dotenv are stubbed in conftest.py.
from memory.stores import neo4j_store as store_module
from memory.stores.neo4j_store import Neo4jStore
from memory.pipeline import MemoryPipeline
from vector_store.activity_logger import ActivityLogger


class FakeSearchStore(Neo4jStore):
    def __init__(self):
        pass

    def run(self, query, **params):
        if query == store_module._SEARCH_EVENTS_CYPHER:
            return [{
                "kind": "event", "id": "event-1", "title": "Editor",
                "text": "Worked on memory explorer", "ts": 20, "rooms": [],
                "score": 100,
            }]
        if query == store_module._SEARCH_ENTITIES_CYPHER:
            return [{
                "kind": "entity", "id": "memory-explorer",
                "title": "Memory Explorer", "text": "project", "ts": None,
                "rooms": [], "score": 150,
            }]
        return []


class FakeQdrant:
    def __init__(self):
        self.deleted = None

    def collection_exists(self, _name):
        return True

    def delete(self, collection_name, points_selector):
        self.deleted = (collection_name, points_selector)


class FakeAssistantStore(Neo4jStore):
    def __init__(self):
        self.saved_summary = None

    def run(self, query, **params):
        if query == store_module._GET_CONVERSATION_CYPHER:
            return [{
                "conversation_id": "c1", "title": "Test", "scope": "all",
                "room_id": None, "from_ts": None, "to_ts": None,
            }]
        if query == store_module._CONVERSATION_MESSAGES_CYPHER:
            return [{
                "message_id": "m1", "role": "assistant", "text": "Answer [1]",
                "citations_json": '[{"number": 1, "kind": "event", "id": "e1"}]',
                "ts": 10,
            }]
        if query == store_module._STOP_FOCUS_CYPHER:
            return [{
                "focus_id": "f1", "goal": "Write tests", "room_id": None,
                "planned_minutes": 25, "started_at": 100, "ended_at": 200,
                "state": "completed",
            }]
        if query == store_module._FOCUS_METRICS_CYPHER:
            return [{
                "events": 3, "active_seconds": 90,
                "applications": ["Editor"],
            }]
        if query == store_module._SAVE_FOCUS_SUMMARY_CYPHER:
            self.saved_summary = params
            return []
        return []


class FakeEventTriageStore(Neo4jStore):
    def __init__(self):
        self.last_update = None

    def run(self, query, **params):
        if query == store_module._UPDATE_EVENT_METADATA_CYPHER:
            self.last_update = params
            return [{
                "event_id": params["event_id"],
                "priority": params["priority"] or "normal",
                "priority_source": "user",
                "flagged": params["flagged"],
                "flag_reason": params["flag_reason"],
                "reviewed_at": 1,
            }]
        if query in {store_module._ROOM_FEED_CYPHER,
                     store_module._ROOM_CONTEXT_EVENTS_CYPHER}:
            return [
                {
                    "event_id": "important", "span_start": 30, "span_end": 40,
                    "summary": "Fixed the camera reconnect loop",
                    "activity_type": "coding", "application": "Editor",
                    "assignment": "primary", "manual": False,
                    "importance": .9, "confidence": .8, "priority": "high",
                    "priority_source": "automatic", "flagged": False,
                    "flag_reason": None,
                },
                {
                    "event_id": "noise", "span_start": 10, "span_end": 20,
                    "summary": "Changed windows briefly",
                    "activity_type": "other", "application": "Desktop",
                    "assignment": "primary", "manual": False,
                    "importance": .1, "confidence": .5, "priority": "low",
                    "priority_source": "automatic", "flagged": False,
                    "flag_reason": None,
                },
            ]
        if query in {
            store_module._ROOM_NOTES_CYPHER,
            store_module._ROOM_CONTEXT_NOTES_CYPHER,
            store_module._ROOM_MESSAGES_CYPHER,
            store_module._ROOM_CONTEXT_ENTITIES_CYPHER,
        }:
            return []
        return []


class MemoryExplorerTests(unittest.TestCase):
    def test_capture_sources_write_to_distinct_memory_domains(self):
        batch = {
            "timestamp": 100.0,
            "window_titles": ["Observation"],
            "process_names": ["source"],
            "extraction": {"summary": "Observed activity"},
        }

        personal = MemoryPipeline(log_context="screen").ingest(batch)
        home = MemoryPipeline(log_context="camera").ingest(batch)

        self.assertEqual(personal.current_event.memory_domain, "personal")
        self.assertEqual(home.current_event.memory_domain, "home")

    def test_search_combines_types_and_orders_by_score(self):
        store = FakeSearchStore()

        results = store.memory_search(
            "memory", kinds=["event", "entity"], limit=10)

        self.assertEqual([item["kind"] for item in results], ["entity", "event"])
        self.assertEqual(results[0]["id"], "memory-explorer")

    def test_search_respects_requested_kinds(self):
        store = FakeSearchStore()

        results = store.memory_search("memory", kinds=["event"], limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["kind"], "event")

    def test_event_vector_delete_uses_log_event_deterministic_id(self):
        client = FakeQdrant()
        logger = ActivityLogger.__new__(ActivityLogger)
        logger.client = client
        logger.collection_name = "activity_log"
        logger._lock = threading.Lock()  # normally set by __init__, bypassed here

        deleted = logger.delete_event("event-42")

        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "event:event-42"))
        self.assertEqual(deleted, expected)
        self.assertEqual(client.deleted, ("activity_log", [expected]))

    def test_conversation_messages_decode_citations(self):
        store = FakeAssistantStore()

        conversation = store.get_conversation("c1")

        self.assertEqual(conversation["messages"][0]["citations"][0]["id"], "e1")
        self.assertNotIn("citations_json", conversation["messages"][0])

    def test_focus_stop_calculates_and_persists_metrics(self):
        store = FakeAssistantStore()

        result = store.stop_focus_session("f1")

        self.assertEqual(result["metrics"]["events"], 3)
        self.assertEqual(store.saved_summary["active_seconds"], 90)

    def test_event_triage_is_saved_as_a_user_override(self):
        store = FakeEventTriageStore()

        result = store.update_event_metadata(
            "event-1", priority="low", flagged=True,
            flag_reason="Repeated window switch")

        self.assertEqual(result["priority"], "low")
        self.assertTrue(result["flagged"])
        self.assertEqual(store.last_update["flag_reason"],
                         "Repeated window switch")

    def test_event_triage_rejects_unknown_priority(self):
        store = FakeEventTriageStore()

        with self.assertRaises(ValueError):
            store.update_event_metadata("event-1", priority="urgent")

    def test_room_feed_can_filter_events_by_priority(self):
        store = FakeEventTriageStore()

        results = store.room_feed_full(
            "project:home-assistant", priorities=["high"])

        self.assertEqual([item["event_id"] for item in results], ["important"])
        self.assertEqual(results[0]["priority_source"], "automatic")

    def test_low_priority_events_do_not_ground_room_chat(self):
        store = FakeEventTriageStore()

        context = store.room_context("project:home-assistant")

        # Each line is prefixed with the app/camera that saw it.
        self.assertEqual(
            context["events"], ["[Editor] Fixed the camera reconnect loop"])


if __name__ == "__main__":
    unittest.main()
