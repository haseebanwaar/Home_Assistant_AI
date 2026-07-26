"""Room chat scope: source chips and time windows must actually narrow context.

A filter the user can see but the assistant ignores is worse than no filter, so
these pin down that the selection reaches the query and the prompt.
"""
import unittest

from memory.stores import neo4j_store as store_module
from memory.stores.neo4j_store import Neo4jStore


def _event(event_id, application, ts, summary=None, priority="normal",
           flagged=False):
    return {"event_id": event_id, "application": application, "span_start": ts,
            "span_end": ts + 60, "summary": summary or f"did something in {application}",
            "activity_type": "coding", "priority": priority,
            "priority_source": "automatic", "flagged": flagged,
            "flag_reason": None, "importance": 0.6, "confidence": 0.6,
            "assignment": "primary", "manual": False}


class FakeRoomStore(Neo4jStore):
    """Records every query + params, and replays canned rows."""

    def __init__(self, events=None, notes=None, messages=None, entities=None):
        self._events = events or []
        self._notes = notes or []
        self._messages = messages or []
        self._entities = entities or []
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if query in (store_module._ROOM_CONTEXT_EVENTS_CYPHER,
                     store_module._ROOM_FEED_CYPHER,
                     store_module._ROOM_FEED_DATED_CYPHER):
            return [dict(e) for e in self._events]
        if query == store_module._ROOM_EVENT_IDS_CYPHER:
            return [{"event_id": e["event_id"]} for e in self._events]
        if query in (store_module._ROOM_NOTES_CYPHER,
                     store_module._ROOM_CONTEXT_NOTES_CYPHER):
            return [dict(n) for n in self._notes]
        if query == store_module._ROOM_MESSAGES_CYPHER:
            return [dict(m) for m in self._messages]
        if query in (store_module._ROOM_ENTITIES_CYPHER,
                     store_module._ROOM_CONTEXT_ENTITIES_CYPHER):
            return [dict(e) for e in self._entities]
        if query == store_module._ROOM_APPLICATIONS_CYPHER:
            return [{"application": "pycharm64.exe", "events": 12,
                     "last_active": 300.0},
                    {"application": "opera.exe", "events": 4, "last_active": 200.0}]
        return []

    def params_for(self, query):
        return next(p for q, p in self.calls if q == query)


class RoomContextScopeTests(unittest.TestCase):
    def test_selected_sources_are_pushed_into_the_event_query(self):
        store = FakeRoomStore(events=[_event("e1", "pycharm64.exe", 100)])

        store.room_context("screen", applications=["PyCharm64.exe", "opera.exe"])

        params = store.params_for(store_module._ROOM_CONTEXT_EVENTS_CYPHER)
        # Lowercased, because that is how the graph stores an application.
        self.assertEqual(params["applications"], ["pycharm64.exe", "opera.exe"])

    def test_no_selection_means_no_application_filter(self):
        store = FakeRoomStore(events=[_event("e1", "pycharm64.exe", 100)])

        store.room_context("screen")

        params = store.params_for(store_module._ROOM_CONTEXT_EVENTS_CYPHER)
        self.assertIsNone(params["applications"])

    def test_empty_selection_is_not_treated_as_a_filter(self):
        """An all-deselected chip row must not silently hide the whole room."""
        store = FakeRoomStore(events=[_event("e1", "pycharm64.exe", 100)])

        store.room_context("screen", applications=[])

        self.assertIsNone(
            store.params_for(store_module._ROOM_CONTEXT_EVENTS_CYPHER)["applications"])

    def test_time_window_is_pushed_into_every_context_query(self):
        store = FakeRoomStore(events=[_event("e1", "opera.exe", 500)],
                              notes=[{"note_id": "n1", "text": "note", "ts": 500}],
                              entities=[{"name": "Qdrant", "c": 2}])

        store.room_context("screen", start=100.0, end=900.0)

        for query in (store_module._ROOM_CONTEXT_EVENTS_CYPHER,
                      store_module._ROOM_CONTEXT_NOTES_CYPHER,
                      store_module._ROOM_CONTEXT_ENTITIES_CYPHER):
            params = store.params_for(query)
            self.assertEqual((params["start"], params["end"]), (100.0, 900.0),
                             msg=f"window missing from {query[:40]!r}")

    def test_activity_lines_are_prefixed_with_their_source(self):
        """The model has to be able to attribute what it is told."""
        store = FakeRoomStore(events=[_event("e1", "ipc-a22e-g", 100,
                                             summary="a van parked outside")])

        ctx = store.room_context("camera")

        self.assertEqual(ctx["events"], ["[ipc-a22e-g] a van parked outside"])

    def test_low_priority_and_flagged_events_stay_out_of_the_prompt(self):
        store = FakeRoomStore(events=[
            _event("keep", "opera.exe", 300),
            _event("noise", "opera.exe", 200, priority="low"),
            _event("later", "opera.exe", 100, flagged=True),
        ])

        ctx = store.room_context("screen")

        self.assertEqual(len(ctx["events"]), 1)
        self.assertIn("[opera.exe]", ctx["events"][0])


class RoomEventIdScopeTests(unittest.TestCase):
    """Retrieval is held to the whole in-scope slice, not the prompt window."""

    def test_ids_cover_more_than_the_prompt_quotes(self):
        store = FakeRoomStore(events=[_event(f"e{i}", "opera.exe", i)
                                      for i in range(30)])

        ids = store.room_event_ids("screen")

        self.assertEqual(len(ids), 30)
        params = store.params_for(store_module._ROOM_EVENT_IDS_CYPHER)
        self.assertGreaterEqual(params["limit"], 2000)

    def test_scope_is_pushed_into_the_id_query(self):
        store = FakeRoomStore(events=[_event("e1", "opera.exe", 100)])

        store.room_event_ids("screen", start=50.0, end=150.0,
                             applications=["Opera.exe"])

        params = store.params_for(store_module._ROOM_EVENT_IDS_CYPHER)
        self.assertEqual(params["start"], 50.0)
        self.assertEqual(params["end"], 150.0)
        self.assertEqual(params["applications"], ["opera.exe"])


class RoomFeedScopeTests(unittest.TestCase):
    """The feed must show the same slice the assistant is given."""

    def _store(self):
        return FakeRoomStore(
            events=[_event("e1", "pycharm64.exe", 100),
                    _event("e2", "opera.exe", 5000),
                    _event("e3", "ipc-a22e-g", 9000)],
            notes=[{"note_id": "n1", "text": "a thought", "ts": 5001}],
            messages=[{"message_id": "m1", "role": "user", "text": "hi",
                       "ts": 5002}])

    def test_applications_filter_events_only(self):
        feed = self._store().room_feed_full("screen", applications=["opera.exe"])

        kinds = {item["kind"] for item in feed}
        events = [item for item in feed if item["kind"] == "event"]
        self.assertEqual([e["event_id"] for e in events], ["e2"])
        # Notes and chat belong to the room as a whole, so they survive.
        self.assertEqual(kinds, {"event", "note", "message"})

    def test_application_match_is_case_insensitive(self):
        feed = self._store().room_feed_full("screen", applications=["OPERA.EXE"])

        self.assertEqual(
            [i["event_id"] for i in feed if i["kind"] == "event"], ["e2"])

    def test_time_window_filters_everything_in_the_feed(self):
        feed = self._store().room_feed_full("screen", start=4000.0, end=6000.0)

        self.assertEqual({item.get("event_id") or item.get("note_id") or "msg"
                          for item in feed}, {"e2", "n1", "msg"})

    def test_start_alone_is_an_open_ended_window(self):
        feed = self._store().room_feed_full("screen", start=6000.0)

        self.assertEqual(
            [i["event_id"] for i in feed if i["kind"] == "event"], ["e3"])

    def test_scope_and_kind_filters_compose(self):
        feed = self._store().room_feed_full(
            "screen", start=4000.0, applications=["opera.exe"], kinds=["event"])

        self.assertEqual([i["event_id"] for i in feed], ["e2"])


class RoomApplicationsTests(unittest.TestCase):
    def test_sources_are_listed_busiest_first(self):
        store = FakeRoomStore()

        sources = store.room_applications("screen")

        self.assertEqual([s["application"] for s in sources],
                         ["pycharm64.exe", "opera.exe"])
        self.assertEqual(sources[0]["events"], 12)

    def test_a_time_window_narrows_the_source_list(self):
        store = FakeRoomStore()

        store.room_applications("screen", start=10.0, end=20.0)

        params = store.params_for(store_module._ROOM_APPLICATIONS_CYPHER)
        self.assertEqual((params["start"], params["end"]), (10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
