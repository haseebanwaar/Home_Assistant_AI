"""Tests for query tokenization and the unified evidence retriever.

The retriever is exercised against fakes, so these run with no Neo4j/Qdrant.
"""
import pytest

from memory.retrieval.evidence import EvidenceRetriever
from memory.retrieval.terms import is_scope_only, tokenize


# -- tokenize --------------------------------------------------------------
def test_drops_question_scaffolding():
    assert tokenize("what was I doing with the camera worker") == ["camera", "worker"]


def test_keeps_identifier_shaped_terms():
    assert tokenize("RTSP reconnect in camera_worker.py") == [
        "rtsp", "reconnect", "camera_worker.py"]


def test_dedupes_preserving_order():
    assert tokenize("qdrant and Qdrant and neo4j") == ["qdrant", "neo4j"]


@pytest.mark.parametrize("query", [
    "what did I do today", "show me yesterday", "what happened last week", "",
])
def test_scope_only_questions_have_no_terms(query):
    assert is_scope_only(query)


def test_temporal_words_are_stripped_but_content_survives():
    assert tokenize("what was I doing in Neo4j yesterday") == ["neo4j"]


# -- fakes -----------------------------------------------------------------
def _event(event_id, text, ts=100.0, score=100):
    return {"kind": "event", "id": event_id, "title": "Code", "text": text,
            "ts": ts, "rooms": [], "score": score}


class FakeGraph:
    """Stands in for Neo4jStore; records how it was called."""

    def __init__(self, hits=None, recent=None, room_members=None):
        self._hits = hits or []
        self._recent = recent or []
        self._room_members = room_members
        self.calls = []

    def memory_search(self, query, limit=40, kinds=None, start=None, end=None,
                      room_id=None, domain=None):
        self.calls.append(
            ("memory_search", query, start, end, room_id, domain))
        # Mirrors the real store: no content terms -> no keyword results.
        return [] if is_scope_only(query) else [dict(h) for h in self._hits]

    def recent_events(self, start=None, end=None, room_id=None, domain=None,
                      limit=20):
        self.calls.append(("recent_events", start, end, room_id, domain))
        return [dict(r) for r in self._recent]

    def events_in_room(self, event_ids, room_id):
        self.calls.append(("events_in_room", tuple(event_ids), room_id))
        if self._room_members is None:
            return set(event_ids)
        return {i for i in event_ids if i in self._room_members}

    def entities_for_events(self, event_ids):
        return {}


class FakeHit:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeQdrant:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.filters = []

    def query(self, collection_name, query_text=None, query_filter=None, limit=10):
        self.filters.append(query_filter)
        return self._hits


def _semantic_hit(event_id, text):
    return FakeHit({"event_id": event_id, "document": text,
                    "span_start": 50.0, "profile": "coding"})


def test_memory_domain_is_forwarded_to_graph_retrieval():
    graph = FakeGraph(hits=[_event("home-1", "Front door opened")])
    retriever = EvidenceRetriever(neo4j_store=graph)

    retriever.retrieve("front door", domain="home", semantic=False)

    assert ("memory_search", "front door", None, None, None, "home") in graph.calls


# -- retrieve --------------------------------------------------------------
def test_keyword_and_semantic_hits_merge_without_duplicates():
    graph = FakeGraph(hits=[_event("e1", "edited camera_worker.py")])
    qdrant = FakeQdrant(hits=[_semantic_hit("e1", "dup"),
                              _semantic_hit("e2", "reviewed the RTSP retry loop")])
    retriever = EvidenceRetriever(neo4j_store=graph, qdrant_client=qdrant)

    results = retriever.retrieve("camera worker", limit=10)

    assert [r["id"] for r in results] == ["e1", "e2"]
    assert results[0]["match"] == "keyword"   # the duplicate keeps its origin
    assert results[1]["match"] == "semantic"


def test_semantic_search_still_runs_when_a_room_scope_is_set():
    """The old code disabled vector search whenever a room/date filter existed."""
    graph = FakeGraph(hits=[], room_members={"e2"})
    qdrant = FakeQdrant(hits=[_semantic_hit("e2", "in-room"),
                              _semantic_hit("e9", "other room")])
    retriever = EvidenceRetriever(neo4j_store=graph, qdrant_client=qdrant)

    results = retriever.retrieve("camera worker", limit=10, room_id="project:ha")

    # Ran, and was scope-filtered through the graph rather than dropped.
    assert [r["id"] for r in results] == ["e2"]
    assert ("events_in_room", ("e2", "e9"), "project:ha") in graph.calls


def test_date_scope_is_pushed_into_the_vector_filter():
    graph = FakeGraph(hits=[])
    qdrant = FakeQdrant(hits=[_semantic_hit("e1", "hit")])
    retriever = EvidenceRetriever(neo4j_store=graph, qdrant_client=qdrant)

    retriever.retrieve("camera worker", start=10.0, end=20.0)

    condition = qdrant.filters[0].must[0]
    assert condition.key == "timestamp"
    assert (condition.range.gte, condition.range.lt) == (10.0, 20.0)


def test_scope_only_question_falls_back_to_chronological_events():
    graph = FakeGraph(
        hits=[_event("e1", "should not be used")],
        recent=[_event("old", "9am", ts=1.0), _event("new", "5pm", ts=9.0)])
    qdrant = FakeQdrant(hits=[_semantic_hit("e5", "noise")])
    retriever = EvidenceRetriever(neo4j_store=graph, qdrant_client=qdrant)

    results = retriever.retrieve("what did I do today", start=0.0, end=100.0)

    assert [r["id"] for r in results] == ["new", "old"]   # newest first
    assert all(r["match"] == "recent" for r in results)
    assert qdrant.filters == []   # no vector search for a contentless query


def test_empty_keyword_and_semantic_result_falls_back_to_scope():
    graph = FakeGraph(hits=[], recent=[_event("r1", "recent")])
    retriever = EvidenceRetriever(neo4j_store=graph, qdrant_client=FakeQdrant())

    results = retriever.retrieve("nothing matches this", room_id="daily")

    assert [r["id"] for r in results] == ["r1"]


def test_reranker_reorders_when_there_are_more_candidates_than_needed():
    graph = FakeGraph(hits=[_event("e1", "first", score=200),
                            _event("e2", "second", score=150),
                            _event("e3", "third", score=100)])

    class Reranker:
        def predict(self, pairs):
            # Favour the record whose text is "third".
            return [9.0 if "third" in pair[1] else 0.1 for pair in pairs]

    retriever = EvidenceRetriever(neo4j_store=graph, reranker=Reranker())
    results = retriever.retrieve("camera worker", limit=2)

    assert results[0]["id"] == "e3"
    assert results[0]["rerank_score"] == 9.0


def test_reranker_is_skipped_for_scope_only_questions():
    """Relevance-ranking a contentless query would scramble the day's order."""
    graph = FakeGraph(recent=[_event("a", "9am", ts=1.0),
                              _event("b", "1pm", ts=5.0),
                              _event("c", "5pm", ts=9.0)])

    class ExplodingReranker:
        def predict(self, pairs):
            raise AssertionError("reranker must not run on a scope-only query")

    retriever = EvidenceRetriever(neo4j_store=graph, reranker=ExplodingReranker())
    results = retriever.retrieve("what did I do today", limit=2)

    assert [r["id"] for r in results] == ["c", "b"]


def test_a_failing_reranker_degrades_to_lexical_order():
    graph = FakeGraph(hits=[_event("e1", "a", score=100),
                            _event("e2", "b", score=300),
                            _event("e3", "c", score=200)])

    class BrokenReranker:
        def predict(self, pairs):
            raise RuntimeError("model unavailable")

    retriever = EvidenceRetriever(neo4j_store=graph, reranker=BrokenReranker())
    results = retriever.retrieve("camera worker", limit=2)

    assert [r["id"] for r in results] == ["e2", "e3"]


def test_store_failures_do_not_raise():
    class BrokenGraph:
        def memory_search(self, *a, **k):
            raise RuntimeError("neo4j down")

        def recent_events(self, *a, **k):
            raise RuntimeError("neo4j down")

    retriever = EvidenceRetriever(neo4j_store=BrokenGraph(),
                                  qdrant_client=FakeQdrant())
    assert retriever.retrieve("camera worker") == []


def test_works_with_no_stores_configured():
    assert EvidenceRetriever().retrieve("anything") == []
