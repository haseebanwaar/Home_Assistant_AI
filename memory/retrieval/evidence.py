"""One retrieval path for every question the assistant answers.

Before this existed each chat surface retrieved differently: the grounded
assistant called the graph's keyword search directly, `/memory/search` merged in
vector hits but only when NO room/date scope was set, and room chat did not
retrieve on the question at all (it pasted the last few events). Same memory,
three answers.

`EvidenceRetriever.retrieve()` is that single path:

  1. graph keyword search (term-scored) over every requested kind;
  2. vector search over event summaries, scoped by the SAME room/date window —
     dates via the Qdrant timestamp payload, rooms by post-filtering through the
     graph (membership lives there, and old points carry no room payload);
  3. de-duplicate, then rank.

Ranking is deliberately conditional. The cross-encoder is a relevance model, so
it is used only when the query HAS content terms and there are more candidates
than we need. A scope-only question ("what did I do today") is answered
chronologically instead — reranking narration by "relevance" to a query with no
content shuffles a day's story into nonsense.
"""
from __future__ import annotations

import logging

from memory.retrieval.terms import tokenize

logger = logging.getLogger("home_assistant")

DEFAULT_KINDS = ("event", "note", "message", "entity", "claim", "room", "insight")


class EvidenceRetriever:
    def __init__(self, neo4j_store=None, qdrant_client=None, reranker=None,
                 collection_name="activity_log"):
        """`reranker` is an optional CrossEncoder (shared with ActivityRetriever)."""
        self.neo4j = neo4j_store
        self.qdrant = qdrant_client
        self.reranker = reranker
        self.collection_name = collection_name

    # -- public ------------------------------------------------------------
    def retrieve(self, query, limit=10, kinds=None, start=None, end=None,
                 room_id=None, domain=None, semantic=True):
        """Ranked evidence for `query` within an optional room/date scope.

        Each item: {kind, id, title, text, ts, rooms, score, match[, entities]}.
        """
        limit = max(1, min(int(limit), 200))
        allowed = [k for k in (kinds or DEFAULT_KINDS) if k in DEFAULT_KINDS]
        if not allowed:
            allowed = list(DEFAULT_KINDS)
        terms = tokenize(query)

        results, seen = [], set()
        if terms:
            for item in self._keyword(
                    query, limit, allowed, start, end, room_id, domain):
                key = (item.get("kind"), item.get("id"))
                if key not in seen:
                    seen.add(key)
                    results.append(item)
            if semantic and "event" in allowed:
                for item in self._semantic(
                        query, limit, start, end, room_id, domain):
                    key = (item.get("kind"), item.get("id"))
                    if key not in seen:
                        seen.add(key)
                        results.append(item)

        # No content terms, or nothing matched: answer from the scope itself.
        if not results and "event" in allowed:
            results = self._recent(start, end, room_id, domain, limit)

        return self._rank(query, results, limit, rerankable=bool(terms))

    # -- sources -----------------------------------------------------------
    def _keyword(self, query, limit, kinds, start, end, room_id, domain):
        if self.neo4j is None:
            return []
        try:
            # Over-fetch so the reranker has candidates to choose between.
            scope = {"domain": domain} if domain is not None else {}
            rows = self.neo4j.memory_search(
                query, limit=limit * 3, kinds=kinds,
                start=start, end=end, room_id=room_id, **scope)
        except Exception as exc:
            logger.warning("keyword retrieval failed: %s", exc)
            return []
        for row in rows:
            row.setdefault("match", "keyword")
        return rows

    def _semantic(self, query, limit, start, end, room_id, domain):
        """Vector hits over event summaries, constrained to the same scope."""
        if self.qdrant is None:
            return []
        from qdrant_client import models as qmodels

        must = [qmodels.IsEmptyCondition(
            is_empty=qmodels.PayloadField(key="session_id"))]
        # Only event-scoped points carry session_id; exclude the legacy ones.
        conditions = {"must_not": must}
        required = []
        if start is not None or end is not None:
            required.append(qmodels.FieldCondition(
                key="timestamp", range=qmodels.Range(gte=start, lt=end)))
        if domain:
            required.append(qmodels.FieldCondition(
                key="context",
                match=qmodels.MatchValue(
                    value="camera" if domain == "home" else "screen")))
        if required:
            conditions["must"] = required
        try:
            hits = self.qdrant.query(
                self.collection_name, query_text=query,
                query_filter=qmodels.Filter(**conditions), limit=limit * 4)
        except Exception as exc:
            logger.warning("semantic retrieval failed: %s", exc)
            return []

        items = []
        for hit in hits:
            meta = hit.metadata or {}
            event_id = meta.get("event_id")
            if not event_id:
                continue
            items.append({
                "kind": "event", "id": event_id,
                "title": meta.get("profile") or "Activity",
                "text": meta.get("document") or "",
                "ts": meta.get("span_start"),
                "span_start": meta.get("span_start"),
                "span_end": meta.get("span_end"), "rooms": [],
                "score": 70, "match": "semantic",
            })
        if not items:
            return []

        # Room membership lives in the graph, so scope-filter there.
        if room_id and self.neo4j is not None:
            try:
                keep = self.neo4j.events_in_room(
                    [i["id"] for i in items], room_id)
                items = [i for i in items if i["id"] in keep]
            except Exception as exc:
                logger.warning("room post-filter failed, dropping semantic hits: %s", exc)
                return []
        if self.neo4j is not None and items:
            try:
                enrich = self.neo4j.entities_for_events([i["id"] for i in items])
                for item in items:
                    item["entities"] = enrich.get(item["id"], [])
            except Exception as exc:
                logger.debug("entity enrichment failed: %s", exc)
        return items

    def _recent(self, start, end, room_id, domain, limit):
        if self.neo4j is None:
            return []
        try:
            scope = {"domain": domain} if domain is not None else {}
            rows = self.neo4j.recent_events(
                start=start, end=end, room_id=room_id, limit=limit, **scope)
        except Exception as exc:
            logger.warning("recency fallback failed: %s", exc)
            return []
        for row in rows:
            row["match"] = "recent"
        return rows

    # -- ranking -----------------------------------------------------------
    @staticmethod
    def _document(item):
        """Rerank on title + text; an entity's `text` is only its type."""
        parts = [str(item.get("title") or ""), str(item.get("text") or "")]
        return " ".join(p for p in parts if p).strip()

    def _rank(self, query, results, limit, rerankable):
        if not results:
            return []
        # Scope-only questions ("what did I do today") read as a chronological
        # narration; relevance-reranking them destroys that order.
        if not rerankable:
            results.sort(key=lambda i: (i.get("ts") or 0), reverse=True)
            return results[:limit]
        if self.reranker is None or len(results) <= limit:
            results.sort(
                key=lambda i: (i.get("score") or 0, i.get("ts") or 0), reverse=True)
            return results[:limit]

        pairs, scored = [], []
        for item in results:
            document = self._document(item)
            if document:
                pairs.append([query, document])
                scored.append(item)
        if not pairs:
            results.sort(
                key=lambda i: (i.get("score") or 0, i.get("ts") or 0), reverse=True)
            return results[:limit]
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as exc:
            logger.warning("rerank failed, falling back to lexical order: %s", exc)
            results.sort(
                key=lambda i: (i.get("score") or 0, i.get("ts") or 0), reverse=True)
            return results[:limit]

        for item, score in zip(scored, rerank_scores):
            item["rerank_score"] = round(float(score), 4)
        scored.sort(
            key=lambda i: (i.get("rerank_score"), i.get("score") or 0), reverse=True)
        return scored[:limit]
