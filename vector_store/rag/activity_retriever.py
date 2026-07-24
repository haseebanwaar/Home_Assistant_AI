import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

load_dotenv()

logger = logging.getLogger("home_assistant")

# Approximate seconds per unit for time-range filtering.
_UNIT_SECONDS = {
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,  # ~30 days
}


class ActivityRetriever:
    def __init__(self, client: QdrantClient,
                 embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
                 reranker_model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")):
        """
        Args:
            client: An existing instance of QdrantClient.
            embedding_model_name: MUST match the one used in ActivityLogger.
            reranker_model_name: Can be changed anytime (e.g., to a BGE or Qwen reranker).
        """
        self.client = client
        self.collection_name = "activity_log"
        self.model_name = embedding_model_name

        # Load the Reranker (Cross-Encoder)
        # This downloads the model locally on first run
        logger.info("Loading reranker: %s...", reranker_model_name)
        self.reranker = CrossEncoder(reranker_model_name)


    #todo, reranker can mess up, esp in temporal quries or hybrid
    def retrieve_memory(self, search_query: str = None, time_value: float = None, time_unit: str = None):
        """
        HYBRID SEARCH:
        Combines Vector Similarity (Content) + Metadata Filtering (Time) + Reranking.
        Missing/incomplete time arguments are handled safely (no time filter applied).
        """
        # 1. Build the Filter List
        filter_conditions = []
        now = time.time()

        # A time filter is only usable when we have BOTH a value and a unit.
        has_time = time_value is not None and bool(time_unit)
        seconds_to_subtract = 0
        if has_time:
            unit = time_unit.lower().strip()
            per_unit = next((sec for key, sec in _UNIT_SECONDS.items() if key in unit), 3600)
            seconds_to_subtract = time_value * per_unit

        if not has_time:                       # semantic-only / no window
            limit = 30
        elif seconds_to_subtract <= 3600:      # <= 1 hour
            limit = 30                          # Enough for a short chat context
        elif seconds_to_subtract <= 86400:     # <= 1 day
            limit = 150                         # Need more chunks to summarize a whole day
        elif seconds_to_subtract <= 604800:    # <= 1 week
            limit = 200                         # Broad overview
        else:                                   # > 1 week
            limit = 800

        # Add Time Filter if requested
        if has_time:
            start_timestamp = now - seconds_to_subtract
            human_readable_start = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug("Searching memory from %s (%s ago)", human_readable_start, time_unit)
            filter_conditions.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.Range(gte=start_timestamp, lte=now)
                )
            )

        # Add Context Filter (Always strictly screen/activity)
        filter_conditions.append(
            models.FieldCondition(key="context", match=models.MatchValue(value="screen"))
        )

        # 2. Construct the Qdrant Filter
        qdrant_filter = models.Filter(must=filter_conditions)

        # 3. Execute Query
        if search_query:
            # Case A: Semantic search, optionally constrained by time.
            # Qdrant finds vectors close to "PowerPoint" BUT only within the timestamp range

            # Retrieve more candidates for reranking
            candidates_limit = limit * 5

            hits = self.client.query(
                collection_name=self.collection_name,
                query_text=search_query,
                query_filter=qdrant_filter,
                limit=candidates_limit
            )

            if not hits:
                return []
            # Rerank with Cross-Encoder
            cross_encoder_inputs = [
                [search_query, hit.metadata["document"]]
                for hit in hits
                if hit.metadata and "document" in hit.metadata
            ]

            if not cross_encoder_inputs:
                return []

            scores = self.reranker.predict(cross_encoder_inputs)

            document_hits = [
                hit for hit in hits if hit.metadata and "document" in hit.metadata
            ]

            ranked_hits = sorted(
                zip(document_hits, scores),
                key=lambda x: x[1],
                reverse=True
            )

            return [
                hit.metadata["document"]
                for hit, score in ranked_hits[:limit]
            ]

        else:
            # Case B: Time Only (e.g., "What happened yesterday?")
            # Standard Scroll because there is no vector query
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True
            )
            # Sort chronologically for storytelling
            points.sort(key=lambda x: x.payload.get('timestamp', 0))
            return [p.payload['document'] for p in points if 'document' in p.payload]
