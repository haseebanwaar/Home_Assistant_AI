import time
from datetime import datetime
from typing import List, Dict
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder


class ActivityRetriever:
    def __init__(self, client: QdrantClient,
                 embedding_model_name="BAAI/bge-small-en-v1.5",
                 reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
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
        print(f"Loading reranker: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)


    #todo, reranker can mess up, esp in temporal quries or hybrid
    def retrieve_memory(self, search_query: str = None, time_value: float =None, time_unit: str = None, date: str = None):
        """
        HYBRID SEARCH:
        Combines Vector Similarity (Content) + Metadata Filtering (Time) + Reranking.
        """
        # 1. Build the Filter List
        filter_conditions = []
        now = time.time()
        seconds_to_subtract = 0

        # Standardize inputs
        unit = time_unit.lower().strip()

        if "minute" in unit:
            seconds_to_subtract = time_value * 60
        elif "hour" in unit:
            seconds_to_subtract = time_value * 3600
        elif "day" in unit:
            seconds_to_subtract = time_value * 86400
        elif "week" in unit:
            seconds_to_subtract = time_value * 604800
        elif "month" in unit:
            # Approximate 30 days per month
            seconds_to_subtract = time_value * 2592000
        else:
            # Default fallback to 1 hour if unit is weird
            seconds_to_subtract = 3600


        start_timestamp = now - seconds_to_subtract

        if seconds_to_subtract <= 3600:       # <= 1 hour
            limit = 30                           # Enough for a short chat context
        elif seconds_to_subtract <= 86400:    # <= 1 day
            limit = 150                          # Need more chunks to summarize a whole day
        elif seconds_to_subtract <= 604800:   # <= 1 week
            limit = 200                          # Broad overview
        else:                                    # > 1 week
            limit = 800
        # Optional: Log for debugging
        human_readable_start = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"DEBUG: Searching from {human_readable_start} ({unit} ago)")

        # Add Time Filter if requested
        if time_value is not None:
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
        if search_query and not time_value:
            # Case A: Semantic + Time (e.g., "PowerPoint yesterday")
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

            scores = self.reranker.predict(cross_encoder_inputs)

            ranked_hits = sorted(
                zip(hits, scores),
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
            points.sort(key=lambda x: x.payload['timestamp'])
            return [p.payload['document'] for p in points]