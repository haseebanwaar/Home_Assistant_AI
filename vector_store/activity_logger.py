import os
import logging
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
logger = logging.getLogger("home_assistant")

class ActivityLogger:
    def __init__(self, client: QdrantClient, embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")):
        """
        Args:
            client: An existing instance of QdrantClient.
            embedding_model_name: The FastEmbed model to use (e.g., "BAAI/bge-small-en-v1.5").
        """
        self.client = client
        self.collection_name = "activity_log"
        self.model_name = embedding_model_name

        self.ensure_collection()

    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            logger.info("Creating activity log using model: %s", self.model_name)
            # Initialize with a dummy document to set the model configuration
            self.client.add(
                collection_name=self.collection_name,
                documents=["init"],
                metadata=[{"type": "init"}],
                ids=[0]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=[0])
        else:
            logger.debug("Activity log collection is ready.")

    def log_activity(self, description: str, timestamp: float, context: str, sub_context: str):
        self.ensure_collection()
        # We explicitly pass the model_name here so Qdrant knows how to vectorize it
        self.client.add(
            collection_name=self.collection_name,
            documents=[description],
            metadata=[{"timestamp": timestamp, "context": context, "sub_context": sub_context}],
            ids=[hash(str(timestamp))],
            parallel=None # Use 0 for auto-detection or 2-4 for speed
        )

    def log_event(self, summary: str, event_id: str, session_id: str,
                  span_start: float, span_end: float, profile: str = None,
                  timestamp: float = None):
        """Step 10: embed an EVENT summary with event-scoped metadata.

        Keeps context="screen" and a timestamp so existing voice retrieval (which
        filters on those) still answers — now over event-scoped summaries. Uses a
        deterministic id (uuid5 of event_id) so replays upsert instead of
        duplicating. Completes the dual store (JSONL + Neo4j + Qdrant).
        """
        if not summary or not summary.strip():
            return None
        self.ensure_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"event:{event_id}"))
        self.client.add(
            collection_name=self.collection_name,
            documents=[summary],
            metadata=[{
                "context": "screen",
                "timestamp": timestamp if timestamp is not None else span_start,
                "session_id": session_id,
                "event_id": event_id,
                "span_start": span_start,
                "span_end": span_end,
                "profile": profile,
            }],
            ids=[point_id],
            parallel=None,
        )
        return point_id

    def reset(self):
        """Clear the single-user activity collection and recreate it for immediate use."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self.ensure_collection()
        logger.info("Activity memory cleared.")
