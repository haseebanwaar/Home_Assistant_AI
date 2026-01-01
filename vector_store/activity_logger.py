import time
from typing import List, Dict
from qdrant_client import QdrantClient, models

class ActivityLogger:
    def __init__(self, client: QdrantClient, embedding_model_name="BAAI/bge-small-en-v1.5"):
        """
        Args:
            client: An existing instance of QdrantClient.
            embedding_model_name: The FastEmbed model to use (e.g., "BAAI/bge-small-en-v1.5").
        """
        self.client = client
        self.collection_name = "activity_log"
        self.model_name = embedding_model_name

        if not self.client.collection_exists(self.collection_name):
            print(f"Creating new activity log using model: {self.model_name}...")
            # Initialize with a dummy document to set the model configuration
            self.client.add(
                collection_name=self.collection_name,
                documents=["init"],
                metadata=[{"type": "init"}],
                ids=[0]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=[0])
        else:
            print("Loaded existing activity log.")

    def log_activity(self, description: str, timestamp: float, context: str, sub_context: str):
        # We explicitly pass the model_name here so Qdrant knows how to vectorize it
        self.client.add(
            collection_name=self.collection_name,
            documents=[description],
            metadata=[{"timestamp": timestamp, "context": context, "sub_context": sub_context}],
            ids=[hash(str(timestamp))],
            parallel=None # Use 0 for auto-detection or 2-4 for speed
        )

    def reset(self):
        """Clear current log - REQUIRED if you change embedding models!"""
        self.client.delete_collection(collection_name=self.collection_name)
        print("Database cleared.")
