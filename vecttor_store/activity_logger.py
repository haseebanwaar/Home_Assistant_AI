import time
from collections import deque
import chromadb
from chromadb.errors import InvalidCollectionException


class ActivityLogger:
    def __init__(self,persist_directory='./chroma_db'):
        """
        Args:
        persist_directory: the directory where the chroma database should be stored
        """
        self.db = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "activity_log"
        try:
            self.collection = self.db.get_collection(self.collection_name)
        except InvalidCollectionException:
            self.collection = self.db.create_collection(self.collection_name)

    def log_activity(self, description, timestamp):
        """Logs a description of activity with a timestamp."""
        metadata = {"timestamp": str(timestamp)}
        self.collection.add(
            documents=[description],
            metadatas=[metadata],
            ids=[str(timestamp)],
        )

    def search_activity(self, query, n_results=5):
        """Searches for past activity descriptions relevant to the query."""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return list(zip(results['documents'][0],results['metadatas'][0]))

    def reset(self):
        """clear current log"""
        self.db.delete_collection(self.collection_name)
        self.collection = self.db.create_collection(self.collection_name)