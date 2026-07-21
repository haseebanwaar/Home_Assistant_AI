# rag/base_retriever.py
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, n_results: int = 5):
        """Return a list of (document, metadata) tuples."""
        pass
