# rag/multi_retriever.py
from vector_store.rag.base_retriever import BaseRetriever

class MultiSourceRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers  # list of (name, retriever_instance)

    def search(self, query: str, n_results: int = 5):
        combined = []
        for name, retriever in self.retrievers:
            try:
                docs = retriever.search(query, n_results)
                for doc, meta in docs:
                    combined.append({
                        "source": name,
                        "document": doc,
                        "metadata": meta
                    })
            except Exception as e:
                combined.append({
                    "source": name,
                    "document": f"[Error retrieving from {name}: {str(e)}]",
                    "metadata": {}
                })

        # Simple heuristic: just truncate to n_results (can be improved with reranking later)
        return combined[:n_results]
