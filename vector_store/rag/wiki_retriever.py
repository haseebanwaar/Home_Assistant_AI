# rag/wiki_retriever.py
import chromadb
from vector_store.rag.base_retriever import BaseRetriever

class WikiRetriever(BaseRetriever):
    def __init__(self, persist_directory="./chroma_db"):
        self.db = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "local_wikipedia"
        try:
            self.collection = self.db.get_collection(self.collection_name)
        except:
            self.collection = self.db.create_collection(self.collection_name)

    def search(self, query: str, n_results: int = 5):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return list(zip(results['documents'][0], results['metadatas'][0]))
