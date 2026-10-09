from src.embeddings.service import NomicEmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

class Retriever:

    def __init__(self,
        # vector_store=VectorStore(),
        # bm25_retriever=BM25Retriever(),
        # embedding_model=GeminiEmbeddingService(),
        # hybrid=HybridRetriever(),
        # reranker=Reranker(),
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60):
        self.embedder = NomicEmbeddingService()
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()
        self.hybrid = HybridRetriever(vector_store=self.vector_store,
                                      bm25_retriever=self.bm25,
                                      embedding_model=self.embedder,
                                      reranker=self.reranker)

    def retrieve(
            self,
            query:str,
            candidate_k:int = 20,
            top_k:int=5
    ):
        results = self.hybrid.search(query, top_k=top_k,)

        return results 