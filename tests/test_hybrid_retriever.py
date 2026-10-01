from src.retrieval.reranker import Reranker
from src.embeddings.service import GeminiEmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever


reranker = Reranker()
embedder = GeminiEmbeddingService()
bm25_retriever= BM25Retriever()
dense_retriever = VectorStore()
hybrid_retriever = HybridRetriever(
    embedding_model=embedder,
    vector_store=dense_retriever,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
)


def test_hybrid_with_reranking():

    query = "How does Spark distribute work across executors?"

    results = hybrid_retriever.search(
        query=query,
        candidate_k=20,
        top_k=5,
    )

    assert len(results) <= 5

    for result in results:
        assert "rrf_score" in result
        assert "rerank_score" in result

    scores = [
        result["rerank_score"]
        for result in results
    ]

    assert scores == sorted(scores, reverse=True)

    print("\nReranked Hybrid Results:")

    for rank, result in enumerate(results, 1):
        print(f"\nRank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"RRF: {result['rrf_score']}")
        print(f"Rerank: {result['rerank_score']}")
        print(f"Text: {result['text'][:300]}")

if __name__ == "__main__":
    test_hybrid_with_reranking()