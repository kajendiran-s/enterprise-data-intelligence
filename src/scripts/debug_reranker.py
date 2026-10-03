from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.embeddings.service import GeminiEmbeddingService
from src.evaluation.retrieval_metrics import recall_at_k, precision_at_k

def main():

    query = "How does Spark distribute work across executors?"

    embedding_model = GeminiEmbeddingService()
    vector_store = VectorStore()
    bm25 = BM25Retriever()

    # Without reranker
    hybrid_rrf = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_model=embedding_model,
        reranker=None,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
        rrf_k=60,
    )

    # With reranker
    reranker = Reranker()

    hybrid_reranked = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_model=embedding_model,
        reranker=reranker,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
        rrf_k=60,
    )

    # Get RRF candidates directly
    query_embedding = embedding_model.embed_query(query)

    dense_results = vector_store.search(
        query_embedding,
        20,
    )

    bm25_results = bm25.search(
        query,
        20,
    )

    candidates = hybrid_rrf._rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        final_top_k=20,
    )

    print("\n" + "=" * 80)
    print("BEFORE RERANKING")
    print("=" * 80)

    for rank, document in enumerate(candidates, start=1):

        print(
            f"{rank}. "
            f"{document['chunk_id']} | "
            f"RRF={document['rrf_score']:.5f}"
        )

    reranked = reranker.rerank(query=query, documents=candidates, top_k=5)

    print("\n" + "=" * 80)
    print("AFTER RERANKING")
    print("=" * 80)

    for rank, document in enumerate(reranked, start=1):

        print(
            f"{rank}. "
            f"{document['chunk_id']} | "
            f"Rerank={document['rerank_score']:.5f}"
        )


if __name__ == "__main__":
    main()