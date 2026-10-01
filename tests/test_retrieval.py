from src.embeddings.service import GeminiEmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever


def print_results(title: str, results: list[dict]):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for rank, result in enumerate(results,start=1):
        print(f"\nRank {rank}")
        print(f"Chunk ID : {result.get('chunk_id')}")
        print(f"Document : {result.get('document_id')}")
        print(f"Source   : {result.get('source')}")
        print(f"Score    : {result.get('score')}")
        print(f"Text     : "f"{result.get('text', '')[:250]}")

        if "retrieval_sources" in result:
            print(f"Sources  : "f"{result['retrieval_sources']}")

def main():
    query = (
        "How does Spark distribute work "
        "across executors?"
    )
    embedder = GeminiEmbeddingService()
    vector_store = VectorStore()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_model=embedder,
        dense_top_k=5,
        sparse_top_k=5,
        final_top_k=5,
    )
    # 1. Dense retrieval
    query_vector = embedder.embed_query(query)
    dense_results = vector_store.search( query_vector=query_vector,top_k=5 )
    print_results(
        "DENSE RETRIEVAL",
        dense_results
    )
    # 2. BM25 retrieval
    bm25_results = bm25.search(query=query,top_k=5)
    print_results(
        "BM25 RETRIEVAL",
        bm25_results
    )
    # 3. Hybrid retrieval
    hybrid_results = hybrid.search( query=query, top_k=5)

    print_results(
        "HYBRID RETRIEVAL + RRF",
        hybrid_results
    )


if __name__ == "__main__":
    main()