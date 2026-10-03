import json
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.embeddings.service import GeminiEmbeddingService
from src.evaluation.retrieval_metrics import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k

def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retriever(name: str,retriever,dataset: list[dict],k: int = 5) -> dict:
    recall_scores = []
    precision_scores = []
    mrr_scores=[]
    ndg_scores=[]
    print("\n" + "=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    for item in dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        results = retriever.search(query=query,top_k=k)

        retrieved_ids = [result["chunk_id"] for result in results]
        recall = recall_at_k(retrieved_ids,relevant_ids,k)
        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = mrr_at_k(retrieved_ids,relevant_ids,k)
        ndg = ndcg_at_k(retrieved_ids,relevant_ids,k)
        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        ndg_scores.append(ndg)
        print("\nQuery:")
        print(query)

        print(f"Retrieved: {retrieved_ids}")
        print(f"Relevant : {relevant_ids}")

        print(f"Recall@{k}: {recall:.2f}")
        print(f"Precision@{k}: {precision:.2f}")

    return {
        "method": name,
        "recall_at_k": (sum(recall_scores) / len(recall_scores) if recall_scores else 0.0),
        "precision_at_k": (sum(precision_scores) / len(precision_scores) if precision_scores else 0.0),
        "mrr_at_k": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "ndcg_at_k": (sum(ndg_scores) / len(ndg_scores) if ndg_scores else 0.0)
    }


def main():

    dataset = load_dataset("data/evaluation/retrieval_eval.json")
    print(f"Loaded {len(dataset)} evaluation queries.")
    # ---------------------------------------------------------
    # Shared components
    # ---------------------------------------------------------
    embedding_model = GeminiEmbeddingService()
    vector_store = VectorStore()
    bm25 = BM25Retriever()
    # ---------------------------------------------------------
    # Dense-only adapter
    # ---------------------------------------------------------
    class DenseRetriever:
        def search(self, query: str, top_k: int = 5):
            query_embedding = embedding_model.embed_query(query)
            return vector_store.search(query_embedding,top_k,)
    dense_retriever = DenseRetriever()
    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------
    class BM25Adapter:
        def search(self, query: str, top_k: int = 5):
            return bm25.search(query,top_k)
    bm25_retriever = BM25Adapter()
    # ---------------------------------------------------------
    # Hybrid RRF
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Hybrid + Reranker
    # ---------------------------------------------------------
    reranker = Reranker()
    hybrid_reranker = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_model=embedding_model,
        reranker=reranker,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
        rrf_k=60,
    )
    # ---------------------------------------------------------
    # Evaluate all methods
    # ---------------------------------------------------------
    results = []
    results.append(
        evaluate_retriever(name="Dense",retriever=dense_retriever,dataset=dataset)
    )
    results.append(
        evaluate_retriever(name="BM25",retriever=bm25_retriever,dataset=dataset)
    )
    results.append(
        evaluate_retriever(name="Hybrid RRF",retriever=hybrid_rrf,dataset=dataset)
    )
    results.append(
        evaluate_retriever(name="Hybrid + Reranker",retriever=hybrid_reranker,dataset=dataset)
    )
    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------
    print("\n\n")
    print("=" * 80)
    print("RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(
        f"{'Method':<25}"
        f"{'Recall@5':<15}"
        f"{'Precision@5':<15}"
        f"{'MRR@5':<15}"
        f"{'NDGC@5':<15}"
    )
    print("-" * 55)
    for result in results:
        print(
            f"{result['method']:<25}"
            f"{result['recall_at_k']:<15.4f}"
            f"{result['precision_at_k']:<15.4f}"
            f"{result['mrr_at_k']:<15.4f}"
            f"{result['ndcg_at_k']:<15.4f}"
        )

if __name__ == "__main__":
    main()