import json

from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.embeddings.service import GeminiEmbeddingService


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    dataset = load_dataset("data/evaluation/retrieval_eval.json")
    print(f"Loaded {len(dataset)} evaluation queries.")
    
    embedding_model = GeminiEmbeddingService()
    vector_store = VectorStore()
    bm25 = BM25Retriever()
    reranker = Reranker()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        embedding_model=embedding_model,
        reranker=reranker,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=5,
        rrf_k=60,
    )
    evaluator = RetrievalEvaluator(dataset)
    result = evaluator.evaluate(name="Hybrid + Reranker",retriever=hybrid_retriever,k=5)

    print("\n")
    print("=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Method       : {result['method']}")
    print(f"Recall@5     : {result['recall_at_k']:.4f}")
    print(f"Precision@5  : {result['precision_at_k']:.4f}")


if __name__ == "__main__":
    main()