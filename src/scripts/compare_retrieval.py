import json
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
)
from src.rag.pipeline import RAGPipeline


def load_dataset(path: str) -> list[dict]:

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_method(name: str, retrieve_function, dataset: list[dict], k: int = 5):
    recall_scores = []
    precision_scores = []

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    for item in dataset:
        query = item["query"]
        relevant_chunk_ids = item["relevant_chunk_ids"]
        results = retrieve_function(
            query,
            k,
        )
        retrieved_chunk_ids = [result["chunk_id"] for result in results]
        recall = recall_at_k(retrieved_chunk_ids,relevant_chunk_ids,k)
        precision = precision_at_k(retrieved_chunk_ids,relevant_chunk_ids,k)
        recall_scores.append(recall)
        precision_scores.append(precision)

        print(f"\nQuery: {query}")
        print(f"Retrieved: {retrieved_chunk_ids}")
        print(f"Recall@{k}: {recall:.2f}")
        print(f"Precision@{k}: {precision:.2f}")

    average_recall = (sum(recall_scores) / len(recall_scores) if recall_scores else 0.0)

    average_precision = (sum(precision_scores) / len(precision_scores) if precision_scores else 0.0)

    return {
        "method": name,
        "recall": average_recall,
        "precision": average_precision,
    }


def print_summary(results: list[dict]):

    print("\n\n")
    print("=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"{'Method':<25}"
        f"{'Recall@5':<15}"
        f"{'Precision@5':<15}"
    )

    print("-" * 55)

    for result in results:
        print(
            f"{result['method']:<25}"
            f"{result['recall']:<15.4f}"
            f"{result['precision']:<15.4f}"
        )


if __name__ == "__main__":
    dataset = load_dataset(
        "data/evaluation/retrieval_eval.json"
    )
    pipeline = RAGPipeline()
    results = []
    def hybrid_retrieval(query, k):
        return pipeline.retriever.retrieve(query=query,candidate_k=20,top_k=k)
    results.append(evaluate_method("Hybrid + Reranker",hybrid_retrieval,dataset,k=5,))
    print_summary(results)