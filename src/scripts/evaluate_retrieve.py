import json

from src.evaluation.retrieval_metrics import recall_at_k
from src.rag.pipeline import RAGPipeline


def load_dataset(path: str) -> list[dict]:

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(
    pipeline: RAGPipeline,
    dataset: list[dict],
    k: int = 5,
):

    scores = []

    for item in dataset:

        query = item["query"]

        relevant_chunk_ids = set(
            item["relevant_chunk_ids"]
        )

        results = pipeline.retriever.retrieve(
            query=query,
            candidate_k=20,
            top_k=k,
        )

        retrieved_chunk_ids = [
            result["chunk_id"]
            for result in results
        ]

        score = recall_at_k(
            retrieved_chunk_ids,
            relevant_chunk_ids,
            k,
        )

        scores.append(score)

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Recall@{k}: {score:.2f}")
        print(
            f"Retrieved: {retrieved_chunk_ids}"
        )
        print(
            f"Relevant:  {list(relevant_chunk_ids)}"
        )

    average_recall = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Recall@{k}: {average_recall:.4f}")


if __name__ == "__main__":

    dataset = load_dataset(
        "data/evaluation/retrieval_eval.json"
    )

    pipeline = RAGPipeline()

    evaluate(
        pipeline,
        dataset,
        k=5,
    )