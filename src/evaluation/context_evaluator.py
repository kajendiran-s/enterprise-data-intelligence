from typing import Any

class ContextEvaluator:

    def __init__(self, dataset: list[dict]):
        self.dataset = dataset

    def evaluate(self, context_builder, retriever, k: int = 5) -> dict[str, Any]:
        recall_scores = []
        precision_scores = []

        for item in self.dataset:
            query = item["query"]
            relevant_ids = set(item["relevant_chunk_ids"])
            # Retrieve documents
            results = retriever.search(query=query, top_k=k)
            context_result = context_builder.build(results)
            context = context_result["context"]
            sources = context_result["sources"]
            # Chunk IDs that actually entered the context
            context_chunk_ids = [source["chunk_id"] for source in sources 
                                 if source.get("chunk_id") is not None]
            context_relevant = (set(context_chunk_ids) & relevant_ids)
            # Context Recall
            recall = (len(context_relevant) / len(relevant_ids) if relevant_ids else 0.0)
            # Context Precision
            precision = (len(context_relevant) / len(context_chunk_ids) if context_chunk_ids else 0.0)
            recall_scores.append(recall)
            precision_scores.append(precision)

            print(f"\nQuery: {query}")
            print(f"Context chunks: {context_chunk_ids}")
            print(f"Relevant chunks: {list(relevant_ids)}")
            print(f"Context Recall: {recall:.4f}")
            print(f"Context Precision: {precision:.4f}")
            print(f"Context chars: {len(context)}")

        return {
            "context_recall": (sum(recall_scores) / len(recall_scores) if recall_scores else 0.0),
            "context_precision": (sum(precision_scores) / len(precision_scores) if precision_scores else 0.0)
        }