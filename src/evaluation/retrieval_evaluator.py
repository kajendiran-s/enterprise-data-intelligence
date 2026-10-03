from src.evaluation.retrieval_metrics import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k

class RetrievalEvaluator:
    def __init__(self, dataset: list[dict]):
        self.dataset = dataset

    def evaluate(self,name: str,retriever,k: int = 5) -> dict:
        recall_scores = []
        precision_scores = []
        mrr_scores=[]
        ndcg_scores = []
        for item in self.dataset:
            query = item["query"]
            relevant_ids = item["relevant_chunk_ids"]
            results = retriever.search(query=query,top_k=k)
            retrieved_ids = [result["chunk_id"] for result in results]
            recall = recall_at_k(retrieved_ids,relevant_ids,k)
            precision = precision_at_k(retrieved_ids,relevant_ids,k)
            mrr = mrr_at_k(retrieved_ids,relevant_ids,k)
            ndcg = ndcg_at_k(retrieved_ids, relevant_ids,k)
            recall_scores.append(recall)
            precision_scores.append(precision)
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)

            print("\n" + "=" * 80)
            print(f"Query: {query}")
            print(f"Method: {name}")
            print(f"Retrieved: {retrieved_ids}")
            print(f"Relevant:  {relevant_ids}")
            print(f"Recall@{k}: {recall:.2f}")
            print(f"Precision@{k}: {precision:.2f}")

        return {
            "method": name,
            "recall_at_k": (sum(recall_scores) / len(recall_scores) if recall_scores else 0.0),
            "precision_at_k": (sum(precision_scores) / len(precision_scores) if precision_scores else 0.0),
            "mrr_at_k": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            "ndcg_at_k": (sum(ndcg_scores) / len(ndcg_scores)if ndcg_scores else 0.0),
        }