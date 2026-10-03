import math

#Recall definition
def recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str],k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    
    retrieved = set(retrieved_chunk_ids[:k])
    relevant = set(relevant_chunk_ids)

    hits = retrieved.intersection(relevant)
    return len(hits) / len(relevant)

#precision definition
def precision_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    
    retrieved = retrieved_chunk_ids[:k]
    relevant = set(relevant_chunk_ids)

    hits = sum(1 for chunk_id in retrieved if chunk_id in relevant)
    return hits / len(retrieved)

#mrr definition
def mrr_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    relevant = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank     
    return 0.0

#ndcg definition
def ndcg_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0

    relevant = set(relevant_chunk_ids)
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k],start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg