def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:

    if not relevant_chunk_ids:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    relevant = set(relevant_chunk_ids)

    hits = retrieved.intersection(relevant)

    return len(hits) / len(relevant)

def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:

    if k <= 0:
        return 0.0

    retrieved = retrieved_chunk_ids[:k]
    relevant = set(relevant_chunk_ids)

    hits = sum(
        1
        for chunk_id in retrieved
        if chunk_id in relevant
    )

    return hits / len(retrieved)