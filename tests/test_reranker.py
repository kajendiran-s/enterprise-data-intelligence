from src.retrieval.reranker import Reranker


def test_reranker():

    documents = [
        {
            "chunk_id": "1",
            "text": (
                "Spark executors run tasks on worker nodes "
                "and process partitions of data."
            )
        },
        {
            "chunk_id": "2",
            "text": (
                "Spark DataFrames provide a structured API "
                "for distributed data processing."
            )
        },
        {
            "chunk_id": "3",
            "text": (
                "The Spark driver coordinates the application "
                "and schedules tasks for executors."
            )
        }
    ]

    query = "How does Spark distribute work across executors?"

    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=2
    )

    assert len(results) == 2

    assert "rerank_score" in results[0]

    assert (
        results[0]["rerank_score"]
        >= results[1]["rerank_score"]
    )

    print("\nReranked results:")

    for result in results:
        print(
            result["chunk_id"],
            result["rerank_score"],
            result["text"]
        )

if __name__ == '__main__':
    test_reranker()