from src.generation.context_builder import ContextBuilder


def test_context_builder():

    documents = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "source": "spark.md",
            "text": (
                "Spark divides applications into stages "
                "and tasks for distributed execution."
            ),
            "rerank_score": 8.5,
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_1",
            "source": "spark.md",
            "text": (
                "Executors run tasks on worker nodes "
                "and process partitions of data."
            ),
            "rerank_score": 7.9,
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "doc_2",
            "source": "spark_cluster.md",
            "text": (
                "The Spark driver coordinates execution "
                "and schedules tasks for executors."
            ),
            "rerank_score": 7.1,
        },
    ]

    builder = ContextBuilder(
        max_context_chars=8000
    )

    result = builder.build(documents)

    assert result["context"]

    assert len(result["sources"]) == 3

    assert "Source 1" in result["context"]
    assert "spark.md" in result["context"]

    assert result["sources"][0]["chunk_id"] == "chunk_1"

    print("\n" + "=" * 80)
    print("GENERATED CONTEXT")
    print("=" * 80)

    print(result["context"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(source)

if __name__ == "__main__":
    test_context_builder()