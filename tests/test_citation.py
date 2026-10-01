from src.rag.pipeline import RAGPipeline


def test_resolve_citations():

    rag = RAGPipeline()

    sources = [
        {
            "chunk_id": "chunk_1",
            "source": "spark.md",
        },
        {
            "chunk_id": "chunk_2",
            "source": "spark_driver.md",
        },
        {
            "chunk_id": "chunk_3",
            "source": "spark_sql.md",
        },
    ]

    resolved = rag._resolve_citations(
        citation_ids=[1, 3],
        sources=sources,
    )

    assert len(resolved) == 2

    assert resolved[0]["chunk_id"] == "chunk_1"
    assert resolved[1]["chunk_id"] == "chunk_3"