from src.retrieval.reranker import Reranker

class HybridRetriever:

    def __init__(
        self,vector_store, bm25_retriever, embedding_model, reranker=None,
        dense_top_k: int = 20, sparse_top_k: int = 20, final_top_k: int = 5, rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def _document_id(self, document: dict) -> str:

        return str(
            document.get(
                "chunk_id",
                document.get("id")
            )
        )

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        final_top_k: int,
    ) -> list[dict]:

        fused = {}

        # Process dense results
        for rank, doc in enumerate(dense_results, start=1):

            doc_id = self._document_id(doc)

            if doc_id not in fused:

                fused[doc_id] = {
                    **doc,
                    "dense_score": doc.get("score"),
                    "bm25_score": None,
                    "rrf_score": 0.0,
                    "retrieval_sources": set(),
                }

            fused[doc_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

            fused[doc_id]["retrieval_sources"].add(
                "dense"
            )

        # Process BM25 results
        for rank, doc in enumerate(bm25_results, start=1):

            doc_id = self._document_id(doc)

            if doc_id not in fused:

                fused[doc_id] = {
                    **doc,
                    "dense_score": None,
                    "bm25_score": doc.get("score"),
                    "rrf_score": 0.0,
                    "retrieval_sources": set(),
                }

            else:
                fused[doc_id]["bm25_score"] = doc.get("score")

            fused[doc_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

            fused[doc_id]["retrieval_sources"].add(
                "bm25"
            )

        results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        for result in results:

            result["retrieval_sources"] = sorted(
                result["retrieval_sources"]
            )
            result["score"] = result["rrf_score"]
            result.pop("retrieval_type", None)

        return results[:final_top_k]

    def search(
        self,
        query: str,
        candidate_k:int=20,
        top_k: int | None = 5,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.final_top_k

        # 1. Dense retrieval
        query_embedding = (self.embedding_model.embed_query(query))

        dense_results = self.vector_store.search( query_embedding, self.dense_top_k,)

        # 2. BM25 retrieval
        bm25_results = self.bm25.search(query, self.sparse_top_k,)

        # 3. RRF fusion
        hybrid_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            final_top_k=top_k,
        )[:candidate_k]

        if self.reranker is not None:
             hybrid_results = self.reranker.rerank(query=query, documents=hybrid_results, top_k=top_k)

        return hybrid_results

