from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self,query: str, documents: list[dict], top_k: int = 5 ) -> list[dict]:
        if not documents:
            return []
        pairs = [(query, document["text"]) for document in documents]
        scores = self.model.predict(pairs)
        reranked_documents = []
        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked_documents.append(result)
        reranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]