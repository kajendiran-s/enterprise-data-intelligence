
import json
from src.rag.pipeline import RAGpipeline

class AgentTools:
    def __init__(self):
        self.rag = RAGpipeline()

    
    def search_knowledge_base(self, question: str) -> str:
        """Search enterprise documents and return traceable source metadata."""
        result = self.rag.answer(
            question=question,
            top_k=5,
        )

        citations = result.get("citations", [])
        sources = result.get("sources", [])

        # The pipeline resolves sources in citation order.
        # Only build an explicit mapping when both lists align.
        source_map = []

        if len(citations) == len(sources):
            for citation_id, source in zip(citations, sources):
                source_map.append({
                    "citation": f"[Source {citation_id}]",
                    "citation_id": citation_id,
                    "document": source.get("source", "Unknown document"),
                    "document_id": source.get("document_id"),
                    "chunk_id": source.get("chunk_id"),
                })

        tool_output = {
            "answer": result["answer"],
            "citations": citations,
            "sources": sources,
            "source_map": source_map,
        }

        return json.dumps(tool_output, ensure_ascii=False, default=str)


    def get_system_capabilities(self) -> str:
        capabilities = {
            "project": "Enterprise Data Intelligence",
            "language": "Python",
            "generation_model": "qwen2.5:7b via Ollama",
            "embedding_model": "nomic-embed-text",
            "retrieval": [
                "Dense vector retrieval using Qdrant",
                "Sparse retrieval using BM25",
                "Hybrid retrieval using Reciprocal Rank Fusion",
                "Cross-encoder reranking",
            ],
            "generation_features": [
                "Context construction",
                "Citation resolution",
                "Answer evaluation",
                "Citation integrity evaluation",
            ],
            "agent_status": "Basic tool-calling loop under development",
        }

        return json.dumps(capabilities, ensure_ascii=False)
