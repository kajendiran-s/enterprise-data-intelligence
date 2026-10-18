
from src.rag.retriever import Retriever
from src.generation.context_builder import ContextBuilder
from src.generation.generator import Generator


class RAGpipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.context_builder = ContextBuilder()
        self.generator = Generator()

    def _resolve_citations(
        self,
        citation_ids: list[int],
        sources: list[dict],
    ) -> list[dict]:
        resolved = []
        seen_ids = set()

        for citation_id in citation_ids:
            # Source numbers in the generated answer are 1-based.
            index = citation_id - 1

            if (
                isinstance(citation_id, int)
                and not isinstance(citation_id, bool)
                and 1 <= citation_id <= len(sources)
                and citation_id not in seen_ids
            ):
                resolved.append(sources[index])
                seen_ids.add(citation_id)

        return resolved

    def answer(self, question: str, top_k: int = 5):
        documents = self.retriever.retrieve(
            query=question,
            candidate_k=20,
            top_k=top_k,
        )

        context_result = self.context_builder.build(documents)
        context = context_result["context"]
        context_sources = context_result["sources"]

        generated = self.generator.generate(
            query=question,
            context=context,
        )

        resolved_sources = self._resolve_citations(
            citation_ids=generated.citations,
            sources=context_sources,
        )

        return {
            "answer": generated.answer,
            "sources": resolved_sources,
            "citations": generated.citations,
            "context": context,
        }
