from src.rag.retriever import Retriever
from src.generation.context_builder import ContextBuilder
from src.generation.generator import Generator

class RAGpipeline:

    def  __init__(self):
        self.retriever = Retriever()
        self.context_builder = ContextBuilder()
        self.generator = Generator()

    def _resolve_citations(self, citation_ids: list[int], sources: list[dict],) -> list[dict]:
        resolved = []
        for citation_id in citation_ids:
            index = citation_id - 1
            if 0 <= index < len(sources):
                resolved.append(sources[index])
        return resolved

    def answer(self,question:str,top_k:int=5):
        documents = self.retriever.retrieve(
                        query=question,
                        candidate_k=20,
                        top_k=5 )


        context_result = self.context_builder.build( documents)
        generated = self.generator.generate(question=question,context=context_result["context"])
        sources = self._resolve_citations(generated.citations,context_result["sources"])
        return { "answer": generated.answer, "sources": sources, "citations": generated.citations}