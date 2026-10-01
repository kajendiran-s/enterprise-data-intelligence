from typing import Any


class ContextBuilder:

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def build(self, documents: list[dict[str, Any]],) -> dict[str, Any]:
        if not documents:
            return {
                "context": "",
                "sources": [],
            }

        selected_documents = self._select_documents(documents)
        context_parts = []
        sources = []
        for index, document in enumerate(selected_documents,start=1):
            source = document.get("source", "Unknown source")
            text = document.get("text", "").strip()
            if not text:
                continue
            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {source}\n"
                f"Content:\n{text}"
            )
            sources.append(
                {
                    "chunk_id": document.get("chunk_id"),
                    "document_id": document.get("document_id"),
                    "source": source,
                    "rerank_score": document.get(
                        "rerank_score"
                    ),
                }
            )
        context = "\n\n".join(context_parts)
        return {
            "context": context,
            "sources": sources,
        }

    def _select_documents( self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = []
        current_length = 0
        seen_chunks = set()

        for document in documents:
            chunk_id = document.get("chunk_id")
            if chunk_id in seen_chunks:
                continue
            text = document.get("text", "").strip()
            if not text:
                continue
            additional_length = len(text)
            if (current_length + additional_length > self.max_context_chars):
                break
            selected.append(document)
            seen_chunks.add(chunk_id)
            current_length += additional_length
        return selected