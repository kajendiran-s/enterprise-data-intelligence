
import ollama


class NomicEmbeddingService:

    def __init__(
        self,
        model_name: str = "nomic-embed-text:latest",
    ):
        self.client = ollama.Client(
            host="http://localhost:11434"
        )
        self.model_name = model_name

    def embed_document(self, text: str) -> list[float]:
        response = self.client.embed(
            model=self.model_name,
            input=f"search_document: {text}",
        )

        return response["embeddings"][0]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embed(
            model=self.model_name,
            input=f"search_query: {query}",
        )

        return response["embeddings"][0]
