from google import genai
from google.genai import types
from src.config import settings

class GeminiEmbeddingService:

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = 'gemini-embedding-2'

    def embed_document(self, text:str) -> list[float]:

        embed_res = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT"
                        )
        )

        return embed_res.embeddings[0].values

    def embed_query(self, query:str) -> list[float]:

        query_res = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        return query_res.embeddings[0].values