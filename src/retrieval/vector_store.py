from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import settings

class VectorStore:

    def __init__(
        self,
        collection_name:str = 'enterprise_documents'
    ):
        self.client = QdrantClient(
            url=settings.QDRANT_URL
        )

        self.collection_name=collection_name


    def create_collections(self, vector_size:int):

        collections = self.client.get_collections()

        existing = {collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size = vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert(self, points: list[PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
            self,
            query_vector:list[float],
            top_k:int = 5,
            document_id:str|None=None
    ):

        query_filter = None
        if document_id is not None:
            query_filter=Filter(
                must=[FieldCondition(
                    key='document_id',
                    match=MatchValue(value=document_id)
                )]
            )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        normalized_results = []

        for point in results.points:
            payload = point.payload or {}
            normalized_results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "source": payload.get( "source"),
                    "text": payload.get("text"),
                    "metadata": payload.get( "metadata",{} ),
                    "score": float(point.score),
                    "retrieval_type": "dense",
                }
            )

        return normalized_results