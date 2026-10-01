from qdrant_client.models import PointStruct

from src.ingestion.loader import load_documents
from src.ingestion.chunking import chunk_document
from src.embeddings.service import GeminiEmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever

def main():
    documents = load_documents(r'data\documents')
    print(documents)
    embeddding_service = GeminiEmbeddingService()
    vector_store = VectorStore()
    bm25 = BM25Retriever()

    all_chunks = []

    for document in documents:
        chunks = chunk_document(document=document)
        print(
            f"Created {len(chunks)} chunks "
            f"for {document.get('document_id')}"
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks found")
        return

    print(
        f"Total chunks created: {len(all_chunks)}"
    )
    first_embedding = embeddding_service.embed_document(
        all_chunks[0]['text']
    )

    vector_store.create_collections(vector_size=len(first_embedding))

    points = []

    for idx, chunk in enumerate(all_chunks):
        vector = embeddding_service.embed_document(chunk['text'])
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    'chunk_id':chunk['chunk_id'],
                    'document_id':chunk['document_id'],
                    'source':chunk['document_source'],
                    'text':chunk['text'],
                    'metadata':chunk['metadata']
                }
            )
        )

    vector_store.upsert(points)

    print(f'Indexed {len(points)} chunks successfully')

    bm25_documents = []

    for chunk in all_chunks:

        bm25_documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "source": chunk["document_source"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
        )

    if not bm25_documents:
        raise RuntimeError(
            "BM25 documents are empty. "
            "Check chunking before building BM25."
        )

    bm25.build(bm25_documents)
    print(
        f"Indexed {len(bm25_documents)} chunks into BM25"
    )

    print("Hybrid indexing completed successfully")

if __name__ == "__main__":
    main()