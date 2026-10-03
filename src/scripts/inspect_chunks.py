from src.ingestion.loader import load_documents
from src.ingestion.chunking import chunk_document


documents = load_documents("data/documents")

for document in documents:

    chunks = chunk_document(document)

    print("\n" + "=" * 100)
    print(f"DOCUMENT: {document.get('source')}")
    print("=" * 100)

    for chunk in chunks:

        print(f"\nChunk ID: {chunk['chunk_id']}")
        print("-" * 80)
        print(chunk["text"])