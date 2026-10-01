from typing import List

def chunk_document(
        document:dict,
        max_chars:int = 1000,
        overlap_chars:int = 150
) -> list[dict]:

    text = document["text"]

    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start+max_chars, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_id": f"{document['document_id']}_chunk_{chunk_index}",
                "document_id" : document["document_id"],
                "document_source": document["source"],\
                "text" : chunk,
                "metadata": {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                }
                
            })

        if end == text_length:
            break

        start = end

        chunk_index+=1
    print(chunks)
    return chunks