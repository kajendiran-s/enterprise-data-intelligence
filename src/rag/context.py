def build_context(results):
    context_parts =[]

    for idx, result in enumerate(results, start=1):
        payload = result.payload

        context_parts.append(
            f"""
            SOURCE: {idx}
            Document: {payload["document_id"]}
            File: {payload['source']}
            Similarity: {result.score:.4f}
            content: {payload['text']}
            """
        )

    return "/n".join(context_parts)
    