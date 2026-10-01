# from src.llm import generate_response


# answer = generate_response(
#     system_prompt=(
#         "You are an enterprise data engineering assistant. "
#         "Answer clearly and technically."
#     ),
#     user_prompt=(
#         "Explain why Spark repartition can cause a shuffle."
#     )
# )

# print(answer)

from src.ingestion import load_documents


documents = load_documents("data/documents")

for document in documents:
    print(document["document_id"])