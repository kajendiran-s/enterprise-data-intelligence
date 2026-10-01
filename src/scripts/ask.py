from src.rag.pipeline import RAGPipeline


rag = RAGPipeline()

result = rag.answer(
    "How does Spark distribute work across executors?"
)

print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)