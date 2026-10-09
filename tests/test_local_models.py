from src.rag.pipeline import RAGpipeline

rag = RAGpipeline()

result = rag.answer(
    "How does Spark distribute work across executors?"
)

print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)