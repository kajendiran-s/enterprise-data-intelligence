
from src.rag.pipeline import RAGpipeline

rag = RAGpipeline()

result = rag.answer(
    question="How does Spark distribute work across executors?",
    top_k=5,
)

print("ANSWER:")
print(result["answer"])

print("\nCITATIONS:")
print(result["citations"])

print("\nRESOLVED SOURCES:")
for source in result["sources"]:
    print(source)

print("\nCONTEXT LENGTH:")
print(len(result["context"]))

assert isinstance(result["context"], str)
assert result["context"].strip(), "Retrieved context is empty"

print("\nSmoke test passed.")
