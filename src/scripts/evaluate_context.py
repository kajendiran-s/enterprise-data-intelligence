import json
from pathlib import Path
from src.embeddings.service import GeminiEmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.context_builder import ContextBuilder
from src.evaluation.context_evaluator import ContextEvaluator


# ---------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------

EVAL_PATH = Path("data/evaluation/retrieval_eval.json")

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# ---------------------------------------------------------
# Initialize retrieval components
# ---------------------------------------------------------

embedding_model = GeminiEmbeddingService()

vector_store = VectorStore()

bm25_retriever = BM25Retriever()

reranker = Reranker()


# ---------------------------------------------------------
# Build Hybrid + Reranker retriever
#
# Flow:
#
# Dense Top-20
#       +
# BM25 Top-20
#       ↓
#     RRF
#       ↓
# 20 candidates
#       ↓
# Cross-Encoder
#       ↓
#    Top-5
# ---------------------------------------------------------

retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    embedding_model=embedding_model,
    reranker=reranker,
    dense_top_k=20,
    sparse_top_k=20,
    final_top_k=5,
)


# ---------------------------------------------------------
# Initialize Context Builder
# ---------------------------------------------------------

context_builder = ContextBuilder(
    max_context_chars=8000
)


# ---------------------------------------------------------
# Evaluate Context
# ---------------------------------------------------------

evaluator = ContextEvaluator(
    dataset=dataset
)

results = evaluator.evaluate(
    context_builder=context_builder,
    retriever=retriever,
    k=5,
)


# ---------------------------------------------------------
# Print final benchmark
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("CONTEXT EVALUATION")
print("=" * 80)

print(
    f"{'Context Recall':<25}"
    f"{'Context Precision':<25}"
)

print("-" * 80)

print(
    f"{results['context_recall']:<25.4f}"
    f"{results['context_precision']:<25.4f}"
)

print("=" * 80)