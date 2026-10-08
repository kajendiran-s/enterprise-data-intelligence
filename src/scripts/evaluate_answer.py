import json
import time
from pathlib import Path

from src.rag.pipeline import RAGpipeline
from src.evaluation.answer_evaluator import AnswerEvaluator


EVAL_PATH = Path("data/evaluation/answer_eval.json")

MAX_RETRIES = 4
RETRY_DELAY = 5


with open(EVAL_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


rag = RAGpipeline()

generated_answers = []

print("=" * 80)
print("GENERATING ANSWERS")
print("=" * 80)


for index, item in enumerate(dataset, start=1):

    query = item["query"]

    print(
        f"\n[{index}/{len(dataset)}] "
        f"Generating answer for: {query}"
    )

    result = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            result = rag.answer(
                question=query,
                top_k=5,
            )

            break

        except Exception as error:

            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed:"
                f" {type(error).__name__}"
            )

            if attempt == MAX_RETRIES:
                print("  Skipping this query.")
                break

            delay = RETRY_DELAY * attempt

            print(
                f"  Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    if result is None:
        continue

    generated_answers.append(
        {
            "query": query,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    )


# ---------------------------------------------------------
# Evaluate generated answers
# ---------------------------------------------------------

evaluator = AnswerEvaluator(
    dataset=dataset
)

results = evaluator.evaluate(
    generated_answers=generated_answers
)


# ---------------------------------------------------------
# Final benchmark
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("ANSWER EVALUATION")
print("=" * 80)

print(
    f"{'Metric':<30}"
    f"{'Score':<15}"
)

print("-" * 80)

print(
    f"{'Concept Coverage':<30}"
    f"{results['answer_concept_score']:<15.4f}"
)

print("=" * 80)