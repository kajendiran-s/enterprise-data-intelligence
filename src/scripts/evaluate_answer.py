
import json
import time
from pathlib import Path

from src.rag.pipeline import RAGpipeline
from src.evaluation.answer_evaluator import AnswerEvaluator
from src.evaluation.citation_check import validate_citations


EVAL_PATH = Path("data/evaluation/answer_eval.json")
OUTPUT_PATH = Path("data/evaluation/generated_answers.json")
RESULTS_PATH = Path("data/evaluation/answer_evaluation_results.json")

MAX_RETRIES = 3
RETRY_DELAY = 3


def save_json(path: Path, data) -> None:
    """Save atomically so an interrupted run doesn't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


with EVAL_PATH.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

if OUTPUT_PATH.exists():
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        saved_answers = json.load(file)
else:
    saved_answers = []

answers_by_query = {
    item["query"]: item for item in saved_answers
}

rag = RAGpipeline()

print("=" * 80)
print("GENERATING ANSWERS AND PRESERVING CITATIONS")
print("=" * 80)

for index, item in enumerate(dataset, start=1):
    query = item["query"]
    existing = answers_by_query.get(query)

    # Skip only records that already contain the citations field.
    # Old records without it must be regenerated to recover real citation IDs.
    if existing is not None and "citations" in existing:
        print(f"\n[{index}/{len(dataset)}] Already completed. Skipping.")
        continue

    if existing is not None:
        print(f"\n[{index}/{len(dataset)}] Repairing old record: {query}")
    else:
        print(f"\n[{index}/{len(dataset)}] {query}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = rag.answer(
                question=query,
                top_k=5,
            )

            record = {
                "query": query,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "context": result.get("context", ""),
                "citations": result.get("citations", []),
            }

            # Replace the old record with this new, complete generation.
            answers_by_query[query] = record

            save_json(
                OUTPUT_PATH,
                list(answers_by_query.values()),
            )

            print(f"Answer saved. Citations: {record['citations']}")
            break

        except Exception as error:
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                print("Query failed. It can be retried on the next run.")


generated_answers = list(answers_by_query.values())

print("\n" + "=" * 80)
print("EVALUATING ANSWERS WITH LOCAL QWEN")
print("=" * 80)

evaluator = AnswerEvaluator(dataset=dataset)
results = evaluator.evaluate(generated_answers)

# Evaluate citations separately from the LLM judge's quality scores.
dataset_queries = {item["query"] for item in dataset}
evaluated_records = [
    record
    for record in generated_answers
    if record.get("query") in dataset_queries
]

citation_count = 0
valid_citation_count = 0
answers_with_citations = 0
answers_with_valid_citations = 0
citation_details = []

for record in evaluated_records:
    citations = record.get("citations", [])
    context = record.get("context", "")

    check = validate_citations(citations, context)

    citation_count += len(citations)
    valid_citation_count += len(check["valid_citation_ids"])

    if citations:
        answers_with_citations += 1

        if check["valid"]:
            answers_with_valid_citations += 1

    citation_details.append({
        "query": record["query"],
        "citations": citations,
        **check,
    })

answer_count = len(evaluated_records)

citation_metrics = {
    "answers_checked": answer_count,
    "answers_with_citations": answers_with_citations,
    "citation_coverage": (
        answers_with_citations / answer_count if answer_count else 0.0
    ),
    "total_citation_ids": citation_count,
    "valid_citation_ids": valid_citation_count,
    "citation_precision": (
        valid_citation_count / citation_count if citation_count else 0.0
    ),
    "answers_with_nonempty_valid_citations": answers_with_valid_citations,
    "valid_citation_answer_rate": (
        answers_with_valid_citations / answer_count
        if answer_count else 0.0
    ),
    "details": citation_details,
}

results["citation_metrics"] = citation_metrics

save_json(RESULTS_PATH, results)

print("\n" + "=" * 80)
print("FINAL ANSWER EVALUATION")
print("=" * 80)

for metric in ("correctness", "completeness", "groundedness"):
    print(f"{metric.capitalize():<20}: {results[metric]:.4f}")

print(f"{'Evaluated':<20}: {results['evaluated_count']}")
print(f"{'Missing':<20}: {results['missing_count']}")
print(f"{'Total':<20}: {results['total_count']}")

print("\n" + "=" * 80)
print("CITATION EVALUATION")
print("=" * 80)

print(f"{'Answers checked':<35}: {answer_count}")
print(f"{'Answers with citations':<35}: {answers_with_citations}")
print(f"{'Citation coverage':<35}: {citation_metrics['citation_coverage']:.2%}")
print(f"{'Total citation IDs':<35}: {citation_count}")
print(f"{'Valid citation IDs':<35}: {valid_citation_count}")
print(f"{'Citation precision':<35}: {citation_metrics['citation_precision']:.2%}")
print(
    f"{'Answers with valid citations':<35}: "
    f"{answers_with_valid_citations}"
)
print(
    f"{'Valid citation answer rate':<35}: "
    f"{citation_metrics['valid_citation_answer_rate']:.2%}"
)

invalid_details = [
    detail for detail in citation_details
    if detail["invalid_citation_ids"] or not detail["citations"]
]

if invalid_details:
    print("\nAnswers needing citation review:")

    for detail in invalid_details:
        print(f"\nQuery: {detail['query']}")
        print(f"Citations: {detail['citations']}")
        print(f"Available source IDs: {detail['available_source_ids']}")
        print(f"Invalid citation IDs: {detail['invalid_citation_ids']}")

print(f"\nGenerated answers: {OUTPUT_PATH}")
print(f"Evaluation results: {RESULTS_PATH}")
