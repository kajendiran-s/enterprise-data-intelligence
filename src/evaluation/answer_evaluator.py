from typing import Any

from pydantic import BaseModel, Field
from google import genai

class AnswerJudgeResult(BaseModel):
    """
    Structured evaluation returned by Gemini.
    Scores:
        correctness:
            Is the answer factually correct?
        completeness:
            Does the answer cover the important concepts?
        groundedness:
            Is the answer supported by the provided context?
    """
    correctness: float = Field(ge=0.0,le=1.0,description="How factually correct the answer is.")
    completeness: float = Field(ge=0.0,le=1.0,description="How completely the answer addresses the expected concepts.")
    groundedness: float = Field(ge=0.0,le=1.0,description="How well the answer is supported by the provided context.")
    explanation: str = Field(description="Brief explanation of the evaluation.")

class AnswerEvaluator:
    def __init__(self,dataset: list[dict],model: str = "gemini-3.8-flash"):
        self.dataset = dataset
        self.model = model
        self.client = genai.Client()

    def evaluate_answer(self,query: str,
                        answer: str,reference_answer: str,
                        required_concepts: list[str],context: str) -> dict[str, Any]:
        concepts = "\n".join(f"- {concept}" for concept in required_concepts)
        prompt = f"""
                    You are an evaluator for an enterprise RAG system.

                    Evaluate the generated answer against the user question,
                    reference answer, required concepts, and retrieved context.

                    Your evaluation must be semantic.

                    Do NOT require exact wording.

                    For example:

                    Required concept:
                    "action triggers a job"

                    Generated answer:
                    "When an action is triggered, Spark creates a job."

                    This should be considered correct.

                    Evaluate the following:

                    QUESTION:
                    {query}

                    REFERENCE ANSWER:
                    {reference_answer}

                    REQUIRED CONCEPTS:
                    {concepts}

                    RETRIEVED CONTEXT:
                    {context}

                    GENERATED ANSWER:
                    {answer}

                    Evaluation criteria:

                    1. CORRECTNESS
                    Determine whether the generated answer is factually correct
                    and answers the question without contradictions.

                    2. COMPLETENESS
                    Determine whether the answer covers the important concepts
                    required by the reference answer and required concepts.

                    3. GROUNDEDNESS
                    Determine whether the claims made by the answer are supported
                    by the retrieved context.

                    Do not penalize the answer merely because it uses different
                    wording from the reference answer.

                    Return scores between 0.0 and 1.0.
                """
        response = self.client.models.generate_content(model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": AnswerJudgeResult,
            },
        )
        result = AnswerJudgeResult.model_validate_json(response.text)
        return result.model_dump()

    def evaluate(self,generated_answers: list[dict]) -> dict[str, Any]:
        generated_by_query = {item["query"]: item for item in generated_answers}
        correctness_scores = []
        completeness_scores = []
        groundedness_scores = []
        evaluated_count = 0
        for item in self.dataset:
            query = item["query"]
            if query not in generated_by_query:
                print(
                    f"\nSkipping evaluation because no answer "
                    f"was generated for: {query}"
                )
                continue
            generated = generated_by_query[query]
            answer = generated["answer"]
            reference_answer = item["reference_answer"]
            required_concepts = item["required_concepts"]
            context = generated.get("context", "")
            result = self.evaluate_answer(
                query=query,
                answer=answer,
                reference_answer=reference_answer,
                required_concepts=required_concepts,
                context=context,
            )
            correctness_scores.append(result["correctness"])
            completeness_scores.append(result["completeness"])
            groundedness_scores.append(result["groundedness"])
            evaluated_count += 1
            print("\n" + "=" * 80)
            print(f"Query: {query}")
            print("-" * 80)
            print(f"Generated Answer:\n{answer}")
            print(f"\nCorrectness: " f"{result['correctness']:.4f}")
            print(f"Completeness: "f"{result['completeness']:.4f}")
            print(f"Groundedness: "f"{result['groundedness']:.4f}")
            print(f"\nJudge Explanation:\n" f"{result['explanation']}")

        return {
            "correctness": (sum(correctness_scores)/ len(correctness_scores) if correctness_scores else 0.0),
            "completeness": (sum(completeness_scores)/ len(completeness_scores) if completeness_scores else 0.0),
            "groundedness": (sum(groundedness_scores)/ len(groundedness_scores) if groundedness_scores else 0.0),
            "evaluated_count": evaluated_count,
            "total_count": len(self.dataset),
        }