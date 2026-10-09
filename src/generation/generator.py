# from google import genai
# from src.config import settings
# from src.generation.schemas import GeneratedAnswer


# class Generator:

#     def __init__(self, model_name: str = "gemini-3.8-flash"):
#         self.client = genai.Client( api_key=settings.GEMINI_API_KEY)
#         self.model_name = model_name

#     def generate(self, query: str, context: str) -> GeneratedAnswer:

#         prompt = f"""
#                     You are an enterprise data intelligence assistant.

#                     Answer the user's question using ONLY the provided context.

#                     Rules:

#                     1. Use only information supported by the context.
#                     2. Do not invent facts.
#                     3. If the context does not contain enough information,
#                     say that the available information is insufficient.
#                     4. Every factual claim should be supported by one or more
#                     source numbers from the provided context.
#                     5. citations must contain only valid source numbers.
#                     6. If the context is insufficient, return an empty citations list.

#                     User Question:
#                     {query}

#                     Context:
#                     {context}
#                 """

#         response = self.client.models.generate_content(model=self.model_name, contents=prompt,
#             config={
#                 "response_mime_type": "application/json",
#                 "response_schema": GeneratedAnswer,
#             },
#         )

#         return GeneratedAnswer.model_validate_json(response.text)



import ollama
from src.generation.schemas import GeneratedAnswer

class Generator:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.client = ollama.Client(host="http://localhost:11434")
        self.model_name = model_name

    def generate(self,query: str,context: str,) -> GeneratedAnswer:

        prompt = f"""
                    You are an enterprise data intelligence assistant.

                    Answer the user's question using ONLY the provided context.

                    Rules:
                    1. Use only information supported by the context.
                    2. Do not invent facts.
                    3. If the context lacks sufficient information, clearly say so.
                    4. Support every factual claim with valid source numbers from the context.
                    5. Use only source numbers that actually appear in the context.
                    6. If the context is insufficient to answer, return an empty citations list.

                    User Question:
                    {query}

                    Context:
                    {context}
                """

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=GeneratedAnswer.model_json_schema(),
            options={"temperature": 0},
        )
        answer_json = response["message"]["content"]
        return GeneratedAnswer.model_validate_json(answer_json)
