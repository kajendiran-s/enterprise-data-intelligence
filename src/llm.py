from google import genai
from google.genai import types

from src.config import settings


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


def generate_response(
    system_prompt: str,
    user_prompt: str
) -> str:

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature=0.2
        )
    )
# response = client.models.generate_content(
#     model="gemini-3.8-flash",
#     contents=user_prompt,
#     config=types.GenerateContentConfig(
#         system_instruction=system_prompt,
#         temperature=0.2
#     )
# )
    return response.text