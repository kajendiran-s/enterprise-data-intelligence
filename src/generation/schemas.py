from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="Answer to the user's question based only on the provided context."
    )

    citations: list[int] = Field(
        description="Source numbers from the provided context that support the answer."
    )