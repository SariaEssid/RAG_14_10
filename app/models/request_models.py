"""
app/models/request_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator


class RAGRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    domains: list[str] = Field(default_factory=list, max_length=10)
    grades: list[str] = Field(default_factory=list, max_length=10)
    nbr_chunks: int = Field(ge=1, le=20)

    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value