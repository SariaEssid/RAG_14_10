"""
app/models/request_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from pydantic import BaseModel, Field, field_validator

from app.core.metadata_config import ALLOWED_DOMAINS, ALLOWED_GRADES

class RAGRequest(BaseModel):
    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=2000)
    domains: list[str] = Field(default_factory=list, max_length=10)
    grades: list[str] = Field(default_factory=list, max_length=10)
    nbr_chunks: int = Field(ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, values: list[str]) -> list[str]:
        invalid_domains = set(values) - ALLOWED_DOMAINS
        if invalid_domains:
            raise ValueError(
                f"Unsupported domains: {sorted(invalid_domains)}"
            )
        return values

    @field_validator("grades")
    @classmethod
    def validate_grades(cls, values: list[str]) -> list[str]:
        invalid_grades = set(values) - ALLOWED_GRADES
        if invalid_grades:
            raise ValueError(
                f"Unsupported grades: {sorted(invalid_grades)}"
            )
        return values

