"""
app/models/request_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    domains: list[str] = Field(default_factory=list)
    grades: list[str] = Field(default_factory=list)
    nbr_chunks: int = Field(ge=1, le=20)