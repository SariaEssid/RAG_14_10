"""
app/models/request_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from typing import Optional
from pydantic import BaseModel

class RAGRequest(BaseModel):
    question: str
    domain: Optional[str] = None
    grade: Optional[str] = None
    k: int = 5
