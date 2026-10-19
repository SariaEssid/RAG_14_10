"""
app/models/test_response_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""
from typing import Optional

from pydantic import BaseModel

class RAGResponse(BaseModel):
    answer: str
    source_file: Optional[str] = None