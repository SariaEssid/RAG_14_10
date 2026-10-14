"""
app/models/test_response_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from pydantic import BaseModel

class RAGResponse(BaseModel):
    answer: str