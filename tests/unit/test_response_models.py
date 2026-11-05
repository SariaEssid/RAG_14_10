"""
tests/unit/test_response_models.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from app.models.response_models import RAGResponse

def test_response_model():
    resp = RAGResponse(answer="my answer")

    assert resp.answer == "my answer"