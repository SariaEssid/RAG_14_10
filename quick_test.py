# quick_test.py
import logging

from pydantic import ValidationError
from pydantic import field_validator

from app.models.request_models import RAGRequest

log = logging.getLogger(__name__)

print(field_validator)
class SARIA_ROMDHANI_ESSID:
    """
    Classe pour nettoyer et formater les réponses textuelles.
    """

    @staticmethod
    def util_values():

        base = {
            "question": "Test question",
            "domains": ["ef"],
            "grades": ["ef"],
            "nbr_chunks": 2,
        }

        # Valid request
        request = RAGRequest(**base)
        print("VALID:", request.model_dump())

        # Invalid domain
        try:
            RAGRequest(**{**base, "domains": ["unknown"]})
        except ValidationError as exc:
            print("INVALID DOMAIN: correctly rejected")

        # Invalid grade
        try:
            RAGRequest(**{**base, "grades": ["unknown"]})
        except ValidationError as exc:
            print("INVALID GRADE: correctly rejected")

        # Blank question
        try:
            RAGRequest(**{**base, "question": "   "})
        except ValidationError as exc:
            print("BLANK QUESTION: correctly rejected")

# --- Zone de Test ---
if __name__ == "__main__":
    SARIA_ROMDHANI_ESSID.util_values()