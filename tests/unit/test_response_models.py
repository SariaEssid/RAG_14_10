from app.models.response_models import RAGResponse


def test_response_model():
    resp = RAGResponse(answer="my answer")

    assert resp.answer == "my answer"