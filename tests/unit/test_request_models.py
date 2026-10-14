from app.models.request_models import RAGRequest


def test_request_model():
    req = RAGRequest(
        question="hello",
        domain="ef",
        grade="efa",
        k=3
    )

    assert req.question == "hello"
    assert req.domain == "ef"
    assert req.grade == "efa"
    assert req.k == 3