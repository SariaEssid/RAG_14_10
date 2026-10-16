"""
tests/unit/test_rag_controller.py
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.rag_controller import (load_lang_chain_context)
from app.main import app

client = TestClient(app)
nbr_chunks = 2

class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def similarity_search(*_args, **_kwargs):
    return [
        DummyDoc(
            page_content="context-1",
            metadata={
                "file": "efa.md",
                "grade": "efa",
                "domain": "ef"
            }
        )
    ]


def similarity_search_with_score(*_args, **_kwargs):
    return [
        (
            DummyDoc(
                page_content="general-context",
                metadata={
                    "file": "general.md"
                }
            ),
            0.1
        ),
        (
            DummyDoc(
                page_content="filtered",
                metadata={
                    "file": "ef_test.md"
                }
            ),
            0.2
        )
    ]


class DummyIndex:
    pass

def test_load_lang_chain_context_with_grades():
    from unittest.mock import patch

    with patch("app.api.rag_controller.get_vector_store", return_value=DummyIndex()), \
         patch("app.api.rag_controller.detect_domain_from_grades", return_value=["ef"]):

        filename, context = load_lang_chain_context(
            question="question",
            domains=[],
            grades=["efa"],
            nbr_chunks=nbr_chunks
        )

    assert filename == "efa.md"
    assert "context-1" in context

def test_empty_index_branch():
    from unittest.mock import patch

    class EmptyIndex:
        def similarity_search_with_score(self, *a, **k):
            return []

    with patch("app.api.rag_controller.get_vector_store", return_value=EmptyIndex()):

        filename, context = load_lang_chain_context(
            question="q",
            domains=[],
            grades=[],
            nbr_chunks=nbr_chunks
        )

    assert filename == "empty"
    assert "Aucun context" in context

def test_grade_and_domain_branch():
    from unittest.mock import patch

    class DummyDoc:
        def __init__(self):
            self.page_content = "X"
            self.metadata = {"file": "efa.md", "grade": "efa", "domain": "ef"}

    class Index:
        def similarity_search(self, *a, **k):
            return [DummyDoc()]

    with patch("app.api.rag_controller.get_vector_store", return_value=Index()), \
         patch("app.api.rag_controller.detect_domain_from_grades", return_value=["ef"]):

        filename, context = load_lang_chain_context(
            "q", [], ["efa"], 2
        )

    assert filename == "efa.md"
    assert "X" in context

def test_domain_only_branch():
    from unittest.mock import patch

    class DummyDoc:
        def __init__(self):
            self.page_content = "D"
            self.metadata = {"file": "efa.md"}

    class Index:
        def similarity_search(self, *a, **k):
            return [DummyDoc()]

    with patch("app.api.rag_controller.get_vector_store", return_value=Index()):

        filename, context = load_lang_chain_context(
            "q", ["ef"], [], 2
        )

    assert filename == "efa.md"

def test_default_branch():
    from unittest.mock import patch

    class DummyDoc:
        def __init__(self):
            self.page_content = "GEN"
            self.metadata = {"file": "general.md"}

    class Index:
        def similarity_search_with_score(self, *a, **k):
            return [(DummyDoc(), 0.1)]

    with patch("app.api.rag_controller.get_vector_store", return_value=Index()):

        filename, context = load_lang_chain_context(
            "q", [], [], 2
        )

    assert filename == "general.md"
    assert "GEN" in context

def test_empty_context_branch():
    from unittest.mock import patch

    class Empty:
        def similarity_search_with_score(self, *a, **k):
            return []

    with patch("app.api.rag_controller.get_vector_store", return_value=Empty()):

        filename, context = load_lang_chain_context(
            "q", [], [], 2
        )

    assert filename == "empty"

def test_get_context_endpoint():
    with patch("app.api.rag_controller.EXPECTED_RAG_TOKEN", "test-service-token"), \
         patch(
        "app.api.rag_controller.load_lang_chain_context",
        return_value=("file.md", "context")
    ):

        response = client.post(
            "/rag/context",
            headers={"X-Service-Token": "test-service-token"},
            json={
                "question": "hello",
                "domains": [],
                "grades": [],
                "nbr_chunks": nbr_chunks
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "context"
    assert response.json()["source_file"] == "file.md"

def test_get_context_endpoint_rejects_missing_service_token():
    response = client.post(
        "/rag/context",
        json={
            "question": "hello",
            "domains": [],
            "grades": [],
            "nbr_chunks": nbr_chunks,
        },
    )

    assert response.status_code == 403

def test_get_context_endpoint_rejects_invalid_service_token():
    with patch("app.api.rag_controller.EXPECTED_RAG_TOKEN", "test-service-token"):
        response = client.post(
            "/rag/context",
            headers={"X-Service-Token": "wrong-token"},
            json={
                "question": "hello",
                "domains": [],
                "grades": [],
                "nbr_chunks": nbr_chunks,
            },
        )

    assert response.status_code == 401