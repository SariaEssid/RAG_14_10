"""Unit tests for the RAG controller."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.api.rag_controller import (
    detect_domain_from_grades,
    load_lang_chain_context,
    rank_and_limit, search_with_scores,
)
from app.main import app


client = TestClient(app)
TEST_TOKEN = "test-service-token"


class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class DummyIndex:
    """Fake with the search method used by the production controller."""

    def __init__(self, scored_docs=None, ntotal=None):
        self.scored_docs = scored_docs or []
        self.index = SimpleNamespace(
            ntotal=len(self.scored_docs) if ntotal is None else ntotal
        )
        self.search_calls = []

    def similarity_search_with_score(self, question, k, filter=None):
        self.search_calls.append(
            {"question": question, "k": k, "filter": filter}
        )

        results = self.scored_docs
        if filter:
            results = [
                (doc, score)
                for doc, score in results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]

        return results[:k]

def test_empty_index_returns_without_searching():
    index = DummyIndex(ntotal=0)

    with patch("app.api.rag_controller.get_vector_store", return_value=index):
        filename, context = load_lang_chain_context(
            question="question",
            domains=[],
            grades=[],
            nbr_chunks=2,
        )

    assert filename == "empty"
    assert "Aucun contexte" in context
    assert index.search_calls == []

def test_detect_domain_from_grades_ef():
    assert detect_domain_from_grades(["efa"]) == ["ef"]


def test_detect_domain_from_grades_elcc():
    assert detect_domain_from_grades(["elcc1"]) == ["elcc"]


def test_detect_domain_from_grades_no_match():
    assert detect_domain_from_grades(["other"]) == []


def test_load_lang_chain_context_with_grade_filter():
    doc = DummyDoc(
        "grade context",
        {"file": "efa.md", "grade": "efa", "domain": "ef"},
    )
    index = DummyIndex([(doc, 0.4)])

    with patch("app.api.rag_controller.get_vector_store", return_value=index):
        filename, context = load_lang_chain_context(
            question="question",
            domains=[],
            grades=["efa"],
            nbr_chunks=2,
        )

    assert filename == "efa.md"
    assert "grade context" in context
    assert any(call["filter"] == {"grade": "efa"} for call in index.search_calls)


def test_load_lang_chain_context_with_domain_filter():
    doc = DummyDoc("domain context", {"file": "ef.md", "domain": "ef"})
    index = DummyIndex([(doc, 0.3)])

    with patch("app.api.rag_controller.get_vector_store", return_value=index):
        filename, context = load_lang_chain_context(
            question="question",
            domains=["ef"],
            grades=[],
            nbr_chunks=2,
        )

    assert filename == "ef.md"
    assert "domain context" in context
    assert any(call["filter"] == {"domain": "ef"} for call in index.search_calls)


def test_load_lang_chain_context_without_filters_ranks_and_limits_results():
    ef_doc = DummyDoc("EF context", {"file": "efa.md"})
    elcc_doc = DummyDoc("ELCC context", {"file": "elcc1.md"})
    index = DummyIndex([
        (ef_doc, 0.1),
        (elcc_doc, 0.4),
    ])

    with patch("app.api.rag_controller.get_vector_store", return_value=index):
        filename, context = load_lang_chain_context(
            question="question",
            domains=[],
            grades=[],
            nbr_chunks=2,
        )

    assert filename == "efa.md, elcc1.md"
    assert "EF context" in context
    assert "ELCC context" in context
    assert index.search_calls[0]["filter"] is None


def test_rank_and_limit_sorts_and_deduplicates_chunks():
    first = DummyDoc("same chunk", {"file": "a.md"})
    duplicate = DummyDoc(" same   chunk ", {"file": "a.md"})
    other = DummyDoc("other chunk", {"file": "b.md"})

    result = rank_and_limit(
        [(first, 0.8), (duplicate, 0.9), (other, 0.2)],
        nbr_chunks=2,
    )

    assert [(doc.metadata["file"], score) for doc, score in result] == [
        ("b.md", 0.2),
        ("a.md", 0.8),
    ]


def test_empty_search_returns_empty_context():
    index = DummyIndex()

    with patch("app.api.rag_controller.get_vector_store", return_value=index):
        filename, context = load_lang_chain_context(
            question="question",
            domains=[],
            grades=[],
            nbr_chunks=2,
        )

    assert filename == "empty"
    assert "Aucun context" in context


def test_get_context_endpoint_returns_response():
    with (
        patch(
            "app.api.rag_controller.get_expected_token",
            return_value=TEST_TOKEN,
        ),
        patch(
            "app.api.rag_controller.load_lang_chain_context",
            return_value=("file.md", "context"),
        ),
    ):
        response = client.post(
            "/rag/context",
            headers={"X-Service-Token": TEST_TOKEN},
            json={
                "question": "hello",
                "domains": [],
                "grades": [],
                "nbr_chunks": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "context",
        "source_file": "file.md",
    }


def test_get_context_endpoint_rejects_missing_service_token():
    response = client.post(
        "/rag/context",
        json={
            "question": "hello",
            "domains": [],
            "grades": [],
            "nbr_chunks": 2,
        },
    )

    assert response.status_code == 403


def test_get_context_endpoint_rejects_invalid_service_token():
    with patch(
        "app.api.rag_controller.get_expected_token",
        return_value=TEST_TOKEN,
    ):
        response = client.post(
            "/rag/context",
            headers={"X-Service-Token": "wrong-token"},
            json={
                "question": "hello",
                "domains": [],
                "grades": [],
                "nbr_chunks": 2,
            },
        )

    assert response.status_code == 401

def test_rank_and_limit_guarantees_one_chunk_per_grade():
    docs = [
        (DummyDoc("ef-1",  {"file": "ef.md",  "grade": "ef"}),  0.25),
        (DummyDoc("ef-2",  {"file": "ef.md",  "grade": "ef"}),  0.26),
        (DummyDoc("efs-1", {"file": "efs.md", "grade": "efs"}), 0.27),
        (DummyDoc("efa-1", {"file": "efa.md", "grade": "efa"}), 0.28),
    ]

    result = rank_and_limit(docs, nbr_chunks=2, grades=["ef", "efs", "efa"])

    grades_returned = {doc.metadata["grade"] for doc, _ in result}
    assert grades_returned == {"ef", "efs", "efa"}
    assert len(result) == 3
    assert [s for _, s in result] == sorted(s for _, s in result)


def test_rank_and_limit_without_grades_keeps_old_behaviour():
    docs = [
        (DummyDoc("a", {"file": "a.md", "grade": "a"}), 0.30),
        (DummyDoc("b", {"file": "b.md", "grade": "b"}), 0.20),
        (DummyDoc("c", {"file": "c.md", "grade": "c"}), 0.10),
    ]

    result = rank_and_limit(docs, nbr_chunks=2)

    assert [doc.page_content for doc, _ in result] == ["c", "b"]

def test_search_with_scores_scans_whole_index_when_filtered():
    index = MagicMock()
    index.index.ntotal = 269
    index.similarity_search_with_score.return_value = []

    search_with_scores(index, "question", 2, {"grade": "efa"})

    index.similarity_search_with_score.assert_called_once_with(
        "question", k=2, filter={"grade": "efa"}, fetch_k=269
    )


def test_search_with_scores_unfiltered_keeps_default_fetch():
    index = MagicMock()
    index.index.ntotal = 269
    index.similarity_search_with_score.return_value = []

    search_with_scores(index, "question", 2)

    index.similarity_search_with_score.assert_called_once_with("question", k=2)