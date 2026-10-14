"""
tests/unit/test_build_rag_index.py
"""

from unittest.mock import mock_open, patch
import app.academic_rag.build_rag_index as mod

import pytest

from app.academic_rag.build_rag_index import build_faiss_index


class DummyDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class DummyFAISS:
    def save_local(self, folder):
        self.folder = folder


class DummyEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name


@patch("app.academic_rag.build_rag_index.pickle.dump")
@patch("builtins.open", new_callable=mock_open)
@patch("app.academic_rag.build_rag_index.FAISS")
@patch("app.academic_rag.build_rag_index.HuggingFaceEmbeddings", DummyEmbeddings)
def test_build_faiss_index_success(
    mock_faiss,
    mocked_open,
    mock_pickle_dump,
):
    docs = [
        DummyDoc("chunk-1", {"id": 1}),
        DummyDoc("chunk-2", {"id": 2}),
    ]

    mock_index = DummyFAISS()
    mock_faiss.from_texts.return_value = mock_index

    with patch(
        "app.academic_rag.build_rag_index.load_markdown_documents",
        return_value=docs,
    ):
        build_faiss_index("kb", "index-folder", "meta.pkl")

    mock_faiss.from_texts.assert_called_once()
    mocked_open.assert_called_once_with("meta.pkl", "wb")
    mock_pickle_dump.assert_called_once()


@patch("app.academic_rag.build_rag_index.load_markdown_documents", return_value=[])
def test_build_faiss_index_no_chunks(mock_loader):
    with pytest.raises(ValueError) as exc:
        build_faiss_index("kb")

    assert "No markdown sections found" in str(exc.value)



def test_build_faiss_index_empty():
    with patch("app.academic_rag.build_rag_index.load_markdown_documents", return_value=[]):
        try:
            mod.build_faiss_index("kb")
        except ValueError as e:
            assert "No markdown" in str(e)