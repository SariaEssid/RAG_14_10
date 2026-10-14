"""
tests/unit/test_langchain_embedding_loader.py
"""

from unittest.mock import patch
from app.academic_rag.langchain_embedding_loader import get_embedder

from app.academic_rag import langchain_embedding_loader as loader


class DummyEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_get_embedder_creates_singleton():
    loader._embedder = None

    with patch.object(loader, "_lock", DummyLock()):
        with patch.object(loader, "HuggingFaceEmbeddings", DummyEmbeddings):
            embedder = loader.get_embedder()

    assert embedder is not None
    assert embedder.model_name == loader.EMBEDDING_MODEL


def test_get_embedder_returns_existing_singleton():
    existing = object()
    loader._embedder = existing

    result = loader.get_embedder()

    assert result is existing

def test_singleton_embedder():
    with patch("app.academic_rag.langchain_embedding_loader.HuggingFaceEmbeddings") as mock:
        mock.return_value = "EMB"

        e1 = get_embedder()
        e2 = get_embedder()

        assert e1 == e2
        assert mock.call_count == 0