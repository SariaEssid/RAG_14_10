"""
tests/unit/test_vector_store.py
"""

from unittest.mock import patch
from app.services.vector_store import get_vector_store

from app.services import vector_store


class DummyEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name


class DummyFAISS:
    pass


def test_get_vector_store_loads_index_once():
    vector_store._index = None

    mock_index = DummyFAISS()

    with patch.object(vector_store, "HuggingFaceEmbeddings", DummyEmbeddings):
        with patch.object(vector_store.FAISS, "load_local", return_value=mock_index) as mock_load:
            result = vector_store.get_vector_store()

    assert result is mock_index
    mock_load.assert_called_once()



def test_get_vector_store_returns_cached_index():
    existing = object()
    vector_store._index = existing

    result = vector_store.get_vector_store()

    assert result is existing

def test_vector_store_singleton():
    with patch("app.services.vector_store.FAISS.load_local") as mock_load, \
         patch("app.services.vector_store.HuggingFaceEmbeddings"):

        mock_load.return_value = "INDEX"

        v1 = get_vector_store()
        v2 = get_vector_store()

        assert v1 == v2
        assert mock_load.call_count == 0