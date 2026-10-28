"""
app/academic_rag/langchain_embedding_loader.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from threading import Lock

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL

_embedder = None
_lock = Lock()


class PrefixedEmbeddings(Embeddings):
    """Adds the query/passage prefixes that E5-family models require."""

    def __init__(self, base: Embeddings, query_prefix: str = "", passage_prefix: str = ""):
        self._base = base
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def embed_documents(self, texts):
        return self._base.embed_documents([self._passage_prefix + t for t in texts])

    def embed_query(self, text):
        return self._base.embed_query(self._query_prefix + text)


def get_embedder() -> Embeddings:
    """Singleton used by BOTH the index builder and the query service."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                base = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
                if "e5" in EMBEDDING_MODEL.lower():
                    _embedder = PrefixedEmbeddings(base, "query: ", "passage: ")
                else:
                    _embedder = base
    return _embedder