"""
app/academic_rag/langchain_embedding_loader.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

# langchain_embedding_loader.py

from langchain_huggingface import HuggingFaceEmbeddings
from threading import Lock

from app.config import EMBEDDING_MODEL

_embedder = None
_lock = Lock()

def get_embedder():
    """Singleton HuggingFaceEmbeddings for RAG."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL
                )
    return _embedder
