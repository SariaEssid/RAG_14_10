"""
app/services/vector_store.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import json
import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.academic_rag.langchain_embedding_loader import get_embedder
from app.config import (
    EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
    QUERY_PREFIX,
    PASSAGE_PREFIX,
)
from app.config import PROJECT_ROOT

log = logging.getLogger(__name__)
RAG_INDEX_FOLDER = os.path.join(PROJECT_ROOT, "faiss_index_rag") # VECTOR_DB_PATH
_index = None

def validate_embedding_manifest(index_folder: str) -> None:
    manifest_path = Path(index_folder) / "embedding_manifest.json"

    if not manifest_path.exists():
        raise RuntimeError(
            f"FAISS index has no embedding manifest: {manifest_path}. "
            "Rebuild the index with the current application version."
        )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
    }

    if saved != expected:
        raise RuntimeError(
            "FAISS index was built with a different embedding configuration. "
            f"Index configuration: {saved}; running configuration: {expected}. "
            "Rebuild the FAISS index before starting the RAG service."
        )

def get_vector_store():
    global _index

    if _index is None:
        log.info("🚀 Loading FAISS index...")

        # Refuse to deserialize an index unless its build-time embedding
        # configuration matches the one used for queries. This must happen
        # before load_local reads the pickle-backed docstore.
        validate_embedding_manifest(RAG_INDEX_FOLDER)

        _index = FAISS.load_local(
            RAG_INDEX_FOLDER,
            get_embedder(),
            # LangChain stores the docstore and ID map in a pickle. Only load
            # index artifacts shipped from a trusted, controlled build.
            allow_dangerous_deserialization=True,
        )

        log.info("✅ FAISS loaded successfully")

    return _index
