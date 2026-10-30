
"""
app/services/vector_store.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import logging
import os
from threading import Lock
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, PROJECT_ROOT, NORMALIZE_EMBEDDINGS, QUERY_PREFIX, PASSAGE_PREFIX

log = logging.getLogger(__name__)

RAG_INDEX_FOLDER = os.path.join(PROJECT_ROOT, "faiss_index_rag")

_index = None
_index_lock = Lock()


def get_vector_store():
    """
    Lazily load the FAISS index once per process.

    The index is not loaded at application startup.
    Concurrent first requests wait for the same initialization.
    """

    global _index

    # Fast path: index is already loaded.
    if _index is not None:
        return _index

    # Only one thread can initialize the index.
    with _index_lock:

        # Double-check after acquiring the lock.
        if _index is not None:
            return _index

        log.info("Loading FAISS index on first retrieval request...")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        loaded_index = FAISS.load_local(
            RAG_INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Publish the index only after successful loading.
        _index = loaded_index

        log.info("FAISS index loaded successfully.")

        return _index

def validate_embedding_manifest(index_folder: str) -> None:
    """
    Validate the embedding configuration stored alongside the FAISS index.

    This function does not load FAISS or initialize the embedding model.
    Raises ValueError or FileNotFoundError if validation fails.
    """

    manifest_path = Path(index_folder) / "embedding_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Embedding manifest not found: {manifest_path}"
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Embedding manifest contains invalid JSON."
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError("Embedding manifest must be a JSON object.")

    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
    }

    for key, expected_value in expected.items():

        if key not in manifest:
            raise ValueError(
                f"Missing '{key}' in embedding manifest."
            )

        if manifest[key] != expected_value:
            raise ValueError(
                f"Embedding configuration mismatch for '{key}'. "
                f"Expected {expected_value!r}, "
                f"got {manifest[key]!r}."
            )