"""
app/services/vector_store.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL, PROJECT_ROOT

log = logging.getLogger(__name__)
RAG_INDEX_FOLDER = os.path.join(PROJECT_ROOT, "faiss_index_rag") # VECTOR_DB_PATH
_index = None


def get_vector_store():
    global _index

    if _index is None:
        log.info("🚀 Loading FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        _index = FAISS.load_local(
            RAG_INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization = True
        )

        log.info("✅ FAISS loaded successfully")

    return _index
