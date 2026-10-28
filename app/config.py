"""
app/config.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-base").strip()

NORMALIZE_EMBEDDINGS = True
QUERY_PREFIX = "query: " if "e5" in EMBEDDING_MODEL.lower() else ""
PASSAGE_PREFIX = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""