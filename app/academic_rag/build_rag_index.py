"""
app/academic_rag/build_rag_index.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

# build_rag_index.py

import logging
import os
import pickle
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.academic_rag.markdown_loader import load_markdown_documents
from app.config import EMBEDDING_MODEL, PROJECT_ROOT

log = logging.getLogger(__name__)

# Make sure Python can find the actions package
sys.path.append(PROJECT_ROOT)

# Charge les variables du fichier .env
load_dotenv()

# Paths
RAG_INDEX_FOLDER = os.path.join(PROJECT_ROOT, "faiss_index_rag")
os.makedirs(RAG_INDEX_FOLDER, exist_ok=True)  # Ensure folder exists
META_FILE = os.path.join(RAG_INDEX_FOLDER, "metadata.pkl")  # Metadata file

KB_PATH = os.getenv("KNOWLEDGE_BASE_FOLDER") or os.path.join(PROJECT_ROOT, "knowledge_base")
INDEX_FILE = os.path.join(RAG_INDEX_FOLDER, "faiss.index")

def build_faiss_index(kb_folder, index_folder=RAG_INDEX_FOLDER, meta_file=META_FILE):
    # Load markdown sections
    chunks = load_markdown_documents(kb_folder)
    if not chunks:
        raise ValueError(f"No markdown sections found in {kb_folder}")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Initialize embeddings
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Build FAISS vector store
    index = FAISS.from_texts(texts, embedder, metadatas=metadatas)

    # Save index and metadata
    index.save_local(index_folder)  # ✅ folder path, not a file path
    with open(meta_file, "wb") as f:
        pickle.dump(metadatas, f)

    log.info(f"✅ RAG FAISS index built successfully at {datetime.now()}")
    log.info(f"Index folder: {index_folder}")
    log.info(f"Metadata path: {meta_file}")
    log.info(f"Chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    # Configure logging to output to the console
    logging.basicConfig(  # NOSONAR
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    build_faiss_index(KB_PATH)
