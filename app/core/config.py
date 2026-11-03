# app/core/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env.dev"

# Load .env.dev before reading or validating environment variables
load_dotenv(dotenv_path=ENV_FILE, override=False)

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

ENV_FILES = {
    "dev": ".env.dev",
    "test": ".env.test",
    "prod": ".env",
}

env_file = ENV_FILES.get(ENVIRONMENT)

if env_file:
    env_path = BASE_DIR / env_file

    if env_path.exists():
        load_dotenv(env_path, override=False)


# Centralized configuration
RAG_MAX_L2_DISTANCE = os.getenv("RAG_MAX_L2_DISTANCE")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")
RAG_INDEX_FOLDER = os.getenv("RAG_INDEX_FOLDER", "faiss_index_rag")