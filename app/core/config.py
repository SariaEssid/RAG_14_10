import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()

ENV_FILES = {
    "dev": ".env.dev",
    "test": ".env.test",
    "prod": ".env",
}

env_file = ENV_FILES.get(ENVIRONMENT)
if env_file is None:
    raise RuntimeError(f"Unsupported ENVIRONMENT: {ENVIRONMENT}")

env_path = BASE_DIR / env_file
if env_path.exists():
    load_dotenv(env_path, override=False)

# Read settings after dotenv has loaded.
RAG_MAX_L2_DISTANCE = os.getenv("RAG_MAX_L2_DISTANCE")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")

raw_index_folder = os.getenv(
    "RAG_INDEX_FOLDER",
    str(BASE_DIR / "app" / "faiss_index_rag"),
)

RAG_INDEX_FOLDER = Path(raw_index_folder)
if not RAG_INDEX_FOLDER.is_absolute():
    RAG_INDEX_FOLDER = BASE_DIR / RAG_INDEX_FOLDER