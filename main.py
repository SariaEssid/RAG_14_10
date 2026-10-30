
"""
app/main.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.api.rag_controller import load_service_token, get_max_l2_distance
from app.api.rag_controller import router
from app.services.vector_store import (
    RAG_INDEX_FOLDER,
    validate_embedding_manifest,
)
from app.services.vector_store import get_vector_store

app = FastAPI(title="LangChain RAG Service")

app.include_router(router)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service_token = load_service_token()  # fail fast if auth is not configured
    get_max_l2_distance()                           # fail fast on an invalid threshold
    get_vector_store()                              # warm-up: loads the index and embedder
    yield

app = FastAPI(title="LangChain RAG Service", lifespan=lifespan)
app.include_router(router)

@app.get("/health", include_in_schema=False)
async def health():
    """
    Liveness probe.
    Confirms that the application process is running.
    Does not load FAISS or initialize embeddings.
    """
    return {
        "status": "alive",
        "service": "langchain-rag-service",
    }


@app.get("/ready", include_in_schema=False)
async def readiness():
    """
    Readiness probe.
    Validates FAISS artifacts and embedding configuration.
    Does not load FAISS or initialize embeddings.
    """

    index_folder = Path(RAG_INDEX_FOLDER)

    required_files = [
        "embedding_manifest.json",
        "index.faiss",
        "index.pkl",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (index_folder / filename).is_file()
    ]

    if missing_files:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": "Missing FAISS index artifacts",
                "missing_files": missing_files,
            },
        )

    try:
        validate_embedding_manifest(str(index_folder))

    except (RuntimeError, ValueError, OSError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": "Invalid or incompatible embedding manifest",
            },
        )

    return {
        "status": "ready",
        "service": "langchain-rag-service",
        "index": "available",
        "embedding_configuration": "valid",
    }


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
)
async def metrics():
    return PlainTextResponse(
        "# LangChain RAG Service\n",
        media_type="text/plain; version=0.0.4",
    )
