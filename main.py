"""
app/main.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status

from app.api.rag_controller import router, load_service_token, get_max_l2_distance
from app.services.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    get_metrics,
)
from app.services.vector_store import (
    RAG_INDEX_FOLDER,
    get_vector_store,
    validate_embedding_manifest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service_token = load_service_token()  # fail fast if RASA_RAG_TOKEN is missing
    get_max_l2_distance()                           # fail fast on an invalid threshold
    get_vector_store()                              # warm-up: loads FAISS index and embedder
    yield


app = FastAPI(title="LangChain RAG Service", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    # Use the route template to avoid unbounded label cardinality
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)

    return response


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness probe: the process is running. Does not touch FAISS."""
    return {"status": "alive", "service": "langchain-rag-service"}


@app.get("/ready", include_in_schema=False)
async def readiness():
    """Readiness probe: FAISS artifacts exist and embedding config is valid."""
    index_folder = Path(RAG_INDEX_FOLDER)
    required_files = ["embedding_manifest.json", "index.faiss", "index.pkl"]

    missing_files = [f for f in required_files if not (index_folder / f).is_file()]
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


@app.get("/metrics", include_in_schema=False)
async def metrics():
    content, content_type = get_metrics()
    return Response(content=content, media_type=content_type)