
"""
app/main.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from app.api.rag_controller import router, load_service_token, get_max_l2_distance
from app.services.vector_store import get_vector_store
from app.services.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    get_metrics,
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
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


@app.get("/health", include_in_schema=False)
async def health():
    return {
        "status": "alive",
        "service": "langchain-rag-service",
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    content, content_type = get_metrics()

    return Response(
        content=content,
        media_type=content_type,
    )