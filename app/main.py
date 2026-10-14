"""
app/main.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from app.api.rag_controller import router

app = FastAPI(title="LangChain RAG Service")
app.include_router(router)


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics():
    """Stub Prometheus /metrics endpoint — prevents 404 log spam from the scraper."""
    return PlainTextResponse("# LangChain RAG Service\n", media_type="text/plain; version=0.0.4")
