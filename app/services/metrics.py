"""
app/services/metrics.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

RAG_RETRIEVAL_COUNT = Counter(
    "rag_retrieval_requests_total",
    "Total RAG retrieval requests",
    ["result"],
)


def get_metrics():
    return generate_latest(), CONTENT_TYPE_LATEST