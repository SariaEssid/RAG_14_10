"""
app/api/rag_controller.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""
from pathlib import Path
from fastapi import Request
from dotenv import load_dotenv

from app.models.request_models import RAGRequest
from app.models.response_models import RAGResponse
from app.services.metrics import RAG_RETRIEVAL_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env.dev"

# Load .env.dev before reading or validating environment variables
load_dotenv(dotenv_path=ENV_FILE, override=False)

import os
import secrets
import re
import unicodedata

from fastapi import APIRouter
from app.services.vector_store import get_vector_store

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

service_token_header = APIKeyHeader(name="X-Service-Token")
router = APIRouter()

def get_max_l2_distance():
    """Return the configured maximum FAISS L2 distance, or None if disabled."""
    raw_value = os.getenv("RAG_MAX_L2_DISTANCE")

    # Leave the threshold disabled until it has been calibrated.
    if raw_value is None or not raw_value.strip():
        return None

    try:
        threshold = float(raw_value)
    except ValueError:
        raise ValueError(
            "RAG_MAX_L2_DISTANCE must be a valid number."
        ) from None

    if threshold < 0:
        raise ValueError(
            "RAG_MAX_L2_DISTANCE must be non-negative."
        )

    return threshold

def filter_by_max_l2_distance(docs_with_scores):
    """Discard chunks whose FAISS distance exceeds the configured limit."""
    max_distance = get_max_l2_distance()

    if max_distance is None:
        return docs_with_scores

    return [
        (doc, score)
        for doc, score in docs_with_scores
        if score <= max_distance
    ]

def load_service_token() -> str:
    """Called once at startup. Raises if the token is missing."""
    token = os.getenv("RASA_RAG_TOKEN")
    if not token or not token.strip():
        raise RuntimeError("RASA_RAG_TOKEN is not set: refusing to start")
    return token

def verify_rasa_service(request: Request, token: str = Depends(service_token_header)) -> None:
    expected = request.app.state.service_token
    if not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.post("/rag/context", response_model=RAGResponse)
def get_context(request: RAGRequest, _authenticated: None = Depends(verify_rasa_service)):
    print("💖  💖   💖   💖   💖   💖   💖   💖   💖  💖   💖   💖   💖   💖   💖   💖   💖   💖  💖  💖   💖   💖   💖   💖   💖   💖   💖  WELCOME to LANGCHAIN 1 ")
    filename, context_text = load_lang_chain_context(request.question,
                                                     request.domains,
                                                     request.grades,
                                                     request.nbr_chunks)
    print("######################## WELCOME to LANGCHAIN 3")

    if filename == "empty":
        RAG_RETRIEVAL_COUNT.labels(result="no_context").inc()
        raise HTTPException(
            status_code=404,
            detail="no_relevant_context",
        )

    RAG_RETRIEVAL_COUNT.labels(result="success").inc()
    return RAGResponse(context=context_text, source_file=filename)

def load_lang_chain_context(question: str, domains: list, grades:list, nbr_chunks:int):
    """Retrieve top-k FAISS chunks for a question."""

    index = get_vector_store()  # FAISS vector store

    if index.index.ntotal == 0:
        return "empty", "Aucun contexte n'est trouvé."

    # get top-k docs with scores
    print("######################## WELCOME to LANGCHAIN 2")
    print(f"######################## grades_rasa : {grades}")
    print(f"######################## domains : {domains}")
    print(f"######################## nbr chunks : {nbr_chunks}")

    docs_with_scores = []

    if grades:
        print("💖 ######################## +GRADE")
        for grade in grades:
            docs_with_scores.extend(
                search_with_scores(index, question, nbr_chunks, {"grade": grade})
            )

        domains = detect_domain_from_grades(grades)
        print(f"######################## domains : {domains}")
        docs_with_scores = get_docs_with_scores(domains, question, index, docs_with_scores)
    elif not grades and domains:
        print("💖 ######################## -GRADE | +DOMAIN")
        for domain in domains:
            docs_with_scores.extend(
                search_with_scores(index, question, nbr_chunks, {"domain": domain})
            )
    else:
        print("💖 ######################## -GRADE | -DOMAIN : " + str(domains))
        docs_with_scores = search_with_scores(index, question, nbr_chunks)

    # TEMPORARY: collect actual FAISS distances for calibration.
    for doc, score in docs_with_scores:
        print(
            f"RETRIEVAL_EVAL | "
            f"question={question!r} | "
            f"distance={float(score):.6f} | "
            f"file={doc.metadata.get('file', 'unknown')} | "
            f"chunk={doc.page_content[:250]!r}"
        )

    # Apply the configured distance threshold once.
    docs_with_scores = filter_by_max_l2_distance(docs_with_scores)

    # Select the final chunks once.
    docs_with_scores = rank_and_limit(
        docs_with_scores,
        nbr_chunks,
        grades
    )

    contexts = []
    print("\n💖   💖   💖   Question : " + question)

    if not docs_with_scores:
        print("\n🚨  NO CONTEXT FOUND")
        return "empty", "Aucun context n'est trouvé."

    source_files = list(dict.fromkeys(
        doc.metadata.get("file", "unknown")
        for doc, _ in docs_with_scores
    ))
    filename = ", ".join(source_files)
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  --- Chunk {i} ---  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸")
        print(f"Score: {score}")
        print("Metadata:", doc.metadata)
        print("------------------------------------------")
        print(doc.page_content + "\n")
        contexts.append(doc.page_content)

    return filename, "\n\n".join(contexts)

def search_with_scores(index, question, k, metadata_filter=None):
    """Return FAISS (doc, distance) pairs. Lower L2 distance is a better match."""
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
        # FAISS filters AFTER fetching fetch_k candidates (default 20).
        # Scan the whole index so the filter never starves the results.
        search_kwargs["fetch_k"] = max(k, index.index.ntotal)
    return list(index.similarity_search_with_score(question, **search_kwargs))

def rank_and_limit(docs_with_scores, nbr_chunks, grades):
    """Deduplicate, keep the best chunk of each requested grade, then fill by distance."""
    unique_docs, seen = [], set()
    for doc, score in docs_with_scores:
        key = (doc.metadata.get("file", "unknown"), " ".join(doc.page_content.split()))
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append((doc, score))

    unique_docs.sort(key=lambda item: item[1])

    selected, chosen_ids = [], set()
    for grade in grades or []:
        best = next((i for i in unique_docs if i[0].metadata.get("grade") == grade), None)
        if best and id(best) not in chosen_ids:
            selected.append(best)
            chosen_ids.add(id(best))

    limit = max(nbr_chunks, len(selected))
    for item in unique_docs:
        if len(selected) >= limit:
            break
        if id(item) not in chosen_ids:
            selected.append(item)
            chosen_ids.add(id(item))

    return sorted(selected, key=lambda item: item[1])

def detect_domain_from_grades(grades: list[str]) -> list[str]:
    domains = []

    if any(grade.lower().startswith("ef") for grade in grades):
        domains.append("ef")

    if any(grade.lower().startswith("elcc") for grade in grades):
        domains.append("elcc")

    return domains

def get_docs_with_scores(domains, question, index, docs_with_scores):
    if domains:
        for domain in domains:
            extra_docs = search_with_scores(
                index,
                question,
                2,
                {"domain": domain},
            )
            print(f"######################## extra_doc - relationship : {extra_docs}")
            docs_with_scores.extend(extra_docs)

    return docs_with_scores