"""
app/api/rag_controller.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""
from dotenv import load_dotenv
load_dotenv()  # Must run before os.getenv(...)

import os
import secrets

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field
from app.services.vector_store import get_vector_store

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

service_token_header = APIKeyHeader(name="X-Service-Token")
router = APIRouter()

EXPECTED_RAG_TOKEN = os.getenv("RASA_RAG_TOKEN")

if not EXPECTED_RAG_TOKEN or not EXPECTED_RAG_TOKEN.strip():
    raise RuntimeError("RASA_RAG_TOKEN must be configured")

def verify_rasa_service(token: str = Depends(service_token_header)):

    print(f"RASA_RAG_TOKEN: {EXPECTED_RAG_TOKEN}")
    if not secrets.compare_digest(token, EXPECTED_RAG_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    domains: List[str] = Field(default_factory=list)
    grades: List[str] = Field(default_factory=list)
    nbr_chunks: int = Field(ge=1, le=20)

class RAGResponse(BaseModel):
    answer: str
    source_file: Optional[str] = None  # 💡 Explicitly allows str or None

@router.post("/rag/context", response_model=RAGResponse)
def get_context(request: RAGRequest, _authenticated: None = Depends(verify_rasa_service)):
    print("💖  💖   💖   💖   💖   💖   💖   💖   💖  💖   💖   💖   💖   💖   💖   💖   💖   💖  💖  💖   💖   💖   💖   💖   💖   💖   💖  WELCOME to LANGCHAIN 1 ")
    filename, context_text = load_lang_chain_context(request.question, request.domains, request.grades, request.nbr_chunks)
    print("######################## WELCOME to LANGCHAIN 3")

    return RAGResponse(answer=context_text, source_file=filename)

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
            docs = index.similarity_search(
                question,
                k=nbr_chunks,
                filter={"grade": grade}
            )
            docs_with_scores.extend([(doc, 0) for doc in docs])

        # domains = detect_domain_from_grades(grades)
        print(f"######################## domains : {domains}")
        # docs_with_scores = get_docs_with_scores(domains, question, index, docs_with_scores)

    elif not grades and domains:
        print("💖 ######################## -GRADE | +DOMAIN")
        for domain in domains:
            docs = index.similarity_search(
                question,
                k=nbr_chunks,
                filter={"domain": domain}
            )
            docs_with_scores.extend([(doc, 0) for doc in docs])
    else:
        print("💖 ######################## -GRADE | -DOMAIN : " + str(domains))

        total_chunks = index.index.ntotal
        if total_chunks == 0:
            return "empty", "Aucun contexte n'est trouvé."

        candidate_k = min(nbr_chunks * 5, total_chunks)
        raw_docs_with_scores = index.similarity_search_with_score(
            question, k=candidate_k
        )

        # 1. Fetch the initial results
        # raw_docs_with_scores = index.similarity_search_with_score(question, k=nbr_chunks)

        # 2. Filter out documents where page_content starts with "ef"
        docs_with_scores = [
            (doc, score) for doc, score in raw_docs_with_scores
            if not doc.metadata.get("file", "").startswith(("ef", "elcc"))        ]

        print("--- Filtered Files ---")
        for doc, score in docs_with_scores:
            print(f"Keeping file: {doc.metadata.get('file')}")

    contexts = []
    print("\n💖   💖   💖   Question : " + question)

    # Remove duplicate chunks returned by multiple searches.
    unique_docs = []
    seen_chunks = set()

    for doc, score in docs_with_scores:
        file_name = doc.metadata.get("file", "unknown")
        normalized_content = " ".join(doc.page_content.split())
        key = (file_name, normalized_content)

        if key not in seen_chunks:
            seen_chunks.add(key)
            unique_docs.append((doc, score))

    docs_with_scores = unique_docs
    if not docs_with_scores:
        print("\n🚨  NO CONTEXT FOUND")
        return "empty", "Aucun context n'est trouvé."

    source_files = list(dict.fromkeys(
        doc.metadata.get("file", "unknown")
        for doc, _ in docs_with_scores
    ))
    for i, (doc, score) in enumerate(docs_with_scores, 1):

        print(f"🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  --- Chunk {i} ---  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸")
        print(f"Score: {score}")
        print("Metadata:", doc.metadata)
        print("------------------------------------------")
        print(doc.page_content + "\n")
        contexts.append(doc.page_content)

    return  ", ".join(source_files), "\n\n".join(contexts)