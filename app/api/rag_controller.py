"""
app/api/rag_controller.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""
from dotenv import load_dotenv
load_dotenv()  # Must run before os.getenv(...)

import os
import secrets
import re
import unicodedata

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field
from app.services.vector_store import get_vector_store

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

service_token_header = APIKeyHeader(name="X-Service-Token")
router = APIRouter()

GRADE_ALIASES = {
    "efm": ["efm", "manager", "enseignant formateur manager"],
    "efs": ["efs", "senior", "enseignant formateur senior"],
    "efa": ["efa", "associe", "enseignant formateur associe"],
    "ef": ["ef", "enseignant formateur"],
    "elcc1": ["elcc1", "classe 1"],
    "elcc2": ["elcc2", "classe 2"],
    "elcc": ["elcc", "enseignant de langue"],
}

def get_expected_token():
    token = os.getenv("RASA_RAG_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="RASA_RAG_TOKEN must be configured")
    return token

def verify_rasa_service(token: str = Depends(service_token_header)):

    expected_token = get_expected_token()
    print(f"RASA_RAG_TOKEN: {expected_token}")
    if not secrets.compare_digest(token, expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

class RAGRequest(BaseModel):
    question: str
    domains: List[str] = Field(default_factory=list)
    grades: List[str] = Field(default_factory=list)
    nbr_chunks: int

class RAGResponse(BaseModel):
    answer: str
    source_file: Optional[str] = None  # 💡 Explicitly allows str or None

@router.post("/rag/context", response_model=RAGResponse)
async def get_context(request: RAGRequest, _authenticated: None = Depends(verify_rasa_service)):
    print("💖  💖   💖   💖   💖   💖   💖   💖   💖  💖   💖   💖   💖   💖   💖   💖   💖   💖  💖  💖   💖   💖   💖   💖   💖   💖   💖  WELCOME to LANGCHAIN 1 ")
    filename, context_text = load_lang_chain_context(request.question, request.domains, request.grades, request.nbr_chunks)
    print("######################## WELCOME to LANGCHAIN 3")

    expected_token = get_expected_token()
    if not expected_token:
        context_text = "RASA_RAG_TOKEN_NOT_EXIST"
    return RAGResponse(answer=context_text, source_file=filename)

def load_lang_chain_context(question: str, domains: list, grades:list, nbr_chunks:int):
    """Retrieve top-k FAISS chunks for a question."""

    index = get_vector_store()  # FAISS vector store

    # get top-k docs with scores
    print("######################## WELCOME to LANGCHAIN 2")
    grades_lol = detect_grades(question)
    print(f"######################## grades_rasa : {grades}")
    print(f"######################## grades_chain : {grades_lol}")
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

        domains = detect_domain_from_grades(grades)
        print(f"######################## domains : {domains}")
        docs_with_scores = get_docs_with_scores(domains, question, index, docs_with_scores)
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
        # 1. Fetch the initial results
        raw_docs_with_scores = index.similarity_search_with_score(question, k=nbr_chunks)

        # 2. Filter out documents where page_content starts with "ef"
        docs_with_scores = [
            (doc, score) for doc, score in raw_docs_with_scores
            if not doc.metadata.get("file", "").startswith(("ef", "elcc"))        ]

        print("--- Filtered Files ---")
        for doc, score in docs_with_scores:
            print(f"Keeping file: {doc.metadata.get('file')}")

    contexts = []
    filename = "empty"
    print("\n💖   💖   💖   Question : " + question)

    if not docs_with_scores:
        print("\n🚨  NO CONTEXT FOUND")
        return "empty", "Aucun context n'est trouvé."

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        filename = doc.metadata.get("file", "unknown")
        print(f"🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  --- Chunk {i} ---  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸  🔸")
        print(f"Score: {score}")
        print("Metadata:", doc.metadata)
        print("------------------------------------------")
        print(doc.page_content + "\n")
        contexts.append(doc.page_content)

    return filename, "\n\n".join(contexts)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def detect_grades(question: str) -> list[str]:
    normalized_question = f" {normalize_text(question)} "
    matches = []

    for grade, aliases in GRADE_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            if f" {normalized_alias} " in normalized_question:
                matches.append((len(normalized_alias), grade))

    if not matches:
        return []

    longest_match_length = max(length for length, _ in matches)
    return sorted({
        grade
        for length, grade in matches
        if length == longest_match_length
    })

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
            extra_docs = index.similarity_search(
                question,
                k=2,
                filter={
                    "domain": domain,
                    "doc_type": "relationship"
                }
            )
            print(f"######################## extra_doc - relationship : {extra_docs}")
            docs_with_scores.extend([(doc, 0) for doc in extra_docs])

            extra_docs = index.similarity_search(
                question,
                k=2,
                filter={
                    "domain": domain,
                    "doc_type": "comparison"
                }
            )
            print(f"######################## extra_doc - comparison : {extra_docs}")
            docs_with_scores.extend([(doc, 0) for doc in extra_docs])
    return docs_with_scores