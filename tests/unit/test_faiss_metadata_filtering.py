"""
tests/unit/test_faiss_metadata_filtering.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

from app.services.vector_store import get_vector_store
from app.api.rag_controller import (
    search_with_scores,
    detect_domain_from_grades,
)

def test_faiss_metadata_filtering():

    index = get_vector_store()

    print("\nFAISS index size:", index.index.ntotal)

    assert index.index.ntotal > 0, "FAISS index is empty"

    question = "Quelles sont les missions de l'enseignant formateur ?"

    # --------------------------------------------------
    # TEST 1: Domain filtering
    # --------------------------------------------------

    domain = "saria"

    domain_docs = search_with_scores(
        index,
        question,
        10,
        {"domain": domain},
    )

    assert domain_docs, f"No documents found for domain={domain}"

    assert all(
        doc.metadata.get("domain") == domain
        for doc, score in domain_docs
    ), "Domain filter returned documents with incorrect metadata"

    print(f"\nPASS: Domain filtering ({domain})")
    print("Returned:", len(domain_docs))

    # --------------------------------------------------
    # TEST 2: Grade filtering
    # --------------------------------------------------

    grade = "ef"

    grade_docs = search_with_scores(
        index,
        question,
        10,
        {"grade": grade},
    )

    assert grade_docs, f"No documents found for grade={grade}"

    assert all(
        doc.metadata.get("grade") == grade
        for doc, score in grade_docs
    ), "Grade filter returned documents with incorrect metadata"

    print(f"\nPASS: Grade filtering ({grade})")
    print("Returned:", len(grade_docs))

    # --------------------------------------------------
    # TEST 3: Grade-to-domain mapping
    # --------------------------------------------------

    inferred_domains = detect_domain_from_grades(
        ["ef", "efa", "efs", "efm"]
    )

    assert inferred_domains == ["ef"], inferred_domains

    print("\nPASS: Grade-to-domain mapping")
    print("Inferred domains:", inferred_domains)

    # --------------------------------------------------
    # TEST 4: Supplementary domain retrieval
    # --------------------------------------------------

    for inferred_domain in inferred_domains:

        extra_docs = search_with_scores(
            index,
            question,
            2,
            {"domain": inferred_domain},
        )

        assert len(extra_docs) <= 2

        assert all(
            doc.metadata.get("domain") == inferred_domain
            for doc, score in extra_docs
        )

        print(
            f"\nPASS: Supplementary retrieval "
            f"({inferred_domain}) - {len(extra_docs)} documents"
        )

    print("\nALL FAISS METADATA TESTS PASSED")


import pytest

import app.api.rag_controller as rag_controller


QUESTION = "Quelles sont les missions de l'enseignant formateur ?"


@pytest.fixture
def disable_distance_threshold(monkeypatch):
    # Isolate retrieval behavior from distance-threshold calibration.
    monkeypatch.setattr(
        rag_controller,
        "get_max_l2_distance",
        lambda: None,
    )


@pytest.mark.parametrize(
    "domains, grades",
    [
        (["saria"], []),  # Domain-only retrieval
        ([], ["ef"]),     # Grade + inferred domain retrieval
        ([], []),         # Unfiltered retrieval
    ],
)
def test_complete_retrieval_pipeline(
    domains,
    grades,
    disable_distance_threshold,
):
    filename, context = rag_controller.load_lang_chain_context(
        question=QUESTION,
        domains=domains,
        grades=grades,
        nbr_chunks=3,
    )

    assert filename != "empty"
    assert filename.strip()
    assert context.strip()

if __name__ == "__main__":
    test_faiss_metadata_filtering()
