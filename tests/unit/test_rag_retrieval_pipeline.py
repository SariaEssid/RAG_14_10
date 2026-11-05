"""
tests/unit/test_faiss_metadata_filtering.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import pytest
import app.api.rag_controller as rag_controller


QUESTION = "Quelles sont les missions de l'enseignant formateur ?"


@pytest.fixture
def disable_distance_threshold(monkeypatch):
    monkeypatch.setattr(
        rag_controller,
        "get_max_l2_distance",
        lambda: None,
    )


@pytest.mark.parametrize(
    "domains, grades",
    [
        (["saria"], []),
        ([], ["ef"]),
        ([], []),
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