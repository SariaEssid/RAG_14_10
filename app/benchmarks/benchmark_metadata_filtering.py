
"""
app/benchmarks/benchmark_metadata_filtering.py

@author ESSID Saria | Copyright (c) 2025. ESPRIT
"""

import math
import random
from collections import Counter
from time import perf_counter

from app.academic_rag.langchain_embedding_loader import get_embedder
from app.config import EMBEDDING_MODEL
from app.services.vector_store import get_vector_store


K = 3
WARMUP = 3
REPEATS = 30

DEFAULT_FETCH_LIMIT = 20

QUERIES = [
    "Quelle est la différence entre EF et EFS ?",
    "Quelles sont les conditions d'accès au grade EFM ?",
    "Quelles sont les missions de l'enseignant formateur ?",
]


def percentile(values, percent):
    """Calculate the nearest-rank percentile."""
    if not values:
        raise ValueError("Cannot calculate percentile from empty values.")

    values = sorted(values)
    position = math.ceil(percent / 100 * len(values)) - 1

    return values[max(0, position)]


def validate_index(index, documents):
    """Validate consistency between FAISS and its document store."""

    ntotal = index.index.ntotal
    docstore_count = len(documents)
    mapping_count = len(index.index_to_docstore_id)

    if ntotal != docstore_count:
        raise AssertionError(
            f"FAISS/docstore mismatch: {ntotal} vectors, "
            f"{docstore_count} documents."
        )

    if ntotal != mapping_count:
        raise AssertionError(
            f"FAISS/document mapping mismatch: {ntotal} vectors, "
            f"{mapping_count} mapped documents."
        )

    print(f"Index consistency: OK ({ntotal} vectors/documents)")


def benchmark(index, query, metadata_filter, fetch_k):
    """Measure filtered retrieval latency and validate returned metadata."""

    # Warm-up
    for _ in range(WARMUP):
        index.similarity_search_with_score(
            query,
            k=K,
            filter=metadata_filter,
            fetch_k=fetch_k,
        )

    durations = []
    results = []

    for _ in range(REPEATS):
        start = perf_counter()

        results = index.similarity_search_with_score(
            query,
            k=K,
            filter=metadata_filter,
            fetch_k=fetch_k,
        )

        durations.append((perf_counter() - start) * 1000)

    metadata_valid = all(
        all(
            doc.metadata.get(key) == value
            for key, value in metadata_filter.items()
        )
        for doc, _ in results
    )

    return {
        "p50_ms": percentile(durations, 50),
        "p95_ms": percentile(durations, 95),
        "returned": len(results),
        "metadata_valid": metadata_valid,
    }

def inspect_embedding_configuration():
    """Inspect the production embedding configuration."""

    embedder = get_embedder()

    query_vector = embedder.embed_query(
        "Quelle est la différence entre EF et EFS ?"
    )

    dimension = len(query_vector)

    norm = math.sqrt(
        sum(value * value for value in query_vector)
    )

    print("\nEMBEDDING CONFIGURATION")
    print("-----------------------")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {dimension}")
    print(f"Query vector L2 norm: {norm:.6f}")
    print(f"Normalized: {math.isclose(norm, 1.0, rel_tol=1e-4)}")

def main():
    inspect_embedding_configuration()
    index = get_vector_store()

    ntotal = index.index.ntotal
    documents = list(index.docstore._dict.values())

    validate_index(index, documents)

    grade_counts = Counter(
        doc.metadata.get("grade")
        for doc in documents
        if doc.metadata.get("grade")
    )

    domain_counts = Counter(
        doc.metadata.get("domain")
        for doc in documents
        if doc.metadata.get("domain")
    )

    print(f"\nTotal FAISS vectors: {ntotal}")
    print(f"Total documents: {len(documents)}")
    print(f"Grades: {grade_counts}")
    print(f"Domains: {domain_counts}")

    filters = (
        [{"grade": value} for value in grade_counts]
        + [{"domain": value} for value in domain_counts]
    )

    # Build all benchmark cases.
    cases = []

    for metadata_filter in filters:
        matching_count = sum(
            all(
                doc.metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
            for doc in documents
        )

        expected = min(K, matching_count)

        for query in QUERIES:
            cases.append({
                "filter": metadata_filter,
                "query": query,
                "expected": expected,
            })

    # Alternate execution order to reduce systematic ordering bias.
    random.Random(42).shuffle(cases)

    default_fetch = min(DEFAULT_FETCH_LIMIT, ntotal)

    failures = 0

    for case in cases:
        metadata_filter = case["filter"]
        query = case["query"]
        expected = case["expected"]

        # Alternate which strategy runs first.
        if random.Random(
            hash(str(metadata_filter) + query)
        ).choice([True, False]):
            default = benchmark(
                index, query, metadata_filter, default_fetch
            )
            full = benchmark(
                index, query, metadata_filter, ntotal
            )
        else:
            full = benchmark(
                index, query, metadata_filter, ntotal
            )
            default = benchmark(
                index, query, metadata_filter, default_fetch
            )

        print("\n----------------------------------------")
        print(f"Filter: {metadata_filter}")
        print(f"Query: {query}")
        print(f"Expected maximum filtered results: {expected}")

        print(
            f"Default fetch={default_fetch}: "
            f"p50={default['p50_ms']:.2f}ms, "
            f"p95={default['p95_ms']:.2f}ms, "
            f"returned={default['returned']}, "
            f"metadata_valid={default['metadata_valid']}"
        )

        print(
            f"Full fetch={ntotal}: "
            f"p50={full['p50_ms']:.2f}ms, "
            f"p95={full['p95_ms']:.2f}ms, "
            f"returned={full['returned']}, "
            f"metadata_valid={full['metadata_valid']}"
        )

        default_starvation = default["returned"] < expected

        full_complete = full["returned"] == expected

        print("Default candidate starvation:", default_starvation)
        print("Full-index result count complete:", full_complete)

        # Full-index correctness assertions.
        try:
            assert full_complete, (
                f"Full-index retrieval incomplete: "
                f"filter={metadata_filter}, "
                f"expected={expected}, "
                f"returned={full['returned']}"
            )

            assert full["metadata_valid"], (
                f"Invalid metadata returned: {metadata_filter}"
            )

        except AssertionError as exc:
            failures += 1
            print(f"VALIDATION FAILURE: {exc}")

    print("\n========================================")
    print("BENCHMARK SUMMARY")
    print("========================================")
    print(f"Total benchmark cases: {len(cases)}")
    print(f"Full-index validation failures: {failures}")

    if failures:
        raise AssertionError(
            f"{failures} benchmark cases failed validation."
        )

    print("Full-index validation: PASSED")


if __name__ == "__main__":
    main()