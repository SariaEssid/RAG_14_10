"""
tests/conftest.py
"""
import os

import pytest
os.environ["RASA_RAG_TOKEN"] = "unit-test-token"

class DummyDoc:
    def __init__(self, page_content="content", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


@pytest.fixture
def dummy_doc():
    return DummyDoc(
        page_content="hello world",
        metadata={
            "file": "efa_test.md",
            "grade": "efa",
            "domain": "ef"
        }
    )


@pytest.fixture
def dummy_docs():
    return [
        DummyDoc(
            page_content="chunk-1",
            metadata={
                "file": "efa_test.md",
                "grade": "efa",
                "domain": "ef"
            }
        ),
        DummyDoc(
            page_content="chunk-2",
            metadata={
                "file": "efs_test.md",
                "grade": "efs",
                "domain": "ef"
            }
        )
    ]