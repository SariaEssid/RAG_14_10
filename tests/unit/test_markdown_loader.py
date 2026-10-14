"""
tests/unit/test_markdown_loader.py
"""

from unittest.mock import mock_open, patch

from app.academic_rag.markdown_loader import (
    extract_grade_from_filename,
    load_markdown_documents,
)


class DummyDocument:
    def __init__(self, metadata=None, page_content="content"):
        self.metadata = metadata or {}
        self.page_content = page_content


class DummySplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_text(self, _content):
        return [
            DummyDocument(metadata={"Header 1": "Title 1"}),
            DummyDocument(metadata={"Header 2": "Title 2"}),
            DummyDocument(metadata={}),
        ]


def test_load_markdown_documents_folder_not_exists():
    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=False):
        result = load_markdown_documents("missing")

    assert result == []


@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_load_markdown_documents_ef_domain():
    mocked_open = mock_open(read_data="# Hello")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["efa_test.md"]), \
         patch("builtins.open", mocked_open):

        result = load_markdown_documents("kb")

    assert len(result) == 3
    assert result[0].metadata["file"] == "efa_test.md"
    assert result[0].metadata["grade"] == "efa_test"
    assert result[0].metadata["domain"] == "ef"
    assert result[0].metadata["title"] == "Title 1"


@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_load_markdown_documents_elcc_domain():
    mocked_open = mock_open(read_data="# Hello")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["elcc1_test.md"]), \
         patch("builtins.open", mocked_open):

        result = load_markdown_documents("kb")

    assert result[0].metadata["domain"] == "elcc"


@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_load_markdown_documents_custom_domain():
    mocked_open = mock_open(read_data="# Hello")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["math_teacher.md"]), \
         patch("builtins.open", mocked_open):

        result = load_markdown_documents("kb")

    assert result[0].metadata["domain"] == "math"


@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_load_markdown_documents_skips_non_markdown_files():
    mocked_open = mock_open(read_data="# Hello")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch(
             "app.academic_rag.markdown_loader.os.listdir",
             return_value=["file.txt", "efa.md"]
         ), \
         patch("builtins.open", mocked_open):

        result = load_markdown_documents("kb")

    assert len(result) == 3


def test_extract_grade_from_filename():
    result = extract_grade_from_filename("EFA.md")

    assert result == "efa"

def test_folder_not_exists():
    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=False):
        result = load_markdown_documents("fake")
    assert result == []

@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_load_markdown_full_flow():
    m = mock_open(read_data="# test")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["efa_test.md"]), \
         patch("builtins.open", m):

        result = load_markdown_documents("kb")

    assert len(result) == 3
    assert result[0].metadata["file"] == "efa_test.md"
    assert result[0].metadata["grade"] == "efa_test"
    assert result[0].metadata["domain"] == "ef"
    assert result[0].metadata["title"] == "Title 1"

def test_skip_non_markdown():
    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["file.txt"]), \
         patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter"):

        result = load_markdown_documents("kb")

    assert result == []

@patch("app.academic_rag.markdown_loader.MarkdownHeaderTextSplitter", DummySplitter)
def test_markdown_full_flow():
    m = mock_open(read_data="# test")

    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["efa_test.md"]), \
         patch("builtins.open", m):

        result = load_markdown_documents("kb")

    assert len(result) == 3
    assert result[0].metadata["file"] == "efa_test.md"
    assert result[0].metadata["domain"] == "ef"
    assert result[0].metadata["title"] == "Title 1"


def test_skip_non_md():
    with patch("app.academic_rag.markdown_loader.os.path.exists", return_value=True), \
         patch("app.academic_rag.markdown_loader.os.listdir", return_value=["file.txt"]):

        result = load_markdown_documents("kb")

    assert result == []