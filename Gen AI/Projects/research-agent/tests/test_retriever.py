
from app.tools import research_tools


def test_document_search_returns_source_and_content(monkeypatch):
    """Retrieved content should include its source metadata."""

    def fake_search_documents(query, k):
        return [
            {
                "content": "StateGraph defines a stateful workflow.",
                "source": "data/documents/langgraph_notes.txt",
                "page": None,
                "relevance": 0.87,
            }
        ]

    monkeypatch.setattr(
        research_tools,
        "search_documents",
        fake_search_documents,
    )

    result = research_tools.document_search(
        "What does StateGraph do?"
    )

    assert "StateGraph defines a stateful workflow." in result
    assert "langgraph_notes.txt" in result
    assert "0.87" in result


def test_document_search_handles_no_matches(monkeypatch):
    """No matches should not be presented as verified evidence."""

    monkeypatch.setattr(
        research_tools,
        "search_documents",
        lambda query, k: [],
    )

    result = research_tools.document_search(
        "Question not covered in the documents"
    )

    assert "NO_RELEVANT_DOCUMENTS" in result


def test_document_search_handles_retrieval_failure(monkeypatch):
    """A retrieval exception should return a controlled error."""

    def fail_search(query, k):
        raise RuntimeError("Simulated database failure")

    monkeypatch.setattr(
        research_tools,
        "search_documents",
        fail_search,
    )

    result = research_tools.document_search("StateGraph")

    assert "RAG_SEARCH_FAILED" in result
    assert "RuntimeError" in result
