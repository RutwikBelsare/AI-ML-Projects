import pytest

from tests.rag_evaluation import evaluate_retrieval


def test_relevant_document_retrieval():
    matches = [
        {
            "content": "This document discusses Python basics.",
            "source": "python.txt",
            "distance": 0.8,
        },
        {
            "content": (
                "StateGraph defines a workflow using "
                "nodes and edges."
            ),
            "source": "langgraph_notes.txt",
            "distance": 0.2,
        },
    ]

    result = evaluate_retrieval(
        matches,
        ["StateGraph", "nodes", "edges"],
        k=2,
    )

    assert result["hit_rate"] == 1.0
    assert result["precision_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_no_relevant_documents():
    matches = [
        {
            "content": "Python is a programming language.",
            "source": "python.txt",
            "distance": 0.9,
        }
    ]

    result = evaluate_retrieval(
        matches,
        ["StateGraph", "nodes", "edges"],
        k=4,
    )

    assert result["hit_rate"] == 0.0
    assert result["precision_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_empty_expected_keywords():
    with pytest.raises(ValueError):
        evaluate_retrieval(
            matches=[],
            expected_keywords=[],
        )