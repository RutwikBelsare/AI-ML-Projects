
from tests.evaluation_dataset import EVALUATION_DATASET


def test_evaluation_dataset_structure():
    assert len(EVALUATION_DATASET) >= 7

    ids = [case["id"] for case in EVALUATION_DATASET]

    # Every evaluation case must have a unique ID.
    assert len(ids) == len(set(ids))

    required_fields = {
        "id",
        "category",
        "question",
        "expected_behavior",
        "expected_tool",
        "expected_answer_contains",
        "requires_human_approval",
    }

    for case in EVALUATION_DATASET:
        assert required_fields.issubset(case.keys())
        assert isinstance(case["question"], str)
        assert isinstance(case["expected_answer_contains"], list)
        assert isinstance(case["requires_human_approval"], bool)


def test_evaluation_categories():
    categories = {
        case["category"] for case in EVALUATION_DATASET
    }

    expected_categories = {
        "direct_answer",
        "calculator",
        "web_search",
        "rag",
        "long_term_memory",
        "human_approval",
        "input_validation",
    }

    assert expected_categories.issubset(categories)