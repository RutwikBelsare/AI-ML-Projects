from tests.evaluation_metrics import evaluate_answer


def test_correct_answer():
    answer = "There are 360 hours in 15 days."

    result = evaluate_answer(
        answer,
        ["360", "hours"],
    )

    assert result["passed"] is True
    assert result["correctness"] == 1.0
    assert result["completeness"] == 1.0


def test_incomplete_answer():
    answer = "There are 360 hours."

    result = evaluate_answer(
        answer,
        ["360", "hours", "15 days"],
    )

    assert result["passed"] is False
    assert result["correctness"] < 1.0
    assert "15 days" in result["missing_keywords"]