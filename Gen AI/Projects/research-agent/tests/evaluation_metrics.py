def evaluate_answer(
    answer: str,
    expected_answer_contains: list[str],
) -> dict:
    """
    Evaluate an answer using expected keywords.

    This is a basic keyword-based evaluation,
    not an advanced semantic or LLM-based evaluation.
    """

    if not isinstance(answer, str) or not answer.strip():
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "passed": False,
            "matched_keywords": [],
            "missing_keywords": expected_answer_contains,
        }

    normalized_answer = answer.lower()

    matched_keywords = [
        keyword
        for keyword in expected_answer_contains
        if keyword.lower() in normalized_answer
    ]

    missing_keywords = [
        keyword
        for keyword in expected_answer_contains
        if keyword.lower() not in normalized_answer
    ]

    if not expected_answer_contains:
        completeness = 1.0
    else:
        completeness = (
            len(matched_keywords) / len(expected_answer_contains)
        )

    relevance = 1.0 if matched_keywords else 0.0

    correctness = completeness

    passed = (
        correctness >= 0.8
        and relevance == 1.0
    )

    return {
        "correctness": round(correctness, 2),
        "relevance": round(relevance, 2),
        "completeness": round(completeness, 2),
        "passed": passed,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
    }