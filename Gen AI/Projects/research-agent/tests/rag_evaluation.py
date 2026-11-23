def evaluate_retrieval(
    matches: list[dict],
    expected_keywords: list[str],
    k: int = 4,
) -> dict:
    """
    Evaluate retrieved document chunks using keyword-based relevance.

    Metrics:
    - Hit Rate
    - Precision@K
    - Mean Reciprocal Rank (MRR)
    """

    if not expected_keywords:
        raise ValueError("Expected keywords cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    top_matches = matches[:k]

    relevant_ranks = []
    evaluated_chunks = []

    for rank, match in enumerate(top_matches, start=1):
        content = str(match.get("content", "")).lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in content
        ]

        relevance_score = (
            len(matched_keywords) / len(expected_keywords)
        )

        # A chunk is considered relevant when it matches
        # at least half of the expected keywords.
        is_relevant = relevance_score >= 0.5

        if is_relevant:
            relevant_ranks.append(rank)

        evaluated_chunks.append({
            "rank": rank,
            "source": match.get("source", "unknown"),
            "distance": match.get("distance"),
            "matched_keywords": matched_keywords,
            "relevance_score": round(relevance_score, 2),
            "is_relevant": is_relevant,
        })

    retrieved_count = len(top_matches)
    relevant_count = len(relevant_ranks)

    hit_rate = 1.0 if relevant_count > 0 else 0.0

    precision_at_k = (
        relevant_count / retrieved_count
        if retrieved_count > 0
        else 0.0
    )

    reciprocal_rank = (
        1 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    return {
        "hit_rate": round(hit_rate, 2),
        "precision_at_k": round(precision_at_k, 2),
        "reciprocal_rank": round(reciprocal_rank, 2),
        "retrieved_count": retrieved_count,
        "relevant_count": relevant_count,
        "chunks": evaluated_chunks,
    }