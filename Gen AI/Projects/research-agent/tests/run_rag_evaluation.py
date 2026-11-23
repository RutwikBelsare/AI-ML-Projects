from app.rag.retriever import search_documents
from tests.rag_evaluation import evaluate_retrieval
from tests.rag_evaluation_dataset import RAG_EVALUATION_DATASET


def run_rag_evaluation():
    print("\n========== RAG EVALUATION ==========\n")

    results = []

    for case in RAG_EVALUATION_DATASET:
        print(f"Running: {case['id']}")
        print(f"Question: {case['question']}")

        matches = search_documents(
            query=case["question"],
            k=case["k"],
        )

        evaluation = evaluate_retrieval(
            matches=matches,
            expected_keywords=case["expected_keywords"],
            k=case["k"],
        )

        result = {
            "id": case["id"],
            "evaluation": evaluation,
        }

        results.append(result)

        print(f"Retrieved chunks: {evaluation['retrieved_count']}")
        print(f"Relevant chunks: {evaluation['relevant_count']}")
        print(f"Hit Rate: {evaluation['hit_rate']}")
        print(f"Precision@K: {evaluation['precision_at_k']}")
        print(f"MRR: {evaluation['reciprocal_rank']}")

        for chunk in evaluation["chunks"]:
            print(
                f"\nRank {chunk['rank']} | "
                f"Relevant: {chunk['is_relevant']} | "
                f"Distance: {chunk['distance']}"
            )
            print(f"Source: {chunk['source']}")
            print(f"Matched keywords: {chunk['matched_keywords']}")

    print("\n========== RAG EVALUATION COMPLETE ==========")

    return results


if __name__ == "__main__":
    run_rag_evaluation()