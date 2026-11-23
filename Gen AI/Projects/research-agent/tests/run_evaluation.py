from uuid import uuid4

from app.agent.graph import run_research
from tests.evaluation_dataset import EVALUATION_DATASET
from tests.evaluation_metrics import evaluate_answer


def run_evaluation():
    results = []

    print("\n========== RESEARCH AGENT EVALUATION ==========\n")

    for case in EVALUATION_DATASET:
        case_id = case["id"]
        category = case["category"]

        print(f"\nRunning {case_id}: {category}")

        thread_id = f"evaluation_{case_id}_{uuid4().hex[:8]}"

        try:
            state = run_research(
                question=case["question"],
                thread_id=thread_id,
                user_id="user1",
            )

        except (ValueError, TypeError) as exc:
            if category == "input_validation":
                result = {
                    "id": case_id,
                    "category": category,
                    "status": "PASSED",
                    "details": f"Invalid input rejected: {exc}",
                }
            else:
                result = {
                    "id": case_id,
                    "category": category,
                    "status": "FAILED",
                    "details": f"Unexpected validation error: {exc}",
                }

            results.append(result)
            print(f"Result: {result['status']}")
            continue

        except Exception as exc:
            result = {
                "id": case_id,
                "category": category,
                "status": "FAILED",
                "details": str(exc),
            }

            results.append(result)
            print("Result: FAILED")
            continue

        status = state.get("status", "unknown")

        # Tool execution may be paused for human approval.
        if status == "tool_requested":
            result = {
                "id": case_id,
                "category": category,
                "status": "PENDING_APPROVAL",
                "details": "Agent is waiting for human approval.",
            }

            results.append(result)
            print("Result: PENDING HUMAN APPROVAL")
            continue

        answer = state.get("final_answer", "")

        evaluation = evaluate_answer(
            answer=answer,
            expected_answer_contains=case[
                "expected_answer_contains"
            ],
        )

        result = {
            "id": case_id,
            "category": category,
            "status": (
                "PASSED" if evaluation["passed"] else "FAILED"
            ),
            "answer": answer,
            "evaluation": evaluation,
        }

        results.append(result)

        print(f"Answer: {answer}")
        print(f"Correctness: {evaluation['correctness']}")
        print(f"Relevance: {evaluation['relevance']}")
        print(f"Completeness: {evaluation['completeness']}")
        print(f"Result: {result['status']}")

    passed = sum(
        result["status"] == "PASSED"
        for result in results
    )

    failed = sum(
        result["status"] == "FAILED"
        for result in results
    )

    pending = sum(
        result["status"] == "PENDING_APPROVAL"
        for result in results
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total cases: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pending approval: {pending}")

    completed = passed + failed

    if completed:
        score = (passed / completed) * 100
        print(f"Pass rate (completed cases): {score:.2f}%")

    return results


if __name__ == "__main__":
    run_evaluation()