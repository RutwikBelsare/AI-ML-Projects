"""Run a two-thread long-term memory integration test in one process."""
from uuid import uuid4

from app.agent.graph import run_research


def main():
    user_id = "memory-integration-demo-user"
    run_tag = uuid4().hex[:8]
    thread_a = f"memory-test-A-{run_tag}"
    thread_b = f"memory-test-B-{run_tag}"

    print("\n=== CONVERSATION A: SAVE MEMORY ===")
    first = run_research(
        "Remember that my project codename is Orion.",
        thread_id=thread_a,
        user_id=user_id,
    )
    print("Answer:", first.get("final_answer"))
    print("Memory write status:", first.get("memory_write_status"))

    print("\n=== CONVERSATION B: RETRIEVE MEMORY ===")
    second = run_research(
        "What is my project codename?",
        thread_id=thread_b,
        user_id=user_id,
    )
    print("Retrieved memories:")
    for memory in second.get("long_term_memories", []):
        print("-", memory)
    print("Answer:", second.get("final_answer"))

    found = any(
        "Orion".lower() in memory.lower()
        for memory in second.get("long_term_memories", [])
    )
    print("\nCROSS-THREAD MEMORY TEST:", "PASSED" if found else "FAILED")


if __name__ == "__main__":
    main()
