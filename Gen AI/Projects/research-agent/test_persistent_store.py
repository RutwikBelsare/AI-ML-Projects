
import sys

from app.memory.store import (
    long_term_store,
    get_user_memory_namespace,
)

USER_A = "user_123"
USER_B = "user_456"

MEMORY_KEY = "persistent_orion_test"


def save_memory():
    namespace = get_user_memory_namespace(USER_A)

    long_term_store.put(
        namespace,
        MEMORY_KEY,
        {
            "category": "project",
            "text": "My project codename is Orion.",
        },
    )

    print("Memory saved successfully.")
    print("User:", USER_A)
    print("Memory: My project codename is Orion.")

    # Immediately retrieve the saved memory
    saved_item = long_term_store.get(
        namespace,
        MEMORY_KEY,
    )

    print("\nIMMEDIATE READ-BACK TEST")

    if saved_item is None:
        print("FAILED: Memory was not found immediately after saving.")
    else:
        print("Retrieved:", saved_item.value)
        print("IMMEDIATE READ-BACK TEST: PASSED")


def verify_memory():
    namespace = get_user_memory_namespace(USER_A)

    item = long_term_store.get(namespace, MEMORY_KEY)

    if item is None:
        raise AssertionError(
            "Memory was not found after application restart."
        )

    print("\nPERSISTENCE TEST")
    print("Retrieved:", item.value["text"])

    assert item.value["text"] == (
        "My project codename is Orion."
    )

    print("PERSISTENCE TEST: PASSED")


def verify_user_isolation():
    namespace = get_user_memory_namespace(USER_B)

    results = long_term_store.search(
        namespace,
        query="What is my project codename?",
        limit=5,
    )

    print("\nUSER ISOLATION TEST")

    for item in results:
        print("Unexpected memory:", item.value)

    assert not any(
        item.key == MEMORY_KEY
        for item in results
    ), "User B accessed User A's memory!"

    print("USER ISOLATION TEST: PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m test_persistent_store [save|verify]")
        raise SystemExit(1)

    action = sys.argv[1].lower()

    if action == "save":
        save_memory()
    elif action == "verify":
        verify_memory()
        verify_user_isolation()
    else:
        raise ValueError("Use save or verify.")
