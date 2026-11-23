from app.memory.store import (
    long_term_store,
    get_user_memory_namespace,
)

user_id = "user_123"
key = "persistent_orion_test"

namespace = get_user_memory_namespace(user_id)

print("Testing PostgreSQL memory retrieval...")

item = long_term_store.get(namespace, key)

if item is None:
    print("Memory NOT FOUND")
else:
    print("Memory FOUND")
    print("Namespace:", item.namespace)
    print("Key:", item.key)
    print("Value:", item.value)