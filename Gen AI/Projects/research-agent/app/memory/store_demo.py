from langgraph.store.memory import InMemoryStore

#initialize
store = InMemoryStore()

#2. Define a namespace for a user
user_id = "demo_user_001"
namespace = (user_id, "memories")

# save long term memories

store.put(namespace,"project_001",{
    "category" : "project",
    "text":(
        "I am building a Research Agent using"
        "LangGraph, LangChain, Groq and ChromaDB."
    )
})

store.put(namespace,"learning_001",{
    "category":"learning",
    "text":(
        "I am learning langgraph checkpointing"
        " conversation memory and long-term memory"
    )
})

store.put(
    namespace,
    "technology_001",
    {
        "category": "technology",
        "text": (
            "My RAG implementation uses ChromaDB "
            "and HuggingFace embeddings."
        ),
    },
)

print("="*3,"GET MEMORY", "===")

memory = store.get(namespace,"project_001")
if memory:
    print("KEY", memory.key)
    print("VALUE",memory.value)

#Retrieving all memories

memories = store.search(namespace, limit = 10)
for item in memories:
    print("\nKEY:", item.key)
    print("\nVALUE:", item.value)

#Demonstrating memory sharing across thread
print("=======Memory Shareing======")
thread_001 = "conversation_001"
thread_002 = "conversation_002"

#Both conversation use the same users memory namespace

print("Thread 1:", thread_001)
print("Thread 2:", thread_002)

shared_memories = store.search(namespace)

for item in shared_memories:
    print("-", item.value["text"])

# update an existing memory
print("\n=========Update Memory===== ")

store.put(
    namespace, "project_001",
    {    
        "text":(
            "I am building a production ready Research agent"
            "using langGraph, LangChain, Groq, ChromaDB "
            "and SQLite checkpointing."
        )
    }
)

updated_memory = store.get(namespace, "project_001")

print("UPDATED", updated_memory.value)

# Delete a memory(demostrate only)
# store.delete(namespace, "technology_001")
# print(store.get(namespace, "technology_001"))
