
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings



# 1. Initialize HuggingFace Embedding Model


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# 2. Initialize Semantic Memory Store


store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 384,
        "fields": ["text"],
    }
)



# 3. Define Namespace


user_id = "demo_user_001"
namespace = (user_id, "memories")



# 4. Store Memories


memories = [
    {
        "category": "project",
        "text": (
            "I am building a Research Agent using "
            "LangGraph, LangChain, Groq and ChromaDB."
        ),
    },
    {
        "category": "learning",
        "text": (
            "I am learning LangGraph checkpointing, "
            "conversation memory and long-term memory."
        ),
    },
    {
        "category": "technology",
        "text": (
            "My RAG implementation uses ChromaDB "
            "and HuggingFace embeddings."
        ),
    },
    {
        "category": "project",
        "text": (
            "My Research Agent supports web search, "
            "calculator tools and document retrieval."
        ),
    },
    {
        "category": "technology",
        "text": (
            "I use Python, FastAPI, LangChain and "
            "LangGraph for AI application development."
        ),
    },
]

for index, memory in enumerate(memories, start=1):
    store.put(
        namespace,
        f"memory_{index}",
        memory,
    )

print("Memories stored successfully!")



# 5. Semantic Search


def search_memory(query: str, limit: int = 3):

    print("\n" + "=" * 50)
    print("USER QUERY:", query)
    print("=" * 50)

    results = store.search(
        namespace,
        query=query,
        limit=limit,
    )

    if not results:
        print("No relevant memories found.")
        return

    for index, item in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("Key:", item.key)
        print("Category:", item.value["category"])
        print("Memory:", item.value["text"])
        print("Score:", item.score)



# 6. Test Semantic Queries


search_memory(
    "Which AI project am I currently developing?"
)

search_memory(
    "What technologies do I use for building AI applications?"
)

search_memory(
    "How does my document retrieval system work?"
)

search_memory(
    "What am I learning about LangGraph?"
)
