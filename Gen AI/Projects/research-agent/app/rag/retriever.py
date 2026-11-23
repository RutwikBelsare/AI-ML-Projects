
from langchain_chroma import Chroma

from app.rag.ingest import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_embeddings,
)


def search_documents(
    query: str,
    k: int = 4,
) -> list[dict]:
    """
    Retrieve the top-k document chunks using semantic similarity.

    Returns content, metadata, and Chroma distance.
    Lower distance generally indicates a closer match.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10.")

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "The Chroma database does not exist. "
            "Run python -m app.rag.ingest first."
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    matches = []

    for document, distance in results:
        matches.append({
            "content": document.page_content,
            "source": document.metadata.get("source", "unknown"),
            "page": document.metadata.get("page"),
            "start_index": document.metadata.get("start_index"),
            "distance": round(float(distance), 4),
            "metadata": document.metadata,
        })

    return matches


if __name__ == "__main__":
    question = input("Ask a question about your documents: ")

    matches = search_documents(question)

    if not matches:
        print("No document chunks found.")
    else:
        for index, match in enumerate(matches, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Source: {match['source']}")
            print(f"Page: {match['page']}")
            print(f"Distance: {match['distance']}")
            print(match["content"])
