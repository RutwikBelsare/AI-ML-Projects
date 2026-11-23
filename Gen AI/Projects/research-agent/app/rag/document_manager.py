from pathlib import Path

from app.rag.dynamic_ingest import (UPLOAD_DIR, PROJECT_ROOT, get_vector_store)


def list_uploaded_documents() -> list[dict]:
    """Return uploaded documents with their actual document IDs."""

    vector_store = get_vector_store()

    result = vector_store.get(
        where={
            "document_type": {
                "$eq": "uploaded"
            }
        },
        include=["metadatas"],
    )

    documents = {}

    for metadata in result.get("metadatas", []):

        # Use the actual UUID stored in ChromaDB metadata.
        document_id = metadata.get("document_id")

        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("filename"),
                "source": metadata.get("source"),
                "chunks_indexed": 0,
                "file_hash": metadata.get("file_hash"),
            }

        documents[document_id]["chunks_indexed"] += 1

    return list(documents.values())

def delete_uploaded_document(document_id: str) -> dict:
    """Delete an uploaded document and its ChromaDB chunks."""

    vector_store = get_vector_store()

    result = vector_store.get(
        where={
            "$and": [
                {
                    "document_id": {
                        "$eq": document_id
                    }
                },
                {
                    "document_type": {
                        "$eq": "uploaded"
                    }
                },
            ]
        },
        include=["metadatas"],
    )

    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        raise ValueError("Uploaded document not found.")

    source = metadatas[0].get("source")

    if not source:
        raise ValueError("Document source metadata is missing.")

    # Resolve the original file and verify it is inside UPLOAD_DIR.
    upload_root = UPLOAD_DIR.resolve()
    file_path = (
        PROJECT_ROOT / source
    ).resolve()

    if not file_path.is_relative_to(upload_root):
        raise ValueError("Invalid document storage path.")

    # Remove all indexed chunks belonging to this document.
    vector_store.delete(ids=ids)

    # Remove the original uploaded file.
    if file_path.exists():
        file_path.unlink()

    return {
        "status": "success",
        "document_id": document_id,
        "filename": metadatas[0].get("filename"),
        "chunks_deleted": len(ids),
    }
