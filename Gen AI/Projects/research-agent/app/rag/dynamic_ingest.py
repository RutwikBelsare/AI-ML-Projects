
from pathlib import Path
import csv
import hashlib
import uuid

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.ingest import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "uploaded_documents"
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def get_vector_store() -> Chroma:
    """Connect to the existing persistent ChromaDB collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def calculate_file_hash(content: bytes) -> str:
    """Generate a SHA-256 hash to identify duplicate files."""
    return hashlib.sha256(content).hexdigest()


def load_uploaded_document(file_path: Path) -> list[Document]:
    """Extract text from a supported uploaded document."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(str(file_path)).load()

    if extension == ".txt":
        return TextLoader(
            str(file_path),
            encoding="utf-8",
        ).load()

    if extension == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError as exc:
            raise ImportError(
                "DOCX support requires docx2txt. "
                "Install it using: pip install docx2txt"
            ) from exc

        return Docx2txtLoader(str(file_path)).load()

    if extension == ".csv":
        documents = []

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            if not reader.fieldnames:
                raise ValueError("CSV file must contain column headers.")

            for row_number, row in enumerate(reader, start=1):
                content = "\n".join(
                    f"{key}: {value or ''}"
                    for key, value in row.items()
                    if key
                )

                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "row": row_number,
                            },
                        )
                    )

        return documents

    raise ValueError(
        f"Unsupported file format: {extension}"
    )


def split_uploaded_documents(
    documents: list[Document],
) -> list[Document]:
    """Split extracted document text into searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No readable text was extracted from the document."
        )

    return chunks


def ingest_uploaded_document(
    filename: str,
    content: bytes,
) -> dict:
    """
    Save an uploaded document permanently and index it
    in the existing ChromaDB collection.
    """

    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("File size cannot exceed 20 MB.")

    # Prevent directory traversal and retain only the filename.
    safe_filename = Path(filename).name

    extension = Path(safe_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_hash = calculate_file_hash(content)

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = get_vector_store()

    
# Check whether this exact file has already been indexed.
    existing = vector_store.get(
        where={
            "$and": [
                {"file_hash": {"$eq": file_hash}},
                {"document_type": {"$eq": "uploaded"}},
            ]
        },
        include=["metadatas"],
    )

    if existing.get("ids"):
        existing_metadata = existing["metadatas"][0]

        return {
            "status": "duplicate",
            "message": "This document has already been uploaded and indexed.",
            "document_id": existing_metadata.get("document_id"),
            "filename": existing_metadata.get("filename"),
            "chunks_indexed": len(existing["ids"]),
        }


    document_id = str(uuid.uuid4())

    stored_filename = f"{document_id}_{safe_filename}"
    file_path = UPLOAD_DIR / stored_filename

    try:
        # Save the original document permanently.
        file_path.write_bytes(content)

        # Extract text.
        documents = load_uploaded_document(file_path)

        if not documents:
            raise ValueError(
                "No readable content was found in the document."
            )

        # Add common metadata to every extracted document.
        for document in documents:
            document.metadata.update({
                "source": str(
                    file_path.relative_to(PROJECT_ROOT)
                ),
                "document_id": document_id,
                "document_type": "uploaded",
                "filename": safe_filename,
                "file_hash": file_hash,
            })

        # Split extracted text into chunks.
        chunks = split_uploaded_documents(documents)

        # Generate deterministic chunk IDs for this document.
        chunk_ids = []

        for index, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()

            identity = (
                f"{document_id}|{index}|{chunk_hash}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            chunk_ids.append(chunk_id)

        # Store chunks in the SAME collection as existing documents.
        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        return {
            "status": "success",
            "message": "Document uploaded and indexed successfully.",
            "document_id": document_id,
            "filename": safe_filename,
            "chunks_indexed": len(chunks),
            "file_hash": file_hash,
        }

    except Exception:
        # Avoid leaving an uploaded file behind if indexing fails.
        if file_path.exists():
            file_path.unlink()

        raise
