from pathlib import Path
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "research_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings()->HuggingFaceEmbeddings:
    """Load the local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" :"cpu"},
        encode_kwargs = {"normalize_embeddings":True}
        )

def load_documents():
    """Load supported documents and attach source metadata."""
    documents = []

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Documents directory not found :{DOCUMENTS_DIR}")

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(str(path),encoding = "utf-8").load()
        else:
            continue

        for document in loaded:
            document.metadata["source"] = str(path.relative_to(PROJECT_ROOT))
            documents.append(document)

    if not documents:
        raise ValueError(f"No PDF or TXT documents found in {DOCUMENTS_DIR}")
    return documents
    
def split_documents(documents):
    """split loaded documents into overlapping text chunks. """
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    add_start_index=True,
)

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")
    return chunks


def stable_chunk_id(document)->str:
    """Create a repeatable ID for a source chunk."""
    source = document.metadata.get("source","")
    page = document.metadata.get("page","")
    start_index = document.metadata.get("start_index","")
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    identity = (
        f"{source} | {page} | {start_index} | {content_hash}"
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

def ingest_documents()-> int:
    """Index documents into the persistent Chroma database."""
    documents = load_documents()
    chunks = split_documents(documents)

    CHROMA_DIR.mkdir(parents = True, exist_ok=True)
    vector_store = Chroma(
        collection_name= COLLECTION_NAME,
        persist_directory= str(CHROMA_DIR),
        embedding_function = get_embeddings(),

    )

    ids = [stable_chunk_id(chunk) for chunk in chunks]
    #Repeated ingestion updates matching IDs instead of intentionally 
    #creating a new copy of every unchanged chunk

    vector_store.add_documents(documents = chunks, ids = ids)

    return len(chunks)

if __name__ == "__main__":
    count = ingest_documents()
    print(f"Indexed {count} document chunks.")
    print(f"Vector database :{CHROMA_DIR}")





