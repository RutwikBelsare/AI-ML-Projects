import os
import atexit
import logging
from contextlib import ExitStack

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

load_dotenv()

DB_URI = os.getenv("LANGGRAPH_STORE_DB_URI")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_store_stack = ExitStack()

long_term_store = None

if DB_URI:
    try:
        long_term_store = _store_stack.enter_context(
            PostgresStore.from_conn_string(
                DB_URI,
                index={
                    "embed": embeddings,
                    "dims": 384,
                    "fields": ["text"],
                },
            )
        )
        long_term_store.setup()
        atexit.register(_store_stack.close)
    except Exception as exc:
        logger.warning(
            "PostgresStore connection failed (%s). Falling back to InMemoryStore.",
            exc,
        )
        long_term_store = None

if long_term_store is None:
    long_term_store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 384,
            "fields": ["text"],
        }
    )



def get_user_memory_namespace(user_id: str):
    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")

    return (user_id, "memories")