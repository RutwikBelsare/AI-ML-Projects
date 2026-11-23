
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "conversations.sqlite"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the conversation and message tables."""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id)
                    REFERENCES conversations(thread_id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id, updated_at)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread
            ON conversation_messages(thread_id, user_id)
        """)


def create_conversation(thread_id: str, user_id: str, title: str):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations
            (thread_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, user_id, title, now, now),
        )


def list_conversations(user_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_conversation(thread_id: str, user_id: str):
    with get_connection() as conn:
        conversation = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE thread_id = ? AND user_id = ?
            """,
            (thread_id, user_id),
        ).fetchone()

        if conversation is None:
            return None

        messages = conn.execute(
            """
            SELECT role, content, created_at
            FROM conversation_messages
            WHERE thread_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (thread_id, user_id),
        ).fetchall()

        result = dict(conversation)
        result["messages"] = [dict(row) for row in messages]

        return result


def add_message(
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conversation = conn.execute(
            """
            SELECT thread_id
            FROM conversations
            WHERE thread_id = ? AND user_id = ?
            """,
            (thread_id, user_id),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation not found.")

        conn.execute(
            """
            INSERT INTO conversation_messages
            (thread_id, user_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, user_id, role, content, now),
        )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE thread_id = ? AND user_id = ?
            """,
            (now, thread_id, user_id),
        )


def delete_conversation(thread_id: str, user_id: str):
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        cursor = conn.execute(
            """
            DELETE FROM conversations
            WHERE thread_id = ? AND user_id = ?
            """,
            (thread_id, user_id),
        )

        return cursor.rowcount > 0


# Initialize database when this module is imported.
initialize_database()
