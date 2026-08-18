"""SQLite-backed chat history for the Streamlit UI.

Persistence only — no LangChain or Streamlit imports, so this stays usable from
the CLI or a future API layer. Messages are stored as plain ("user"/"assistant",
text) rows and mapped to LangChain message objects by the caller.
"""

import os
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(os.getenv("CHAT_DB_PATH", Path(__file__).parent / "chat_history.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    TEXT NOT NULL REFERENCES conversations(id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
"""


def _connect():
    # A fresh connection per call: Streamlit reruns across threads, so a shared
    # one would need locking for no real gain at this scale.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)


def create_conversation(title):
    conv_id = uuid.uuid4().hex
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
    return conv_id


def list_conversations():
    """Every conversation, most recently used first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def conversation_exists(conv_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return row is not None


def load_messages(conv_id):
    """Messages in send order, as {"role", "content"} dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conv_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def append_message(conv_id, role, content):
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


def delete_conversation(conv_id):
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
