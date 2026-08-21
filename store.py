"""SQLite-backed chat history for the Streamlit UI.

Persistence only — no LangChain or Streamlit imports, so this stays usable from
the CLI or a future API layer. Messages are stored as plain ("user"/"assistant",
text) rows and mapped to LangChain message objects by the caller.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
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


@contextmanager
def _session():
    """One connection per call, committed and then closed.

    Streamlit reruns across threads, so a shared connection would need locking
    for no real gain at this scale. `with conn` alone commits but leaves the
    connection open, which leaks descriptors in a long-running server — hence
    the explicit close.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _migrate(conn):
    """Additive, idempotent schema catch-up for databases created before a column.

    `sources` holds the JSON citation list for an assistant message. Added rather
    than baked into SCHEMA so existing chat_history.db files keep their history:
    the column is nullable, so pre-RAG messages simply have no citations.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    if "sources" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")


def init_db():
    with _session() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def create_conversation(title, conv_id=None):
    """Persist a new conversation.

    The caller may supply the id so a thread id minted before the first message
    (for LangSmith grouping) becomes this conversation's primary key.
    """
    conv_id = conv_id or uuid.uuid4().hex
    now = time.time()
    with _session() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
    return conv_id


def list_conversations():
    """Every conversation, most recently used first."""
    with _session() as conn:
        rows = conn.execute(
            "SELECT id, title FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def conversation_exists(conv_id):
    with _session() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return row is not None


def load_messages(conv_id):
    """Messages in send order, as {"role", "content", "sources"} dicts.

    `sources` is the decoded citation list, or [] for user messages and for any
    assistant message stored before citations existed.
    """
    with _session() as conn:
        rows = conn.execute(
            "SELECT role, content, sources FROM messages WHERE conv_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
        }
        for row in rows
    ]


def append_message(conv_id, role, content, sources=None):
    now = time.time()
    with _session() as conn:
        conn.execute(
            "INSERT INTO messages (conv_id, role, content, created_at, sources) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, now, json.dumps(sources) if sources else None),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


def delete_conversation(conv_id):
    with _session() as conn:
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def clear_all():
    """Wipe every conversation and message. Not reachable from the UI."""
    with _session() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")


def vacuum():
    """Shrink the file on disk.

    Deletes free pages for reuse but never truncate the file, so chat_history.db
    keeps its size until this runs. Cosmetic — VACUUM cannot run inside a
    transaction, so it needs its own connection.
    """
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


def stats():
    """Row counts plus any messages orphaned by a partial delete."""
    with _session() as conn:
        return {
            "conversations": conn.execute("SELECT count(*) FROM conversations").fetchone()[0],
            "messages": conn.execute("SELECT count(*) FROM messages").fetchone()[0],
            "orphan_messages": conn.execute(
                "SELECT count(*) FROM messages "
                "WHERE conv_id NOT IN (SELECT id FROM conversations)"
            ).fetchone()[0],
        }


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "list"
    init_db()

    if command == "list":
        print(f"{DB_PATH}\n")
        for conv in list_conversations():
            count = len(load_messages(conv["id"]))
            print(f"  {conv['id'][:8]}  {count:>3} msgs  {conv['title']}")
        print(f"\n{stats()}")
    elif command == "delete" and len(sys.argv) > 2:
        prefix = sys.argv[2]
        matches = [c for c in list_conversations() if c["id"].startswith(prefix)]
        if len(matches) != 1:
            sys.exit(f"{len(matches)} conversations match {prefix!r}; need exactly one")
        delete_conversation(matches[0]["id"])
        print(f"deleted {matches[0]['title']!r}")
    elif command == "clear":
        clear_all()
        vacuum()
        print("cleared all chat history")
    elif command == "vacuum":
        vacuum()
        print("vacuumed")
    else:
        sys.exit("usage: python store.py [list | delete <id-prefix> | clear | vacuum]")
