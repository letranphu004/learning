"""Session-scoped conversation history, stdlib sqlite3, no ORM - one small
module is simpler than pulling in SQLAlchemy for a handful of queries."""

import sqlite3

from app.config import settings


class ConversationStore:
    def __init__(self, db_path: str | None = None):
        self._path = str(db_path or settings.db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL DEFAULT (datetime('now')),
                seq INTEGER
            )"""
        )
        self._conn.commit()

    def append(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            """INSERT INTO messages (session_id, role, content, seq)
               VALUES (?, ?, ?, (
                   SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?
               ))""",
            (session_id, role, content, session_id),
        )
        self._conn.commit()

    def recent(self, session_id: str, n: int = 6) -> list[dict]:
        """Last n messages for a session, oldest first (ready to prepend to
        a prompt's messages list)."""
        rows = self._conn.execute(
            """SELECT role, content FROM messages
               WHERE session_id = ? ORDER BY seq DESC LIMIT ?""",
            (session_id, n),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]
