"""Best-effort structured logging of every RAG request, same stdlib
sqlite3-no-ORM pattern as memory.py - shares the one `app.db` file (a
separate `interactions` table) rather than adding a second database file."""

import json
import sqlite3
import sys

from app.config import settings


class InteractionLogger:
    def __init__(self, db_path: str | None = None):
        self._path = str(db_path or settings.db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS interactions (
                ts TEXT NOT NULL DEFAULT (datetime('now')),
                session_id TEXT,
                query TEXT NOT NULL,
                retrieved_chunk_ids TEXT NOT NULL,
                retrieved_scores TEXT NOT NULL,
                response TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                prompt_chars INTEGER NOT NULL,
                response_chars INTEGER NOT NULL
            )"""
        )
        self._conn.commit()

    def log_interaction(
        self,
        session_id: str | None,
        query: str,
        chunks: list[dict],
        response: str,
        latency_ms: float,
        prompt_chars: int,
    ) -> None:
        """Never raises - a logging bug must not break the chat response it
        describes, per the phase's non-functional requirement. Failures go
        to stderr instead."""
        try:
            self._conn.execute(
                """INSERT INTO interactions
                   (session_id, query, retrieved_chunk_ids, retrieved_scores,
                    response, latency_ms, prompt_chars, response_chars)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    query,
                    json.dumps([c.get("chunk_id") for c in chunks]),
                    json.dumps([c.get("distance") for c in chunks]),
                    response,
                    latency_ms,
                    prompt_chars,
                    len(response),
                ),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001 - the "never raises" contract
            # covers more than sqlite3.Error (e.g. a non-serializable chunk
            # value reaching json.dumps), so the catch-all is intentional.
            print(f"observability: failed to log interaction: {exc}", file=sys.stderr)
