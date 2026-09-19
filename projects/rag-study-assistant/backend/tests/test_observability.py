import json
import sqlite3

from app.observability import InteractionLogger


def test_log_interaction_writes_a_row(tmp_path):
    logger = InteractionLogger(db_path=tmp_path / "obs.db")
    logger.log_interaction(
        session_id="s1",
        query="What is RAG?",
        chunks=[{"chunk_id": "doc::0", "distance": 0.12}],
        response="Retrieval-augmented generation.",
        latency_ms=42.5,
        prompt_chars=100,
    )

    conn = sqlite3.connect(tmp_path / "obs.db")
    row = conn.execute(
        "SELECT session_id, query, retrieved_chunk_ids, retrieved_scores, "
        "response, latency_ms, prompt_chars, response_chars FROM interactions"
    ).fetchone()

    assert row[0] == "s1"
    assert row[1] == "What is RAG?"
    assert json.loads(row[2]) == ["doc::0"]
    assert json.loads(row[3]) == [0.12]
    assert row[4] == "Retrieval-augmented generation."
    assert row[5] == 42.5
    assert row[6] == 100
    assert row[7] == len("Retrieval-augmented generation.")


def test_log_interaction_failure_does_not_raise(tmp_path):
    logger = InteractionLogger(db_path=tmp_path / "obs.db")
    logger._conn.close()  # simulate a broken/locked database

    # Must not raise - a logging failure must never break the chat response.
    logger.log_interaction(
        session_id="s1",
        query="anything",
        chunks=[],
        response="anything",
        latency_ms=1.0,
        prompt_chars=1,
    )
