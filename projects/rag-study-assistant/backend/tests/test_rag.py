import json
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


# Real integration test against a running Ollama + a scratch Chroma dir.
# Skips (does not fail) if Ollama isn't running.
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not running")
def test_chat_answers_from_ingested_document(tmp_path):
    import app.main as main_module
    from app.observability import InteractionLogger
    from app.vectorstore import ChromaStore

    original_store = main_module.store
    original_observability = main_module.observability
    main_module.store = ChromaStore(persist_dir=tmp_path / "chroma")
    main_module.observability = InteractionLogger(db_path=tmp_path / "obs.db")
    client = TestClient(app)
    try:
        client.post(
            "/documents/ingest",
            data={
                "text": "The study assistant project uses Ollama with the llama3.2:3b model for generation.",
                "source": "fixture-doc",
            },
        )

        resp = client.post("/chat", json={"message": "Which chat model does this project use?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "llama3.2" in body["answer"]
        assert any(s["source"] == "fixture-doc" for s in body["sources"])
    finally:
        main_module.store = original_store
        main_module.observability = original_observability
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not running")
def test_chat_stream_reassembles_tokens_into_grounded_answer(tmp_path):
    import app.main as main_module
    from app.memory import ConversationStore
    from app.observability import InteractionLogger
    from app.vectorstore import ChromaStore

    original_store = main_module.store
    original_memory = main_module.memory
    original_observability = main_module.observability
    main_module.store = ChromaStore(persist_dir=tmp_path / "chroma")
    main_module.memory = ConversationStore(db_path=tmp_path / "conv.db")
    main_module.observability = InteractionLogger(db_path=tmp_path / "obs.db")
    client = TestClient(app)
    try:
        client.post(
            "/documents/ingest",
            data={
                "text": "The study assistant project uses Ollama with the llama3.2:3b model for generation.",
                "source": "fixture-doc",
            },
        )

        resp = client.post(
            "/chat/stream",
            json={"session_id": "test-session", "message": "Which chat model does this project use?"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = [line for line in resp.text.split("\n\n") if line.strip()]
        message_events = [e for e in events if e.startswith("data:")]
        done_events = [e for e in events if e.startswith("event: done")]
        assert message_events, "expected at least one token event"
        assert len(done_events) == 1

        full_answer = "".join(
            json.loads(e[len("data:") :].strip())["token"] for e in message_events
        )
        assert "llama3.2" in full_answer

        history = main_module.memory.recent("test-session", n=6)
        assert history[-2] == {"role": "user", "content": "Which chat model does this project use?"}
        assert history[-1] == {"role": "assistant", "content": full_answer}
    finally:
        main_module.store = original_store
        main_module.memory = original_memory
        main_module.observability = original_observability
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not running")
def test_chat_stream_uses_history_for_followup_question(tmp_path):
    import app.main as main_module
    from app.memory import ConversationStore
    from app.observability import InteractionLogger
    from app.vectorstore import ChromaStore

    original_store = main_module.store
    original_memory = main_module.memory
    original_observability = main_module.observability
    main_module.store = ChromaStore(persist_dir=tmp_path / "chroma")
    main_module.memory = ConversationStore(db_path=tmp_path / "conv.db")
    main_module.observability = InteractionLogger(db_path=tmp_path / "obs.db")
    client = TestClient(app)
    try:
        client.post(
            "/documents/ingest",
            data={
                "text": "The study assistant project uses Ollama with the llama3.2:3b model for generation.",
                "source": "fixture-doc",
            },
        )

        client.post(
            "/chat/stream",
            json={"session_id": "followup-session", "message": "Which chat model does this project use?"},
        )

        resp = client.post(
            "/chat/stream",
            json={"session_id": "followup-session", "message": "What did I just ask you?"},
        )
        events = [line for line in resp.text.split("\n\n") if line.strip() and line.startswith("data:")]
        full_answer = "".join(json.loads(e[len("data:") :].strip())["token"] for e in events)
        assert "model" in full_answer.lower()
    finally:
        main_module.store = original_store
        main_module.memory = original_memory
        main_module.observability = original_observability
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not running")
def test_chat_says_it_does_not_know_when_unrelated(tmp_path):
    import app.main as main_module
    from app.observability import InteractionLogger
    from app.vectorstore import ChromaStore

    original_store = main_module.store
    original_observability = main_module.observability
    main_module.store = ChromaStore(persist_dir=tmp_path / "chroma")
    main_module.observability = InteractionLogger(db_path=tmp_path / "obs.db")
    client = TestClient(app)
    try:
        client.post(
            "/documents/ingest",
            data={"text": "The study assistant project uses Ollama for generation.", "source": "fixture-doc"},
        )

        resp = client.post("/chat", json={"message": "What is the capital of France?"})
        assert resp.status_code == 200
        assert "don't know" in resp.json()["answer"].lower()
    finally:
        main_module.store = original_store
        main_module.observability = original_observability
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
