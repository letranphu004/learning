import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.ingestion import chunk_text
from app.main import app


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


# --- chunk_text: pure function, no external dependency, always runs ---

def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    text = "short text"
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_produces_overlapping_windows():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


# --- Real integration test against a running Ollama + a scratch Chroma dir.
# Skips (does not fail) if Ollama isn't running - a real test against real
# services, not a mock, but must not block local runs without Ollama up. ---

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not running")
def test_ingest_and_list_round_trip(tmp_path):
    import app.main as main_module
    from app.vectorstore import ChromaStore

    original_store = main_module.store
    main_module.store = ChromaStore(persist_dir=tmp_path / "chroma")
    client = TestClient(app)
    try:
        resp = client.post(
            "/documents/ingest",
            data={"text": "RAG combines retrieval with generation.", "source": "unit-test-doc"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "unit-test-doc"
        assert body["chunks"] > 0

        listed = client.get("/documents").json()
        assert any(d["source"] == "unit-test-doc" for d in listed)

        # Re-ingesting the same source must not duplicate chunks.
        resp2 = client.post(
            "/documents/ingest",
            data={"text": "RAG combines retrieval with generation.", "source": "unit-test-doc"},
        )
        count_after_first = body["chunks"]
        count_after_second = resp2.json()["chunks"]
        assert count_after_second == count_after_first
    finally:
        main_module.store = original_store
        shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
