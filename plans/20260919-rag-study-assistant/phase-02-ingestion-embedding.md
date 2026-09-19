---
phase: 2
title: "Ingestion + chunking + Chroma embedding pipeline"
status: pending
priority: P1
dependencies: [1]
---

# Phase 2: Ingestion + chunking + Chroma embedding pipeline

## Overview
Turn raw documents (txt/md/pdf) into embedded, searchable chunks in a local
Chroma collection, exposed via an ingest API.

## Requirements
- Functional: `POST /documents/ingest` accepts multipart/form-data with
  either an optional `file` upload or a `text` form field (+ optional
  `source` id), chunks it, embeds via Ollama, and upserts into Chroma.
  (Deviation from the original "JSON or file" wording: FastAPI can't
  accept a JSON body and a file upload on the same route since they use
  different Content-Types; multipart-with-optional-file achieves the same
  "either/or" requirement without branching on Content-Type.)
  `GET /documents` lists ingested sources with chunk counts.
- Non-functional: chunking must be deterministic and re-runnable
  (re-ingesting the same source replaces its old chunks, not duplicates).

## Architecture
```
app/
  llm_client.py     # OllamaClient: .embed(texts: list[str]) -> list[list[float]]
                     #               .chat(messages, stream=False) -> str  (used later)
  vectorstore.py     # ChromaStore: .upsert(source, chunks, embeddings)
                     #              .query(embedding, k) -> list[Chunk]
                     #              .list_sources() -> list[{source, chunk_count}]
  ingestion.py        # load_text(path|str) -> str
                       # chunk_text(text, chunk_size=800, overlap=100) -> list[str]
  main.py              # POST /documents/ingest, GET /documents
```
- PDF support via `pypdf` text extraction; txt/md read directly.
- Chunking: simple fixed-size sliding window on characters (not tokens) to
  avoid a tokenizer dependency — chunk_size=800 chars, overlap=100. Document
  in code comment why char-based (simplicity) vs token-based (accuracy) is
  the deliberate tradeoff for a learning project.
- `OllamaClient.embed` calls `POST {OLLAMA_BASE_URL}/api/embed` with
  `{"model": EMBED_MODEL, "input": [chunk, ...]}` — verified against the
  running Ollama instance (0.34.2) that this endpoint accepts a batch array
  and returns `{"embeddings": [[...], [...]]}` in the same order, so a
  single HTTP call embeds an entire document's chunks (no client-side
  concurrency needed; simpler than originally assumed). The older
  `/api/embeddings` (singular `prompt`, one vector back) still works but is
  superseded by `/api/embed` for this use case.
- Chroma: `PersistentClient(path=CHROMA_DIR)`, one collection
  `study_documents`. Use `source` (filename or provided id) as a metadata
  field so `upsert` can `collection.delete(where={"source": source})` before
  re-adding, giving idempotent re-ingestion.

## Related Code Files
- Create: `projects/rag-study-assistant/backend/app/llm_client.py`
- Create: `projects/rag-study-assistant/backend/app/vectorstore.py`
- Create: `projects/rag-study-assistant/backend/app/ingestion.py`
- Modify: `projects/rag-study-assistant/backend/app/main.py` (add ingest/list routes)
- Modify: `projects/rag-study-assistant/backend/requirements.txt` (add `chromadb`, `pypdf`)
- Create: `projects/rag-study-assistant/backend/tests/test_ingestion.py`

## Implementation Steps
1. Add `chromadb`, `pypdf` to requirements; reinstall.
2. Implement `ingestion.load_text` (dispatch on extension) and `chunk_text`.
3. Implement `OllamaClient.embed` (httpx async client, concurrency-limited).
4. Implement `ChromaStore` wrapping `chromadb.PersistentClient`.
5. Wire `POST /documents/ingest`: accept `UploadFile` or JSON body, load ->
   chunk -> embed -> upsert, return `{"source": ..., "chunks": N}`.
6. Wire `GET /documents`: return `collection.list_sources()`.
7. Tests: chunk boundary cases (empty text, text shorter than chunk_size),
   and an ingestion round-trip test that skips gracefully (pytest.skip) if
   Ollama isn't reachable in the test environment — real integration test,
   not a mock, but must not hard-fail CI-less local runs.

## Success Criteria
- [ ] Ingesting a sample `.md` file produces >0 chunks in Chroma (verified via `GET /documents`).
- [ ] Re-ingesting the same source doesn't duplicate chunks (count stays stable).
- [ ] PDF ingestion works on at least one real sample PDF.
- [ ] `pytest backend/tests/test_ingestion.py` passes when Ollama is running.

## Risk Assessment
- Ollama embedding latency on 8GB RAM could be slow for large docs — mitigate
  with the concurrency cap and by documenting expected ingest time in README.
- Chroma's embedded mode writes to disk; ensure `CHROMA_DIR` is gitignored
  (already handled in phase 1) so the repo doesn't bloat.
