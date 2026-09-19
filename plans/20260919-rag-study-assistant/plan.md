---
status: in-progress
mode: fast
project: rag-study-assistant
target: projects/rag-study-assistant/
---

# Plan: RAG Study Assistant

Solo-learner project to practice AI engineering by building, not just reading: a
local-first RAG chat app over the user's own documents.

## Goal

Two-folder app (`backend/`, `frontend/`) where the frontend calls a FastAPI
backend over HTTP. Backend does full RAG: ingest docs -> chunk -> embed
(Ollama `nomic-embed-text`) -> store (Chroma, local persistent) -> retrieve ->
generate (Ollama `llama3.2:3b`, sized for 8GB RAM). Adds streaming,
conversation memory, request-level observability, and a retrieval eval script
on top, staged across phases so each phase is independently runnable and
demonstrably working before the next starts.

## Stack

- Backend: FastAPI + Uvicorn, Python 3.14 (`/opt/homebrew/bin/python3.14`), venv
- LLM/embeddings: Ollama (local, no API key) — `llama3.2:3b` chat, `nomic-embed-text` embeddings
- Vector store: ChromaDB (embedded, persistent dir, no external service)
- Frontend: plain HTML/CSS/vanilla JS, no build tool, served via `python -m http.server`
- Observability/memory: SQLite (stdlib `sqlite3`, no ORM)

## Phases

| # | Phase | File | Depends on |
|---|-------|------|------------|
| 1 | Backend skeleton + health check ✅ | `phase-01-backend-skeleton.md` | - |
| 2 | Ingestion + chunking + Chroma embedding pipeline ✅ | `phase-02-ingestion-embedding.md` | 1 |
| 3 | RAG query endpoint + frontend chat UI ✅ | `phase-03-rag-query-frontend.md` | 2 |
| 4 | SSE streaming + conversation memory ✅ | `phase-04-streaming-memory.md` | 3 |
| 5 | Observability logging + retrieval eval script ✅ | `phase-05-observability-eval.md` | 3 |
| 6 | (Stretch) docker-compose for Ollama+app | `phase-06-docker-stretch.md` | 3 |
| 7 | LangChain reimplementation for comparison ✅ | `phase-07-langchain-comparison.md` | 3 |

Phase 5 only needs phase 3 (not 4) — can run before/parallel with phase 4 if desired.
Phase 6 is optional and skippable entirely without breaking anything.
Phase 7 only needs phase 3 (the raw RAG baseline) — deliberately independent of
4/5/6 so it can run any time after there's something to compare against.

## Acceptance Criteria

- `uvicorn app.main:app` boots; `GET /health` returns 200.
- Dropping a `.txt`/`.md` file into `data/documents/` and calling the ingest
  endpoint results in searchable chunks in Chroma.
- Opening `frontend/index.html` (served statically) lets the user type a
  question and get a grounded answer citing which chunks were used.
- Ollama runs fully locally — no external API key required anywhere.
- `scripts/eval_retrieval.py` runs standalone and prints precision@k against
  `scripts/eval_dataset.json`.
- No mocks/fake data in shipped code — Ollama and Chroma calls are real.

## Prerequisites (user machine, one-time)

```bash
brew install ollama
brew services start ollama   # or: ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Why no LangChain in phases 1-6

Deliberate: phases 1-6 hand-roll chunking/embedding/retrieval/prompting so the
user learns the actual mechanics of RAG, not a framework's API surface.
Phase 7 then reimplements the same feature set with LangChain specifically to
make the comparison concrete — see that phase for the reasoning on why it's
additive rather than a replacement.

## Open Questions

- None — decisions (Ollama local, vanilla JS frontend, fast mode, LangChain as
  an added comparison phase rather than a replacement) confirmed by user.
