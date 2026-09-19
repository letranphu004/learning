# RAG Study Assistant

A local-first RAG chat app for learning AI engineering by building it, not just
reading about it. Ask questions about your own documents; the backend
retrieves relevant chunks and generates a grounded answer with a local LLM
(Ollama) — no API key, no cloud cost.

See `plans/20260919-rag-study-assistant/` (repo root) for the full phased plan.

## Prerequisites

```bash
brew install ollama
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Run Ollama manually when you're working on this project (not as an
always-on `brew services` background service — keeps it off your RAM
budget the rest of the time on an 8GB machine):

```bash
ollama serve   # leave running in its own terminal/tab while developing
```

## Backend setup

```bash
cd projects/rag-study-assistant/backend
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .
```

Verify: `curl http://127.0.0.1:8000/health` — should return
`{"status": "ok", "ollama_reachable": true}` (or `false` if Ollama isn't
running yet; the server still starts either way). Use `127.0.0.1`, not
`localhost` — Ollama binds IPv4 only, and on some machines `localhost`
resolves IPv6 first, adding a multi-second stall before falling back.

Run tests: `pytest` (from `backend/`, with the venv active).

## Frontend setup

```bash
cd projects/rag-study-assistant
python3 -m http.server 8080 --directory frontend
```

Open `http://localhost:8080/index.html` with the backend (`uvicorn`, port
8000) already running. Use the file picker at the top of the page to
ingest a `.pdf`/`.txt`/`.md` document (calls `POST /documents/ingest`
directly — same endpoint curl or the `/docs` Swagger page hit), then ask a
question in the chat box. `API_BASE` in `frontend/app.js` points at
`http://localhost:8000`; change it if the backend runs elsewhere.

### Chat behavior

Chat responses stream in real-time via Server-Sent Events (SSE). Conversation history is stored server-side in SQLite (`backend/data/app.db`, auto-created on first run) and persisted per browser via localStorage; page refreshes continue the same session. The model receives context from the last 6 messages (3 conversation turns); older history remains in the database but is not sent to the model.

## Debugging & Evaluation

Every `/chat` and `/chat/stream` call logs a row (timestamp, session, query,
retrieved chunk ids + distances, response, latency, prompt/response char
counts) to the `interactions` table in `backend/data/app.db` — logging is
best-effort and never breaks a chat response if it fails. Inspect recent
calls with:

```bash
sqlite3 backend/data/app.db "select ts, query, latency_ms from interactions order by rowid desc limit 5"
```

Measure retrieval quality against a small hand-labeled dataset
(`backend/scripts/eval_dataset.json`, 5 questions over the shipped
`sample.md`) with:

```bash
python backend/scripts/eval_retrieval.py
```

Prints a precision@k line per question plus an aggregate score. This is a
starting point for building eval intuition on a tiny corpus, not a
rigorous benchmark — extend `eval_dataset.json` as you ingest more of your
own documents.

## LangChain comparison

`backend_langchain/` reimplements the same ingest + non-streaming chat
contract (`GET /health`, `POST /documents/ingest`, `POST /chat`) using
LangChain instead of hand-rolled code, as a second backend running
alongside the raw one — purely additive, doesn't touch `backend/` or
`frontend/`. See `docs/langchain-comparison.md` for what changed, what it
cost, and real latency/answer numbers from an actual run.

```bash
cd projects/rag-study-assistant/backend_langchain
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir . --port 8001
```

With both backends running (raw on :8000, LangChain on :8001) and the same
document ingested into both:

```bash
python projects/rag-study-assistant/scripts/compare_answers.py "your question"
```

Prints both backends' answer and latency side by side. No `--reload` file
watching conflict since they're separate processes on separate ports.

### Troubleshooting: `chroma-hnswlib` fails to build (`'iostream' file not found`)

On a fresh macOS + Command Line Tools setup, `pip install` can fail building
`chromadb`'s C++ extension because the CLT's own
`usr/include/c++/v1` is incomplete while the real libc++ headers live under
the SDK path instead. Fix by pointing the compiler at the SDK's copy before
installing:

```bash
export CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
pip install -r requirements.txt
```

(Root cause, not a project bug — reinstalling Command Line Tools
`xcode-select --install` after removing the old one also fixes it
permanently, but the env var is faster and non-disruptive.)

## Status

- [x] Phase 1 — backend skeleton + health check
- [x] Phase 2 — ingestion + chunking + Chroma embedding pipeline
- [x] Phase 3 — RAG query endpoint + frontend chat UI
- [x] Phase 4 — SSE streaming + conversation memory
- [x] Phase 5 — observability logging + retrieval eval script
- [ ] Phase 6 — (stretch) docker-compose
- [x] Phase 7 — LangChain reimplementation for comparison
