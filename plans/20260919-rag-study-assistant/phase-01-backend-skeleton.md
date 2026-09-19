---
phase: 1
title: "Backend skeleton + health check"
status: completed
priority: P1
dependencies: []
---

# Phase 1: Backend skeleton + health check

## Overview
Stand up the FastAPI app skeleton, project layout, venv, and dependency
pinning so every later phase has a running server to build on.

## Requirements
- Functional: `GET /health` returns `{"status": "ok", "ollama_reachable": bool}`.
- Non-functional: runs on Python 3.14 in an isolated venv; no global installs.

## Architecture
```
backend/
  app/
    __init__.py
    main.py        # FastAPI() instance, route registration, startup check
    config.py       # Settings via pydantic-settings or plain dataclass + os.environ
  data/
    documents/.gitkeep
    chroma/.gitkeep
  tests/
    test_health.py
  requirements.txt
  .env.example
```
`config.py` centralizes: `OLLAMA_BASE_URL` (default `http://localhost:11434`),
`CHAT_MODEL` (`llama3.2:3b`), `EMBED_MODEL` (`nomic-embed-text`),
`CHROMA_DIR` (`./data/chroma`), `DOCUMENTS_DIR` (`./data/documents`).

`main.py` startup event does a lightweight `GET {OLLAMA_BASE_URL}/api/tags`
with a short timeout to populate `ollama_reachable` — never blocks server
boot if Ollama is down, just reports it.

## Related Code Files
- Create: `projects/rag-study-assistant/backend/app/__init__.py`
- Create: `projects/rag-study-assistant/backend/app/main.py`
- Create: `projects/rag-study-assistant/backend/app/config.py`
- Create: `projects/rag-study-assistant/backend/requirements.txt`
- Create: `projects/rag-study-assistant/backend/.env.example`
- Create: `projects/rag-study-assistant/backend/tests/test_health.py`
- Create: `projects/rag-study-assistant/backend/data/documents/.gitkeep`
- Create: `projects/rag-study-assistant/backend/data/chroma/.gitkeep`
- Create: `projects/rag-study-assistant/.gitignore`
- Create: `projects/rag-study-assistant/README.md` (setup steps only for now; expanded in later phases)

## Implementation Steps
1. `python3.14 -m venv backend/.venv` (documented in README, not committed).
2. `requirements.txt`: `fastapi`, `uvicorn[standard]`, `httpx`, `pydantic-settings`, `pytest`, `pytest-asyncio` (pin minor versions current as of Sep 2026).
3. Write `config.py` with a `Settings` class reading env vars with the defaults above.
4. Write `main.py`: create app, CORS middleware allowing the frontend's static-server origin (e.g. `http://localhost:8080`), `/health` route.
5. Write `.gitignore`: `backend/.venv/`, `backend/data/chroma/*` (keep `.gitkeep`), `*.db`, `.env`, `__pycache__/`.
6. Write `test_health.py` using `TestClient` — assert 200 and `status == "ok"` regardless of Ollama availability.
7. README: venv creation, `pip install -r requirements.txt`, `uvicorn app.main:app --reload --app-dir backend`.

## Success Criteria
- [x] `uvicorn app.main:app --app-dir backend --reload` starts without errors.
- [x] `curl localhost:8000/health` returns 200 with a JSON body.
- [x] `pytest backend/tests/test_health.py` passes.
- [x] Works identically whether or not Ollama is running (health check degrades gracefully — verified with Ollama not installed on this machine).

## Risk Assessment
Low risk — no external calls block startup. Main pitfall: CORS misconfig
blocking the phase-3 frontend later; keep the allowed-origins list in
`config.py` so it's a one-line change.
