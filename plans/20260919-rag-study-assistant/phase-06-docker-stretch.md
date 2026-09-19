---
phase: 6
title: "(Stretch) docker-compose for Ollama + app"
status: pending
priority: P3
dependencies: [3]
---

# Phase 6: (Stretch, optional) docker-compose for Ollama + backend

## Overview
Package Ollama + backend as containers for portability practice. Purely
optional — the app works fully without Docker; this phase exists to
practice containerizing an AI service, a common real-world requirement.

## Requirements
- Functional: `docker compose up` starts an `ollama` service and a `backend`
  service that can talk to it; frontend remains a static folder the user
  opens/serves separately (no need to containerize static files).
- Non-functional: model pulls happen once into a named volume, not baked
  into the image (keeps image small, avoids re-downloading on rebuild).

## Architecture
```
docker-compose.yml
backend/Dockerfile
```
- `ollama` service: official `ollama/ollama` image, volume `ollama_data:/root/.ollama`,
  port `11434` exposed only to the compose network (backend reaches it via
  service name `http://ollama:11434`).
- `backend` service: built from `backend/Dockerfile` (python:3.14-slim base,
  `pip install -r requirements.txt`, `uvicorn app.main:app --host 0.0.0.0`),
  env var `OLLAMA_BASE_URL=http://ollama:11434` overriding the local default.
- Model pulling: document as a one-time `docker compose exec ollama ollama pull llama3.2:3b`
  (and the embed model) rather than automating it in an entrypoint script —
  keeps the compose file simple (YAGNI); automate later only if this becomes
  a recurring friction point.

## Related Code Files
- Create: `projects/rag-study-assistant/docker-compose.yml`
- Create: `projects/rag-study-assistant/backend/Dockerfile`
- Modify: `projects/rag-study-assistant/README.md` (add optional "Run with Docker" section)

## Implementation Steps
1. Write `backend/Dockerfile` (slim base, copy app, install deps, expose 8000).
2. Write `docker-compose.yml` with the two services + named volumes for
   `ollama_data` and (bind-mount) `backend/data` so Chroma/SQLite persist
   across container restarts.
3. Update `config.py`'s `OLLAMA_BASE_URL` default resolution to prefer the
   env var (already the case from phase 1) — no code change needed, just
   confirm.
4. Document the one-time model pull step in README.
5. Smoke test: `docker compose up -d`, pull models, hit `/health` on
   `localhost:8000`, confirm `ollama_reachable: true`.

## Success Criteria
- [ ] `docker compose up` brings up both services without errors.
- [ ] `/health` reports `ollama_reachable: true` when hit from the host.
- [ ] Chroma/SQLite data persists across `docker compose down && up`.

## Risk Assessment
- 8GB host RAM makes running Ollama in Docker on top of everything else
  tight — note in README that local (non-Docker) Ollama is recommended for
  this machine, and this phase is for practicing the containerization
  pattern rather than daily use.
