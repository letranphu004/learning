---
phase: 3
title: "RAG query endpoint + frontend chat UI"
status: done
priority: P1
dependencies: [2]
---

# Phase 3: RAG query endpoint + frontend chat UI

## Overview
Wire retrieval + generation into a single non-streaming `/chat` endpoint, and
build the vanilla-JS frontend that calls it. This is the first end-to-end
"ask a question about my docs, get an answer" milestone.

## Requirements
- Functional: `POST /chat {"message": str}` embeds the message, retrieves
  top-k chunks from Chroma, builds a grounded prompt, calls Ollama chat, and
  returns `{"answer": str, "sources": [{"source": str, "text": str}]}`.
- Functional: frontend shows a simple chat log, an input box, and renders
  which source chunks backed each answer (transparency into RAG, not just a
  black-box chatbot — this is the point of the learning exercise).
- Non-functional: k configurable via `config.py` (default 4).

## Architecture
```
app/
  rag.py     # build_prompt(question, chunks) -> messages
             # answer_question(question) -> {"answer", "sources"}
  main.py     # POST /chat
frontend/
  index.html  # chat log + input form
  style.css    # minimal, readable
  app.js        # fetch("/chat", {method: "POST", body: JSON.stringify({message})})
                # renders answer + collapsible source list
```
Prompt template (kept in `rag.py` as a plain string, no template engine —
YAGNI): system message instructs the model to answer only from provided
context and say "I don't know from the given documents" if the context is
insufficient. This teaches grounding/faithfulness as a concrete constraint,
not just a nice-to-have.

Frontend calls the backend at `http://localhost:8000` (configurable via a
`const API_BASE` at the top of `app.js`). Served separately via
`python -m http.server 8080 --directory frontend`.

## Related Code Files
- Create: `projects/rag-study-assistant/backend/app/rag.py`
- Modify: `projects/rag-study-assistant/backend/app/main.py` (add `/chat`)
- Create: `projects/rag-study-assistant/frontend/index.html`
- Create: `projects/rag-study-assistant/frontend/style.css`
- Create: `projects/rag-study-assistant/frontend/app.js`
- Create: `projects/rag-study-assistant/backend/tests/test_rag.py`

## Implementation Steps
1. Implement `rag.build_prompt(question, chunks)` returning an Ollama-chat
   `messages` list (system + user, with retrieved chunks interpolated into
   the system message under a `Context:` section).
2. Implement `rag.answer_question`: embed question -> `vectorstore.query` ->
   `build_prompt` -> `llm_client.chat(messages)` -> shape response.
3. Wire `POST /chat` in `main.py`.
4. Build `index.html` with a message list `<div>` and a `<form>` input.
5. `app.js`: on submit, POST to `/chat`, append user + assistant messages to
   the log, render `sources` as a small expandable list under the answer.
6. `style.css`: simple, legible, no framework.
7. Test: `test_rag.py` hits `/chat` against a fixture-ingested doc (uses the
   same skip-if-no-Ollama pattern as phase 2) and asserts the answer
   references content actually present in the fixture doc.

## Success Criteria
- [x] Ingest a doc, ask a question about it, get a relevant answer with
      visible source chunks (verified via curl against a running backend;
      no Chrome extension available this session to click through the UI).
- [x] Asking something unrelated to ingested docs yields an honest
      "I don't know" rather than a hallucinated answer (asserted in
      `test_chat_says_it_does_not_know_when_unrelated`).
- [x] `pytest backend/tests/test_rag.py` passes when Ollama is running.
- [x] No CORS errors between `localhost:8080` frontend and `localhost:8000`
      backend (verified via OPTIONS preflight).

## Risk Assessment
- Small local models (`llama3.2:3b`) may not always follow the "say I don't
  know" instruction perfectly — acceptable for a learning project; note this
  as an observed limitation in README rather than over-engineering guardrails.
- Frontend/backend port mismatch is the most likely setup friction — document
  both ports explicitly in README.
