---
phase: 4
title: "SSE streaming + conversation memory"
status: done
priority: P2
dependencies: [3]
---

# Phase 4: SSE streaming + conversation memory

## Overview
Upgrade `/chat` to stream tokens as they're generated (SSE) and add
session-scoped multi-turn memory so follow-up questions have context. These
are the two patterns most production LLM apps need and most tutorials skip.

## Requirements
- Functional: `GET /chat/stream?session_id=...&message=...` (SSE, `text/event-stream`)
  streams answer tokens as they arrive from Ollama.
- Functional: conversation history per `session_id` persisted in SQLite,
  included in the prompt for follow-up turns (last N turns, truncated to
  avoid unbounded context growth).
- Frontend: switch to `EventSource` (or fetch + ReadableStream, since
  `EventSource` doesn't support POST bodies) for the chat call; generate a
  `session_id` (crypto.randomUUID()) once per browser session and reuse it.

## Architecture
```
app/
  memory.py       # ConversationStore(sqlite): .append(session_id, role, content)
                   #                            .recent(session_id, n=6) -> list[dict]
  llm_client.py     # add .chat_stream(messages) -> AsyncIterator[str]
  rag.py             # build_prompt gains history param
  main.py             # add streaming route
frontend/
  app.js               # switch to fetch() + ReadableStream reader, append tokens incrementally
```
- SQLite file: `backend/data/conversations.db` (gitignored, stdlib `sqlite3`,
  no ORM — one small module, matches YAGNI).
- Streaming implementation: FastAPI `StreamingResponse` with
  `media_type="text/event-stream"`, wrapping `llm_client.chat_stream` which
  consumes Ollama's `POST /api/chat` with `"stream": true` (newline-delimited
  JSON) and yields SSE-formatted `data: {token}\n\n` chunks.
- Because `EventSource` can't send a POST body, use `fetch()` with a POST
  request and read `response.body.getReader()` on the client instead of a
  literal `EventSource` object — still SSE-formatted payload, just consumed
  manually. Document this choice in code comment (common gotcha).

## Related Code Files
- Create: `projects/rag-study-assistant/backend/app/memory.py`
- Modify: `projects/rag-study-assistant/backend/app/llm_client.py` (add `chat_stream`)
- Modify: `projects/rag-study-assistant/backend/app/rag.py` (history-aware prompt)
- Modify: `projects/rag-study-assistant/backend/app/main.py` (streaming route)
- Modify: `projects/rag-study-assistant/frontend/app.js` (streaming fetch, session id)
- Modify: `projects/rag-study-assistant/backend/.gitignore` if needed (should already cover `*.db`)

## Implementation Steps
1. `memory.py`: create table `messages(session_id, role, content, ts)` on
   first use (`CREATE TABLE IF NOT EXISTS`); `append` and `recent`.
2. `llm_client.chat_stream`: async generator over Ollama's streamed NDJSON
   response, yielding just the content delta each iteration.
3. `rag.py`: `build_prompt` accepts `history: list[dict]`, prepends prior
   turns before the current question in the messages list.
4. `main.py`: new POST route `/chat/stream` — reads session_id + message,
   loads history, streams tokens via `StreamingResponse`, appends the final
   full answer + user message to `memory` after the stream completes.
5. `app.js`: generate/reuse `session_id` in `localStorage`, POST to
   `/chat/stream`, read the stream incrementally, append tokens to the
   current assistant bubble as they arrive.
6. Manual test: ask a question, then a follow-up using "it"/"that" — confirm
   the model resolves the reference using stored history.

## Success Criteria
- [x] Tokens appear incrementally in the browser (not all at once) — verified
      via curl SSE stream and via `test_chat_stream_reassembles_tokens_into_grounded_answer`.
- [x] A follow-up question referencing prior context gets a coherent answer —
      verified via curl ("Who built it?" resolved from history) and
      `test_chat_stream_uses_history_for_followup_question`.
- [x] Refreshing the page keeps the same session (via localStorage) and
      history continues correctly — `session_id` persisted via
      `localStorage` in `app.js`.
- [x] History length is capped at last 6 messages (3 turns) so prompts
      don't grow unbounded — `ConversationStore.recent(n=6)`.

Code review (`plans/20260919-rag-study-assistant/reports/phase-04-code-review.md`)
found one real bug: `OllamaClient.chat_stream` silently swallowed an
Ollama-side error payload mid-stream instead of surfacing it, resulting in
an empty assistant turn persisted with no visible failure. Fixed: an
`"error"` key in a streamed chunk now raises `RuntimeError`, which
`/chat/stream` catches and forwards as an SSE `event: error` (rendered in
the frontend instead of silently truncating the answer).

## Risk Assessment
- Streaming + FastAPI CORS interplay can be fiddly with some browsers'
  buffering; if `StreamingResponse` appears to arrive all at once, disable
  any reverse proxy buffering (not applicable here since there's no proxy,
  but note it) and ensure `X-Accel-Buffering: no` isn't needed for uvicorn
  dev server (it isn't).
- Unbounded SQLite growth over long sessions — acceptable for a learning
  project; note as a known limitation rather than adding TTL/pruning logic
  now (YAGNI).
