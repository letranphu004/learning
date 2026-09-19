# Phase 4 Code Review — SSE Streaming + Conversation Memory

## Scope

- Created: `backend/app/memory.py`, `backend/tests/test_memory.py`
- Modified: `backend/app/llm_client.py`, `backend/app/rag.py`, `backend/app/main.py`,
  `backend/tests/test_rag.py`, `frontend/app.js`
- Verification performed: full read of all six files, cross-referenced against
  `plans/20260919-rag-study-assistant/phase-04-streaming-memory.md`, plus an
  actual test run (`pytest tests/test_memory.py tests/test_rag.py -v`) against
  a live local Ollama instance — all 8 tests passed, including the two new
  Ollama-dependent integration tests (`test_chat_stream_reassembles_tokens_into_grounded_answer`,
  `test_chat_stream_uses_history_for_followup_question`).
- The POST `/chat/stream` interpretation (vs. the spec bullet's `GET`) is
  correctly and consistently implemented as POST + JSON body throughout
  `main.py`, `test_rag.py`, and `app.js` — no inconsistency found, per the
  task's guidance to treat this as intended.

## Overall Assessment

The implementation is sound. The two things the task flagged as likely bug
locations — `ConversationStore.recent()` ordering and the SSE
event-parsing symmetry between `main.py` and `app.js` — are both **correct**,
verified by tracing the logic and by a passing round-trip test that
exercises exactly this path against a real model. One real, plausible
robustness gap exists in `chat_stream`'s NDJSON handling (silent swallowing
of an Ollama-reported error mid-stream). No regressions found in the
non-streaming `/chat` or `/documents/ingest` paths.

## Critical Issues

None.

## High Priority

### 1. `chat_stream` silently swallows Ollama-reported errors instead of surfacing them

`backend/app/llm_client.py:57-63`:

```python
async for line in resp.aiter_lines():
    if not line:
        continue
    chunk = json.loads(line)
    content = chunk.get("message", {}).get("content", "")
    if content:
        yield content
```

`resp.raise_for_status()` only catches a non-2xx *HTTP* status. Ollama can
return HTTP 200 and still emit an error condition inside the streamed body
(e.g. `{"error": "model \"wrong-name\" not found"}` for a bad
`chat_model` config value, or a mid-generation failure) with no `message`
key. `chunk.get("message", {}).get("content", "")` silently degrades this to
`""`, the `if content:` guard drops it, and the loop completes normally.

Effect: the user sees an empty assistant bubble with no error message, the
route still reaches `memory.append(req.session_id, "assistant", "")` and
persists an empty assistant turn into history, and no exception is raised
anywhere — this is a real failure mode a learner will actually hit (e.g. a
typo'd `chat_model` in `.env`), not a theoretical one.

Suggested fix (small, no new abstraction):
```python
chunk = json.loads(line)
if "error" in chunk:
    raise RuntimeError(chunk["error"])
content = chunk.get("message", {}).get("content", "")
```
This surfaces the error as an exception during generation; since
`StreamingResponse` has already committed a 200 status by the time the
generator runs, the client will see the fetch's reader throw (already
handled by `app.js`'s existing `try/catch` → `Error: ...` bubble) instead of
a silently empty answer. This is consistent with the project's existing
error-handling style (no new middleware or retry layer needed).

## Medium Priority

### 2. Known limitation, not a blocker: mid-stream exceptions can't change the HTTP status

Because `StreamingResponse` sends headers (200 OK) before the generator body
runs, any exception raised inside `event_stream()` (including the fix above,
or a genuine `httpx` failure like Ollama going down mid-generation) will
terminate the connection after a 200 has already been sent — the client only
learns about the failure via a broken read, not a proper error status. This
matches the FastAPI/Starlette streaming model in general and isn't something
worth engineering around for a single-user local learning app (matches the
phase spec's own "note as a known limitation" stance on similar risks). Flagging
for awareness only — no action required.

### 3. Weak assertion in the new follow-up-history integration test

`backend/tests/test_rag.py:97-131`
(`test_chat_stream_uses_history_for_followup_question`) only asserts
`"model" in full_answer.lower()` for the follow-up turn. This is a fairly
loose signal that history was actually used (the word "model" could appear
even without context resolution, given the ingested fixture text itself
contains "model"). It does exercise the real code path with a real LLM call
(not a phantom test), and it's consistent with the loose-assertion style
already used elsewhere in this file (e.g. `test_chat_says_it_does_not_know_when_unrelated`
just checks for the substring "don't know"), so this is not a new problem
introduced by this diff — just worth noting it's a weak proof of the
"coherent answer" acceptance criterion. A stronger version would ask something
that can only be answered correctly if history was loaded (e.g. "What model
did I just ask about?" checked against the specific model name), but this is
optional polish, not a blocker.

## Low Priority

- `backend/app/config.py:21` names the sqlite file `app.db`, while the phase
  spec's Architecture section says `backend/data/conversations.db`. Purely
  cosmetic — `*.db` is already gitignored at the repo root and the file is a
  single shared name used only by `ConversationStore`. Not worth a rename in
  a solo project, mentioning only because the spec explicitly named it.
- `ConversationStore.append()` (`backend/app/memory.py:24`) has no
  docstring, unlike `recent()` which explains its oldest-first contract. Minor
  stylistic inconsistency with the project's "docstring explains why" convention
  observed elsewhere (e.g. `rag.py`, `llm_client.py` module docstrings).
- `backend/tests/test_memory.py:23-38`
  (`test_recent_caps_to_n_most_recent_messages`) manually imports `tempfile`
  and `pathlib.Path` inside the test body instead of using the `tmp_path`
  pytest fixture used by every other test in the same file. Inconsistent
  style, no functional difference — worth aligning for DRY/readability but not
  urgent.
- `n=6` in `ConversationStore.recent(session_id, n=6)` caps to 6 *messages*
  (3 user/assistant turn-pairs), whereas the phase spec's Success Criteria
  bullet says "last 6 turns." The spec itself hedges with "(e.g. last 6
  turns)" so this isn't a contract violation, just worth confirming the
  interpretation (6 messages, not 6 turns) was an intentional and accepted
  reading rather than an oversight.

## Verified-Correct Items (explicitly checked per the review request)

### `ConversationStore.recent()` ordering — correct, not buggy

```sql
SELECT role, content FROM messages
WHERE session_id = ? ORDER BY seq DESC LIMIT ?
```
followed by `reversed(rows)` in Python. Walking through it: `ORDER BY seq
DESC LIMIT n` selects the *n most recent* rows (highest `seq` first), then
`reversed()` flips that back to ascending `seq` order (oldest of the
selected window first). This is the correct standard pattern for "last N,
oldest-first" and matches `test_recent_caps_to_n_most_recent_messages`,
which asserts exactly `["message 6", "message 7", "message 8", "message
9"]` out of 10 inserted rows when `n=4` — verified passing. No off-by-one:
`seq` is computed per-session via `COALESCE(MAX(seq), -1) + 1` inside the
same synchronous `INSERT` statement in `append()`, so the first message gets
`seq=0`, and there's no gap/duplicate risk since the max-then-insert happens
in one non-yielding SQL statement (see concurrency note below).

### Shared sqlite3 connection / `check_same_thread=False` — correct for this design, not a real race

- All `ConversationStore` calls from `main.py`'s routes are plain synchronous
  calls made directly inside `async def` route handlers — they are never
  dispatched to a worker thread (FastAPI only does that for handlers it
  detects as sync `def`, and both `/chat` and `/chat/stream` are `async
  def`). Under a normal single-process `uvicorn` dev server, every
  `ConversationStore` call therefore runs on the one asyncio event-loop
  thread, and each `execute()`/`commit()` pair completes without an
  intervening `await`, so no other coroutine can interleave mid-statement.
  There is no real read-then-write race for the `seq` computation in this
  deployment model.
- `check_same_thread=False` is not defensive over-engineering here — it is
  actually load-bearing for the test suite: Starlette's `TestClient` runs the
  ASGI app through an `anyio` blocking portal, which executes the app (and
  therefore all `ConversationStore` calls made from `main.py`'s
  module-level `memory` singleton) on a different OS thread than the one
  that constructed the `sqlite3.Connection`. Without
  `check_same_thread=False`, `tests/test_rag.py`'s two new streaming tests
  would raise `sqlite3.ProgrammingError` immediately. Confirmed by running
  the suite — it passes. This is a case where a check the reviewer might
  reflexively flag as "unnecessary paranoia" is in fact necessary and correct.
- One real (but low-severity, and inherent to the product design rather than
  a code bug) sequencing note: `memory.recent()` is read in the route body
  *before* the `await retrieve_chunks(...)` call that precedes it, and the
  await point means a second concurrent request for the same `session_id`
  could read history before the first request's turn is appended. This can't
  happen through the shipped frontend (the form disables the input field for
  the duration of the fetch, so a single browser tab can't produce
  overlapping requests), so it doesn't need defensive code per the project's
  YAGNI stance — noting it only so it's a documented, not accidental,
  limitation.

### SSE format symmetry between `main.py` and `app.js` — correct, including the `event: done` case

Server emits, per token: `data: {json}\n\n` (single line, default event
type). Final event: `event: done\ndata: {json}\n\n` (two lines in one
block). Client's `consumeSseEvents` (`frontend/app.js:43-57`):

```js
const parts = buffer.split("\n\n");
const tail = parts.pop();
for (const part of parts) {
  const lines = part.split("\n");
  let eventType = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (data) onEvent(eventType, JSON.parse(data));
}
return tail;
```

Traced both cases:
- Token block `"data: {...}"` → splits to one line → `eventType` stays
  default `"message"`, `data` set → `onEvent("message", ...)` → matches the
  `if (eventType === "message")` branch that appends the token. Correct.
- Done block `"event: done\ndata: {...}"` → splits to two lines in the
  **same** block (both lines belong to one `\n\n`-delimited part, so they're
  processed together, not across two `consumeSseEvents` calls) → `eventType`
  becomes `"done"` from the first line, `data` set from the second line →
  `onEvent("done", ...)` fires with the parsed sources payload. Correct — the
  scenario the task specifically asked to check (event line and data line
  landing in the same block) works because splitting happens on `\n\n`
  first, then `\n` only within an already-complete block.
- Partial-chunk buffering: `buffer.split("\n\n")` + `pop()` to hold back an
  incomplete trailing block is the standard, correct pattern — a network
  chunk boundary landing anywhere inside a block (even between `event:` and
  `data:` lines) is preserved in `tail` and re-processed whole on the next
  `read()`, since the split point is the double-newline, not any single
  line. Token content itself can't introduce a stray bare `\n` into the SSE
  framing because `json.dumps` escapes newlines within the JSON string.
  Confirmed no buffering bug.

### Persistence order in `/chat/stream` — correct

`backend/app/main.py:103-113`: `memory.append(session_id, "user", ...)` is
called before `memory.append(session_id, "assistant", ...)`, both after the
`async for token in llm.chat_stream(...)` loop completes and before the
final `done` event is yielded. Since `seq` is assigned monotonically per
`append()` call, this guarantees the user turn always gets a lower `seq`
than its paired assistant turn, so a later `recent()` call returns correct
chronological role ordering. Verified directly by the passing
`test_chat_stream_reassembles_tokens_into_grounded_answer` assertion:
`history[-2]` is the user message, `history[-1]` is the assistant's full
reassembled answer.

### Regression check on `/chat` and `/documents/ingest` — no regression

`rag.py`'s `answer_question` now delegates retrieval to the new
`retrieve_chunks(question, store, llm)` helper but the sequence of
operations (embed → query → build prompt with no history → non-streaming
`llm.chat`) is unchanged, and `build_prompt`'s `history` param defaults to
`None` (`*(history or [])` becomes a no-op), so `/chat`'s message shape is
byte-for-byte identical to before. Confirmed empirically:
`test_chat_answers_from_ingested_document` and
`test_chat_says_it_does_not_know_when_unrelated` (pre-existing behavior)
both still pass unchanged. `/documents/ingest` was not touched by this diff
at all.

## Edge Cases Considered (Scout)

- Concurrent requests to the same `session_id` racing on `memory.recent()`
  vs. `memory.append()` — addressed above; not exploitable through the
  shipped frontend, and not a data-corruption risk given the single-threaded
  event-loop execution model.
- Client disconnects mid-stream — the async generator has no
  `request.is_disconnected()` check, so on a broken pipe the generator will
  either keep running to completion (still persisting history, arguably
  desirable) or be interrupted by a `GeneratorExit` at the next `yield`,
  which would skip the trailing `memory.append()` calls. Edge case, low
  severity, no action needed for a local single-user tool.
- Empty-token filtering (`if content:` in `chat_stream`, `if (data)` in
  `consumeSseEvents`) — traced through and consistent; the server never
  emits an empty-data SSE line, so the client-side guard is a no-op safety
  net, not masking a real case.
- `ChatStreamRequest.session_id` has no validation (empty string, absurdly
  long value, etc.) — acceptable given the local-only threat model (no auth
  layer exists anywhere in this app yet, CORS is already restricted to
  `localhost:8080`/`127.0.0.1:8080`); not a trust-boundary gap worth adding
  validation for at this stage.

## Positive Observations

- The two new integration tests in `test_rag.py` are genuine, not phantom —
  they run a real Ollama model, ingest a real document, and assert on the
  actual reassembled/history content, catching real regressions (verified by
  actually running them against a live Ollama instance).
- Module docstrings continue the established "explain why, not what"
  convention (`memory.py`'s no-ORM rationale, `llm_client.py`'s NDJSON vs.
  SSE translation note, `main.py`'s POST-vs-EventSource comment) — good
  continuity with phases 1-3.
- No new abstractions, config layers, or premature generalization were
  introduced; `ConversationStore` is a single small class matching the
  spec's YAGNI guidance.

## Recommended Actions

1. (High) Add an explicit check for an `"error"` key in each streamed NDJSON
   chunk in `llm_client.chat_stream` and raise, so a misconfigured model name
   or Ollama-side failure surfaces as a visible error instead of a silently
   empty, persisted assistant turn.
2. (Optional/Low) Strengthen the follow-up-history test assertion if/when
   convenient — not blocking given it does exercise the real code path.
3. (Optional/Low) Align `test_recent_caps_to_n_most_recent_messages` to use
   the `tmp_path` fixture like its neighbors, for consistency.

## Plan Status (Phase 4 Success Criteria)

- [x] Tokens appear incrementally — confirmed by generator design
      (`async for token in llm.chat_stream(...): yield ...` per token, not
      buffered) and by the multi-`data:`-event structure observed in the
      passing test.
- [x] Follow-up question referencing prior context gets a coherent answer —
      confirmed via a live-model integration test and by tracing
      `history` load → `build_prompt` prepend order.
- [x] Refreshing the page keeps the same session — `getSessionId()` reads
      `localStorage` first, only generates a new UUID if absent.
- [x] History length capped — `memory.recent(session_id, n=6)` bounds the
      prompt to the 6 most recent messages per session.

All four Phase 4 acceptance criteria are met. Recommend addressing the High
item (error swallowing) before treating this phase as fully hardened, but it
does not block moving on functionally — the streaming and memory features
work correctly end-to-end today.

## Metrics

- Files reviewed: 6 (2 new, 4 modified)
- Test run: `pytest tests/test_memory.py tests/test_rag.py -v` → 8 passed, 0
  failed, 0 skipped (Ollama was reachable locally)
- Type coverage / lint: not separately run (no lint/type command found
  configured beyond standard project conventions already exercised by the
  passing tests)

## Unresolved Questions

- Confirm whether `n=6` meaning "6 messages" (3 turns) vs. the spec's "6
  turns" wording was an intentional simplification — flagged as Low priority
  above, not blocking.
