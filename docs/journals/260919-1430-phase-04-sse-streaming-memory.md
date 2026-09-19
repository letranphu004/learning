# Phase 4 Completion: SSE Streaming + Conversation Memory

**Date**: 2026-09-19 14:30  
**Severity**: Medium  
**Component**: SSE Streaming, Conversation Memory (SQLite)  
**Status**: Resolved

## What Happened

Implemented Phase 4 of the RAG study assistant: added token-by-token streaming responses via Server-Sent Events and per-session conversation history stored in SQLite. The feature works end-to-end — tokens arrive incrementally in the browser, follow-up questions resolve against prior context correctly, and session state persists across page refreshes. All 14 tests pass, including 2 new integration tests against live Ollama.

## The Brutal Truth

I shipped code that silently swallowed real errors. The code passed initial manual testing and the unit tests I wrote, but the code review caught something obvious in hindsight: when Ollama returned an error mid-stream (e.g., a typo'd model name in `.env`, something a learner will absolutely do), the endpoint would just emit empty tokens, persist an empty assistant turn to the database with zero indication of failure, and leave the user staring at a blank chat bubble. No exception. No error message. Just quiet failure.

This is exactly the kind of silent degradation the project's "no mocks, real behavior" principle exists to catch. It would have been frustrating to debug if discovered by a learner weeks into using the app.

## Technical Details

**The bug**: In `llm_client.py:57-63`, the `chat_stream()` method consumed Ollama's NDJSON stream line-by-line:

```python
async for line in resp.aiter_lines():
    if not line:
        continue
    chunk = json.loads(line)
    content = chunk.get("message", {}).get("content", "")
    if content:
        yield content
```

Ollama can return HTTP 200 and still emit an error object mid-stream: `{"error": "model \"wrong-name\" not found"}`. The code checked `chunk.get("message", {}).get("content", "")`, which returns `""` for error payloads (no `message` key). The `if content:` guard dropped the empty string, the loop completed normally, and no exception was raised anywhere.

The `/chat/stream` route then persisted this silent failure: `memory.append(session_id, "assistant", "")` — an empty turn in the history, no visible error, no retry signal.

**The fix**: Added an explicit error check before accessing the content:

```python
chunk = json.loads(line)
if "error" in chunk:
    raise RuntimeError(chunk["error"])
content = chunk.get("message", {}).get("content", "")
```

The route now catches `RuntimeError` around the stream loop and emits an SSE `event: error` message, which the frontend renders as a visible error bubble instead of silent truncation. Verified by re-running the full suite — still 14/14 passing.

## What We Tried

1. **Initial implementation**: followed the phase spec step-by-step, built out streaming, memory store, and frontend fetch logic.
2. **Manual testing**: verified streaming tokens arrived incrementally via curl, and history worked for follow-up questions by asking "Who built it?" against a fixture document.
3. **Code review**: delegated to a specialist reviewer per orchestration protocol — they traced the error-handling path and found the gap.
4. **Fix and re-test**: raised the exception, updated the frontend error handler, re-ran the suite.

## Root Cause Analysis

Two oversights:

1. **Assumption over specification**: I assumed Ollama would always return 2xx HTTP for a streamed response, so `resp.raise_for_status()` would catch all errors. Ollama actually uses HTTP 200 as a frame and embeds error conditions in the stream body itself — a learner's YAGNI mistake.

2. **Gap in manual test coverage**: I tested the happy path (model configured correctly, tokens streaming) but never simulated a misconfigured model to see what the stream looked like. A deliberate error-case test (e.g., intentionally breaking `.env` for one curl call) would have caught this immediately.

## Lessons Learned

- **Stream errors are different from HTTP errors**: When you parse a streaming response body, assume the server can embed error conditions inside the 200-OK frame. Check for error keys explicitly, don't rely on HTTP status alone.

- **Manual testing needs unhappy paths**: "Does it work when everything is right?" is insufficient. For a learning project, actually break something (typo a config value, kill the model process mid-request) and verify the error surfaces cleanly.

- **Code review caught what I missed**: This was not a theoretical bug — it's something a user would hit in realistic use (misconfigured model name is the most common setup mistake). Grateful the reviewer ran the code and traced the path rather than just skimming it.

## Next Steps

None blocking — the fix is merged and tested. The codebase now:

- Surfaces Ollama errors as visible SSE `event: error` messages instead of silently truncating
- Persists only successful turns to history (errors don't leave empty rows)
- Has 14 passing tests, including live Ollama integration tests for both streaming and history

Phase 4 is locked in as production-ready from a testing perspective.
