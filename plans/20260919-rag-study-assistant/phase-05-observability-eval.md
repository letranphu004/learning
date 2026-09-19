---
phase: 5
title: "Observability logging + retrieval eval script"
status: done
priority: P2
dependencies: [3]
---

# Phase 5: Observability logging + retrieval eval script

## Overview
Add the two things that separate "a demo" from "an AI engineer's project":
structured logging of every RAG request, and a repeatable way to measure
retrieval quality. Depends only on phase 3 (the `/chat` endpoint existing) —
can be built before or in parallel with phase 4.

## Requirements
- Functional: every `/chat` (and `/chat/stream`) call logs
  `{timestamp, session_id, query, retrieved_chunk_ids, retrieved_scores,
  response, latency_ms, prompt_chars, response_chars}` to SQLite.
- Functional: `scripts/eval_retrieval.py` loads a hand-labeled
  `scripts/eval_dataset.json` (list of `{"question": str, "expected_source": str}`),
  runs retrieval only (no generation) for each, computes precision@k
  (fraction of retrieved chunks whose `source` matches `expected_source`),
  and prints a per-question + aggregate report.
- Non-functional: logging must never block or fail the chat response (best
  effort — wrap in try/except, log to stderr on failure).

## Architecture
```
app/
  observability.py   # log_interaction(...) -> None (SQLite insert, table `interactions`)
  main.py              # call log_interaction after each /chat and /chat/stream response
scripts/
  eval_retrieval.py     # standalone script, imports app.vectorstore + app.llm_client directly
  eval_dataset.json       # small (5-10 entry) hand-labeled Q/A-to-source mapping
```
- `observability.py` uses the same lightweight sqlite3-stdlib pattern as
  `memory.py` from phase 4 — separate table (`interactions`), can even live
  in the same `.db` file (`backend/data/app.db`) to avoid multiple files.
  Rename `conversations.db` -> `app.db` if phase 4 already shipped, keeping
  both tables (`messages`, `interactions`) in one file (simpler, matches DRY —
  document this consolidation as a one-line migration note in README).
- Token estimate: no tokenizer dependency — approximate with
  `len(text) // 4` and label the field `*_chars` (not `*_tokens`) to be
  honest about the approximation rather than implying real tokenization.
- Eval script is a plain CLI (`python scripts/eval_retrieval.py`), not a
  pytest test — it's a research/measurement tool the user re-runs by hand
  when tuning chunk size, k, or embedding model, matching the "vừa làm vừa
  research" goal.

## Related Code Files
- Create: `projects/rag-study-assistant/backend/app/observability.py`
- Modify: `projects/rag-study-assistant/backend/app/main.py` (call logger after chat responses)
- Modify: `projects/rag-study-assistant/backend/app/memory.py` (rename db path if phase 4 shipped first; else skip)
- Create: `projects/rag-study-assistant/backend/scripts/eval_retrieval.py`
- Create: `projects/rag-study-assistant/backend/scripts/eval_dataset.json`
- Modify: `projects/rag-study-assistant/README.md` (document how to inspect logs + run eval)

## Implementation Steps
1. `observability.py`: `CREATE TABLE IF NOT EXISTS interactions (...)`, `log_interaction(**fields)`.
2. Wire into `main.py`'s `/chat` and `/chat/stream` handlers (wrap in try/except).
3. Write `eval_dataset.json` with 5-10 real Q/A pairs against whatever sample
   doc the user ingests during phase 2/3 testing (ship one sample doc under
   `backend/data/documents/sample.md` for this purpose so eval is runnable
   out of the box).
4. `eval_retrieval.py`: for each entry, embed question, query Chroma top-k,
   check if `expected_source` appears in the returned sources, accumulate
   precision@k, print per-question pass/fail + aggregate score at the end.
5. README: add a "Debugging & Evaluation" section — `sqlite3 backend/data/app.db "select * from interactions order by ts desc limit 5"` and `python backend/scripts/eval_retrieval.py`.

## Success Criteria
- [x] After a few chat interactions, `interactions` table has matching rows
      with plausible latency numbers — verified live via `sqlite3 backend/data/app.db`.
- [x] Logging failure does not break `/chat` — `test_log_interaction_failure_does_not_raise`;
      `except sqlite3.Error` broadened to `except Exception` per code review
      (the original narrower clause only worked by accident of current data shapes).
- [x] `eval_retrieval.py` runs standalone and prints a precision@k score
      between 0 and 1 for the shipped sample dataset.

Code review (`plans/20260919-rag-study-assistant/reports/phase-05-code-review.md`)
found a real bug: `precision@k` divided by the *requested* k instead of the
*actual* number of chunks Chroma returned, which it silently caps below k
when the collection has fewer vectors — producing an identical,
non-discriminating score across all 5 questions on a small corpus. Fixed:
divide by `len(results)`, matching the spec's literal "fraction of
retrieved chunks" wording; re-running against the (now larger) real store
produces varied scores (0.25/0.50) instead of a flat 0.50.

## Risk Assessment
- Hand-labeled eval sets of 5-10 items are statistically thin — acceptable
  for a learning tool; README should note this is a starting point for
  building eval intuition, not a rigorous benchmark.
- SQLite single-writer contention is a non-issue at this scale (single user,
  local dev) — no need for a queue or async writer (YAGNI).
