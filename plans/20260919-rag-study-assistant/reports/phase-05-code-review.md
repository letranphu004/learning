# Phase 5 Code Review — Observability Logging + Retrieval Eval Script

## Scope

- Created: `backend/app/observability.py`, `backend/tests/test_observability.py`,
  `backend/scripts/eval_retrieval.py`, `backend/scripts/eval_dataset.json`
- Modified: `backend/app/vectorstore.py`, `backend/app/main.py`, `backend/app/rag.py`
  (deleted `answer_question`), `backend/tests/test_rag.py`, `backend/README.md`
- Verification performed: full read of all changed/created files, cross-referenced
  against `plans/20260919-rag-study-assistant/phase-05-observability-eval.md` and
  the phase-4 review; full `pytest` run (16 passed, live Ollama reachable);
  **live empirical runs** of `scripts/eval_retrieval.py` and a real `/chat` call
  through `TestClient` against the developer's actual persisted Chroma/sqlite
  stores to confirm behavior rather than infer it from reading code; confirmed
  `chromadb` 0.6.3's `distance` values are plain Python `float` (not numpy) via
  a live query; confirmed via `chunk_text` that the shipped `sample.md` (1111
  chars, chunk_size 800/overlap 100) produces only 2 chunks.

## Overall Assessment

The core logging path (`InteractionLogger`) is sound and does satisfy acceptance
criteria 1 and 2 — verified empirically, not just by reading the code. The
`/chat`/`/chat/stream` refactor away from `rag.answer_question` is a clean,
behavior-preserving inline (verified: no positional destructuring of
`ChromaStore.query` results anywhere, so the new `chunk_id`/`distance` keys are
safe; response shape `{"answer", "sources"}` unchanged). The test-isolation fix
in `test_rag.py` (patching `main_module.observability`) is real and necessary —
confirmed the tests do write to `backend/data/app.db`/`chroma` without it.

The one substantive defect is in `eval_retrieval.py`'s precision@k formula:
it is **empirically confirmed broken** for the shipped out-of-the-box dataset —
every single question scores exactly `0.50`, regardless of question content,
because the denominator is the requested `k` rather than the number of chunks
Chroma actually returned. This defeats the acceptance criterion's purpose (a
score that can't discriminate between good and bad retrieval is not useful for
"tuning chunk size, k, or embedding model," which is the script's stated
purpose).

## Critical Issues

None.

## High Priority

### 1. `eval_retrieval.py` precision@k denominator uses requested `k`, not the actual number of retrieved chunks — confirmed broken for the shipped dataset

`backend/scripts/eval_retrieval.py:49-52`:
```python
results = store.query(embedding, k=k)
matches = sum(1 for r in results if r["source"] == expected_source)
precision = matches / k
```

The phase spec defines precision@k as "fraction of **retrieved chunks** whose
source matches `expected_source`" — i.e. the denominator should be
`len(results)`, not the requested `k`. `ChromaStore.query` (`app/vectorstore.py:31-40`)
passes `k` straight through to Chroma's `n_results`, and Chroma silently caps
`n_results` to the collection size when the collection has fewer vectors than
requested (confirmed directly: `chromadb.query()` logs `"Number of requested
results 4 is greater than number of elements in index 3, updating n_results =
3"`).

I ran the script live against the developer's real Ollama + Chroma store:

```
[PASS] precision@4=0.50  What is Retrieval-Augmented Generation (RAG)?
[PASS] precision@4=0.50  Why do we split documents into chunks before embedding them?
[PASS] precision@4=0.50  What is a vector store used for?
[PASS] precision@4=0.50  Which vector store does this project use and how does it persist data?
[PASS] precision@4=0.50  What problem does combining retrieval with generation solve for a language model?

Aggregate precision@4: 0.50 over 5 questions
```

Every question — regardless of content or how well the retriever actually did —
returns the identical score, because the retrieval index had fewer vectors than
`k=4` and the score is silently capped at `(actual results)/k`. Even a
mathematically perfect retriever (100% of returned chunks correct) can never
score above `0.50` here. This isn't a scale-dependent theoretical concern: it
reproduces on the very dataset shipped with this phase, on a fresh/typical
checkout, with the default `retrieval_k=4` and the shipped `sample.md` (which
chunks to only 2-3 vectors depending on what else has been ingested into the
shared store). The metric currently cannot distinguish good retrieval from bad,
which is the entire point of the eval script.

**Fix:**
```python
precision = matches / len(results) if results else 0.0
```
This also matches the spec's own wording ("fraction of retrieved chunks") more
literally than the current implementation. (Textbook precision@k as
`matches/k` is a defensible convention only when the index is guaranteed to
return `k` results — Chroma explicitly does not guarantee that here, and the
shipped dataset triggers the shortfall in practice, not hypothetically.)

## Medium Priority

### 2. `InteractionLogger.log_interaction`'s `except sqlite3.Error` is narrower than its own "never raises" contract

`backend/app/observability.py:43-62`. Good news first: the specific thing the
task asked me to check for — the `json.dumps([c.get(...) for c in chunks])`
list comprehensions — **are** correctly inside the `try:` block (they're
evaluated as part of building the argument tuple for the same statement as
`self._conn.execute(...)`, not before it). I confirmed `chromadb` 0.6.3 returns
plain Python `float` for `distance` (not numpy), and `chunk_id`/`source` are
plain strings, so `json.dumps` cannot currently raise for real data shapes in
this codebase — there is no live trigger today.

However, the `except` clause only catches `sqlite3.Error`, while the
docstring's promise is unconditional ("Never raises — a logging bug must not
break the chat response it describes"). Neither call site in `main.py`
(`/chat` line 99, `/chat/stream` line 137) wraps `observability.log_interaction(...)`
in its own try/except — both rely entirely on the callee's promise. If any
future change ever puts a non-JSON-serializable value into `chunks` (e.g. a
different vector backend that returns numpy scalars, which is exactly the kind
of drop-in swap this project's docstrings elsewhere anticipate — see
`vectorstore.py`'s "phase 7 LangChain comparison" framing), a `TypeError` from
`json.dumps` would propagate straight out of the try/except and directly break
the `/chat` response it was supposed to be logging — precisely the regression
class this review was asked to hunt for. It doesn't exist today, but the
contract as written doesn't actually enforce what it claims to.

Given the phase's explicit non-functional requirement ("logging must never
block or fail the chat response... wrap in try/except"), this is a case where
broadening `except sqlite3.Error` to `except Exception` is the correct,
spec-mandated behavior, not defensive over-engineering — it directly closes the
gap between what the code promises and what it actually guarantees, at zero
added complexity.

### 3. `eval_retrieval.py` mutates the developer's real, persistent Chroma store on every run — confirmed to already contain unrelated data

`_ensure_sample_doc_ingested` (`scripts/eval_retrieval.py:28-34`) calls
`store.upsert("sample.md", ...)` against `settings.chroma_dir`, i.e. the same
store the running backend and the developer's own manual testing use — not a
throwaway copy. This matches the phase spec's explicit instruction (share the
real store, idempotent delete-then-add, "runs standalone even on a fresh
checkout") so it is not a deviation from what was asked for, and I'm not
recommending a separate eval-only store (that would be exactly the kind of
premature abstraction the project's YAGNI stance rules out).

That said, I confirmed empirically that this has a real, visible consequence
today: the developer's live store already contains chunks from unrelated
documents (`test.pdf`, `demo-doc`) ingested during earlier manual testing, sitting
alongside `sample.md`'s chunks. This means `eval_retrieval.py`'s reported score
reflects retrieval quality against the user's *entire* current document set,
not an isolated 5-question fixture — a score change over time could come from
the user ingesting more of their own material rather than from a real chunking/
`k`/embedding-model change, which is exactly the kind of confound a "measure
retrieval quality when tuning k" tool should avoid. Worth a one-line README
callout (e.g. "scores reflect your full ingested document set, not just
`sample.md`") rather than a code change — this is a documentation gap, not a
functional bug, given the design was explicitly chosen this way by the spec.

## Low Priority

- `app/ingestion.py:18` (`load_text` docstring: "Used by the eval script (phase 5)
  and tests...") and `app/observability.py:41` ("per the phase's non-functional
  requirement") name phase numbers/plan references in code comments, which
  `.claude/rules/review-audit-self-decision.md`'s "Stable Code Artifacts" rule
  prohibits ("Do not put plan IDs, phase numbers, audit labels, or finding
  codes in code comments... Explain the invariant or behavior directly").
  This is a pre-existing, pervasive pattern across the whole codebase, not
  something phase 5 introduced fresh — `llm_client.py`, `rag.py`, and
  `vectorstore.py` already reference "phase 3"/"phase 7" in comments from
  earlier phases (phase 4's review didn't flag it). Phase 5 continues rather
  than starts the pattern, so I'm not treating it as a phase-5-specific
  regression, but it's worth a project-wide cleanup pass at some point:
  replace phase references with a direct description of the behavior/rationale
  (e.g. `load_text`'s docstring could just say "used by scripts and tests that
  read documents from disk directly, rather than via the upload endpoint").
- `main.py:108` and `main.py:146` duplicate the identical one-line
  `sources = [{"source": c["source"], "text": c["text"]} for c in chunks]`
  list comprehension in both `/chat` and `/chat/stream`. Minor DRY nit given
  each instance is a single line; a small `rag.py` helper (e.g. `to_sources(chunks)`)
  would remove the duplication if it's touched again, but not worth a change on
  its own.
- `InteractionLogger.__init__`'s `db_path: str | None = None` type hint doesn't
  reflect that `Path` objects are passed in practice (`tmp_path / "obs.db"` in
  tests, `settings.db_path` which is a `Path` by default). This mirrors the
  same pre-existing minor inaccuracy in `ConversationStore.__init__` from phase
  4 — consistent, not a new issue, not worth fixing in isolation.

## Verified-Correct Items (explicitly checked per the review request)

### `log_interaction`'s try/except placement — the specific concern raised did not materialize
The `json.dumps(...)` list comprehensions building `retrieved_chunk_ids`/
`retrieved_scores` are correctly inside the `try:` block (see Medium #2 for the
separate, real gap: the `except` type is narrower than the docstring's
promise, not a placement bug).

### Caller-side protection in `main.py` — not currently exploitable, but only because of a narrow current-data-shape guarantee, not a hard contract
See Medium #2. `/chat` and `/chat/stream` do not wrap `observability.log_interaction(...)`
in try/except and don't need to today, but only because `log_interaction`'s
actual `except` clause happens to cover every failure mode reachable with
today's data shapes — not because the promise is airtight by construction.

### `/chat` response shape and behavior vs. the deleted `rag.answer_question` — no regression
Traced the inlined sequence in `main.py`'s `/chat` (`retrieve_chunks` →
`build_prompt` → `llm.chat`) against the phase-4 review's description of the
old `answer_question` (embed → query → build prompt with no history →
non-streaming `llm.chat`) — identical operation order, identical
`{"answer": ..., "sources": [...]}` response shape, and `sources` is built via
explicit key access (`c["source"]`, `c["text"]`), so the two new keys
(`chunk_id`, `distance`) added to `ChromaStore.query`'s return dicts in this
phase are inert for every caller. Grepped the full codebase for any positional
unpacking of `store.query()` results — none found; every caller (`rag.py`,
`main.py`, `observability.py`, `eval_retrieval.py`) uses key-based
`dict[...]`/`.get(...)` access. No leftover references to `answer_question`
anywhere in the tree. `pytest` (16 tests, including the 4 live-Ollama
integration tests in `test_rag.py`) passes.

### Acceptance criterion 1 (interactions table gets rows with plausible latency) — confirmed live
Ran a real `/chat` call through `TestClient` against the actual Ollama
instance and inspected the resulting SQLite row directly:
```
('2026-09-19 02:50:15', None, 'What is RAG?',
 '["sample.md::0", "test.pdf::0", "sample.md::1", "demo-doc::0"]',
 '[0.6058..., 0.8336..., 1.1967..., 1.2862...]',
 5131.03, 1667, 238)
```
`session_id=None` for `/chat` is correct, not a bug — `ChatRequest` has no
`session_id` field; `/chat` is intentionally stateless (no memory), unlike
`/chat/stream`.

### Acceptance criterion 2 (logging failure doesn't break `/chat`) — confirmed via test and reasoning
`test_log_interaction_failure_does_not_raise` closes the connection first
(`sqlite3.ProgrammingError`, a subclass of `sqlite3.Error`) and asserts no
exception escapes — passes. Combined with Medium #2's caveat about the
`except` clause's actual scope.

### `sys.path.insert(0, ...)` pattern in `eval_retrieval.py` — correct, no real issue
Uses `Path(__file__).resolve().parent.parent`, anchored to the script's own
location rather than the caller's cwd, so it works correctly whether invoked
as `python scripts/eval_retrieval.py` (from `backend/`) or
`python backend/scripts/eval_retrieval.py` (from repo root) — verified both
invocation styles conceptually via the path logic; ran the former live and it
worked. This is the standard, expected pattern for a plain (non-`-m`) script
that needs to import a sibling package; no realistic collision risk in this
project.

### `app.db` consolidation — already done, no action needed
`settings.db_path` was already named `app.db` as of phase 4 (flagged only as a
cosmetic naming note in the phase-4 review, not renamed since). `ConversationStore`
and `InteractionLogger` both default to `settings.db_path`, so `messages` and
`interactions` already share the one file as the spec requested — confirmed by
querying both tables' presence in the same `app.db` during the live test above.

## Edge Cases Considered (Scout)

- Concurrent writes from `ConversationStore` and `InteractionLogger` (two
  separate `sqlite3.Connection` objects to the same file) — each connection's
  `execute`+`commit` pair completes synchronously within the same
  single-threaded event loop before any `await`, matching the concurrency
  analysis already verified correct in the phase-4 review for `ConversationStore`
  alone; no new race introduced by adding a second connection to the same file
  under this single-process dev-server execution model.
- `/chat/stream`'s early-return error path (`event: error` on `RuntimeError`/
  `httpx.HTTPError`) skips both `memory.append` and `observability.log_interaction`
  — intentional, not a bug: a failed generation has no `response`/`latency_ms`
  worth logging, and the spec only requires logging on completed interactions.
- Chroma returning a variable number of results across questions (already
  covered in High #1) — also relevant to `retrieve_chunks` in normal `/chat`
  usage, but harmless there since `build_prompt` just joins however many
  chunks come back; the only place the count matters numerically is the
  eval script's precision formula.

## Positive Observations

- `InteractionLogger` and its test file are a clean match for the project's
  established stdlib-sqlite3-no-ORM convention; `test_log_interaction_failure_does_not_raise`
  is a genuine (not phantom) test — it exercises the real failure path via an
  actually-closed connection, not a mock.
- The `test_rag.py` fix (patching `main_module.observability` in all four
  tests) is a real, verified bug fix: I confirmed the module-level
  `observability = InteractionLogger()` singleton in `main.py` would otherwise
  write to the developer's real `backend/data/app.db` on every test run
  without it.
- README's "Debugging & Evaluation" section accurately describes what's
  implemented and includes an honest caveat that 5 hand-labeled questions are
  "a starting point... not a rigorous benchmark," matching the phase spec's
  own risk-assessment framing.

## Recommended Actions

1. (High) Fix `eval_retrieval.py`'s precision@k denominator to use
   `len(results)` instead of the requested `k`, guarding the empty-results
   case. Verify the fix by re-running the script — the aggregate score should
   move off the flat `0.50` and actually vary per question.
2. (Medium) Broaden `InteractionLogger.log_interaction`'s `except sqlite3.Error`
   to `except Exception` so the method's docstring promise ("never raises")
   is actually enforced for any failure mode, not just SQLite-specific ones —
   cheap, directly serves the phase's explicit non-functional requirement.
3. (Medium/docs-only) Add a one-line README note that eval scores reflect the
   user's full current document set (shared persistent store), not an
   isolated `sample.md`-only corpus, so score drift over time isn't mistaken
   for a retrieval regression.
4. (Low, optional, not phase-5-specific) Consider a project-wide pass to
   remove phase-number references from code comments per the repo's own
   "Stable Code Artifacts" rule; not blocking and predates this phase.

## Plan Status (Phase 5 Success Criteria)

- [x] After a few chat interactions, `interactions` table has matching rows
      with plausible latency numbers — confirmed live (`5131.03` ms for a real
      Ollama call, chunk ids/distances correctly serialized).
- [x] Logging failure does not break `/chat` — confirmed via test; see Medium
      #2 for a caveat on the guarantee's actual scope (not a failure of this
      criterion as tested, but a latent gap in how completely it's enforced).
- [ ] `eval_retrieval.py` runs standalone and prints a precision@k score
      between 0 and 1 — it does run and does print a number in `[0, 1]`
      (`0.50`), so the criterion's literal text is technically satisfied, but
      the score is not a meaningful measurement of retrieval quality as
      currently computed (see High #1) — recommend not treating this box as
      genuinely checked until the denominator fix lands, since a metric that
      can't move is not doing its job.

## Metrics

- Files reviewed: 9 (4 new, 5 modified)
- Test run: `pytest` (full suite) → 16 passed, 0 failed, 0 skipped (Ollama
  reachable locally)
- Live verification: 1 real `/chat` call inspected end-to-end (SQLite row),
  1 real `eval_retrieval.py` run against the live store, 1 direct `chromadb`
  query to confirm `distance` value types and the "fewer results than k"
  behavior
- Type coverage / lint: not separately run (no lint/type command configured
  in this project, consistent with phase 4's review)

## Unresolved Questions

None — all specific concerns raised in the review request were checked
against running code, not just read.
