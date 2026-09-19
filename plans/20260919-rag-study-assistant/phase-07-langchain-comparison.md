---
phase: 7
title: "LangChain reimplementation for comparison"
status: done
priority: P2
dependencies: [3]
---

# Phase 7: LangChain reimplementation for comparison

## Overview
Reimplement the same ingest + RAG-chat behavior from phases 2-3 using
LangChain, as a second, independent backend running alongside the raw one.
Goal is not to replace the hand-rolled implementation — it's to let the user
directly compare "what I wrote by hand" vs "what a framework gives me for
free", which is the concrete learning payoff of doing both.

## Requirements
- Functional: a second FastAPI app (`backend_langchain/`) exposing the same
  contract as the raw backend — `GET /health`, `POST /documents/ingest`,
  `POST /chat` — built with LangChain primitives instead of hand-rolled code.
- Functional: `scripts/compare_answers.py` sends the same question to both
  backends (raw on :8000, LangChain on :8001) and prints answer + latency
  side by side.
- Functional: a short written comparison (`docs/langchain-comparison.md`)
  mapping each hand-rolled piece to its LangChain equivalent and noting real
  differences observed (code volume, latency, answer quality, ease of
  swapping components).
- Non-functional: does not modify anything in `backend/` or `frontend/` —
  fully additive, isolated dependency set (own `requirements.txt`), so the
  raw implementation stays untouched as the "ground truth" reference.

## Architecture
```
backend_langchain/
  app/
    main.py       # FastAPI app on port 8001, same 3 routes as backend/app/main.py
    chain.py        # LangChain pieces: loader, splitter, embeddings, vectorstore, LCEL chain
  data/
    chroma/.gitkeep  # separate persistent dir/collection — do not share the raw backend's collection
  requirements.txt
docs/
  langchain-comparison.md
scripts/
  compare_answers.py   # lives at project root, hits both backends over HTTP
```
Mapping (hand-rolled -> LangChain):
- `ingestion.load_text` + `chunk_text` -> `TextLoader`/`PyPDFLoader` +
  `RecursiveCharacterTextSplitter`
- `llm_client.OllamaClient.embed` -> `langchain_ollama.OllamaEmbeddings`
- `vectorstore.ChromaStore` -> `langchain_chroma.Chroma` (as a `VectorStore`,
  used via `.as_retriever(search_kwargs={"k": ...})`)
- `rag.build_prompt` + `llm_client.chat` -> LCEL chain:
  `retriever | format_docs | prompt | ChatOllama | StrOutputParser()`
- (No memory/streaming parity required here — phase 4's raw streaming/memory
  is out of scope for this comparison; keep phase 7 scoped to the phase-3
  baseline to avoid scope creep.)

Use the same models (`llama3.2:3b`, `nomic-embed-text`) so the comparison
isolates "framework overhead" rather than "different model" as a variable.

## Related Code Files
- Create: `projects/rag-study-assistant/backend_langchain/app/__init__.py`
- Create: `projects/rag-study-assistant/backend_langchain/app/main.py`
- Create: `projects/rag-study-assistant/backend_langchain/app/chain.py`
- Create: `projects/rag-study-assistant/backend_langchain/requirements.txt`
- Create: `projects/rag-study-assistant/backend_langchain/data/chroma/.gitkeep`
- Create: `projects/rag-study-assistant/scripts/compare_answers.py`
- Create: `projects/rag-study-assistant/docs/langchain-comparison.md`
- Modify: `projects/rag-study-assistant/.gitignore` (add `backend_langchain/.venv/`, `backend_langchain/data/chroma/*`)
- Modify: `projects/rag-study-assistant/README.md` (add "LangChain comparison" section: setup + how to run both + how to run compare script)

## Implementation Steps
1. `backend_langchain/requirements.txt`: `fastapi`, `uvicorn[standard]`,
   `langchain`, `langchain-community`, `langchain-ollama`, `langchain-chroma`,
   `langchain-text-splitters`, `pypdf` (pin current versions as of Sep 2026 —
   LangChain's package split is fast-moving, verify exact import paths
   against installed version during implementation, not from memory).
2. `chain.py`: build the ingestion pipeline (loader -> splitter -> embeddings
   -> Chroma `.add_documents`) and the LCEL query chain, both as plain
   functions (`ingest_source(path_or_text, source_id)`, `answer(question)`) —
   no unnecessary class wrapping, matches project's own KISS preference even
   inside the "framework" demo.
3. `main.py`: same 3 routes as the raw backend, calling into `chain.py`,
   running on port 8001 (`uvicorn app.main:app --app-dir backend_langchain --port 8001`).
4. `scripts/compare_answers.py`: given a question (CLI arg or a small fixed
   list), POST to both `:8000/chat` and `:8001/chat`, print answer text +
   wall-clock latency for each, no assertions — it's a manual comparison
   tool, not a test.
5. Ingest the same sample doc (`backend/data/documents/sample.md`) into both
   backends, run 3-5 questions through `compare_answers.py`, capture
   observations.
6. Write `docs/langchain-comparison.md`: the mapping table above, plus a
   short "what LangChain bought us" and "what it cost us" section grounded
   in the actual observed run (line count diff, latency diff, any answer
   quality difference, and one concrete thing that was harder to customize
   inside the LCEL chain vs the raw prompt string).

## Success Criteria
- [x] `backend_langchain` boots on :8001 independently of `backend` on :8000
      — verified live, both running simultaneously.
- [x] Same sample doc ingested into both produces comparable retrieval —
      `sample.md` ingested into both, both correctly answer/cite it.
- [x] `compare_answers.py` runs and prints both answers for a real question
      — ran against 3 real questions, output captured in the comparison doc.
- [x] `docs/langchain-comparison.md` contains real numbers/observations from
      an actual run, not generic/hypothetical claims — includes real
      latencies, real answers, and a real `langchain-community` deprecation
      warning observed during install.
- [x] `backend/` and `frontend/` are untouched by this phase — only
      `backend_langchain/`, `scripts/compare_answers.py`,
      `docs/langchain-comparison.md`, and `README.md` were touched.

## Risk Assessment
- LangChain's package layout changes frequently between versions — pin
  versions in `requirements.txt` and verify import paths at implementation
  time rather than trusting any specific import shown here.
- Running both backends plus two Ollama-backed pipelines on 8GB RAM may be
  slow/tight — fine for sequential comparison runs, not meant to run
  side-by-side under load; note this in the comparison doc.
- Scope discipline: this phase should NOT grow into "port everything to
  LangChain" — cap it at ingest + non-streaming chat, matching phase 3 only.
