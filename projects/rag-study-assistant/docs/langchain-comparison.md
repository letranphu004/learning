# Hand-rolled RAG vs. LangChain: what changed

Same models (`llama3.2:3b` chat, `nomic-embed-text` embeddings), same chunk
size (800/100 overlap), same k (4), same local Ollama instance — the only
variable is whether the ingest+retrieve+generate pipeline is hand-rolled
(`backend/`, port 8000) or built with LangChain (`backend_langchain/`, port
8001). Scoped to the phase-3 baseline only: ingest + non-streaming chat, no
streaming/memory/observability on either side of this comparison.

## Mapping (hand-rolled → LangChain)

| Hand-rolled | LangChain equivalent |
|---|---|
| `ingestion.load_text`/`extract_upload_text` + `chunk_text` (58 lines) | `PyPDFLoader`/`Document(page_content=...)` + `RecursiveCharacterTextSplitter` |
| `vectorstore.ChromaStore` (47 lines: manual `chromadb` collection wrapper) | `langchain_chroma.Chroma` used directly — same `add_documents`/`delete(where=...)`/`get()` primitives, just not hand-wrapped |
| `llm_client.OllamaClient.embed`/`.chat` | `langchain_ollama.OllamaEmbeddings`/`ChatOllama` |
| `rag.build_prompt` + manual message list | `ChatPromptTemplate.from_messages([...])` |
| — | `StrOutputParser()` to unwrap the chat model's `AIMessage` into plain text |

## What LangChain bought us

- **Chunking and PDF loading are one import away.** `RecursiveCharacterTextSplitter`
  and `PyPDFLoader` replace `ingestion.py`'s hand-rolled sliding-window
  chunker and `pypdf.PdfReader` loop entirely — genuinely less code to
  reason about for that specific piece.
- **The vector store wrapper disappears.** `ChromaStore`'s 47 lines existed
  only to translate between our own `{"text", "source"}` dict shape and
  raw `chromadb` collection calls. `langchain_chroma.Chroma` exposes the
  same underlying `chromadb` collection with `Document` objects as the
  currency instead — no hand-rolled translation layer needed.
- **Idempotent re-ingest is identical either way.** Both use the exact same
  `collection.delete(where={"source": ...})` then `add(...)` pattern —
  LangChain doesn't add or remove anything here, it's the same Chroma API
  underneath.

## What it cost us

- **Total line count didn't drop as much as expected.** `chain.py` + `main.py`
  together are 162 lines vs. 206 for the raw backend's equivalent modules
  (`ingestion.py` + `vectorstore.py` + `llm_client.py` + `rag.py`) — and the
  raw number includes phase 4/5 code (streaming, logging) that has no
  LangChain counterpart here, so the real gap is smaller than 206→162
  suggests, maybe even a wash. The FastAPI route boilerplate, prompt
  construction, and response shaping are effectively the same amount of
  code either way — LangChain saves you the *retrieval internals*, not the
  *application glue*.
- **Getting both an answer and its source chunks out of one LCEL chain is
  awkward.** The natural single-pipe chain (`retriever | format | prompt |
  llm | parser`) throws away the retrieved `Document` objects before they
  reach the response — exposing them for the `sources` field needs
  `RunnableParallel`/`RunnablePassthrough.assign` plumbing. We ended up
  just calling `retriever.invoke()` directly and skipping the single-chain
  composition for this reason. In the hand-rolled version this was free —
  `chunks` was already a local variable in scope.
- **`PyPDFLoader` only reads from a filesystem path, not bytes.** A PDF
  arriving as a FastAPI `UploadFile` has to be written to a temp file first
  (`app/chain.py`'s `ingest_source`) just to hand a path to the loader.
  `pypdf.PdfReader(io.BytesIO(...))` in the hand-rolled version reads bytes
  directly — one fewer moving part.
- **Dependency churn, observed live, not hypothetical.** Installing
  `langchain-community` (needed for `PyPDFLoader`) prints: *"langchain-community
  is being sunset and is no longer actively maintained... migration guidance
  toward standalone integration packages."* Confirms the phase's own risk
  note — pin versions, verify import paths at implementation time, expect
  this package layout to keep moving.

## Real numbers from an actual run

`python scripts/compare_answers.py`, both backends warm, same `sample.md`
ingested into both:

| Question | raw latency | raw answer (truncated) | langchain latency | langchain answer (truncated) |
|---|---|---|---|---|
| What is RAG? | 7158 ms | "...combines a retrieval step... with a generation step... never trained on." | 3867 ms | "...combines a retrieval step... with a generation step..." (shorter, dropped the last clause) |
| Why chunk documents? | 6839 ms | "...limited context windows... find the most relevant slice..." | 3758 ms | "...limited context windows... find the most relevant slice..." (near-identical content) |
| What is a vector store? | 4150 ms | "...indexes embeddings... find the most similar ones already stored." | 3186 ms | "...quickly find the most similar vectors... for NLP tasks such as text similarity search." (slightly more generic, less grounded in the doc's own wording) |

LangChain was faster in every question in this run (3.2–3.9s vs. 4.1–7.2s),
but this is **not** a reliable "framework overhead" signal — both hit the
same real Ollama process, and `llama3.2:3b`'s generation time varies with
output length more than with which HTTP client called it. A rigorous
latency comparison would need many repeated runs averaged, not three
one-shot questions; noting the honest limitation here rather than
overclaiming a performance difference the data doesn't support.

The bigger qualitative difference: the raw backend's answers stuck closer
to the source document's exact phrasing (same system prompt, so this is
likely sampling variance between two separate `ChatOllama` connections
rather than a framework effect) — worth re-running with a fixed seed if
this project ever needs a more rigorous eval here.

## Bottom line

For a small, single-purpose RAG app like this one, LangChain mainly buys
back the *ingestion primitives* (loader + splitter) and a slightly higher-level
vector-store interface — not a dramatic reduction in total code, and not a
free lunch on the LCEL composition side once you need something (like
exposing sources) that isn't the framework's default happy path.
