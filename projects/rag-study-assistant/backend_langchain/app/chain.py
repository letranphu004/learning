"""LangChain reimplementation of the raw backend's ingest + non-streaming
chat pipeline (phase 3 baseline only - no streaming/memory parity, see the
phase 7 spec's scope note). Plain functions, not classes, matching the
project's own KISS preference even inside this "framework" demo.

Same models and chunk settings as the raw backend so the comparison isolates
framework overhead rather than a different model or chunking strategy.
"""

import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the context "
    "below. If the context does not contain enough information to answer, "
    "say \"I don't know from the given documents\" instead of guessing.\n\n"
    "Context:\n{context}"
)

# Module-level singletons, same rationale as the raw backend: one embedding
# client, one chat client, one persistent Chroma collection for the process
# lifetime.
_embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
_llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)
_store = Chroma(
    collection_name="study_documents",
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)
_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", "{question}")])


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


def ingest_source(source_id: str, *, text: str | None = None, pdf_bytes: bytes | None = None) -> int:
    """Idempotent re-ingest: delete-then-add per source, matching the raw
    backend's ChromaStore.upsert (chunk boundaries can shift on re-ingest).

    PyPDFLoader only reads from a filesystem path, not bytes - a PDF upload
    is written to a temp file for the loader, then discarded.
    """
    if pdf_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            docs = PyPDFLoader(tmp.name).load()
    elif text is not None:
        docs = [Document(page_content=text)]
    else:
        raise ValueError("Provide either text or pdf_bytes")

    for doc in docs:
        doc.metadata["source"] = source_id

    chunks = _splitter.split_documents(docs)
    _store.delete(where={"source": source_id})
    if not chunks:
        return 0
    ids = [f"{source_id}::{i}" for i in range(len(chunks))]
    _store.add_documents(chunks, ids=ids)
    return len(chunks)


def list_sources() -> list[dict]:
    items = _store.get()
    counts: dict[str, int] = {}
    for meta in items["metadatas"] or []:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": source, "chunk_count": count} for source, count in counts.items()]


def answer_question(question: str) -> dict:
    # Retrieval is invoked directly (not folded into the LCEL chain below)
    # so the retrieved chunks are available for the `sources` response field
    # without a second retrieval call - getting both an answer AND its
    # source docs out of one composed chain needs extra plumbing
    # (RunnableParallel/RunnablePassthrough.assign); see
    # docs/langchain-comparison.md for the observed trade-off.
    retriever = _store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(question)

    messages = _prompt.invoke({"context": _format_docs(docs), "question": question})
    answer = (_llm | StrOutputParser()).invoke(messages)

    sources = [{"source": d.metadata["source"], "text": d.page_content} for d in docs]
    return {"answer": answer, "sources": sources}
