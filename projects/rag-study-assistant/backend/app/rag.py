"""Retrieval + prompt-building shared by the /chat and /chat/stream routes.

Kept separate from main.py so the RAG logic (prompt shape, grounding rule)
is testable without going through HTTP, and separate from vectorstore/
llm_client so those stay swappable (relevant for phase 7's LangChain
comparison, which reimplements this module against the same interfaces).
"""

from app.config import settings
from app.llm_client import OllamaClient
from app.vectorstore import ChromaStore

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the context "
    "below. If the context does not contain enough information to answer, "
    "say \"I don't know from the given documents\" instead of guessing.\n\n"
    "Context:\n{context}"
)


def build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *(history or []),
        {"role": "user", "content": question},
    ]


async def retrieve_chunks(question: str, store: ChromaStore, llm: OllamaClient) -> list[dict]:
    [embedding] = await llm.embed([question])
    return store.query(embedding, k=settings.retrieval_k)
