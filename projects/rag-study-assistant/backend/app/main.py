import json
import time

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.ingestion import chunk_text, extract_upload_text
from app.llm_client import OllamaClient
from app.memory import ConversationStore
from app.observability import InteractionLogger
from app.rag import build_prompt, retrieve_chunks
from app.vectorstore import ChromaStore


class ChatRequest(BaseModel):
    message: str


class ChatStreamRequest(BaseModel):
    session_id: str
    message: str


app = FastAPI(title="RAG Study Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons: one Chroma connection and one Ollama client for
# the process lifetime, matching how a real service would avoid reopening
# a persistent DB handle or an HTTP connection pool per request.
store = ChromaStore()
llm = OllamaClient()
memory = ConversationStore()
observability = InteractionLogger()


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ollama_reachable": await _check_ollama()}


@app.post("/documents/ingest")
async def ingest_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    source: str | None = Form(None),
) -> dict:
    """Accepts either a multipart file upload or a plain text form field -
    not raw JSON, so a single endpoint can hold both paths without branching
    on Content-Type (FastAPI can't mix a JSON body and file upload on one
    route; multipart-with-optional-file is the simpler equivalent)."""
    if file is not None:
        raw_bytes = await file.read()
        source_id = source or file.filename or "untitled"
        content = extract_upload_text(file.filename or "", raw_bytes)
    elif text is not None:
        source_id = source or "untitled"
        content = text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'text'")

    chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    embeddings = await llm.embed(chunks)
    chunk_count = store.upsert(source_id, chunks, embeddings)
    return {"source": source_id, "chunks": chunk_count}


@app.get("/documents")
async def list_documents() -> list[dict]:
    return store.list_sources()


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    start = time.perf_counter()
    chunks = await retrieve_chunks(req.message, store, llm)
    messages = build_prompt(req.message, chunks)
    answer = await llm.chat(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    observability.log_interaction(
        session_id=None,
        query=req.message,
        chunks=chunks,
        response=answer,
        latency_ms=latency_ms,
        prompt_chars=sum(len(m["content"]) for m in messages),
    )

    sources = [{"source": c["source"], "text": c["text"]} for c in chunks]
    return {"answer": answer, "sources": sources}


@app.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    """SSE-formatted stream of answer tokens, followed by a final `sources`
    event once generation completes. POST (not a literal EventSource GET)
    because EventSource can't carry a request body - the frontend consumes
    this with fetch() + a ReadableStream reader instead."""
    chunks = await retrieve_chunks(req.message, store, llm)
    history = memory.recent(req.session_id, n=6)
    messages = build_prompt(req.message, chunks, history=history)
    prompt_chars = sum(len(m["content"]) for m in messages)

    async def event_stream():
        start = time.perf_counter()
        full_answer = ""
        try:
            async for token in llm.chat_stream(messages):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except (RuntimeError, httpx.HTTPError) as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return
        latency_ms = (time.perf_counter() - start) * 1000

        memory.append(req.session_id, "user", req.message)
        memory.append(req.session_id, "assistant", full_answer)
        observability.log_interaction(
            session_id=req.session_id,
            query=req.message,
            chunks=chunks,
            response=full_answer,
            latency_ms=latency_ms,
            prompt_chars=prompt_chars,
        )

        sources = [{"source": c["source"], "text": c["text"]} for c in chunks]
        yield f"event: done\ndata: {json.dumps({'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
