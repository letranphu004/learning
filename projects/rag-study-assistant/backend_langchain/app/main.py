from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import chain

app = FastAPI(title="RAG Study Assistant (LangChain)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/ingest")
def ingest_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    source: str | None = Form(None),
) -> dict:
    """Same request contract as the raw backend's /documents/ingest -
    multipart file upload or a plain text form field."""
    if file is not None:
        raw_bytes = file.file.read()
        source_id = source or file.filename or "untitled"
        if (file.filename or "").lower().endswith(".pdf"):
            chunk_count = chain.ingest_source(source_id, pdf_bytes=raw_bytes)
        else:
            chunk_count = chain.ingest_source(source_id, text=raw_bytes.decode("utf-8"))
    elif text is not None:
        source_id = source or "untitled"
        chunk_count = chain.ingest_source(source_id, text=text)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'text'")

    return {"source": source_id, "chunks": chunk_count}


@app.get("/documents")
def list_documents() -> list[dict]:
    return chain.list_sources()


@app.post("/chat")
def chat_endpoint(req: ChatRequest) -> dict:
    return chain.answer_question(req.message)
