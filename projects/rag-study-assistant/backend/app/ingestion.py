"""Turn a raw document into plain text, then into overlapping chunks.

Chunking here is character-based, not token-based. That's a deliberate
simplification for a learning project: token-accurate chunking needs a
tokenizer matching the embedding model, which is one more moving part to
learn/debug. Character count is a close-enough proxy for "how much context
fits" and keeps this module dependency-free.
"""

import io
from pathlib import Path

from pypdf import PdfReader


def load_text(source: str | Path) -> str:
    """Load text from a filesystem path (.txt/.md/.pdf). Used by the eval
    script (phase 5) and tests, which read documents already on disk."""
    path = Path(source)
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path.read_bytes())
    return path.read_text(encoding="utf-8")


def extract_upload_text(filename: str, raw_bytes: bytes) -> str:
    """Load text from an HTTP-uploaded file's raw bytes, dispatching on
    extension. Kept separate from `load_text` so the ingest endpoint never
    needs to write the upload to a temp file just to read it back."""
    if filename.lower().endswith(".pdf"):
        return _extract_pdf(raw_bytes)
    return raw_bytes.decode("utf-8")


def _extract_pdf(raw_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Fixed-size sliding-window chunking over characters.

    overlap must be < chunk_size or the window never advances.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
