#!/usr/bin/env python3
"""Standalone retrieval-quality check - not a pytest test, a tool you
re-run by hand while tuning chunk size, k, or the embedding model.

Usage: python scripts/eval_retrieval.py (from backend/), or
       python backend/scripts/eval_retrieval.py (from repo root).
"""

import asyncio
import json
import sys
from pathlib import Path

# Run as a plain script (not `python -m`), so put the backend/ directory
# (the parent of this file's `scripts/` folder) on sys.path to import the
# app package regardless of the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ingestion import chunk_text, load_text  # noqa: E402
from app.llm_client import OllamaClient  # noqa: E402
from app.vectorstore import ChromaStore  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
SAMPLE_DOC = settings.documents_dir / "sample.md"


async def _ensure_sample_doc_ingested(store: ChromaStore, llm: OllamaClient) -> None:
    """Ingest the shipped sample doc so the eval runs standalone even on a
    fresh checkout - upsert is idempotent (delete-then-add per source)."""
    text = load_text(SAMPLE_DOC)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = await llm.embed(chunks)
    store.upsert("sample.md", chunks, embeddings)


async def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    store = ChromaStore()
    llm = OllamaClient()
    k = settings.retrieval_k

    await _ensure_sample_doc_ingested(store, llm)

    precisions = []
    for item in dataset:
        question, expected_source = item["question"], item["expected_source"]
        [embedding] = await llm.embed([question])
        results = store.query(embedding, k=k)
        matches = sum(1 for r in results if r["source"] == expected_source)
        # Divide by the actual number of chunks retrieved, not the requested
        # k - Chroma silently caps n_results when the collection has fewer
        # vectors than k, and the spec's "fraction of retrieved chunks"
        # means the denominator is what came back, not what was asked for.
        precision = matches / len(results) if results else 0.0
        precisions.append(precision)

        status = "PASS" if matches > 0 else "FAIL"
        note = f" (only {len(results)}/{k} chunks in index)" if len(results) < k else ""
        print(f"[{status}] precision@{k}={precision:.2f}{note}  {question}")

    aggregate = sum(precisions) / len(precisions) if precisions else 0.0
    print(f"\nAggregate precision@{k}: {aggregate:.2f} over {len(dataset)} questions")


if __name__ == "__main__":
    asyncio.run(main())
