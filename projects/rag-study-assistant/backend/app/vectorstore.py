"""Wraps a single persistent Chroma collection. Kept deliberately thin -
functions the query/retrieval and ingestion code actually needs, not a
generic vector-store abstraction (that generality is what phase 7's
LangChain VectorStore interface buys you; here we're studying what it
buys, not pre-building it)."""

import chromadb

from app.config import settings

COLLECTION_NAME = "study_documents"


class ChromaStore:
    def __init__(self, persist_dir: str | None = None):
        self._client = chromadb.PersistentClient(path=str(persist_dir or settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def upsert(self, source: str, chunks: list[str], embeddings: list[list[float]]) -> int:
        """Replace all chunks for `source` with the given ones (idempotent
        re-ingestion: delete-then-add rather than a real upsert, since chunk
        boundaries can shift between re-ingests of an edited document)."""
        self._collection.delete(where={"source": source})
        if not chunks:
            return 0
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self._collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        return len(chunks)

    def query(self, embedding: list[float], k: int) -> list[dict]:
        result = self._collection.query(query_embeddings=[embedding], n_results=k)
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result.get("distances") else [None] * len(documents)
        return [
            {"chunk_id": cid, "text": doc, "source": meta["source"], "distance": dist}
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    def list_sources(self) -> list[dict]:
        all_items = self._collection.get()
        counts: dict[str, int] = {}
        for meta in all_items["metadatas"] or []:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return [{"source": source, "chunk_count": count} for source, count in counts.items()]
