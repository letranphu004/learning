"""Thin wrapper around Ollama's HTTP API. No SDK - Ollama's API is small
enough that a raw httpx client is simpler to read and debug than adding a
dependency for it (matches the project's "no framework I don't understand"
learning goal, and mirrors what the LangChain comparison in phase 7 will
otherwise hide behind `OllamaEmbeddings`/`ChatOllama`)."""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings


class OllamaClient:
    def __init__(self, base_url: str = settings.ollama_base_url):
        self.base_url = base_url

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single call via /api/embed.

        Ollama's /api/embed accepts a list under "input" and returns
        "embeddings" in the same order - no client-side batching needed.
        """
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": settings.embed_model, "input": texts},
            )
            resp.raise_for_status()
        return resp.json()["embeddings"]

    async def chat(self, messages: list[dict]) -> str:
        """Non-streaming chat completion. Used starting phase 3."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": settings.chat_model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming chat completion. Ollama's streaming response is
        newline-delimited JSON (one object per line), not SSE - we
        translate each line's content delta into plain text chunks and let
        the caller (the /chat/stream route) re-wrap them as SSE."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": settings.chat_model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    # Ollama can return HTTP 200 and stream an error object
                    # mid-response (e.g. an invalid chat_model in .env) -
                    # without this check it silently becomes an empty
                    # answer instead of a visible failure.
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
