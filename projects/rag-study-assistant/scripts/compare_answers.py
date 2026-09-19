#!/usr/bin/env python3
"""Manual comparison tool, not a test - sends the same question(s) to the
raw backend (:8000) and the LangChain backend (:8001) and prints answer +
wall-clock latency side by side. Requires both backends already running
and at least one document already ingested into both.

Usage: python scripts/compare_answers.py "your question here"
       python scripts/compare_answers.py   (runs a small fixed list)
"""

import sys
import time
import urllib.error
import urllib.request
import json

RAW_URL = "http://127.0.0.1:8000/chat"
LANGCHAIN_URL = "http://127.0.0.1:8001/chat"

DEFAULT_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "Why do we split documents into chunks before embedding them?",
    "What is a vector store used for?",
]


def ask(url: str, question: str) -> tuple[str, float]:
    body = json.dumps({"message": question}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        return f"<request failed: {exc}>", 0.0
    latency_ms = (time.perf_counter() - start) * 1000
    return data["answer"], latency_ms


def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS

    for question in questions:
        print(f"\n=== {question}")

        raw_answer, raw_ms = ask(RAW_URL, question)
        print(f"[raw        {raw_ms:7.0f}ms] {raw_answer}")

        lc_answer, lc_ms = ask(LANGCHAIN_URL, question)
        print(f"[langchain  {lc_ms:7.0f}ms] {lc_answer}")


if __name__ == "__main__":
    main()
