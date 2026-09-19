from app.memory import ConversationStore


def test_recent_returns_empty_for_unknown_session(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.db")
    assert store.recent("no-such-session") == []


def test_append_then_recent_round_trip_in_order(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.db")
    store.append("s1", "user", "What is RAG?")
    store.append("s1", "assistant", "Retrieval-augmented generation.")
    store.append("s1", "user", "And chunking?")

    history = store.recent("s1", n=6)
    assert history == [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "Retrieval-augmented generation."},
        {"role": "user", "content": "And chunking?"},
    ]


def test_recent_caps_to_n_most_recent_messages():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(db_path=Path(tmp) / "conv.db")
        for i in range(10):
            store.append("s1", "user", f"message {i}")

        history = store.recent("s1", n=4)
        assert [m["content"] for m in history] == [
            "message 6",
            "message 7",
            "message 8",
            "message 9",
        ]


def test_sessions_are_isolated(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.db")
    store.append("s1", "user", "from session 1")
    store.append("s2", "user", "from session 2")

    assert store.recent("s1") == [{"role": "user", "content": "from session 1"}]
    assert store.recent("s2") == [{"role": "user", "content": "from session 2"}]
