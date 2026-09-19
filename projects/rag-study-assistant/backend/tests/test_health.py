from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_regardless_of_ollama():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["ollama_reachable"], bool)
