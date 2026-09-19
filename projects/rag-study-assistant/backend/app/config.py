"""Central settings for the backend. Single source of truth for anything
that changes between local dev, Docker, and the LangChain comparison stack."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # 127.0.0.1, not "localhost": on this machine "localhost" resolves to
    # IPv6 first and Ollama only binds IPv4, adding a ~2s stall per call
    # while the IPv6 attempt times out before falling back.
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"

    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    documents_dir: Path = BACKEND_DIR / "data" / "documents"
    db_path: Path = BACKEND_DIR / "data" / "app.db"

    retrieval_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Frontend is served separately (plain static server) - list every origin
    # it might run on so CORS doesn't silently block requests.
    cors_origins: list[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
