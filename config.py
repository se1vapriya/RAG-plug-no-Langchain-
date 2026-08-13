"""Single source of truth for every knob in the pipeline.

Everything is read from environment variables (loaded from .env) with sane
defaults, so a new RAG project usually needs nothing but a fresh .env.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = _env(key)
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else default


@dataclass
class Config:
    # --- project identity -------------------------------------------------
    project_name: str = _env("PROJECT_NAME", "rag-project")

    # --- data -------------------------------------------------------------
    # Folder scanned by `python ingest.py`. Relative paths resolve against
    # this file's directory.
    data_dir: Path = field(
        default_factory=lambda: (BASE_DIR / _env("DATA_DIR", "data")).resolve()
    )
    supported_extensions: List[str] = field(
        default_factory=lambda: _env_list("SUPPORTED_EXTENSIONS", [".pdf", ".txt", ".md"])
    )

    # --- chunking ---------------------------------------------------------
    chunk_size: int = _env_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 150)

    # --- embeddings -------------------------------------------------------
    embedding_model: str = _env("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimensions: int = _env_int("EMBEDDING_DIMENSIONS", 1536)
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)

    # --- LLM --------------------------------------------------------------
    # Tried in order; a transient 5xx/429 falls through to the next one.
    llm_models: List[str] = field(
        default_factory=lambda: _env_list(
            "LLM_MODELS",
            ["gemini-2.5-flash", "gemini-2.0-flash"],
        )
    )
    temperature: float = _env_float("TEMPERATURE", 0.4)
    max_retries_per_model: int = _env_int("MAX_RETRIES_PER_MODEL", 2)

    # --- provider endpoints ----------------------------------------------
    google_api_key: str = _env("GOOGLE_API_KEY")
    api_base_url: str = _env(
        "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # --- vector store -----------------------------------------------------
    pinecone_api_key: str = _env("PINECONE_API_KEY")
    pinecone_index_name: str = _env("PINECONE_INDEX_NAME")
    pinecone_index_host: str = _env("PINECONE_INDEX_HOST")
    namespace: str = _env("PINECONE_NAMESPACE", "")
    upsert_batch_size: int = _env_int("UPSERT_BATCH_SIZE", 100)

    # --- retrieval --------------------------------------------------------
    top_k: int = _env_int("TOP_K", 4)
    min_score: float = _env_float("MIN_SCORE", 0.0)

    # --- prompt -----------------------------------------------------------
    system_prompt: str = _env(
        "SYSTEM_PROMPT",
        "You are a helpful assistant that answers questions using only the "
        "provided context. Do not make assumptions beyond the context. If the "
        "context is insufficient, say so plainly and do not invent an answer. "
        "Cite the source filename when you use a passage.",
    )

    def validate(self) -> None:
        """Fail fast with a readable message instead of a provider 401."""
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not (self.pinecone_index_name or self.pinecone_index_host):
            missing.append("PINECONE_INDEX_NAME or PINECONE_INDEX_HOST")
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + f"\nCopy .env.example to .env in {BASE_DIR} and fill them in."
            )


config = Config()
