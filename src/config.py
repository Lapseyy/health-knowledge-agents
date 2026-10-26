"""Settings read from the environment (and a local .env file in development)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # RAG
    corpus_path: Path
    eval_path: Path
    chroma_dir: Path
    chroma_collection: str
    embedding_model: str
    embedding_cache_dir: Path
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    # Generation providers
    nvidia_model: str
    nvidia_fast_model: str
    gemini_model: str
    gemini_reasoning_effort: str
    gemini_use_vertex: bool
    google_cloud_project: str | None
    google_cloud_location: str
    # Tools
    tavily_api_key: str | None
    patient_db_path: Path
    # Startup
    warm_up: bool


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


@lru_cache
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        corpus_path=_path("CORPUS_PATH", "data/ap2e_sections.jsonl"),
        eval_path=_path("EVAL_PATH", "data/ap2e_eval_candidates.jsonl"),
        chroma_dir=_path("CHROMA_PERSIST_DIR", "chroma_index"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "ap2e"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_cache_dir=_path("EMBEDDING_CACHE_DIR", "models"),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        nvidia_model=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        nvidia_fast_model=os.getenv("NVIDIA_FAST_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        gemini_reasoning_effort=os.getenv("GEMINI_REASONING_EFFORT", "low"),
        gemini_use_vertex=_flag("GOOGLE_GENAI_USE_VERTEXAI", "true"),
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        patient_db_path=_path("PATIENT_DB_PATH", "data/patients.db"),
        warm_up=_flag("WARM_UP", "true"),
    )
