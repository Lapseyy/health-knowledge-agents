"""Settings read from the environment (and a local .env file in development)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    corpus_path: Path
    eval_path: Path
    chroma_dir: Path
    chroma_collection: str
    embedding_model: str
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


@lru_cache
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        corpus_path=_path("CORPUS_PATH", "data/ap2e_sections.jsonl"),
        eval_path=_path("EVAL_PATH", "data/ap2e_eval_candidates.jsonl"),
        chroma_dir=_path("CHROMA_PERSIST_DIR", "chroma_index"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "ap2e"),
        embedding_model=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
