"""Embedding model and Chroma vector store shared by ingestion and retrieval."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from src.config import Settings

MANIFEST = "manifest.json"


def get_embeddings(settings: Settings) -> Embeddings:
    # NVIDIAEmbeddings reads NVIDIA_API_KEY from the environment and sends
    # input_type="query" for queries and "passage" for documents automatically.
    return NVIDIAEmbeddings(model=settings.embedding_model, truncate="END")


def open_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def write_manifest(settings: Settings, chunk_count: int) -> None:
    manifest = {
        "embedding_model": settings.embedding_model,
        "collection": settings.chroma_collection,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks": chunk_count,
    }
    (settings.chroma_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(chroma_dir: Path) -> dict:
    path = chroma_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"No vector index at {chroma_dir}. Build it first: python -m src.rag.ingest"
        )
    return json.loads(path.read_text(encoding="utf-8"))
