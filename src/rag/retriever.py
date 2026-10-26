"""Top-k retrieval over the Chroma index, plus the context block for prompts."""

from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.rag.store import get_embeddings, open_store, read_manifest


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    citation: str
    heading: str
    text: str
    score: float  # cosine similarity, higher is closer


@lru_cache
def default_store() -> Chroma:
    settings = get_settings()
    return load_store(settings, get_embeddings(settings))


def load_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    manifest = read_manifest(settings.chroma_dir)
    if manifest["embedding_model"] != settings.embedding_model:
        # Querying with a different model than the index was built with returns
        # garbage (or a dimension error), so fail loudly instead.
        raise RuntimeError(
            f"Index was built with {manifest['embedding_model']} but NVIDIA_EMBEDDING_MODEL is "
            f"{settings.embedding_model}. Rebuild with: python -m src.rag.ingest --rebuild"
        )
    return open_store(settings, embeddings)


def retrieve(query: str, k: int | None = None, store: Chroma | None = None) -> list[RetrievedChunk]:
    store = store or default_store()
    k = k or get_settings().retrieval_k
    results = store.similarity_search_with_score(query, k=k)
    return [
        RetrievedChunk(
            chunk_id=doc.metadata["chunk_id"],
            source_id=doc.metadata["source_id"],
            citation=doc.metadata["citation"],
            heading=doc.metadata["heading"],
            text=doc.page_content,
            score=round(1 - distance, 4),
        )
        for doc, distance in results
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered context block; the generation prompt asks the model to cite [n]."""
    return "\n\n".join(
        f"[{n}] {chunk.citation} ({chunk.chunk_id})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )
