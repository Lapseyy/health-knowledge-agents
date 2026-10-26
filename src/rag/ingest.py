"""Build the Chroma index: load sections, chunk, embed, store.

Resumable: chunk IDs are deterministic, so a rerun after an interruption skips
chunks already stored. Use --rebuild after changing the chunking or the
embedding model.

    python -m src.rag.ingest            # build or resume
    python -m src.rag.ingest --rebuild  # start over
"""

import argparse
import time

from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.rag.chunking import build_chunks, load_sections
from src.rag.store import get_embeddings, open_store, write_manifest

BATCH_SIZE = 128


def ingest(settings: Settings, embeddings: Embeddings, rebuild: bool = False) -> int:
    chunks = build_chunks(load_sections(settings.corpus_path), settings.chunk_size, settings.chunk_overlap)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    store = open_store(settings, embeddings)
    if rebuild:
        store.delete_collection()
        store = open_store(settings, embeddings)

    existing = set(store.get(include=[])["ids"])
    pending = [chunk for chunk in chunks if chunk.id not in existing]
    print(f"{len(chunks)} chunks, {len(existing)} already stored, {len(pending)} to embed")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[doc.id for doc in batch])
        print(f"  {start + len(batch)}/{len(pending)}")

    write_manifest(settings, len(chunks))
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the collection and re-embed everything")
    args = parser.parse_args()
    settings = get_settings()
    started = time.perf_counter()
    count = ingest(settings, get_embeddings(settings), rebuild=args.rebuild)
    print(f"Index ready: {count} chunks in {settings.chroma_dir} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
