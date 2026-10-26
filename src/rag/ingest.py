"""Build the Chroma index: load sections, chunk, embed, store.

Resumable: chunk IDs are deterministic, so a rerun after a rate-limit failure
skips chunks already stored. Use --rebuild after changing the chunking or the
embedding model.

    python -m src.rag.ingest            # build or resume
    python -m src.rag.ingest --rebuild  # start over
"""

import argparse
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.rag.chunking import build_chunks, load_sections
from src.rag.store import get_embeddings, open_store, write_manifest

BATCH_SIZE = 50  # one embeddings request per batch
MAX_RETRIES = 5


def add_with_retry(store, batch: list[Document]) -> None:
    for attempt in range(MAX_RETRIES):
        try:
            store.add_documents(batch, ids=[doc.id for doc in batch])
            return
        except Exception as error:  # the NVIDIA client raises plain exceptions on 429/5xx
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  embedding request failed ({error.__class__.__name__}: {error}); retrying in {wait}s")
            time.sleep(wait)


def ingest(settings: Settings, embeddings: Embeddings, rebuild: bool = False, pause: float = 1.6) -> int:
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
        add_with_retry(store, pending[start:start + BATCH_SIZE])
        done = min(start + BATCH_SIZE, len(pending))
        print(f"  {done}/{len(pending)}")
        if done < len(pending):
            time.sleep(pause)  # stays under the free tier's ~40 requests/minute

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
