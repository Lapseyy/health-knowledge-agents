import json
from dataclasses import replace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.agents.runtime import load_sections
from src.config import get_settings
from src.rag.ingest import ingest
from src.rag.retriever import load_store

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("index")
    sections = load_sections(SETTINGS)
    corpus = tmp / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(sections[k]) for k in list(sections)[:4]), encoding="utf-8")
    settings = replace(SETTINGS, corpus_path=corpus, chroma_dir=tmp / "chroma")
    embeddings = DeterministicFakeEmbedding(size=64)
    ingest(settings, embeddings)
    return settings, load_store(settings, embeddings), sections
