import json
from dataclasses import replace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import get_settings
from src.rag.chunking import build_chunks, heading_path, load_sections
from src.rag.ingest import ingest
from src.rag.retriever import load_store, retrieve

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def sections():
    return load_sections(SETTINGS.corpus_path)


@pytest.fixture(scope="module")
def chunks(sections):
    return build_chunks(sections, SETTINGS.chunk_size, SETTINGS.chunk_overlap)


def test_heading_path_tracks_nesting():
    text = "intro\n\n## Heart\n\nbody\n\n### Valves\n\nmore\n\n## Vessels\n\nend"
    assert heading_path(text, 0) == []
    assert heading_path(text, text.index("more")) == ["Heart", "Valves"]
    assert heading_path(text, text.index("end")) == ["Vessels"]


def test_corpus_covers_whole_book(sections):
    assert len({s["chapter"] for s in sections}) == 28
    assert all(s["text"] for s in sections)


def test_chunks_are_bounded_and_citable(chunks):
    for chunk in chunks:
        body = chunk.page_content.split("\n\n", 1)[1]
        assert len(body) <= SETTINGS.chunk_size
        assert chunk.metadata["citation"].startswith("OpenStax A&P 2e §")
        assert chunk.metadata["chunk_id"] == chunk.id


def test_chunk_ids_are_unique_and_stable(sections, chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    rebuilt = build_chunks(sections, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    assert ids == [c.id for c in rebuilt]


def test_eval_questions_are_not_in_the_index(sections):
    corpus = "\n".join(s["text"] for s in sections)
    with open(SETTINGS.eval_path, encoding="utf-8") as f:
        questions = [q["question"] for q in map(json.loads, f) if len(q["question"]) > 60]
    leaked = [q for q in questions if q in corpus]
    assert not leaked, f"{len(leaked)} eval questions appear verbatim in the corpus"


@pytest.fixture
def tiny_index(tmp_path, sections):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(s) for s in sections[:3]), encoding="utf-8")
    settings = replace(SETTINGS, corpus_path=corpus, chroma_dir=tmp_path / "chroma")
    embeddings = DeterministicFakeEmbedding(size=64)
    count = ingest(settings, embeddings, pause=0)
    return settings, embeddings, count


def test_ingest_then_retrieve_round_trip(tiny_index):
    settings, embeddings, count = tiny_index
    store = load_store(settings, embeddings)
    stored = store.get(include=["documents"])
    assert len(stored["ids"]) == count

    target_id, target_text = stored["ids"][3], stored["documents"][3]
    top = retrieve(target_text, k=3, store=store)[0]
    assert top.chunk_id == target_id
    assert top.score == pytest.approx(1.0, abs=1e-3)


def test_ingest_resumes_without_duplicates(tiny_index, capsys):
    settings, embeddings, count = tiny_index
    ingest(settings, embeddings, pause=0)
    assert "0 to embed" in capsys.readouterr().out
    assert len(load_store(settings, embeddings).get(include=[])["ids"]) == count


def test_model_mismatch_fails_loudly(tiny_index):
    settings, embeddings, _ = tiny_index
    with pytest.raises(RuntimeError, match="Rebuild"):
        load_store(replace(settings, embedding_model="some/other-model"), embeddings)
