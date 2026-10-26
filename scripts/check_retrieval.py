"""Retrieval sanity check against the book's own end-of-section questions.

For a fixed random sample of critical-thinking questions, checks whether the
top-k retrieved chunks come from the section the question was written for.
This tests retrieval alone, before any LLM is involved, and gives a baseline
number to compare chunking or embedding changes against.

    python -m scripts.check_retrieval            # 50 questions, k from settings
    python -m scripts.check_retrieval -n 100 -k 8
"""

import argparse
import json
import random
import statistics
import time

from src.config import get_settings
from src.rag.retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50, help="number of questions to sample")
    parser.add_argument("-k", type=int, default=None, help="chunks to retrieve per question")
    parser.add_argument("--seed", type=int, default=599)
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.retrieval_k
    with open(settings.eval_path, encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["type"] == "free-response"]
    sample = random.Random(args.seed).sample(questions, min(args.n, len(questions)))

    hits_top1 = hits_topk = chapter_hits = 0
    reciprocal_ranks, latencies, misses = [], [], []
    for q in sample:
        started = time.perf_counter()
        chunks = retrieve(q["question"], k=k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources = [c.source_id for c in chunks]
        target_chapter = q["source_id"].split(".")[0]
        rank = sources.index(q["source_id"]) + 1 if q["source_id"] in sources else None
        hits_top1 += rank == 1
        hits_topk += rank is not None
        chapter_hits += any(s.split(".")[0] == target_chapter for s in sources)
        reciprocal_ranks.append(1 / rank if rank else 0)
        if rank is None:
            misses.append((q["source_id"], q["question"], sources[:3]))

    n = len(sample)
    print(f"{n} questions, k={k}, embedding model {settings.embedding_model}")
    print(f"  hit@1 (right section ranked first): {hits_top1 / n:.0%}")
    print(f"  hit@{k} (right section anywhere in top {k}): {hits_topk / n:.0%}")
    print(f"  MRR: {statistics.mean(reciprocal_ranks):.3f}")
    print(f"  right chapter in top {k}: {chapter_hits / n:.0%}")
    print(f"  retrieval latency: median {statistics.median(latencies):.0f} ms, "
          f"max {max(latencies):.0f} ms (first query includes connection setup)")
    print("\nSample misses (expected section, question, top 3 retrieved):")
    for expected, question, got in misses[:5]:
        print(f"  {expected}: {question[:90]}\n    -> {got}")


if __name__ == "__main__":
    main()
