"""Write the fixed evaluation sets used by scripts/evaluate.py.

    python scripts/build_eval_sets.py

data/eval_ragas.jsonl    30 OpenStax critical-thinking questions with the book's own written
                         answers as references, spread across chapters (seed 599). Questions
                         that depend on a figure, or whose answer is too short to grade
                         recall against, are skipped.
data/eval_routing.jsonl  Messages for the router comparison, each with the agents it should
                         get, covering every agent, two-part questions, greetings, an
                         off-topic question, and follow-ups that need the conversation.

Both files are committed, so every run evaluates the same questions.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 599
RAGAS_SIZE = 30
MIN_REFERENCE_CHARS = 80
VISUAL = re.compile(r"\b(figure|diagram|image|picture|illustrat|shown|photo|micrograph)", re.IGNORECASE)

ROUTING = [
    # Textbook anatomy and physiology
    ("What does the left ventricle do?", ["research"]),
    ("How do neurons transmit a signal across a synapse?", ["research"]),
    ("What is the function of the loop of Henle?", ["research"]),
    ("How does the diaphragm help us breathe?", ["research"]),
    # Current information (web search)
    ("What are the latest guidelines for treating high blood pressure?", ["research"]),
    ("Has the FDA approved any new Alzheimer's drugs recently?", ["research"]),
    ("What is the current CDC recommendation for the flu vaccine?", ["research"]),
    ("What have recent studies found about GLP-1 drugs and heart disease?", ["research"]),
    # Summaries
    ("Summarize chapter 2.", ["summary"]),
    ("Summarize the textbook section on the cardiac cycle.", ["summary"]),
    ("Give me an overview of chapter 19 of the textbook.", ["summary"]),
    ("Summarize this article: https://www.cdc.gov/flu/about/index.html", ["summary"]),
    # Synthetic patient database
    ("How many patients have diabetes?", ["data"]),
    ("What is the average BMI of living patients?", ["data"]),
    ("Which three conditions are most common in the patient database?", ["data"]),
    ("What percentage of patients are over 65?", ["data"]),
    # Two distinct parts
    ("What is hypertension, and how many patients in the database have it?", ["research", "data"]),
    ("Explain how insulin works, and give the average HbA1c of diabetic patients.", ["research", "data"]),
    # No agents: direct reply
    ("Hi!", []),
    ("Thanks, that was helpful.", []),
    ("What can you do?", []),
    ("What do you think about the Epstein files?", []),
]

# Follow-ups: (history, message, expected agents). The first one reproduces the
# misroute fixed on Oct 25, 2026, when an earlier question leaked into the plan.
FOLLOW_UPS = [
    ([("user", "What are the latest guidelines for treating high blood pressure?"),
      ("assistant", "The 2025 ACC/AHA guideline sets a treatment target below 130/80 mm Hg for most adults "
                    "and recommends starting medication at stage 2 hypertension.")],
     "What does the left ventricle do?", ["research"]),
    ([("user", "What does the left ventricle do?"),
      ("assistant", "The left ventricle pumps oxygen-rich blood into the aorta for the systemic circuit.")],
     "And the right one?", ["research"]),
    ([("user", "How many patients have diabetes?"),
      ("assistant", "In this synthetic population, 92 patients have a recorded diagnosis of diabetes.")],
     "What is their average BMI?", ["data"]),
    ([("user", "Summarize the textbook section on the cardiac cycle."),
      ("assistant", "The cardiac cycle is the sequence of atrial and ventricular systole and diastole.")],
     "Now summarize chapter 20.", ["summary"]),
]


def ragas_set() -> list[dict]:
    rows = [json.loads(line) for line in open(ROOT / "data/ap2e_eval_candidates.jsonl", encoding="utf-8")]
    usable = [r for r in rows if r["type"] == "free-response" and len(r["reference_answer"]) >= MIN_REFERENCE_CHARS
              and not VISUAL.search(r["question"]) and not VISUAL.search(r["reference_answer"])]
    by_chapter: dict[int, list[dict]] = defaultdict(list)
    for row in usable:
        by_chapter[int(row["source_id"].split("-")[1].split(".")[0])].append(row)
    rng = random.Random(SEED)
    for chapter_rows in by_chapter.values():
        rng.shuffle(chapter_rows)
    chosen: list[dict] = []
    chapters = sorted(by_chapter)
    rng.shuffle(chapters)
    while len(chosen) < RAGAS_SIZE and any(by_chapter.values()):  # round-robin: chapters take turns
        for chapter in chapters:
            if by_chapter[chapter] and len(chosen) < RAGAS_SIZE:
                chosen.append(by_chapter[chapter].pop())
    return [{"id": f"q{i:02d}", "question": r["question"], "reference": r["reference_answer"],
             "source_id": r["source_id"], "citation": r["citation"]} for i, r in enumerate(chosen, 1)]


def routing_set() -> list[dict]:
    items = [{"id": f"r{i:02d}", "history": [], "message": m, "expected": e} for i, (m, e) in enumerate(ROUTING, 1)]
    start = len(items) + 1
    items += [{"id": f"r{i:02d}", "history": [{"role": role, "text": text} for role, text in history],
               "message": m, "expected": e} for i, (history, m, e) in enumerate(FOLLOW_UPS, start)]
    return items


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    ragas, routing = ragas_set(), routing_set()
    write(ROOT / "data/eval_ragas.jsonl", ragas)
    write(ROOT / "data/eval_routing.jsonl", routing)
    chapters = sorted({int(r["source_id"].split("-")[1].split(".")[0]) for r in ragas})
    print(f"data/eval_ragas.jsonl: {len(ragas)} questions from {len(chapters)} chapters")
    print(f"data/eval_routing.jsonl: {len(routing)} messages")


if __name__ == "__main__":
    main()
