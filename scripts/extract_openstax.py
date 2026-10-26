"""Extract OpenStax Anatomy & Physiology 2e into a RAG-ready corpus.

Reads the book's CNXML source (github.com/openstax/osbooks-anatomy-physiology)
instead of the 476 MB PDF: same text, no images, clean section structure.

Outputs (JSON Lines):
  ap2e_sections.jsonl        one record per book section, for ingestion/chunking
  ap2e_eval_candidates.jsonl end-of-section questions with the book's own answers,
                             as seed material for the RAGAS test set

License: the book is CC BY-NC-SA 4.0 (OpenStax, Rice University). Keep use
non-commercial, attribute OpenStax, and share derived corpus files under the
same license.

Usage:
  git clone --depth 1 https://github.com/openstax/osbooks-anatomy-physiology.git ../osbooks-anatomy-physiology
  python scripts/extract_openstax.py ../osbooks-anatomy-physiology data/
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CNXML = "{http://cnx.rice.edu/cnxml}"
COLXML = "{http://cnx.rice.edu/collxml}"
MDML = "{http://cnx.rice.edu/mdml}"

# Section classes that hold end-of-section exercises or recaps, kept out of the
# corpus so the RAGAS questions are not sitting verbatim in the index.
EXCLUDED_SECTION_CLASSES = {"multiple-choice", "free-response", "summary", "interactive-exercise"}
LETTERS = "abcdefghij"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def inline_text(elem: ET.Element) -> str:
    """Text of an element, rendering cross-reference links as the word 'Figure'."""
    parts = [elem.text or ""]
    for child in elem:
        if child.tag == f"{CNXML}link" and not (child.text or "").strip() and len(child) == 0:
            parts.append("Figure")
        else:
            parts.append(inline_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def render_blocks(elem: ET.Element, depth: int = 2) -> list[str]:
    """Flatten a CNXML content tree into markdown-ish text blocks."""
    blocks = []
    for child in elem:
        tag = child.tag.replace(CNXML, "")
        if tag == "para":
            blocks.append(clean(inline_text(child)))
        elif tag == "list":
            items = [f"- {clean(inline_text(i))}" for i in child.findall(f"{CNXML}item")]
            blocks.append("\n".join(items))
        elif tag == "section":
            if set(child.get("class", "").split()) & EXCLUDED_SECTION_CLASSES:
                continue
            title = child.find(f"{CNXML}title")
            if title is not None:
                blocks.append(f"{'#' * depth} {clean(inline_text(title))}")
            blocks.extend(render_blocks(child, depth + 1))
        elif tag == "note":
            title = child.find(f"{CNXML}title")
            label = clean(inline_text(title)) if title is not None else "Note"
            body = render_blocks(child, depth + 1)
            if body:
                blocks.append(f"[{label}] " + " ".join(body))
        elif tag == "figure":
            title = child.find(f"{CNXML}title")
            caption = child.find(f"{CNXML}caption")
            text = " ".join(clean(inline_text(e)) for e in (title, caption) if e is not None)
            if text:
                blocks.append(f"[Figure] {text}")
        elif tag == "table":
            rows = []
            for row in child.iter(f"{CNXML}row"):
                cells = [clean(inline_text(e)) for e in row.findall(f"{CNXML}entry")]
                rows.append(" | ".join(cells))
            title = child.find(f"{CNXML}title")
            head = f"[Table] {clean(inline_text(title))}\n" if title is not None else "[Table]\n"
            blocks.append(head + "\n".join(rows))
        elif tag == "equation":
            blocks.append(clean(inline_text(child)))
    return [b for b in blocks if b]


def extract_exercises(content: ET.Element, source: dict) -> list[dict]:
    records = []
    for section in content.iter(f"{CNXML}section"):
        kind = section.get("class", "")
        if kind not in {"multiple-choice", "free-response"}:
            continue
        for ex in section.findall(f"{CNXML}exercise"):
            problem = ex.find(f"{CNXML}problem")
            solution = ex.find(f"{CNXML}solution")
            if problem is None or solution is None:
                continue
            question = " ".join(clean(inline_text(p)) for p in problem.findall(f"{CNXML}para"))
            answer = " ".join(clean(inline_text(p)) for p in solution.findall(f"{CNXML}para"))
            record = {"type": kind, "question": question, "reference_answer": answer, **source}
            options = problem.find(f"{CNXML}list")
            if options is not None:
                opts = [clean(inline_text(i)) for i in options.findall(f"{CNXML}item")]
                record["options"] = opts
                letter = answer.strip().lower()[:1]
                if letter in LETTERS[: len(opts)]:
                    record["reference_answer"] = opts[LETTERS.index(letter)]
                    record["answer_letter"] = letter.upper()
            records.append(record)
    return records


def main(repo: Path, out_dir: Path) -> None:
    collection = next((repo / "collections").glob("*.collection.xml"))
    root = ET.parse(collection).getroot()
    out_dir.mkdir(parents=True, exist_ok=True)

    sections, exercises = [], []
    chapter_num = 0
    for unit in root.find(f"{COLXML}content").findall(f"{COLXML}subcollection"):
        for chapter in unit.find(f"{COLXML}content").findall(f"{COLXML}subcollection"):
            chapter_num += 1
            chapter_title = clean(chapter.find(f"{MDML}title").text)
            modules = chapter.find(f"{COLXML}content").findall(f"{COLXML}module")
            for section_num, module in enumerate(modules):  # module 0 is the chapter intro
                module_id = module.get("document")
                doc = ET.parse(repo / "modules" / module_id / "index.cnxml").getroot()
                title = clean(doc.find(f"{CNXML}title").text)
                content = doc.find(f"{CNXML}content")
                section_id = f"{chapter_num}.{section_num}"
                source = {
                    "source_id": f"ap2e-{section_id}",
                    "citation": f"OpenStax A&P 2e §{section_id} {title}",
                }
                sections.append({
                    **source,
                    "chapter": chapter_num,
                    "chapter_title": chapter_title,
                    "section": section_num,
                    "section_title": title,
                    "module_id": module_id,
                    "text": "\n\n".join(render_blocks(content)),
                })
                exercises.extend(extract_exercises(content, source))

    for name, rows in (("ap2e_sections.jsonl", sections), ("ap2e_eval_candidates.jsonl", exercises)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    words = sum(len(s["text"].split()) for s in sections)
    free = sum(e["type"] == "free-response" for e in exercises)
    print(f"{chapter_num} chapters, {len(sections)} sections, {words:,} words")
    print(f"{len(exercises)} exercises ({free} free-response with written answers)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
