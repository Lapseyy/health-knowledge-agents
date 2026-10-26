"""Heading-aware chunking for the OpenStax A&P 2e corpus.

Each book section is split preferring its own subsection headings, then
paragraphs, then sentences. Every chunk keeps its heading path and citation so
retrieved text can always be traced back to a section of the book.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING = re.compile(r"^(#{2,6}) (.+)$", re.MULTILINE)
# Regex separators, tried in order: subsection headings, paragraphs, lines,
# sentence ends (lookbehind keeps the period with its own sentence), words.
SEPARATORS = [r"\n(?=## )", r"\n(?=### )", r"\n(?=#### )", r"\n\n", r"\n", r"(?<=[.!?]) ", " ", ""]


def load_sections(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def heading_path(text: str, position: int) -> list[str]:
    """Subsection headings in effect at a character position of a section's text."""
    stack: dict[int, str] = {}
    for match in HEADING.finditer(text):
        if match.start() > position:
            break
        level = len(match.group(1))
        stack = {lvl: title for lvl, title in stack.items() if lvl < level}
        stack[level] = match.group(2).strip()
    return [stack[level] for level in sorted(stack)]


def chunk_section(section: dict, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    text = section["text"]
    chunks = []
    for index, piece in enumerate(splitter.create_documents([text])):
        path = [section["chapter_title"], section["section_title"]]
        path += heading_path(text, piece.metadata["start_index"])
        breadcrumb = " > ".join(path)
        chunks.append(Document(
            id=f"{section['source_id']}#{index}",
            # The breadcrumb is part of the embedded text so a chunk that never
            # names its topic ("These cells...") still matches queries about it.
            page_content=f"{breadcrumb}\n\n{piece.page_content}",
            metadata={
                "chunk_id": f"{section['source_id']}#{index}",
                "source_id": section["source_id"],
                "citation": section["citation"],
                "chapter": section["chapter"],
                "section": section["section"],
                "heading": breadcrumb,
            },
        ))
    return chunks


def build_chunks(sections: list[dict], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        is_separator_regex=True,
        add_start_index=True,
    )
    return [chunk for section in sections for chunk in chunk_section(section, splitter)]
