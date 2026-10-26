"""Tools for each specialist agent.

Each agent gets its own tool set (distinct tool access is what makes the
agents specialists rather than prompts over the same call):

    Research  search_textbook (RAG), search_web (Tavily MCP)
    Summary   find_sections, read_section (whole textbook sections),
              summarize_chapter (every section read in full, condensed in parallel),
              search_web, extract_webpage (Tavily MCP)
    Data      run_sql (read-only SQLite over synthetic patients)

search_web is the same MCP tool in two agents: Research uses it to ground
answers in current sources, Summary uses it to locate an article to condense.

Tools are built per agent run so they can number sources into that turn's
SourceRegistry and tag trace events with the agent that called them. Each
factory also receives the agent's model (route, client), which
summarize_chapter uses for its per-section calls.
"""

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from src.agents import prompts
from src.agents.events import emit, timer
from src.agents.sources import SourceRegistry
from src.agents.web import accepted_args, content_text, parse_results, redact
from src.llm import call_options, usage
from src.rag.retriever import retrieve

if TYPE_CHECKING:
    from src.agents.runtime import Runtime

MAX_WEB_SEARCHES = 2  # per agent per turn: each costs ~2.5 s and adds sources to the answer prompt
MAX_TEXTBOOK_SEARCHES = 1  # Research's searches beyond the passages prefetched for the question
WEB_RESULTS = 3
MAX_WEB_PAGES = 2  # full-page reads per turn: each adds a whole article to the answer prompt
MAX_SQL_ROWS = 50
SQL_TIMEOUT_S = 5
MAX_SECTION_CHARS = 30000
DIGEST_MAX_CHARS = 70000  # the longest section is ~65k chars, so chapter summaries read every section in full
DIGEST_CONCURRENCY = 6  # parallel section calls; keeps a 12-section chapter within NVIDIA's free-tier burst
DIGEST_TIMEOUT_S = 45  # per section: a stuck call drops that section instead of stalling the chapter
SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


# --- Textbook (RAG) ---------------------------------------------------------

def textbook_search(runtime: "Runtime", registry: SourceRegistry, agent: str, query: str) -> str:
    with timer() as elapsed:
        chunks = retrieve(query, store=runtime.store)
    numbered = [registry.add("textbook", c.chunk_id, c.citation, c.text) for c in chunks]
    emit({
        "type": "rag_retrieval", "agent": agent, "query": query, "ms": elapsed["ms"],
        "results": [{"n": n, "chunk_id": c.chunk_id, "citation": c.citation, "score": c.score}
                    for n, c in zip(numbered, chunks)],
    })
    return registry.render(numbered)


def research_tools(runtime: "Runtime", registry: SourceRegistry, model=None) -> list[BaseTool]:
    searches = {"count": 0}

    @tool
    async def search_textbook(query: str) -> str:
        """Search the OpenStax Anatomy & Physiology 2e textbook. Returns numbered passages
        with section citations. Use focused queries about anatomy or physiology."""
        if searches["count"] >= MAX_TEXTBOOK_SEARCHES:
            return ("Textbook search limit reached; the textbook passages you have are what it covers. "
                    "Use search_web or answer with the sources you have.")
        searches["count"] += 1
        return textbook_search(runtime, registry, "research", query)

    return [search_textbook, *web_tools(runtime, registry, "research", extract=False)]


# --- Whole sections (Summary) -----------------------------------------------

def chapter_sections(runtime: "Runtime", chapter: int) -> list[dict]:
    return sorted((s for s in runtime.sections.values() if s["chapter"] == chapter), key=lambda s: s["section"])


def summary_tools(runtime: "Runtime", registry: SourceRegistry, model=None) -> list[BaseTool]:
    """`model` is the agent's (route, client): summarize_chapter runs its per-section calls on it."""
    read: dict[str, int] = {}  # sections read this turn -> source number

    @tool
    async def find_sections(topic: str) -> str:
        """Find textbook sections about a topic. Returns section IDs (like ap2e-19.1) and titles
        to pass to read_section."""
        with timer() as elapsed:
            chunks = retrieve(topic, k=10, store=runtime.store)
        seen: dict[str, str] = {}
        for chunk in chunks:
            seen.setdefault(chunk.source_id, chunk.citation)
        emit({"type": "rag_retrieval", "agent": "summary", "query": topic, "ms": elapsed["ms"],
              "results": [{"source_id": s, "citation": c} for s, c in list(seen.items())[:5]]})
        return "\n".join(f"{s}: {c}" for s, c in list(seen.items())[:5]) or "No matching sections."

    @tool
    async def read_section(section_id: str) -> str:
        """Read a whole textbook section by ID, e.g. ap2e-19.1 (or 19.1)."""
        key = section_id.strip()
        key = key if key.startswith("ap2e-") else f"ap2e-{key.lstrip('§')}"
        section = runtime.sections.get(key)
        if section is None:
            return f"No section {section_id}. Use find_sections to get valid IDs."
        if key in read:  # re-reading would paste the whole section into the context again
            return f"Already read: [{read[key]}] {section['citation']} is in your sources. Do not read it again."
        text = section["text"][:MAX_SECTION_CHARS]
        n = read[key] = registry.add("textbook", key, section["citation"], text, max_chars=MAX_SECTION_CHARS)
        emit({"type": "rag_retrieval", "agent": "summary", "query": f"read {key}", "ms": 0,
              "results": [{"n": n, "source_id": key, "citation": section["citation"]}]})
        return f"[{n}] {section['citation']}\n{text}"

    @tool
    async def summarize_chapter(chapter: int) -> str:
        """Read every section of a textbook chapter in full and return a digest of each, for
        summarizing a whole chapter. Takes the chapter number, e.g. 19 (ap2e-19.3 is in chapter 19)."""
        sections = chapter_sections(runtime, chapter)
        if not sections:
            return f"No chapter {chapter}. The textbook has chapters 1-28."
        route, llm = model
        digester = llm.bind(**call_options(route)).with_config(tags=["digest"], run_name="summary_digest")
        gate = asyncio.Semaphore(DIGEST_CONCURRENCY)

        async def digest(section: dict) -> str:
            async with gate:
                with timer() as elapsed:
                    reply = await asyncio.wait_for(digester.ainvoke([
                        SystemMessage(prompts.SECTION_DIGEST),
                        HumanMessage(f"{section['citation']}\n\n{section['text'][:DIGEST_MAX_CHARS]}"),
                    ]), DIGEST_TIMEOUT_S)
            emit({"type": "llm_call", "agent": "summary", "provider": route.provider, "model": route.model,
                  "stage": "digest", "section": f"§{chapter}.{section['section']}", "ms": elapsed["ms"],
                  **usage(reply)})
            return reply.text.strip()

        # Map: one call per section, in parallel. Reduce: the agent's answer call combines the digests.
        with timer() as elapsed:
            digests = await asyncio.gather(*(digest(s) for s in sections), return_exceptions=True)
        numbers, missing = [], []
        for section, result in zip(sections, digests):
            if isinstance(result, BaseException) or not result:
                missing.append(section["citation"])
                continue
            numbers.append(registry.add("textbook", section["source_id"], section["citation"], section["text"],
                                        context=f"Digest of the full section:\n{result}"))
        emit({"type": "rag_retrieval", "agent": "summary", "query": f"read chapter {chapter}", "ms": elapsed["ms"],
              "results": [{"n": n, "source_id": s["source_id"], "citation": s["citation"]}
                          for n, s in zip(numbers, (s for s in sections if s["citation"] not in missing))]})
        if not numbers:
            raise RuntimeError(f"Could not read any section of chapter {chapter}.")
        header = f"Chapter {chapter}: {sections[0]['chapter_title']}. Each section was read in full and condensed."
        if missing:
            header += f" These sections could not be read: {'; '.join(missing)}."
        return f"{header}\n\n{registry.render(numbers)}"

    return [find_sections, read_section, summarize_chapter, *web_tools(runtime, registry, "summary", extract=True)]


# --- Web (Tavily Remote MCP, shared) ----------------------------------------

def web_tools(runtime: "Runtime", registry: SourceRegistry, agent: str, extract: bool) -> list[BaseTool]:
    mcp_search, mcp_extract = runtime.web.find("search"), runtime.web.find("extract")
    tools: list[BaseTool] = []

    async def call_mcp(mcp_tool: BaseTool, args: dict) -> list[dict]:
        args = accepted_args(mcp_tool, args)
        with timer() as elapsed:
            try:
                results = parse_results(content_text(await mcp_tool.ainvoke(args)))
                status = "ok"
            except Exception as error:  # tool execution error or malformed response
                results, status = [], redact(f"error: {type(error).__name__}: {error}")[:300]
        emit({"type": "mcp_tool_call", "agent": agent, "server": "tavily", "tool": mcp_tool.name,
              "args": args, "status": status, "results": len(results), "ms": elapsed["ms"]})
        if status != "ok":
            raise RuntimeError(f"Web tool failed ({status}). Continue without it.")
        return results

    def number(results: list[dict]) -> str:
        numbered = [registry.add("web", r["url"] or r["title"], r["title"], r["content"], url=r["url"])
                    for r in results]
        return registry.render(numbered) if numbered else "No results."

    if mcp_search:
        searches = {"count": 0}

        @tool
        async def search_web(query: str) -> str:
            """Search the live web for current health information: news, guidelines, drug
            approvals, statistics, or anything newer than or missing from the textbook."""
            if searches["count"] >= MAX_WEB_SEARCHES:
                return f"Web search limit ({MAX_WEB_SEARCHES} per turn) reached. Answer with the results you have."
            searches["count"] += 1
            return number(await call_mcp(mcp_search, {"query": query, "max_results": WEB_RESULTS}))

        tools.append(search_web)

    if extract and mcp_extract:
        pages = {"count": 0}

        @tool
        async def extract_webpage(url: str) -> str:
            """Read the full text of a web page by URL."""
            if pages["count"] >= MAX_WEB_PAGES:
                return f"Page limit ({MAX_WEB_PAGES} per turn) reached. Answer with the pages you have read."
            pages["count"] += 1
            return number(await call_mcp(mcp_extract, {"urls": [url], "url": url}))

        tools.append(extract_webpage)
    return tools


# --- SQL over synthetic patients (Data) -------------------------------------

def run_readonly_query(db_path: Path, sql: str) -> dict:
    """Run one SELECT against a read-only connection. Raises ValueError on anything else."""
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only one statement is allowed.")
    if not SELECT_ONLY.match(sql):
        raise ValueError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    # mode=ro makes the connection itself read-only, so even a query that
    # slipped past the checks above could not modify the database.
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    deadline = time.monotonic() + SQL_TIMEOUT_S
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
        columns = [d[0] for d in cursor.description or []]
    finally:
        conn.close()
    return {"columns": columns, "rows": [list(r) for r in rows[:MAX_SQL_ROWS]],
            "truncated": len(rows) > MAX_SQL_ROWS}


def data_tools(runtime: "Runtime", registry: SourceRegistry, model=None) -> list[BaseTool]:
    db_path = runtime.settings.patient_db_path

    @tool
    async def run_sql(query: str) -> str:
        """Run one read-only SQLite SELECT query on the synthetic patient database.
        Returns up to 50 rows as JSON. Fix the query and retry if it returns an error."""
        with timer() as elapsed:
            try:
                result = run_readonly_query(db_path, query)
                status = "ok"
            except (ValueError, sqlite3.Error) as error:
                result, status = None, f"error: {error}"
        emit({"type": "sql_query", "agent": "data", "sql": query, "status": status, "ms": elapsed["ms"],
              "rows": len(result["rows"]) if result else 0})
        if result is None:
            return f"SQL {status}"
        payload = json.dumps(result, default=str)
        n = registry.add("sql", query, "Synthetic patient database (Synthea) query", f"SQL: {query}\nResult: {payload}")
        return f"[{n}] {payload}"

    return [run_sql]


TABLE_NOTES = {
    "patients": "one row per patient; deathdate IS NULL means living",
    "conditions": "diagnoses; description like 'Essential hypertension (disorder)'; stop IS NULL means ongoing",
    "medications": "prescriptions; reason is the condition treated; stop IS NULL means current",
    "encounters": "visits; encounter_class is ambulatory, outpatient, inpatient, emergency, wellness, urgentcare",
    "immunizations": "vaccines given",
    "allergies": "type is allergy or intolerance; category is food, medication, or environment",
    "latest_observations": "each patient's most recent numeric lab or vital sign; value is REAL with units",
}


COMMON_OBSERVATIONS = [
    "Body mass index (BMI) [Ratio]", "Body Weight", "Body Height", "Systolic Blood Pressure",
    "Diastolic Blood Pressure", "Heart rate", "Hemoglobin A1c/Hemoglobin.total in Blood",
    "Glucose [Mass/volume] in Blood", "Cholesterol [Mass/volume] in Serum or Plasma",
    "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "Creatinine [Mass/volume] in Blood",
]


def describe_database(db_path: Path) -> str:
    """Schema summary for the Data agent's prompt, read from the database itself."""
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        lines = []
        for table, note in TABLE_NOTES.items():
            columns = ", ".join(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            lines.append(f"- {table}({columns}): {count:,} rows; {note}")
        lines.append("  latest_observations descriptions include: "
                     + "; ".join(f"'{d}'" for d in COMMON_OBSERVATIONS))
    finally:
        conn.close()
    return "\n".join(lines)
