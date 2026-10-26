"""Trace events streamed to the UI while the graph runs.

Event types: llm_route, agent_plan, agent_route, llm_call, llm_cancelled,
rag_retrieval, mcp_tool_call, sql_query, answer_reset, error. Each is a plain dict with a
"type" key; the API layer forwards them as SSE `trace` frames.
"""

import time
from contextlib import contextmanager

from langgraph.config import get_stream_writer


def emit(event: dict) -> None:
    """Send a trace event to the stream; a no-op when not running inside the graph."""
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):  # outside any runnable, or inside one that isn't a graph
        return
    writer(event)


@contextmanager
def timer():
    """Yields a dict whose "ms" key is filled in when the block exits."""
    elapsed = {"ms": 0}
    started = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed["ms"] = round((time.perf_counter() - started) * 1000)
