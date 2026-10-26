"""Turn one graph run into a Server-Sent Events stream.

Frames (each is `event: <name>` plus one JSON `data:` line):

    token     {"text"}          a piece of the final answer, as it is generated
    trace     {"type", ..., "t_ms"}  a routing decision, tool call, or error
    final     {"text"}          the complete answer, sent only when it differs from
                                the streamed tokens (direct replies, failover, errors)
    citation  {"n", "kind", "key", "title", "url", "snippet"}
    error     {"message"}       the request itself failed
    done      {"ttft_ms", "total_ms", "plan", "agents", "instance"}

instance is {"turn", "uptime_s"}: which request this server process is serving
and how long it has been up, so a latency probe can tell a cold start (turn 1,
seconds of uptime) from a warm one.

t_ms on trace frames is milliseconds since the request started, measured on
the server, so the UI's timeline is not skewed by network buffering.
"""

import json
import logging
import time
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_turn(graph, query: str, session_id: str, instance: dict | None = None) -> AsyncIterator[str]:
    started = time.perf_counter()

    def elapsed() -> int:
        return round((time.perf_counter() - started) * 1000)

    config = {"configurable": {"thread_id": session_id}}
    first_token_ms: int | None = None
    streamed: list[str] = []
    try:
        async for mode, item in graph.astream({"messages": [HumanMessage(query)]}, config,
                                              stream_mode=["custom", "messages"]):
            if mode == "custom":
                if item.get("type") == "answer_reset":
                    first_token_ms, streamed = None, []
                yield sse("trace", {**item, "t_ms": elapsed()})
                continue
            chunk, metadata = item
            if "answer" in metadata.get("tags", []) and (text := chunk.text):
                first_token_ms = first_token_ms or elapsed()
                streamed.append(text)
                yield sse("token", {"text": text})

        state = (await graph.aget_state(config)).values
        answer = state.get("answer", "")
        if "".join(streamed).strip() != answer.strip():
            first_token_ms = first_token_ms or elapsed()
            yield sse("final", {"text": answer})
        for citation in state.get("citations", []):
            yield sse("citation", citation)
        agents = [{k: r.get(k) for k in ("agent", "provider", "model", "tools", "ms", "error")}
                  for r in state.get("completed", [])]
        yield sse("done", {"ttft_ms": first_token_ms, "total_ms": elapsed(), "plan": state.get("plan"),
                           "agents": agents, "instance": instance})
    except Exception as error:  # never leave the client hanging on a half-open stream
        log.exception("Turn failed for session %s", session_id)
        yield sse("error", {"message": f"The request failed: {type(error).__name__}. Try again."})
        yield sse("done", {"ttft_ms": first_token_ms, "total_ms": elapsed(), "plan": None, "agents": [],
                           "instance": instance})
