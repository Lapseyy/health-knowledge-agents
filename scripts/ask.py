"""Ask the multi-agent system questions from the terminal, with the live trace.

    python -m scripts.ask "What does the left ventricle do?"
    python -m scripts.ask              # interactive; the session remembers earlier turns

Trace lines show each routing decision, provider, and tool call as it happens;
the answer streams after them, followed by citations and timing.
"""

import asyncio
import logging
import sys
import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.graph import build_graph
from src.agents.runtime import build_runtime
from src.config import get_settings


def describe(event: dict) -> str:
    kind = event["type"]
    if kind == "llm_route":
        role = "FALLBACK " if event["fallback"] else ""
        why = f" because {event['trigger']}" if event.get("trigger") else ""
        return f"llm_route    {event['agent']} -> {role}{event['provider']} {event['model']} [{event['dimension']}]{why}"
    if kind == "agent_plan":
        hedged = ", hedged" if event.get("hedged") else ""
        return f"agent_plan   {event['plan']} by {event['planner']} ({event['ms']} ms{hedged}): {event['reason']}"
    if kind == "llm_cancelled":
        return f"cancelled    {event['agent']} on {event['provider']} after {event['ms']} ms: {event['reason']}"
    if kind == "agent_route":
        return f"agent_route  supervisor -> {event['to']} (step {event['step']} of {event['of']})"
    if kind == "llm_call":
        calls = f", {event['tool_calls']} tool calls" if event["stage"] == "tools" else ""
        section = f" {event['section']}" if event.get("section") else ""
        tokens = (f", {event['input_tokens']} in / {event['output_tokens']} out tokens"
                  if event.get("input_tokens") is not None else "")
        return (f"llm_call     {event['agent']} on {event['provider']}: {event['stage']}{section}{calls} "
                f"({event['ms']} ms{tokens})")
    if kind == "rag_retrieval":
        return f"rag          {event['agent']}: \"{event['query'][:60]}\" -> {len(event['results'])} results ({event['ms']} ms)"
    if kind == "mcp_tool_call":
        return f"mcp          {event['agent']}: tavily/{event['tool']} {event['status']}, {event['results']} results ({event['ms']} ms)"
    if kind == "sql_query":
        return f"sql          {event['sql'][:90]} -> {event['status']}, {event['rows']} rows ({event['ms']} ms)"
    if kind == "answer_reset":
        return f"answer_reset {event['agent']}: {event['reason']}"
    if kind == "error":
        provider = f" ({event['provider']})" if event.get("provider") else ""
        took = f" after {event['ms']} ms" if event.get("ms") is not None else ""
        return f"ERROR        {event['agent']}{provider} [{event['stage']}]{took}: {event['message']}"
    return str(event)


async def ask(graph, question: str, thread: str) -> None:
    config = {"configurable": {"thread_id": thread}}
    started = time.perf_counter()
    first_token, streamed = None, []
    print()
    async for mode, item in graph.astream({"messages": [HumanMessage(question)]}, config,
                                          stream_mode=["custom", "messages"]):
        if mode == "custom":
            if item["type"] == "answer_reset" and streamed:
                print("\n  [partial answer discarded]")
                first_token, streamed = None, []
            print(f"  - {describe(item)}", flush=True)
            continue
        chunk, metadata = item
        if "answer" in metadata.get("tags", []) and chunk.text:
            if first_token is None:
                first_token = time.perf_counter()
                print("\nAnswer:")
            streamed.append(chunk.text)
            print(chunk.text, end="", flush=True)

    state = (await graph.aget_state(config)).values
    if "".join(streamed).strip() != state["answer"].strip():
        # Direct replies and failure messages are not streamed token by token.
        first_token = first_token or time.perf_counter()
        print(f"\nAnswer:\n{state['answer']}", end="")
    total = time.perf_counter() - started
    print("\n")
    for citation in state.get("citations", []):
        where = citation["url"] or citation["key"]
        print(f"  [{citation['n']}] {citation['title']}  ({where})")
    print(f"\n  time to first token {first_token - started:.1f} s, total {total:.1f} s")


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    runtime = await build_runtime(get_settings())
    web = runtime.web
    print(f"Tavily MCP: {'connected, tools ' + ', '.join(sorted(web.tools)) if web.tools else 'unavailable (' + str(web.last_error) + ')'}")
    graph = build_graph(runtime, InMemorySaver())
    thread = str(uuid.uuid4())

    if len(sys.argv) > 1:
        await ask(graph, " ".join(sys.argv[1:]), thread)
        return
    print("Ask a question (blank line to quit). Follow-ups in this session remember context.")
    while question := input("\nYou: ").strip():
        await ask(graph, question, thread)


if __name__ == "__main__":
    asyncio.run(main())
