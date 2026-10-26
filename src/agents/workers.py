"""Specialist agents. Each runs in two phases:

1. Tool loop: the agent's LLM decides which of its own tools to call (up to
   MAX_TOOL_ROUNDS), and every result is numbered into the turn's sources.
2. Answer: a separate call with no tools writes the user-facing answer from
   the numbered sources, citing them as [n]. Only this call is tagged
   "answer", so the API streams the answer's tokens and nothing else
   (no tool-call arguments, no "DONE" markers).
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.agents import prompts
from src.agents.events import emit, timer
from src.agents.sources import SourceRegistry
from src.agents.tools import data_tools, research_tools, summary_tools, textbook_search
from src.llm import call_options

if TYPE_CHECKING:
    from src.agents.runtime import Runtime
    from src.llm import Route

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 4  # per agent per turn; extra calls are answered with "budget used up"
BUDGET_USED = "Skipped: the tool budget for this turn is used up. Answer with the evidence you have."
HISTORY_MESSAGES = 6  # the last three exchanges


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    tool_prompt: str
    answer_prompt: str
    make_tools: Callable[["Runtime", SourceRegistry, tuple], list[BaseTool]]  # (runtime, registry, (route, llm))
    prefetch_textbook: bool = False


WORKERS = {
    "research": WorkerSpec("research", prompts.RESEARCH_TOOLS, prompts.RESEARCH_ANSWER, research_tools,
                           prefetch_textbook=True),
    "summary": WorkerSpec("summary", prompts.SUMMARY_TOOLS, prompts.SUMMARY_ANSWER, summary_tools),
    "data": WorkerSpec("data", prompts.DATA_TOOLS, prompts.DATA_ANSWER, data_tools),
}


def recent_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Earlier exchanges in this session (excluding the current question), oldest first."""
    prior = [m for m in messages[:-1] if isinstance(m, (HumanMessage, AIMessage))][-HISTORY_MESSAGES:]
    while prior and not isinstance(prior[0], HumanMessage):
        prior.pop(0)  # Gemini expects the conversation to start with a user turn
    return prior


def prior_findings(completed: list[dict]) -> str:
    return "\n\n".join(f"{r['agent']} agent: {r['answer']}" for r in completed if r.get("answer"))


async def run_tool(tools: dict[str, BaseTool], call: dict, agent: str) -> str:
    tool = tools.get(call["name"])
    if tool is None:
        return f"Unknown tool {call['name']}. Available: {', '.join(tools)}"
    try:
        return str(await tool.ainvoke(call["args"]))
    except Exception as error:  # a failed tool becomes a message the agent can react to
        emit({"type": "error", "agent": agent, "stage": f"tool {call['name']}", "message": str(error)[:300]})
        return f"Tool error: {error}"


async def run_worker(spec: WorkerSpec, state: dict, runtime: "Runtime", route: "Route",
                     llm: BaseChatModel) -> dict:
    emit(route.as_event())
    options = call_options(route)
    registry = SourceRegistry(state.get("sources"))
    history = recent_history(state["messages"])
    completed = state.get("completed", [])
    is_final = len(completed) + 1 >= len(state["plan"])

    task = f"Question: {state['question']}"
    if state.get("standalone") and state["standalone"] != state["question"]:
        task += f"\n(In context: {state['standalone']})"
    if findings := prior_findings(completed):
        task += f"\n\nFindings from earlier agents in this turn:\n{findings}"

    await runtime.web.connect()  # no-op when connected; retries a failed connection at most once a minute
    tools = spec.make_tools(runtime, registry, (route, llm))
    tools_used: list[str] = []
    with timer() as tool_time:
        loop_input = task
        if spec.prefetch_textbook:
            query = state.get("standalone") or state["question"]
            passages = textbook_search(runtime, registry, spec.name, query)
            loop_input += f"\n\nTextbook passages already retrieved:\n{passages}"
        messages: list[BaseMessage] = [
            SystemMessage(spec.tool_prompt.format(schema=runtime.db_schema)), *history, HumanMessage(loop_input),
        ]
        bound = llm.bind_tools(tools, **options).with_config(tags=["tool_loop"], run_name=f"{spec.name}_tools")
        by_name = {tool.name: tool for tool in tools}
        for _ in range(MAX_TOOL_ROUNDS):
            with timer() as call_time:
                reply = await bound.ainvoke(messages)
            emit({"type": "llm_call", "agent": spec.name, "provider": route.provider, "model": route.model,
                  "stage": "tools", "tool_calls": len(reply.tool_calls), "ms": call_time["ms"]})
            messages.append(reply)
            if not reply.tool_calls:
                break
            budget = MAX_TOOL_CALLS - len(tools_used)
            allowed, skipped = reply.tool_calls[:budget], reply.tool_calls[budget:]
            # Calls requested in the same round are independent, so they run concurrently.
            results = await asyncio.gather(*(run_tool(by_name, call, spec.name) for call in allowed))
            tools_used += [call["name"] for call in allowed]
            for call, result in [*zip(allowed, results), *((call, BUDGET_USED) for call in skipped)]:
                messages.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            if len(tools_used) >= MAX_TOOL_CALLS:
                break

    tag = "answer" if is_final else "intermediate"
    answer_llm = llm.bind(**options).with_config(tags=[tag], run_name=f"{spec.name}_answer")
    with timer() as answer_time:
        response = await answer_llm.ainvoke([
            SystemMessage(spec.answer_prompt),
            *history,
            HumanMessage(f"{task}\n\nSources:\n{registry.render()}"),
        ])
    emit({"type": "llm_call", "agent": spec.name, "provider": route.provider, "model": route.model,
          "stage": tag, "ms": answer_time["ms"]})
    answer = response.text.strip()

    result = {
        "agent": spec.name, "provider": route.provider, "model": route.model, "answer": answer,
        "tools": tools_used, "ms": {"tools": tool_time["ms"], "answer": answer_time["ms"]}, "error": None,
    }
    return {
        "completed": [*completed, result],
        "sources": registry.sources,
        "answer": answer if is_final else state.get("answer", ""),
    }
