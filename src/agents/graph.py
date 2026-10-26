"""Supervisor orchestration as a LangGraph state graph.

    START -> prepare -> supervisor -> research | summary | data -> supervisor -> ... -> finish -> END

The supervisor makes one LLM call per turn to produce a plan (an ordered
list of agents plus a reason), stored in state. Gemini plans by default; if it
is rate-limited or slower than SUPERVISOR_TIMEOUT_S, a fast Nemotron model
plans instead, and the trace shows which router decided and why. Agents get
the same treatment: if an agent's primary provider fails, the agent reruns on
the other provider before the turn gives up. After each agent returns, the
supervisor sends the turn to the next agent in the plan, or to finish when
the plan is done, so every transition is driven by inspectable state fields
(plan, completed, next). A failed agent is recorded in `completed` with its
error and the plan continues; finish falls back to the best answer available.
"""

import asyncio
import json
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agents import prompts
from src.agents.events import emit, timer
from src.agents.runtime import Runtime
from src.agents.workers import WORKERS, recent_history, run_worker
from src.llm import call_options

AgentName = Literal["research", "summary", "data"]
MAX_PLAN = 3
SUPERVISOR_TIMEOUT_S = 8.0  # Gemini's warm routing time is ~3 s; beyond 8 s, the fallback is faster
FALLBACK_TIMEOUT_S = 10.0
WORKER_TIMEOUT_S = 90.0  # per provider attempt, so a hung call fails over instead of stalling
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
FALLBACK_ANSWER = ("Sorry, I couldn't complete that request because a service I depend on failed. "
                   "Please try again in a moment.")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory, per session
    question: str  # the user's latest message, as typed
    standalone: str  # the same question rewritten to stand alone, for retrieval
    plan: list[str] | None  # agents to run this turn, in order
    route_reason: str  # the supervisor's one-sentence rationale
    reply: str | None  # direct reply when the plan is empty
    next: str  # where the supervisor routes next
    completed: list[dict]  # one record per agent run: provider, tools, timings, error
    sources: list[dict]  # numbered evidence for this turn
    answer: str
    citations: list[dict]  # the sources the final answer actually cites


class RoutePlan(BaseModel):
    agents: list[AgentName] = Field(default_factory=list, description="Agents to run, in order (0 to 3).")
    reason: str = Field(description="One sentence on why these agents fit the message.")
    standalone: str | None = Field(default=None, description=(
        "The user's latest message rewritten as a self-contained question, resolving references to "
        "earlier turns (for example 'which one' -> 'which heart chamber'). Omit if already self-contained."))
    reply: str | None = Field(default=None, description="Only when agents is empty: a short direct reply.")


def normalize_plan(raw) -> RoutePlan:
    plan = raw if isinstance(raw, RoutePlan) else RoutePlan.model_validate(raw)
    agents = list(dict.fromkeys(a for a in plan.agents if a in WORKERS))[:MAX_PLAN]
    if not agents and not plan.reply:
        agents = ["research"]
    return RoutePlan(agents=agents, reason=plan.reason, standalone=plan.standalone,
                     reply=None if agents else plan.reply)


def build_graph(runtime: Runtime, checkpointer=None):
    async def prepare(state: AgentState) -> dict:
        return {
            "question": state["messages"][-1].text, "standalone": "", "plan": None, "route_reason": "",
            "reply": None, "next": "", "completed": [], "sources": [], "answer": "", "citations": [],
        }

    async def ask_planner(route, llm, messages: list) -> RoutePlan:
        emit(route.as_event())
        config = {"tags": ["supervisor"], "run_name": "supervisor_plan"}
        if route.provider == "google":
            return normalize_plan(await llm.with_structured_output(RoutePlan).ainvoke(messages, config=config))
        # Nemotron replies with plain JSON: its structured-output mode can't turn reasoning off.
        system = SystemMessage(f"{prompts.SUPERVISOR}\n\n{prompts.PLAN_AS_JSON}")
        reply = await llm.bind(**call_options(route)).ainvoke([system, *messages[1:]], config=config)
        if not (match := JSON_OBJECT.search(reply.text)):
            raise ValueError(f"Router reply had no JSON object: {reply.text[:120]!r}")
        return normalize_plan(json.loads(match.group(0)))

    async def plan_turn(state: AgentState) -> RoutePlan:
        messages = [SystemMessage(prompts.SUPERVISOR), *recent_history(state["messages"]),
                    HumanMessage(state["question"])]
        for (route, llm), timeout in zip(runtime.attempts("supervisor"), (SUPERVISOR_TIMEOUT_S, FALLBACK_TIMEOUT_S)):
            try:
                with timer() as elapsed:
                    plan = await asyncio.wait_for(ask_planner(route, llm, messages), timeout)
            except Exception as error:  # rate limit, timeout, or malformed output: try the next router
                reason = f"timed out after {timeout:.0f} s" if isinstance(error, TimeoutError) else str(error)
                emit({"type": "error", "agent": "supervisor", "provider": route.provider, "stage": "plan",
                      "message": reason[:300]})
                continue
            emit({"type": "agent_plan", "planner": f"{route.provider} {'fallback' if route.fallback else 'primary'}",
                  "plan": plan.agents, "reason": plan.reason, "standalone": plan.standalone, "ms": elapsed["ms"]})
            return plan
        return RoutePlan(agents=["research"], reason="Both routers were unavailable; defaulted to Research.")

    async def supervisor(state: AgentState) -> dict:
        update: dict = {}
        if state.get("plan") is None:
            plan = await plan_turn(state)
            update = {"plan": plan.agents, "route_reason": plan.reason, "reply": plan.reply,
                      "standalone": plan.standalone or state["question"]}
        agents = update.get("plan", state.get("plan")) or []
        step = len(state.get("completed", []))
        target = agents[step] if step < len(agents) else "finish"
        if target != "finish":
            emit({"type": "agent_route", "from": "supervisor", "to": target, "step": step + 1,
                  "of": len(agents), "reason": update.get("route_reason", state.get("route_reason"))})
        return {**update, "next": target}

    def worker(name: str):
        async def node(state: AgentState) -> dict:
            for attempt, (route, llm) in enumerate(runtime.attempts(name)):
                try:
                    return await asyncio.wait_for(run_worker(WORKERS[name], state, runtime, route, llm),
                                                  WORKER_TIMEOUT_S)
                except Exception as error:  # one provider failing must not take down the turn
                    message = (f"timed out after {WORKER_TIMEOUT_S:.0f} s" if isinstance(error, TimeoutError)
                               else f"{type(error).__name__}: {error}")[:300]
                    emit({"type": "error", "agent": name, "provider": route.provider, "stage": "agent",
                          "message": message})
                    if attempt == 0:  # tell the client to discard any partially streamed answer
                        emit({"type": "answer_reset", "agent": name, "reason": f"{route.provider} failed; "
                              f"retrying on {runtime.fallbacks[name].provider}"})
            failed = {"agent": name, "provider": None, "model": None, "answer": "", "tools": [], "ms": {},
                      "error": message}
            return {"completed": [*state.get("completed", []), failed]}
        return node

    async def finish(state: AgentState) -> dict:
        earlier = [r["answer"] for r in state.get("completed", []) if r.get("answer")]
        answer = state.get("answer") or state.get("reply") or (earlier[-1] if earlier else FALLBACK_ANSWER)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        citations = [
            {k: s[k] for k in ("n", "kind", "key", "title", "url")} | {"snippet": s["text"][:300]}
            for s in state.get("sources", []) if s["n"] in cited
        ]
        return {"messages": [AIMessage(answer)], "answer": answer, "citations": citations}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("supervisor", supervisor)
    for name in WORKERS:
        graph.add_node(name, worker(name))
        graph.add_edge(name, "supervisor")
    graph.add_node("finish", finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next"], {**{n: n for n in WORKERS}, "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
