"""Supervisor orchestration as a LangGraph state graph.

    START -> prepare -> supervisor -> research | summary | data -> supervisor -> ... -> finish -> END

The supervisor makes one LLM call per turn to produce a plan (an ordered
list of agents plus a reason), stored in state. Gemini plans by default. If it
errors (for example a 429 from Vertex AI's shared capacity), a fast Nemotron
model plans at once; if it is merely slow, Nemotron starts after HEDGE_AFTER_S
and runs alongside it (a hedged request), the first plan back wins, and the
other call is cancelled. The trace shows which router decided and why. Agents get
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
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from src.agents import prompts
from src.agents.events import emit
from src.agents.plan import RoutePlan
from src.agents.runtime import Runtime
from src.agents.workers import WORKERS, recent_history, run_worker
from src.llm import PROVIDER_NAMES, call_options, describe_failure

MAX_PLAN = 3
HEDGE_AFTER_S = 4.0  # Gemini usually plans in 1.5-3 s; past 4 s, start the backup router alongside it
SUPERVISOR_TIMEOUT_S = 12.0  # hard limit on the primary router, which keeps running after the hedge starts
FALLBACK_TIMEOUT_S = 10.0  # hard limit on the backup router
WORKER_TIMEOUT_S = 90.0  # per provider attempt, so a hung call fails over instead of stalling
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
PLAN_HISTORY_CHARS = 300  # per earlier message: enough to resolve "it" or "the left one"
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


def normalize_plan(raw) -> RoutePlan:
    plan = raw if isinstance(raw, RoutePlan) else RoutePlan.model_validate(raw)
    agents = list(dict.fromkeys(a for a in plan.agents if a in WORKERS))[:MAX_PLAN]
    if not agents and not plan.reply:
        agents = ["research"]
    return RoutePlan(agents=agents, reason=plan.reason, standalone=plan.standalone,
                     reply=None if agents else plan.reply)


def planner_request(messages: list[AnyMessage], question: str) -> HumanMessage:
    """The router's input: earlier turns as clipped, labeled context, then the one message to route.

    Passed as raw chat turns, history let the small backup router plan for an earlier question
    as if it were part of the new one, so the message to route is now stated explicitly.
    """
    lines = []
    for message in recent_history(messages):
        text = re.sub(r"\s*\[\d+\]", "", message.text).strip()  # citation markers mean nothing here
        if len(text) > PLAN_HISTORY_CHARS:
            text = text[:PLAN_HISTORY_CHARS].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: {text}")
    context = ("Earlier conversation (already answered; use it only to resolve references in the "
               "latest message):\n" + "\n".join(lines) + "\n\n") if lines else ""
    return HumanMessage(f"{context}Latest message, the only one to route:\n{question}")


def snippet(source: dict, length: int = 300) -> str:
    """Readable preview of a source: no chunk breadcrumb, no markdown heading marks."""
    text = source["text"]
    if source["kind"] == "textbook" and "#" in source["key"]:  # a chunk: first paragraph is its breadcrumb
        text = text.split("\n\n", 1)[-1]
    return re.sub(r"(^|\n)#{2,6} ", r"\1", text).strip()[:length]


def build_graph(runtime: Runtime, checkpointer=None):
    async def prepare(state: AgentState) -> dict:
        return {
            "question": state["messages"][-1].text, "standalone": "", "plan": None, "route_reason": "",
            "reply": None, "next": "", "completed": [], "sources": [], "answer": "", "citations": [],
        }

    async def ask_planner(route, llm, messages: list, trigger: str | None = None) -> RoutePlan:
        emit({**route.as_event(), **({"trigger": trigger} if trigger else {})})
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
        messages = [SystemMessage(prompts.SUPERVISOR), planner_request(state["messages"], state["question"])]
        (primary, primary_llm), (backup, backup_llm) = runtime.attempts("supervisor")
        primary_name = PROVIDER_NAMES[primary.provider]
        running: dict[asyncio.Task, tuple] = {}  # task -> (route, start time)

        async def attempt(route, llm, timeout: float, trigger: str | None):
            """Returns (plan, message, detail) rather than raising, so one router failing never stops the other."""
            try:
                return await asyncio.wait_for(ask_planner(route, llm, messages, trigger), timeout), None, None
            except Exception as error:  # rate limit, timeout, or malformed output
                return None, describe_failure(error, route.provider, timeout), str(error)

        def start(route, llm, timeout: float, trigger: str | None = None) -> None:
            running[asyncio.create_task(attempt(route, llm, timeout, trigger))] = (route, time.perf_counter())

        def elapsed_ms(started: float) -> int:
            return round((time.perf_counter() - started) * 1000)

        start(primary, primary_llm, SUPERVISOR_TIMEOUT_S)
        backup_started = False
        try:
            while running:
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED,
                                             timeout=None if backup_started else HEDGE_AFTER_S)
                if not done:  # the primary is still working: hedge with the backup instead of waiting
                    start(backup, backup_llm, FALLBACK_TIMEOUT_S,
                          f"{primary_name} had not answered after {HEDGE_AFTER_S:g} s, so both ran "
                          "and the first plan back was used")
                    backup_started = True
                    continue
                finished = [(running.pop(task), task.result()) for task in done]
                for (route, started), (plan, message, detail) in finished:
                    if plan is None:
                        emit({"type": "error", "agent": "supervisor", "provider": route.provider, "stage": "plan",
                              "message": message, "detail": detail[:600], "ms": elapsed_ms(started)})
                winners = [(route, started, plan) for (route, started), (plan, _, _) in finished if plan is not None]
                if winners:
                    route, started, plan = min(winners, key=lambda w: w[0].fallback)  # primary wins a tie
                    for loser, (other, other_started) in running.items():
                        loser.cancel()
                        emit({"type": "llm_cancelled", "agent": "supervisor", "provider": other.provider,
                              "model": other.model, "stage": "plan", "ms": elapsed_ms(other_started),
                              "reason": f"{PROVIDER_NAMES[route.provider]} answered first"})
                    emit({"type": "agent_plan",
                          "planner": f"{route.provider} {'fallback' if route.fallback else 'primary'}",
                          "hedged": backup_started, "plan": plan.agents, "reason": plan.reason,
                          "standalone": plan.standalone, "ms": elapsed_ms(started)})
                    return plan
                if not backup_started:  # the primary failed outright: fail over now
                    start(backup, backup_llm, FALLBACK_TIMEOUT_S, f"{primary_name} failed: {finished[0][1][1]}")
                    backup_started = True
        finally:
            for task in running:
                task.cancel()
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
                started = time.perf_counter()
                try:
                    return await asyncio.wait_for(run_worker(WORKERS[name], state, runtime, route, llm),
                                                  WORKER_TIMEOUT_S)
                except Exception as error:  # one provider failing must not take down the turn
                    message = describe_failure(error, route.provider, WORKER_TIMEOUT_S)
                    emit({"type": "error", "agent": name, "provider": route.provider, "stage": "agent",
                          "message": message, "detail": str(error)[:600],
                          "ms": round((time.perf_counter() - started) * 1000)})
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
        citations = [{k: s[k] for k in ("n", "kind", "key", "title", "url")} | {"snippet": snippet(s)}
                     for s in state.get("sources", []) if s["n"] in cited]
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
