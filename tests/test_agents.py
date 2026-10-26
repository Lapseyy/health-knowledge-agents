"""Graph tests with scripted LLMs: routing, tools, failure handling, and memory.

No network calls: chat models are replaced by ScriptedModel, which returns
pre-written replies in order, and the vector index uses fake embeddings.
SQL tests run against the real synthetic patient database.
"""

import asyncio
import json
import sqlite3
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import src.agents.graph as graph_module
from src.agents.graph import (FALLBACK_ANSWER, HEDGE_AFTER_S, PLAN_HISTORY_CHARS, RoutePlan, build_graph,
                              normalize_plan, planner_request)
from src.agents.runtime import Runtime, warm_up
from src.agents.tools import run_readonly_query
from src.agents.web import TavilyMCP, parse_results
from src.config import get_settings
from src.llm import describe_failure, fallback_table, routing_table, tool_round_options
from tests.helpers import TAVILY_TEXT, ScriptedModel, fake_web, make_runtime, tool_call

SETTINGS = get_settings()


def run_turn(runtime: Runtime, question: str, thread: str = "t1", checkpointer=None):
    """Runs one turn; returns (final state, trace events, (chunk, metadata) token pairs)."""
    graph = build_graph(runtime, checkpointer or InMemorySaver())
    config = {"configurable": {"thread_id": thread}}

    async def go():
        events, tokens = [], []
        async for mode, item in graph.astream({"messages": [HumanMessage(question)]}, config,
                                              stream_mode=["custom", "messages"]):
            (events if mode == "custom" else tokens).append(item)
        return (await graph.aget_state(config)).values, events, tokens

    return asyncio.run(go())


def types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def test_routing_table_uses_two_providers_with_cross_provider_fallbacks():
    routes, fallbacks = routing_table(SETTINGS), fallback_table(SETTINGS)
    assert {r.provider for r in routes.values()} == {"google", "nvidia"}
    assert routes["supervisor"].provider == "google" and routes["research"].provider == "nvidia"
    assert set(fallbacks) == set(routes)
    assert all(fallbacks[a].provider != routes[a].provider and fallbacks[a].fallback for a in routes)
    assert all(r.reason and r.dimension in {"cost", "capability", "hybrid"} for r in routes.values())


def test_research_turn_produces_cited_answer(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="An anatomy question.")),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("The heart pumps blood [1].")]),
    )
    state, events, tokens = run_turn(runtime, "What does the heart do?")

    assert state["messages"][-1].content == "The heart pumps blood [1]."
    assert state["citations"][0]["kind"] == "textbook" and state["citations"][0]["n"] == 1
    routes = {e["agent"]: e["provider"] for e in events if e["type"] == "llm_route"}
    assert routes == {"supervisor": "google", "research": "nvidia"}  # both providers in a normal turn
    assert {"agent_plan", "agent_route", "rag_retrieval"} <= set(types(events))
    calls = [(e["agent"], e["provider"], e["stage"]) for e in events if e["type"] == "llm_call"]
    assert calls == [("research", "nvidia", "tools"), ("research", "nvidia", "answer")]  # every model call is timed
    answer_tokens = [c for c, meta in tokens if "answer" in meta.get("tags", [])]
    assert answer_tokens and "".join(c.text for c in answer_tokens) == "The heart pumps blood [1]."


def test_two_agent_plan_runs_real_sql_and_last_agent_answers(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research", "data"], reason="Explain, then count.")),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Hypertension is high blood pressure [1].")]),
        data=ScriptedModel(replies=[
            tool_call("run_sql", query="SELECT COUNT(*) AS n FROM patients"),
            AIMessage("DONE"),
            AIMessage("The database has 1126 synthetic patients."),
        ]),
    )
    state, events, tokens = run_turn(runtime, "What is hypertension and how many patients are there?")

    assert [r["agent"] for r in state["completed"]] == ["research", "data"]
    sql = next(e for e in events if e["type"] == "sql_query")
    assert sql["status"] == "ok" and sql["rows"] == 1
    assert state["answer"] == "The database has 1126 synthetic patients."
    tags = {tag for _, meta in tokens for tag in meta.get("tags", [])}
    assert {"intermediate", "answer"} <= tags  # research's draft is not streamed as the answer
    data_prompt = runtime.models["data"].seen[-1][-1].content
    assert "Hypertension is high blood pressure" in data_prompt  # later agents see earlier findings


def test_agent_fails_over_to_the_other_provider(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[AIMessage("DONE"), RuntimeError("[504] deadline exceeded")]),
        research_fallback=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("The heart pumps blood [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert state["answer"] == "The heart pumps blood [1]."
    assert state["completed"][0]["provider"] == "google"  # the fallback's answer is the one kept
    kinds = types(events)
    assert kinds.index("error") < kinds.index("answer_reset")  # client is told to drop the partial answer
    research_routes = [e for e in events if e["type"] == "llm_route" and e["agent"] == "research"]
    assert [(e["provider"], e["fallback"]) for e in research_routes] == [("nvidia", False), ("google", True)]


def test_failed_agent_on_both_providers_does_not_crash_the_turn(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[RuntimeError("[503] provider down")]),
        research_fallback=ScriptedModel(replies=[RuntimeError("429 resource exhausted")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert state["answer"] == FALLBACK_ANSWER
    assert [(e["provider"], e["message"]) for e in events if e["type"] == "error"] == [
        ("nvidia", "provider unavailable (503)"),
        ("google", "rate-limited (429) because Vertex AI's shared Gemini capacity was busy")]
    assert state["completed"][0]["error"].startswith("rate-limited (429)")


def test_supervisor_failure_uses_the_fast_fallback_router(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RuntimeError("429 Too Many Requests: Resource exhausted")),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('Plan: {"agents": ["data"], "reason": "A count."}')]),
        data=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("There are 1126 patients.")]),
    )
    started = time.perf_counter()
    state, events, _ = run_turn(runtime, "How many patients are there?")

    assert time.perf_counter() - started < HEDGE_AFTER_S  # failed over at once, no waiting for the hedge
    assert state["plan"] == ["data"]
    error = next(e for e in events if e["type"] == "error")
    assert error["agent"] == "supervisor" and error["message"].startswith("rate-limited (429) because Vertex AI")
    assert "Too Many Requests" in error["detail"]  # the raw error stays available in the trace
    plan_event = next(e for e in events if e["type"] == "agent_plan")
    assert plan_event["planner"] == "nvidia fallback"
    assert plan_event["backup"] == "failed" and plan_event["hedged"] is False  # failed over, not hedged
    assert plan_event["wall_ms"] >= plan_event["ms"]
    supervisor_routes = [e for e in events if e["type"] == "llm_route" and e["agent"] == "supervisor"]
    assert [(e["provider"], e["fallback"]) for e in supervisor_routes] == [("google", False), ("nvidia", True)]
    assert supervisor_routes[1]["trigger"].startswith("Gemini failed: rate-limited (429)")


def test_slow_supervisor_is_hedged_and_the_faster_plan_wins(index, monkeypatch):
    monkeypatch.setattr(graph_module, "HEDGE_AFTER_S", 0.05)
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["summary"], reason="Slow."), delay=1.0),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["research"], "reason": "Anatomy."}')]),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    started = time.perf_counter()
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert time.perf_counter() - started < 1.0  # did not wait for the slow router
    assert state["plan"] == ["research"] and state["answer"] == "Answer [1]."
    backup = [e for e in events if e["type"] == "llm_route" and e["agent"] == "supervisor"][1]
    assert "had not answered after 0.05 s" in backup["trigger"]
    cancelled = next(e for e in events if e["type"] == "llm_cancelled")
    assert cancelled["provider"] == "google" and cancelled["reason"] == "Nemotron answered first"
    plan_event = next(e for e in events if e["type"] == "agent_plan")
    assert plan_event["planner"] == "nvidia fallback" and plan_event["backup"] == "slow" and plan_event["hedged"]
    assert plan_event["wall_ms"] >= 50 + plan_event["ms"] - 5  # the wait before the hedge counts toward planning
    assert not any(e["type"] == "error" for e in events)  # slow is not a failure


def test_primary_router_still_wins_if_it_answers_during_the_hedge(index, monkeypatch):
    monkeypatch.setattr(graph_module, "HEDGE_AFTER_S", 0.05)
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy."), delay=0.15),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["data"], "reason": "Slow."}')], delay=1.0),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert state["plan"] == ["research"]
    plan_event = next(e for e in events if e["type"] == "agent_plan")
    assert plan_event["planner"] == "google primary" and plan_event["hedged"] is True
    cancelled = next(e for e in events if e["type"] == "llm_cancelled")
    assert cancelled["provider"] == "nvidia" and cancelled["reason"] == "Gemini answered first"


def test_routers_that_never_answer_time_out_to_research(index, monkeypatch):
    monkeypatch.setattr(graph_module, "HEDGE_AFTER_S", 0.02)
    monkeypatch.setattr(graph_module, "SUPERVISOR_TIMEOUT_S", 0.1)
    monkeypatch.setattr(graph_module, "FALLBACK_TIMEOUT_S", 0.1)
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["data"], reason="Hung."), delay=2.0),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["data"], "reason": "Hung."}')], delay=2.0),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    started = time.perf_counter()
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert time.perf_counter() - started < 1.0
    assert state["plan"] == ["research"] and state["answer"] == "Answer [1]."
    errors = [e for e in events if e["type"] == "error"]
    assert [(e["provider"], e["message"]) for e in errors] == [
        ("google", "timed out after 0.1 s"), ("nvidia", "timed out after 0.1 s")]


def test_both_routers_failing_defaults_to_research(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=ValueError("down")),
        supervisor_fallback=ScriptedModel(replies=[AIMessage("I think research is best.")]),  # no JSON
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    assert state["plan"] == ["research"] and state["answer"] == "Answer [1]."
    assert [e["provider"] for e in events if e["type"] == "error"] == ["google", "nvidia"]


def test_tool_calls_in_one_round_run_in_parallel(index):
    calls = [{"name": "search_web", "args": {"query": f"q{i}"}, "id": f"c{i}"} for i in range(2)]
    runtime = make_runtime(
        index,
        web=fake_web(TAVILY_TEXT, delay=0.4),
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="News.")),
        research=ScriptedModel(replies=[AIMessage("", tool_calls=calls), AIMessage("DONE"), AIMessage("Done [1].")]),
    )
    started = time.perf_counter()
    _, events, _ = run_turn(runtime, "Latest heart news?")
    elapsed = time.perf_counter() - started

    assert len([e for e in events if e["type"] == "mcp_tool_call"]) == 2
    assert elapsed < 0.75  # two 0.4 s searches run concurrently, not back to back


def test_web_searches_are_capped_per_agent(index):
    calls = [{"name": "search_web", "args": {"query": f"q{i}"}, "id": f"c{i}"} for i in range(3)]
    research = ScriptedModel(replies=[AIMessage("", tool_calls=calls), AIMessage("DONE"), AIMessage("Done [1].")])
    runtime = make_runtime(index, web=fake_web(TAVILY_TEXT), research=research,
                           supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="News.")))
    _, events, _ = run_turn(runtime, "Latest heart news?")

    assert len([e for e in events if e["type"] == "mcp_tool_call"]) == 2
    tool_replies = [m.content for m in research.seen[1] if m.type == "tool"]
    assert sum("limit" in reply for reply in tool_replies) == 1


def test_tool_budget_caps_calls_per_agent(index):
    def round_of(n, start):  # run_sql has no per-tool cap, so only the overall budget applies
        return AIMessage("", tool_calls=[{"name": "run_sql", "args": {"query": f"SELECT {i}"}, "id": f"c{i}"}
                                         for i in range(start, start + n)])
    data = ScriptedModel(replies=[round_of(3, 0), round_of(3, 3), AIMessage("Answer [1].")])
    runtime = make_runtime(index, data=data,
                           supervisor=ScriptedModel(plan=RoutePlan(agents=["data"], reason="A count.")))
    state, events, _ = run_turn(runtime, "How many patients are there?")

    queries = [e for e in events if e["type"] == "sql_query"]
    assert len(queries) == 4  # the four-call budget; the last two requested calls were refused
    assert state["completed"][0]["tools"] == ["run_sql"] * 4
    assert state["answer"] == "Answer [1]."


def test_read_section_passes_the_whole_section(index):
    from src.agents.sources import SourceRegistry
    from src.agents.tools import summary_tools

    runtime = make_runtime(index)
    registry = SourceRegistry()
    read_section = next(t for t in summary_tools(runtime, registry) if t.name == "read_section")
    asyncio.run(read_section.ainvoke({"section_id": "19.3"}))

    assert registry.sources[0]["key"] == "ap2e-19.3"
    assert len(registry.sources[0]["text"]) == len(runtime.sections["ap2e-19.3"]["text"]) > 6000


def test_read_section_reads_each_section_once(index):
    from src.agents.sources import SourceRegistry
    from src.agents.tools import summary_tools

    registry = SourceRegistry()
    read_section = next(t for t in summary_tools(make_runtime(index), registry) if t.name == "read_section")
    first = asyncio.run(read_section.ainvoke({"section_id": "19.3"}))
    again = asyncio.run(read_section.ainvoke({"section_id": "ap2e-19.3"}))

    assert len(first) > 6000 and again.startswith("Already read: [1]") and len(again) < 200
    assert len(registry.sources) == 1


def test_chapter_summary_reads_every_section_in_parallel(index):
    sections = [k for k in index[2] if k.startswith("ap2e-2.")]
    summary = ScriptedModel(delay=0.2, replies=[
        tool_call("summarize_chapter", chapter=2),
        *[AIMessage("- A key point from this section.") for _ in sections],
        AIMessage("DONE"),
        AIMessage("Chapter 2 covers the chemistry of life [1][6]."),
    ])
    runtime = make_runtime(index, summary=summary,
                           supervisor=ScriptedModel(plan=RoutePlan(agents=["summary"], reason="A chapter summary.")))
    started = time.perf_counter()
    state, events, _ = run_turn(runtime, "Summarize chapter 2")
    elapsed = time.perf_counter() - started

    digests = [e for e in events if e["type"] == "llm_call" and e["stage"] == "digest"]
    assert sorted(e["section"] for e in digests) == [f"§2.{i}" for i in range(len(sections))] and len(sections) == 6
    assert elapsed < 0.2 * (3 + len(sections)) - 0.4  # the six section calls overlapped
    assert [s["key"] for s in state["sources"]] == sorted(sections, key=lambda k: int(k.split(".")[1]))
    assert all(s["context"].startswith("Digest of the full section") for s in state["sources"])
    answer_input = summary.seen[-1][-1].content
    assert answer_input.count("A key point from this section.") == 6
    longest = max(len(index[2][k]["text"]) for k in sections)
    assert len(answer_input) < longest  # the answer call reads digests, not six whole sections
    chapter_read = next(e for e in events if e["type"] == "rag_retrieval" and e["query"] == "read chapter 2")
    assert len(chapter_read["results"]) == 6
    assert state["answer"] == "Chapter 2 covers the chemistry of life [1][6]."


def test_unknown_chapter_is_reported_to_the_agent(index):
    from src.agents.sources import SourceRegistry
    from src.agents.tools import summary_tools

    tools = summary_tools(make_runtime(index), SourceRegistry(), model=(None, None))
    summarize = next(t for t in tools if t.name == "summarize_chapter")
    assert asyncio.run(summarize.ainvoke({"chapter": 42})) == "No chapter 42. The textbook has chapters 1-28."


def test_greeting_gets_direct_reply_without_agents(index):
    runtime = make_runtime(index, supervisor=ScriptedModel(
        plan=RoutePlan(agents=[], reason="A greeting.", reply="Hi! Ask me about anatomy.")))
    state, events, _ = run_turn(runtime, "hello")

    assert state["answer"] == "Hi! Ask me about anatomy." and state["completed"] == []
    assert "agent_route" not in types(events)


def test_memory_carries_across_turns_in_a_session(index):
    standalone = "Which heart chamber pumps blood to the body?"
    supervisor = ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.", standalone=standalone))
    runtime = make_runtime(index, supervisor=supervisor, research=ScriptedModel(replies=[
        AIMessage("DONE"), AIMessage("The heart has four chambers [1]."),
        AIMessage("DONE"), AIMessage("The left ventricle pumps to the body [1]."),
    ]))
    saver = InMemorySaver()
    run_turn(runtime, "How many chambers does the heart have?", thread="s1", checkpointer=saver)
    state, events, _ = run_turn(runtime, "Which one pumps to the body?", thread="s1", checkpointer=saver)

    assert len(state["messages"]) == 4
    retrieval = next(e for e in events if e["type"] == "rag_retrieval")
    assert retrieval["query"] == standalone  # follow-ups are retrieved with the resolved question
    system, request = supervisor.seen[-1]  # one labeled message, not raw chat turns
    assert "User: How many chambers" in request.content and "Assistant: The heart has four chambers." in request.content
    assert request.content.endswith("Latest message, the only one to route:\nWhich one pumps to the body?")


def test_planner_request_labels_and_clips_history():
    long_answer = "The left ventricle " + "pumps blood to the body " * 40 + "[3]"
    history = [HumanMessage("Latest BP guidelines?"), AIMessage(long_answer), HumanMessage("What does the left ventricle do?")]
    content = planner_request(history, "What does the left ventricle do?").content

    earlier, latest = content.split("Latest message, the only one to route:\n")
    assert latest == "What does the left ventricle do?"
    assert "User: Latest BP guidelines?" in earlier and "[3]" not in earlier
    assert len(earlier) < PLAN_HISTORY_CHARS + 250  # the long answer was clipped
    assert planner_request([HumanMessage("Hi")], "Hi").content == "Latest message, the only one to route:\nHi"


def test_research_gets_one_extra_textbook_search(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[
            AIMessage("", tool_calls=[{"name": "search_textbook", "args": {"query": q}, "id": f"c{i}"}
                                      for i, q in enumerate(["valves", "ventricle wall"])]),
            AIMessage("DONE"), AIMessage("Answer [1]."),
        ]),
    )
    state, events, _ = run_turn(runtime, "What does the left ventricle do?")

    searches = [e for e in events if e["type"] == "rag_retrieval" and e["agent"] == "research"]
    assert len(searches) == 2  # the prefetch plus one extra search; the second extra was refused
    tool_results = [m.content for m in runtime.models["research"].seen[1] if m.type == "tool"]
    assert any("Textbook search limit reached" in r for r in tool_results)


def test_same_mcp_tool_is_shared_by_two_agents(index):
    runtime = make_runtime(
        index,
        web=fake_web(TAVILY_TEXT),
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research", "summary"], reason="Find, then summarize.")),
        research=ScriptedModel(replies=[tool_call("search_web", query="hypertension guideline 2026"),
                                        AIMessage("DONE"), AIMessage("A new guideline exists [5].")]),
        summary=ScriptedModel(replies=[tool_call("search_web", query="2026 hypertension guideline"),
                                       AIMessage("DONE"), AIMessage("- Lower threshold [5].")]),
    )
    state, events, _ = run_turn(runtime, "Summarize the newest hypertension guideline")

    calls = [e for e in events if e["type"] == "mcp_tool_call"]
    assert {(e["agent"], e["tool"]) for e in calls} == {("research", "tavily_search"), ("summary", "tavily_search")}
    web_sources = [s for s in state["sources"] if s["kind"] == "web"]
    assert len(web_sources) == 1 and web_sources[0]["url"] == "https://example.org/bp"  # deduplicated


def test_mcp_failure_is_reported_and_survived(index):
    runtime = make_runtime(
        index,
        web=fake_web(ConnectionError("MCP transport closed")),
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Current events.")),
        research=ScriptedModel(replies=[tool_call("search_web", query="news"), AIMessage("DONE"),
                                        AIMessage("I couldn't reach the web; from the textbook [1].")]),
    )
    state, events, _ = run_turn(runtime, "Any news about heart disease?")

    call = next(e for e in events if e["type"] == "mcp_tool_call")
    assert call["status"].startswith("error")
    assert state["answer"].startswith("I couldn't reach the web")
    tool_message = runtime.models["research"].seen[1][-1]
    assert "Tool error" in tool_message.content  # the agent was told, and could adapt


def test_tavily_key_never_reaches_the_trace(index):
    leaked = ("Client error '401 Unauthorized' for url "
              "'https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-dev-SECRET123&x=1'")
    runtime = make_runtime(
        index,
        web=fake_web(RuntimeError(leaked)),
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Current events.")),
        research=ScriptedModel(replies=[tool_call("search_web", query="news"), AIMessage("DONE"),
                                        AIMessage("From the textbook [1].")]),
    )
    state, events, _ = run_turn(runtime, "Any news about heart disease?")

    call = next(e for e in events if e["type"] == "mcp_tool_call")
    assert "tavilyApiKey=***&x=1" in call["status"]
    everything = json.dumps(events, default=str) + str(runtime.models["research"].seen)
    assert "SECRET123" not in everything  # not in the UI trace, not in what the model saw


def test_warm_up_pings_each_distinct_client_and_survives_failures(index):
    shared = ScriptedModel(replies=[AIMessage("OK")])
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(replies=[AIMessage("OK")]),
        supervisor_fallback=ScriptedModel(replies=[RuntimeError("[429] Too Many Requests")]),
    )
    runtime.models.update(research=shared, summary=shared, data=shared)  # one client for one model

    timings = asyncio.run(warm_up(runtime))

    assert len(timings) == 3  # router, backup router, and the agents' shared Nemotron client
    assert any(r.startswith("failed (rate-limited (429)") for r in timings.values())
    assert shared.replies == []  # pinged once, not once per agent
    assert sum(r.endswith(" s") for r in timings.values()) == 2


def fake_mcp_client(monkeypatch, outcomes: list):
    """Replaces the MCP client: each connect attempt pops an outcome (an exception, or "ok")."""
    import langchain_mcp_adapters.client as client_module

    attempts = []

    class FakeClient:
        def __init__(self, config):
            pass

        async def get_tools(self, server_name):
            attempts.append(server_name)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return list(fake_web(TAVILY_TEXT).tools.values())

    monkeypatch.setattr(client_module, "MultiServerMCPClient", FakeClient)
    return attempts


def test_mcp_connects_on_a_freshly_booted_container(monkeypatch):
    # time.monotonic() counts from boot: a new Cloud Run container is seconds old.
    attempts = fake_mcp_client(monkeypatch, ["ok"])
    mcp = TavilyMCP("tvly-test")
    mcp.clock = lambda: 3.0

    assert asyncio.run(mcp.connect()) is True and attempts == ["tavily"]
    assert "tavily_search" in mcp.tools


def test_mcp_retries_a_failed_connection_at_most_once_a_minute(monkeypatch):
    attempts = fake_mcp_client(monkeypatch, [ConnectionError("down"), "ok"])
    mcp = TavilyMCP("tvly-test")
    now = {"t": 3.0}
    mcp.clock = lambda: now["t"]

    assert asyncio.run(mcp.connect()) is False  # first attempt fails
    now["t"] = 30.0
    assert asyncio.run(mcp.connect()) is False and len(attempts) == 1  # too soon to retry
    now["t"] = 64.0
    assert asyncio.run(mcp.connect()) is True and len(attempts) == 2


def test_web_tools_are_absent_when_mcp_is_unavailable(index):
    from src.agents.sources import SourceRegistry
    from src.agents.tools import research_tools

    names = [t.name for t in research_tools(make_runtime(index), SourceRegistry())]
    assert names == ["search_textbook"]


def test_sql_tool_is_read_only():
    db = SETTINGS.patient_db_path
    assert run_readonly_query(db, "SELECT COUNT(*) FROM patients;")["rows"] == [[1126]]
    for bad in ("DELETE FROM patients", "SELECT 1; DROP TABLE patients", "PRAGMA writable_schema = 1"):
        with pytest.raises(ValueError):
            run_readonly_query(db, bad)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        run_readonly_query(db, "WITH x AS (SELECT 1) INSERT INTO allergies (description) SELECT * FROM x")
    assert run_readonly_query(db, "SELECT id FROM patients")["truncated"] is True


def test_normalize_plan_dedupes_caps_and_defaults():
    assert normalize_plan({"agents": ["data", "data", "research"], "reason": "x"}).agents == ["data", "research"]
    assert normalize_plan({"agents": [], "reason": "x"}).agents == ["research"]
    assert normalize_plan({"agents": [], "reason": "x", "reply": "hi"}).reply == "hi"


def test_parse_tavily_results_from_text_and_json():
    assert parse_results(TAVILY_TEXT)[0]["url"] == "https://example.org/bp"
    as_json = '{"results": [{"title": "T", "url": "https://a.b", "content": "C"}]}'
    assert parse_results(as_json) == [{"title": "T", "url": "https://a.b", "content": "C"}]
    assert parse_results("no structure here") == []


def test_tool_rounds_cap_nemotron_output_only():
    routes = routing_table(SETTINGS)
    assert tool_round_options(routes["research"]) == {"thinking_mode": False, "max_tokens": 600}
    assert tool_round_options(fallback_table(SETTINGS)["research"]) == {}  # Gemini: reasoning shares the budget


def test_model_calls_report_output_size_and_tokens(index):
    tokens = {"input_tokens": 1200, "output_tokens": 3, "total_tokens": 1203}
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[AIMessage("DONE", usage_metadata=tokens), AIMessage("Answer [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    tools_call, answer_call = [e for e in events if e["type"] == "llm_call"]
    assert tools_call["output_chars"] == 4 and tools_call["input_tokens"] == 1200 and tools_call["output_tokens"] == 3
    assert answer_call["output_chars"] == len("Answer [1].") and "input_tokens" not in answer_call  # not reported
    plan = next(e for e in events if e["type"] == "agent_plan")
    assert plan["input_tokens"] == 900 and plan["output_tokens"] == 60  # survives structured parsing


def test_unparseable_gemini_plan_fails_over(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=None),  # structured output came back empty
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["research"], "reason": "Anatomy."}')]),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    error = next(e for e in events if e["type"] == "error")
    assert error["provider"] == "google" and "could not be parsed" in error["message"]
    assert next(e for e in events if e["type"] == "agent_plan")["planner"] == "nvidia fallback"


def test_failures_are_described_briefly():
    class APIError(Exception):
        code = 429

    assert describe_failure(TimeoutError(), "google", 12) == "timed out after 12 s"
    assert describe_failure(APIError("quota"), "google").startswith("rate-limited (429) because Vertex AI")
    assert describe_failure(Exception("[429] Too Many Requests"), "nvidia") == (
        "rate-limited (429) because NVIDIA's free-tier request limit was reached")
    assert describe_failure(RuntimeError("504 DEADLINE_EXCEEDED"), "google") == "provider deadline exceeded (504)"
    # a malformed router reply that happens to contain a number is not an HTTP error
    assert describe_failure(ValueError("no JSON in: 500 patients"), "nvidia") == "ValueError: no JSON in: 500 patients"
