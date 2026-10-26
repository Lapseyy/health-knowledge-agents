"""Graph tests with scripted LLMs: routing, tools, failure handling, and memory.

No network calls: chat models are replaced by ScriptedModel, which returns
pre-written replies in order, and the vector index uses fake embeddings.
SQL tests run against the real synthetic patient database.
"""

import asyncio
import sqlite3
from dataclasses import replace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agents.graph import FALLBACK_ANSWER, RoutePlan, build_graph, normalize_plan
from src.agents.runtime import Runtime, load_sections
from src.agents.tools import describe_database, run_readonly_query
from src.agents.web import TavilyMCP, parse_results
from src.config import get_settings
from src.llm import fallback_table, routing_table
from src.rag.ingest import ingest
from src.rag.retriever import load_store

SETTINGS = get_settings()


class ScriptedModel(BaseChatModel):
    """Returns queued replies in order; `plan` answers structured-output calls."""

    replies: list = Field(default_factory=list)
    plan: object = None
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self.bind(**kwargs)

    def with_structured_output(self, schema, **kwargs):
        def plan(messages):
            self.seen.append(messages)
            if isinstance(self.plan, Exception):
                raise self.plan
            return self.plan

        async def slow_plan(messages):  # plan="slow" simulates a provider that never answers in time
            if self.plan == "slow":
                await asyncio.sleep(5)
            return plan(messages)
        return RunnableLambda(plan, afunc=slow_plan)


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


def fake_web(search_result: str | Exception, delay: float = 0) -> TavilyMCP:
    async def search(query: str, max_results: int = 5) -> str:
        await asyncio.sleep(delay)
        if isinstance(search_result, Exception):
            raise search_result
        return search_result

    web = TavilyMCP(None)
    web.tools = {"tavily_search": StructuredTool.from_function(
        coroutine=search, name="tavily_search", description="Tavily web search")}
    return web


TAVILY_TEXT = ("Detailed Results:\n\nTitle: New hypertension guideline\nURL: https://example.org/bp\n"
               "Content: The 2026 guideline lowers the treatment threshold.\n")


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("index")
    sections = load_sections(SETTINGS)
    corpus = tmp / "corpus.jsonl"
    corpus.write_text("\n".join(__import__("json").dumps(sections[k]) for k in list(sections)[:4]), encoding="utf-8")
    settings = replace(SETTINGS, corpus_path=corpus, chroma_dir=tmp / "chroma")
    embeddings = DeterministicFakeEmbedding(size=64)
    ingest(settings, embeddings)
    return settings, load_store(settings, embeddings), sections


def make_runtime(index, web=None, **models: ScriptedModel) -> Runtime:
    """Scripted models per agent: pass `research=...` for a primary, `research_fallback=...` for its backup."""
    settings, store, sections = index
    routes = routing_table(settings)
    primary = {agent: models.get(agent, ScriptedModel()) for agent in routes}
    backup = {agent: models.get(f"{agent}_fallback", ScriptedModel()) for agent in routes}
    return Runtime(
        settings=settings, routes=routes, fallbacks=fallback_table(settings), models=primary,
        fallback_models=backup, store=store, sections=sections, web=web or TavilyMCP(None),
        db_schema=describe_database(settings.patient_db_path),
    )


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
    assert state["completed"][0]["error"].startswith("RuntimeError")
    assert [e["provider"] for e in events if e["type"] == "error"] == ["nvidia", "google"]


def test_supervisor_failure_uses_the_fast_fallback_router(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=ValueError("429 Resource exhausted")),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('Plan: {"agents": ["data"], "reason": "A count."}')]),
        data=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("There are 1126 patients.")]),
    )
    state, events, _ = run_turn(runtime, "How many patients are there?")

    assert state["plan"] == ["data"]
    assert any(e["type"] == "error" and e["agent"] == "supervisor" for e in events)
    plan_event = next(e for e in events if e["type"] == "agent_plan")
    assert plan_event["planner"] == "nvidia fallback"
    supervisor_routes = [e for e in events if e["type"] == "llm_route" and e["agent"] == "supervisor"]
    assert [(e["provider"], e["fallback"]) for e in supervisor_routes] == [("google", False), ("nvidia", True)]


def test_slow_supervisor_times_out_to_the_fallback(index, monkeypatch):
    import src.agents.graph as graph_module

    monkeypatch.setattr(graph_module, "SUPERVISOR_TIMEOUT_S", 0.05)
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan="slow"),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["research"], "reason": "Anatomy."}')]),
        research=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Answer [1].")]),
    )
    state, events, _ = run_turn(runtime, "What does the heart do?")

    error = next(e for e in events if e["type"] == "error")
    assert error["agent"] == "supervisor" and "timed out" in error["message"]
    assert state["answer"] == "Answer [1]."


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
    started = __import__("time").perf_counter()
    _, events, _ = run_turn(runtime, "Latest heart news?")
    elapsed = __import__("time").perf_counter() - started

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
    def round_of(n, start):
        return AIMessage("", tool_calls=[{"name": "search_textbook", "args": {"query": f"q{i}"}, "id": f"c{i}"}
                                         for i in range(start, start + n)])
    research = ScriptedModel(replies=[round_of(3, 0), round_of(3, 3), AIMessage("Answer [1].")])
    runtime = make_runtime(index, research=research,
                           supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")))
    state, events, _ = run_turn(runtime, "What does the heart do?")

    searches = [e for e in events if e["type"] == "rag_retrieval"]
    assert len(searches) == 1 + 4  # the prefetch plus the four-call budget
    assert state["completed"][0]["tools"] == ["search_textbook"] * 4
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
    second_plan_input = " ".join(m.content for m in supervisor.seen[-1])
    assert "How many chambers" in second_plan_input and "four chambers" in second_plan_input


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
