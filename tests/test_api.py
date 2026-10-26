"""API tests: SSE framing, session memory, failure handling, validation, health, static UI."""

import json
import logging

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agents.graph import RoutePlan
from src.api.app import configure_logging, create_app
from tests.helpers import ScriptedModel, make_runtime


def parse(body: str) -> list[tuple[str, dict]]:
    frames = []
    for block in body.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        frames.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return frames


def client_for(runtime) -> TestClient:
    async def factory(settings):
        return runtime
    return TestClient(create_app(runtime_factory=factory))


def ask(client: TestClient, query: str, session: str = "s1") -> list[tuple[str, dict]]:
    response = client.post("/api/chat", json={"query": query, "session_id": session})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse(response.text)


def research_runtime(index, *answers: str, supervisor: ScriptedModel | None = None):
    """A runtime that always routes to Research, which answers with `answers` in order."""
    replies = [message for answer in answers for message in (AIMessage("DONE"), AIMessage(answer))]
    supervisor = supervisor or ScriptedModel(plan=RoutePlan(agents=["research"], reason="An anatomy question."))
    return make_runtime(index, research=ScriptedModel(replies=replies), supervisor=supervisor)


def test_chat_streams_tokens_then_trace_citations_and_done(index):
    with client_for(research_runtime(index, "The heart pumps blood [1].")) as client:
        frames = ask(client, "What does the heart do?")

    names = [name for name, _ in frames]
    assert "".join(d["text"] for n, d in frames if n == "token") == "The heart pumps blood [1]."
    assert "final" not in names  # the streamed tokens already are the answer
    assert all("t_ms" in d for n, d in frames if n == "trace")
    assert {d["type"] for n, d in frames if n == "trace"} >= {"llm_route", "agent_plan", "agent_route", "rag_retrieval"}
    assert next(d for n, d in frames if n == "citation")["n"] == 1
    name, done = frames[-1]
    assert name == "done" and done["plan"] == ["research"] and done["ttft_ms"] <= done["total_ms"]
    assert done["agents"][0]["provider"] == "nvidia"


def test_session_id_keys_conversation_memory(index):
    supervisor = ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy."))
    runtime = research_runtime(index, "Four chambers [1].", "The left ventricle [1].", "Fresh start [1].",
                               supervisor=supervisor)
    with client_for(runtime) as client:
        ask(client, "How many chambers does the heart have?", session="alpha")
        ask(client, "Which one pumps to the body?", session="alpha")
        same_session = " ".join(m.content for m in supervisor.seen[-1])
        ask(client, "Which one pumps to the body?", session="beta")
        other_session = " ".join(m.content for m in supervisor.seen[-1])

    assert "How many chambers" in same_session
    assert "How many chambers" not in other_session


def test_done_frame_reports_instance_turn_for_cold_start_labeling(index):
    runtime = research_runtime(index, "First [1].", "Second [1].")
    with client_for(runtime) as client:
        first = ask(client, "What does the heart do?", session="a")[-1][1]
        second = ask(client, "And the lungs?", session="b")[-1][1]

    assert first["instance"]["turn"] == 1 and second["instance"]["turn"] == 2
    assert 0 <= first["instance"]["uptime_s"] <= second["instance"]["uptime_s"]


def test_direct_reply_is_sent_as_one_final_frame(index):
    runtime = make_runtime(index, supervisor=ScriptedModel(
        plan=RoutePlan(agents=[], reason="A greeting.", reply="Hi! Ask me about the body.")))
    with client_for(runtime) as client:
        frames = ask(client, "hello")

    assert [d["text"] for n, d in frames if n == "final"] == ["Hi! Ask me about the body."]
    assert not [n for n, _ in frames if n == "token"]


def test_failover_is_visible_in_the_stream(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[AIMessage("DONE"), RuntimeError("[504] deadline exceeded")]),
        research_fallback=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Recovered answer [1].")]),
    )
    with client_for(runtime) as client:
        frames = ask(client, "What does the heart do?")

    trace_types = [d["type"] for n, d in frames if n == "trace"]
    assert trace_types.index("error") < trace_types.index("answer_reset")
    assert "".join(d["text"] for n, d in frames if n == "token") == "Recovered answer [1]."
    assert frames[-1][1]["agents"][0]["provider"] == "google"


def test_unexpected_failure_becomes_an_error_frame(index):
    class BrokenGraph:
        async def astream(self, *args, **kwargs):
            raise RuntimeError("checkpointer unavailable")
            yield  # makes this an async generator

    with client_for(make_runtime(index)) as client:
        client.app.state.graph = BrokenGraph()
        frames = ask(client, "What does the heart do?")

    assert [n for n, _ in frames] == ["error", "done"]
    assert "RuntimeError" in frames[0][1]["message"]


def test_invalid_requests_are_rejected(index):
    with client_for(make_runtime(index)) as client:
        assert client.post("/api/chat", json={"query": "", "session_id": "s1"}).status_code == 422
        assert client.post("/api/chat", json={"query": "x" * 2001, "session_id": "s1"}).status_code == 422
        assert client.post("/api/chat", json={"query": "hi", "session_id": "../etc"}).status_code == 422


def test_health_reports_routes_and_fallbacks(index):
    with client_for(make_runtime(index)) as client:
        health = client.get("/api/health").json()

    assert health["status"] == "ok" and health["web_search"] is False
    assert health["routes"]["supervisor"]["provider"] == "google"
    assert health["fallbacks"]["supervisor"]["provider"] == "nvidia"


def test_ui_is_served_as_static_files(index):
    with client_for(make_runtime(index)) as client:
        page = client.get("/")
        script = client.get("/app.js")

    assert page.status_code == 200 and "Health knowledge agents" in page.text
    assert script.status_code == 200 and "text/event-stream" in script.text


def test_request_urls_are_not_logged():
    configure_logging()  # the Tavily MCP URL carries the API key as a query parameter
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpx2").getEffectiveLevel() >= logging.WARNING
