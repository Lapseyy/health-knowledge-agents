"""Evaluation harness tests: trace breakdown, SSE parsing, the generate/score/router/report steps
on scripted models, and the deployed latency probe against a fake server. No network.

The RAGAS tests use the real ragas 0.4.3 metric classes with a scripted judge; they are skipped
when requirements-eval.txt isn't installed.
"""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import scripts.evaluate as evaluate
import scripts.measure_latency as measure_latency
from src.agents.graph import RoutePlan, build_graph
from src.evaluation import SSEReader, breakdown, parse_sse, plan_backup, plan_wall_ms
from tests.helpers import ScriptedModel, make_runtime

TOKENS = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}


def test_sse_reader_handles_frames_split_across_chunks():
    stream = 'event: token\ndata: {"text": "Hel"}\n\nevent: token\ndata: {"text": "lo"}\n\nevent: done\ndata: {"ttft_ms": 5}\n\n'
    reader, frames = SSEReader(), []
    for i in range(0, len(stream), 7):  # arbitrary network chunking
        frames += reader.feed(stream[i:i + 7])
    assert frames == [("token", {"text": "Hel"}), ("token", {"text": "lo"}), ("done", {"ttft_ms": 5})]
    assert parse_sse('event: done\ndata: {"total_ms": 9}') == [("done", {"total_ms": 9})]


def test_breakdown_sums_time_and_tokens_per_model():
    events = [
        {"type": "agent_plan", "planner": "google primary", "model": "gemini-3.8-flash", "hedged": False, "ms": 2100,
         "input_tokens": 900, "output_tokens": 60, "output_chars": 120, "plan": ["research"]},
        {"type": "llm_route", "agent": "research", "fallback": False},
        {"type": "rag_retrieval", "agent": "research", "query": "q", "ms": 6},
        {"type": "llm_call", "agent": "research", "provider": "nvidia", "model": "nvidia/ultra", "stage": "tools",
         "ms": 800, "input_tokens": 3000, "output_tokens": 40, "output_chars": 90},
        {"type": "mcp_tool_call", "tool": "tavily_search", "ms": 2000},
        {"type": "mcp_tool_call", "tool": "tavily_search", "ms": 2200},
        {"type": "llm_call", "agent": "research", "provider": "nvidia", "model": "nvidia/ultra", "stage": "answer",
         "ms": 4000, "output_chars": 1500},
    ]
    b = breakdown(events)
    assert (b["plan_ms"], b["planner"], b["retrieval_ms"], b["web_ms"], b["web_searches"]) == (
        2100, "google primary", 6, 4200, 2)
    assert b["llm_ms"] == 4800 and b["llm_by_stage"]["answer"] == {"calls": 1, "ms": 4000}
    assert b["tokens_by_model"]["gemini-3.8-flash"]["input_tokens"] == 900
    ultra = b["tokens_by_model"]["nvidia/ultra"]
    assert (ultra["calls"], ultra["input_tokens"], ultra["unreported"]) == (2, 3000, 1)
    assert b["plan_backup"] is None and b["errors"] == 0 and b["error_messages"] == []


def test_breakdown_tells_a_slow_primary_router_from_a_failed_one():
    plan = {"type": "agent_plan", "planner": "nvidia fallback", "model": "nvidia/lightning", "ms": 900}
    failure = {"type": "error", "agent": "supervisor", "stage": "plan", "message": "rate-limited (429)"}
    current = breakdown([failure, {**plan, "hedged": False, "backup": "failed", "wall_ms": 1500}])
    assert (current["plan_backup"], current["hedged"], current["plan_wall_ms"]) == ("failed", False, 1500)
    assert current["error_messages"] == ["supervisor plan: rate-limited (429)"]
    # Older traces set hedged whenever the backup ran; an earlier plan error means the primary failed.
    assert breakdown([failure, {**plan, "hedged": True}])["plan_backup"] == "failed"
    older_slow = breakdown([{**plan, "hedged": True}])
    assert (older_slow["plan_backup"], older_slow["hedged"]) == ("slow", True)
    # Breakdowns saved before plan_backup existed: inferred from the planner, errors, and agent fallbacks.
    saved = {"hedged": True, "planner": "nvidia fallback", "errors": 1, "agent_fallbacks": [], "plan_ms": 900}
    assert plan_backup(saved) == "failed"
    assert plan_backup({**saved, "agent_fallbacks": ["research"]}) == "slow"  # the error was the agent's
    assert plan_backup({**saved, "planner": "google primary"}) == "slow"  # the primary's plan was used
    assert plan_backup({**saved, "hedged": False}) is None
    # Planning wall time: reported, or estimated as the hedge wait plus the winning call.
    assert plan_wall_ms(current) == 1500
    assert plan_wall_ms(older_slow) == 900 + 4000
    assert plan_wall_ms(breakdown([failure, {**plan, "hedged": True}])) == 900


def test_answer_one_records_answer_contexts_and_timing(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research=ScriptedModel(replies=[AIMessage("DONE", usage_metadata=TOKENS),
                                        AIMessage("The heart pumps blood [1].", usage_metadata=TOKENS)]),
    )
    graph = build_graph(runtime, InMemorySaver())
    item = {"id": "q01", "question": "What does the heart do?", "reference": "It pumps blood."}
    record = asyncio.run(evaluate.answer_one(graph, item))

    assert record["answer"] == "The heart pumps blood [1]." and record["cited"] == [1]
    assert record["contexts"] and all(isinstance(c, str) and c for c in record["contexts"])
    assert len(record["contexts"]) == len(record["sources"]) == runtime.settings.retrieval_k  # the prefetched chunks
    assert record["plan"] == ["research"] and record["ttft_ms"] is not None and record["total_ms"] >= record["ttft_ms"]
    assert record["breakdown"]["tokens_by_model"][runtime.routes["research"].model]["input_tokens"] == 2000


def test_research_on_gemini_swaps_primary_and_fallback(index):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["research"], reason="Anatomy.")),
        research_fallback=ScriptedModel(replies=[AIMessage("DONE"), AIMessage("Gemini's answer [1].")]),
    )
    evaluate.use_gemini_for_research(runtime)
    graph = build_graph(runtime, InMemorySaver())
    record = asyncio.run(evaluate.answer_one(graph, {"id": "q01", "question": "What does the heart do?",
                                                     "reference": "r"}))

    assert runtime.routes["research"].provider == "google" and not runtime.routes["research"].fallback
    assert runtime.fallbacks["research"].provider == "nvidia" and runtime.fallbacks["research"].fallback
    assert record["answer"] == "Gemini's answer [1]." and record["agents"][0]["provider"] == "google"


def test_router_comparison_scores_both_routers(index, tmp_path, monkeypatch):
    runtime = make_runtime(
        index,
        supervisor=ScriptedModel(plan=RoutePlan(agents=["data"], reason="A count.")),
        supervisor_fallback=ScriptedModel(replies=[AIMessage('{"agents": ["research"], "reason": "Guess."}',
                                                             usage_metadata=TOKENS)]),
    )

    async def fake_build_runtime(settings):
        return runtime

    routing = tmp_path / "routing.jsonl"
    routing.write_text(json.dumps({"id": "r01", "history": [{"role": "user", "text": "Hi"}],
                                   "message": "How many patients have diabetes?", "expected": ["data"]}) + "\n")
    monkeypatch.setattr(evaluate, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(evaluate, "ROUTING_SET", routing)
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    args = type("Args", (), {"limit": 0, "pause": 0, "timeout": 5})()
    asyncio.run(evaluate.router(args))

    rows = {r["router"]: r for r in evaluate.read_jsonl(tmp_path / "router.jsonl")}
    assert rows["gemini"]["correct"] and rows["gemini"]["input_tokens"] == 900  # usage from include_raw
    assert not rows["nemotron-lightning"]["correct"] and rows["nemotron-lightning"]["input_tokens"] == 1000
    sent = runtime.models["supervisor"].seen[-1][-1].content  # the production router input format
    assert sent.startswith("Earlier conversation") and sent.endswith("How many patients have diabetes?")


def test_report_prints_tables_from_saved_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    b = breakdown([{"type": "agent_plan", "planner": "google primary", "model": "gemini-3.8-flash", "ms": 2000,
                    "input_tokens": 1000, "output_tokens": 100},
                   {"type": "mcp_tool_call", "tool": "tavily_search", "ms": 1500}])
    for i in range(3):
        evaluate.append_jsonl(tmp_path / "default.jsonl", {"id": f"q{i}", "ttft_ms": 4000 + i, "total_ms": 9000,
                                                           "breakdown": b})
        evaluate.append_jsonl(tmp_path / "default.scored.jsonl", {
            "id": f"q{i}", "scores": {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7,
                                      "context_recall": None if i == 0 else 0.6},
            "judge": {"calls": 10, "input_tokens": 20000, "output_tokens": 1000, "per_answer": True}})
    evaluate.append_jsonl(tmp_path / "router.jsonl", {"id": "r01", "router": "gemini", "model": "g", "message": "m",
                                                      "expected": ["data"], "agents": ["data"], "correct": True,
                                                      "exact": True, "error": None, "ms": 2100,
                                                      "input_tokens": 900, "output_tokens": 60})
    evaluate.append_jsonl(tmp_path / "latency.jsonl", {"category": "textbook", "cold": True, "client_ttft_ms": 20000,
                                                       "client_total_ms": 25000, "errors": [], "breakdown": b})
    evaluate.report(None)
    out = capsys.readouterr().out

    assert "0.90 ± 0.00 (n=3)" in out and "0.60 ± 0.00 (n=2)" in out  # a missing score isn't counted as 0
    assert "| 3 | 4.0 / 4.0 | 9.0 / 9.0 |" in out and "100%" in out
    # cost per query: Gemini 1,000 in + 100 out tokens, one web search, 9 s of Cloud Run
    expected = (1000 * 0.75 + 100 * 3.75) / 1e6 + 0.008 + 9 * evaluate.CLOUD_RUN_PER_SECOND + 0.4e-6
    assert f"${expected:.5f}" in out
    assert "| gemini | g | 1 | 1/1 | 1/1 | 0 |" in out and "20.0 / 20.0 (n=1)" in out
    assert "Gemini planned 1 of 1 requests" in out and "wall time median / p90 2.0 / 2.0 s" in out
    judge_cost = 3 * (20000 * 0.75 + 1000 * 3.75) / 1e6
    assert f"| 30 | ${judge_cost:.3f} |" in out


def test_report_hides_judge_usage_recorded_before_per_answer_tracking(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    scores = {m: 0.5 for m in evaluate.METRICS}
    evaluate.append_jsonl(tmp_path / "default.scored.jsonl", {"id": "q1", "scores": scores,
                                                               "judge": {"calls": 40, "input_tokens": 9, "output_tokens": 9}})
    evaluate.report(None)
    out = capsys.readouterr().out
    assert "| not measured | not measured |" in out and "scored before it was tracked per answer" in out


def test_judge_usage_is_attributed_to_the_answer_that_used_it(tmp_path, monkeypatch):
    """Answers are scored concurrently; each record must count only its own judge calls."""
    pytest.importorskip("ragas")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    calls_per_metric = {"q1": 1, "q2": 3}
    for qid in calls_per_metric:
        evaluate.append_jsonl(tmp_path / "default.jsonl", {"id": qid, "question": qid, "answer": "a [1]",
                                                           "contexts": ["c"], "reference": "r"})
    judge = evaluate.make_judge(None, chat=ScriptedModel(plan=RoutePlan(agents=["research"], reason="x"), delay=0.01))

    class Metric:
        async def ascore(self, user_input, **kwargs):
            for _ in range(calls_per_metric[user_input]):  # interleaves with the other answer's calls
                await judge.agenerate("prompt", RoutePlan)
            return type("Result", (), {"value": 1.0})()

    monkeypatch.setattr(evaluate, "make_judge", lambda settings: judge)
    monkeypatch.setattr(evaluate, "make_embeddings", lambda: None)
    monkeypatch.setattr(evaluate, "make_metrics", lambda j, embeddings: {m: Metric() for m in evaluate.METRICS})
    args = type("Args", (), {"config": "default", "limit": 0, "concurrency": 2, "retry_errors": False})()
    asyncio.run(evaluate.score(args))

    usage = {r["id"]: r["judge"] for r in evaluate.read_jsonl(tmp_path / "default.scored.jsonl")}
    assert usage["q1"] == {"calls": 4, "input_tokens": 4 * 900, "output_tokens": 4 * 60, "per_answer": True}
    assert usage["q2"]["calls"] == 12 and judge.calls == 16


def test_latency_probe_labels_cold_starts_from_the_server(monkeypatch):
    def respond(request):
        body = ('event: trace\ndata: {"type": "rag_retrieval", "agent": "research", "ms": 7}\n\n'
                'event: token\ndata: {"text": "Hi"}\n\n'
                'event: done\ndata: {"ttft_ms": 900, "total_ms": 1500, "plan": ["research"], '
                '"instance": {"turn": 1, "uptime_s": 4.2}}\n\n')
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await measure_latency.ask(client, "https://example.run.app", "textbook", "What does the heart do?")

    record = asyncio.run(run())
    assert record["cold"] and record["plan"] == ["research"] and record["server_ttft_ms"] == 900
    assert record["client_ttft_ms"] is not None and record["client_total_ms"] >= record["client_ttft_ms"]
    assert not record["errors"]
    assert record["events"] == [{"type": "rag_retrieval", "agent": "research", "ms": 7}]  # the trace is kept
    assert record["breakdown"]["retrievals"] == 1


# --- RAGAS wiring (real ragas 0.4.3 metric classes, scripted judge) ----------

def scripted_answers():
    from ragas.metrics.collections.answer_relevancy.util import AnswerRelevanceOutput
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput
    from ragas.metrics.collections.context_recall.util import ContextRecallClassification, ContextRecallOutput
    from ragas.metrics.collections.faithfulness.util import (NLIStatementOutput, StatementFaithfulnessAnswer,
                                                             StatementGeneratorOutput)

    return {
        StatementGeneratorOutput: StatementGeneratorOutput(statements=["The heart pumps blood."]),
        NLIStatementOutput: NLIStatementOutput(statements=[StatementFaithfulnessAnswer(
            statement="The heart pumps blood.", reason="Stated in context 1.", verdict=1)]),
        AnswerRelevanceOutput: AnswerRelevanceOutput(question="What does the heart do?", noncommittal=0),
        ContextPrecisionOutput: ContextPrecisionOutput(reason="Useful.", verdict=1),
        ContextRecallOutput: ContextRecallOutput(classifications=[ContextRecallClassification(
            statement="It pumps blood.", reason="In context.", attributed=1)]),
    }


class SchemaJudgeModel(ScriptedModel):
    """A chat model whose structured output returns a canned instance of whatever schema is asked for."""

    answers: dict = {}

    def with_structured_output(self, schema, include_raw: bool = False, **kwargs):
        from langchain_core.runnables import RunnableLambda

        async def respond(prompt):
            raw = AIMessage("{}", usage_metadata={"input_tokens": 500, "output_tokens": 20, "total_tokens": 520})
            return {"raw": raw, "parsed": self.answers[schema], "parsing_error": None}
        return RunnableLambda(lambda p: None, afunc=respond)


def test_ragas_metrics_score_through_the_langchain_judge():
    pytest.importorskip("ragas")
    from ragas.embeddings.base import BaseRagasEmbedding

    class SameVector(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return [1.0, 0.0, 0.0]

        async def aembed_text(self, text, **kwargs):
            return [1.0, 0.0, 0.0]

    judge = evaluate.make_judge(None, chat=SchemaJudgeModel(answers=scripted_answers()))
    metrics = evaluate.make_metrics(judge, SameVector())
    record = {"question": "What does the heart do?", "answer": "The heart pumps blood [1].",
              "contexts": ["The heart pumps blood through the body.", "Valves prevent backflow."],
              "reference": "It pumps blood."}
    scores = asyncio.run(evaluate.score_one(metrics, record))

    assert scores == pytest.approx({"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0,
                                    "context_recall": 1.0})  # RAGAS adds a tiny epsilon to precision
    # 2 faithfulness calls, 3 relevancy questions, 1 precision call per context, 1 recall call
    assert judge.calls == 2 + 3 + 2 + 1 and judge.input_tokens == 500 * judge.calls


def test_judge_retries_then_reports_the_error(monkeypatch):
    pytest.importorskip("ragas")
    from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput

    monkeypatch.setattr(evaluate, "JUDGE_BACKOFF_S", 0)
    judge = evaluate.make_judge(None, chat=ScriptedModel(plan=RuntimeError("429 Resource exhausted")))
    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(judge.agenerate("prompt", ContextPrecisionOutput))


def test_unanswered_question_is_not_scored():
    record = {"question": "q", "answer": "", "contexts": [], "reference": "r"}
    scores = asyncio.run(evaluate.score_one({}, record))
    assert all(scores[m] is None for m in evaluate.METRICS) and scores["_errors"]


def test_retry_errors_rescores_only_answers_with_a_failed_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    for i in (1, 2):
        evaluate.append_jsonl(tmp_path / "default.jsonl", {"id": f"q{i}", "question": "q", "answer": "a [1]",
                                                           "contexts": ["c"], "reference": "r"})
    good = {m: 0.5 for m in evaluate.METRICS}
    evaluate.append_jsonl(tmp_path / "default.scored.jsonl", {"id": "q1", "scores": good, "judge": {}})
    evaluate.append_jsonl(tmp_path / "default.scored.jsonl", {
        "id": "q2", "scores": {**good, "faithfulness": None, "_errors": {"faithfulness": "429"}}, "judge": {}})

    class Judge:
        calls = input_tokens = output_tokens = 0

    class Metric:
        async def ascore(self, **kwargs):
            return type("Result", (), {"value": 1.0})()

    monkeypatch.setattr(evaluate, "make_judge", lambda settings: Judge())
    monkeypatch.setattr(evaluate, "make_embeddings", lambda: None)
    monkeypatch.setattr(evaluate, "make_metrics", lambda judge, embeddings: {m: Metric() for m in evaluate.METRICS})
    args = type("Args", (), {"config": "default", "limit": 0, "concurrency": 2, "retry_errors": True})()
    asyncio.run(evaluate.score(args))

    rows = {r["id"]: r["scores"] for r in evaluate.read_jsonl(tmp_path / "default.scored.jsonl")}
    assert rows["q1"] == good  # untouched
    assert rows["q2"] == {m: 1.0 for m in evaluate.METRICS}  # re-scored in full, error gone


def test_report_lists_missing_scores_after_the_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    evaluate.append_jsonl(tmp_path / "default.scored.jsonl", {
        "id": "q7", "scores": {"faithfulness": None, "answer_relevancy": 0.9, "context_precision": 0.8,
                               "context_recall": 0.7, "_errors": {"faithfulness": "GoogleRateLimitError: 429"}},
        "judge": {"calls": 5, "input_tokens": 1000, "output_tokens": 50}})
    evaluate.report(None)
    lines = capsys.readouterr().out.splitlines()

    table_end = max(i for i, line in enumerate(lines[:12]) if line.startswith("|"))
    note = next(i for i, line in enumerate(lines) if "q7: faithfulness not scored" in line)
    assert note > table_end and not lines[table_end + 1].startswith("|")  # the table isn't split by the note
