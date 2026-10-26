"""Evaluate the system: RAGAS quality, latency and tokens, and both provider comparisons.

    python -m scripts.evaluate generate                          # answer the 30 RAGAS questions
    python -m scripts.evaluate generate --config research-gemini # same, Research on Gemini
    python -m scripts.evaluate score default                     # RAGAS scores, Gemini judge
    python -m scripts.evaluate score research-gemini
    python -m scripts.evaluate router                            # Gemini vs Nemotron Lightning routing
    python -m scripts.evaluate report                            # markdown tables for docs/evaluation.md

Results go to docs/eval/. `generate`, `score`, and `router` resume where they stopped, so an
interrupted run continues instead of starting over; delete a file to redo it. Add --limit 3 for
a quick check before a full run. Needs: pip install -r requirements-eval.txt

What each step measures:
- generate runs every question through the full system in-process (supervisor, agents, tools,
  the same streaming code as the API) and saves the answer, the exact sources the answer call
  saw (RAGAS contexts), the plan, and a timing and token breakdown from the trace.
- score grades each answer with four RAGAS 0.4 metrics (faithfulness, answer relevancy,
  context precision and context recall against the book's reference answer). The judge is
  Gemini 3.8 Flash on Vertex AI at temperature 0, wrapped as a RAGAS LLM through LangChain's
  structured output; answer relevancy embeds with the same local bge-small model as retrieval.
- router sends each routing-set message to both routers with the production prompt and input
  format, and checks the agents each chose against the expected ones.
"""

import argparse
import asyncio
import contextvars
import io
import json
import math
import os
import statistics
import time
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no usage analytics from the RAGAS library

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from src.agents import prompts  # noqa: E402
from src.agents.graph import build_graph, planner_request, request_plan  # noqa: E402
from src.agents.runtime import build_runtime  # noqa: E402
from src.api.stream import stream_turn  # noqa: E402
from src.config import ROOT, get_settings  # noqa: E402
from src.evaluation import breakdown, parse_sse, plan_backup, plan_wall_ms  # noqa: E402
from src.llm import FAILOVER_REASON, Route, chat_model  # noqa: E402

OUT = ROOT / "docs" / "eval"
RAGAS_SET = ROOT / "data" / "eval_ragas.jsonl"
ROUTING_SET = ROOT / "data" / "eval_routing.jsonl"
CONFIGS = {
    "default": "As deployed: Research on Nemotron 3 Ultra (NVIDIA)",
    "research-gemini": "Provider comparison: Research on Gemini 3.8 Flash (Google)",
}
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# Prices checked Oct 26, 2026 (re-check before reusing; see docs/evaluation.md for sources).
PRICES_AS_OF = "2026-10-26"
TOKEN_PRICES = {  # USD per 1M tokens: (input, output incl. reasoning)
    "gemini-3.8-flash": (0.75, 3.75),  # Vertex AI global endpoint, introductory rate to Dec 31, 2026
}
FREE_MODELS = ("nvidia/",)  # NVIDIA API catalog free developer tier: no per-token charge
TAVILY_PER_SEARCH = 0.008  # 1 credit per basic search, pay-as-you-go; first 1,000 a month free
CLOUD_RUN_PER_SECOND = 1 * 0.000024 + 2 * 0.0000025  # 1 vCPU + 2 GiB, request-based billing, us-west1
CLOUD_RUN_PER_REQUEST = 0.40 / 1_000_000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- generate ----------------------------------------------------------------

def use_gemini_for_research(runtime) -> None:
    """Swap Research's primary and fallback, so Gemini answers and Nemotron is the backup."""
    nemotron, gemini = runtime.routes["research"], runtime.fallbacks["research"]
    runtime.routes["research"] = replace(gemini, fallback=False, dimension="evaluation",
                                         reason="Evaluation run: Research on Gemini, for the provider comparison.")
    runtime.fallbacks["research"] = replace(nemotron, fallback=True, dimension="failover", reason=FAILOVER_REASON)
    runtime.models["research"], runtime.fallback_models["research"] = (
        runtime.fallback_models["research"], runtime.models["research"])


async def answer_one(graph, item: dict) -> dict:
    """One question through the full system, exactly as the API streams it."""
    thread = f"eval-{item['id']}-{uuid.uuid4().hex[:8]}"
    frames = []
    async for chunk in stream_turn(graph, item["question"], thread):
        frames += parse_sse(chunk)
    state = (await graph.aget_state({"configurable": {"thread_id": thread}})).values
    done = next((data for event, data in reversed(frames) if event == "done"), {})
    sources = state.get("sources", [])
    return {
        **item,
        "answer": state.get("answer", ""),
        "contexts": [s.get("context") or s["text"] for s in sources],  # what the answer call saw
        "sources": [{k: s.get(k) for k in ("n", "kind", "key", "title", "url")} for s in sources],
        "cited": [c["n"] for c in state.get("citations", [])],
        "plan": state.get("plan"),
        "agents": done.get("agents", []),
        "ttft_ms": done.get("ttft_ms"),
        "total_ms": done.get("total_ms"),
        "breakdown": breakdown([data for event, data in frames if event == "trace"]),
        "errors": [data.get("message") for event, data in frames if event == "error"],
    }


async def generate(args) -> None:
    runtime = await build_runtime(get_settings())
    if args.config == "research-gemini":
        use_gemini_for_research(runtime)
    graph = build_graph(runtime, InMemorySaver())
    path = OUT / f"{args.config}.jsonl"
    done_ids = {r["id"] for r in read_jsonl(path)}
    items = [r for r in read_jsonl(RAGAS_SET)[: args.limit or None] if r["id"] not in done_ids]
    print(f"{CONFIGS[args.config]}: {len(items)} questions to answer ({len(done_ids)} already in {path.name})")
    for item in items:
        record = {**await answer_one(graph, item), "config": args.config}
        append_jsonl(path, record)
        b = record["breakdown"]
        print(f"  {item['id']}  first words {(record['ttft_ms'] or 0) / 1000:5.1f} s"
              f"  total {(record['total_ms'] or 0) / 1000:5.1f} s  {str(b['planner']):<15}  plan {record['plan']}"
              f"  sources {len(record['contexts'])}  web {b['web_searches']}"
              f"{'  ERRORS' if record['errors'] or b['errors'] else ''}")
        await asyncio.sleep(args.pause)


# --- score -------------------------------------------------------------------

JUDGE_BACKOFF_S = 2.0
# Judge usage for the answer being scored. Answers are scored concurrently, so shared counters
# can't say which calls belonged to which answer; each scoring task sets its own dict here, and
# the metric calls it starts inherit it.
JUDGE_USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("judge_usage", default=None)


def make_judge(settings, chat=None):
    """Gemini on Vertex AI as a RAGAS 0.4 LLM, via LangChain structured output (the same path the
    router uses in production), with retries for Vertex AI's shared-capacity 429s."""
    from ragas.llms.base import InstructorBaseRagasLLM

    if chat is None:
        route = Route("judge", "google", settings.gemini_model, "evaluation", "RAGAS judge")
        chat = chat_model(route, settings, temperature=0.0, max_retries=3)

    class LangChainJudge(InstructorBaseRagasLLM):
        def __init__(self):
            self.totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

        calls = property(lambda self: self.totals["calls"])
        input_tokens = property(lambda self: self.totals["input_tokens"])
        output_tokens = property(lambda self: self.totals["output_tokens"])

        async def agenerate(self, prompt, response_model):
            structured = chat.with_structured_output(response_model, include_raw=True)
            error: Exception | None = None
            for attempt in range(4):
                try:
                    result = await structured.ainvoke(prompt)
                    meta = getattr(result.get("raw"), "usage_metadata", None) or {}
                    for counter in (self.totals, JUDGE_USAGE.get()):
                        if counter is not None:
                            counter["calls"] += 1
                            counter["input_tokens"] += meta.get("input_tokens") or 0
                            counter["output_tokens"] += meta.get("output_tokens") or 0
                    if result.get("parsed") is not None:
                        return result["parsed"]
                    error = ValueError(f"unparseable judge reply: {result.get('parsing_error')}")
                except Exception as exc:  # 429s and timeouts: back off and retry
                    error = exc
                await asyncio.sleep(JUDGE_BACKOFF_S * 2 ** attempt)
            raise error

        def generate(self, prompt, response_model):
            return asyncio.run(self.agenerate(prompt, response_model))

    return LangChainJudge()


def make_embeddings():
    """The local bge-small model used for retrieval, as RAGAS 0.4 embeddings."""
    from ragas.embeddings.base import BaseRagasEmbedding

    from src.rag.store import get_embeddings

    local = get_embeddings(get_settings())

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return local.embed_query(text)

        async def aembed_text(self, text, **kwargs):
            return await asyncio.to_thread(local.embed_query, text)

        def embed_texts(self, texts, **kwargs):
            return [local.embed_query(t) for t in texts]

        async def aembed_texts(self, texts, **kwargs):
            return await asyncio.to_thread(self.embed_texts, texts)

    return LocalEmbeddings()


def make_metrics(judge, embeddings) -> dict:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


async def score_one(metrics: dict, record: dict) -> dict:
    question, answer, contexts, reference = (record["question"], record["answer"], record["contexts"],
                                             record["reference"])
    if not answer or not contexts:
        return {name: None for name in METRICS} | {"_errors": {"all": "no answer or no contexts"}}
    calls = {
        "faithfulness": metrics["faithfulness"].ascore(user_input=question, response=answer,
                                                       retrieved_contexts=contexts),
        "answer_relevancy": metrics["answer_relevancy"].ascore(user_input=question, response=answer),
        "context_precision": metrics["context_precision"].ascore(user_input=question, reference=reference,
                                                                 retrieved_contexts=contexts),
        "context_recall": metrics["context_recall"].ascore(user_input=question, retrieved_contexts=contexts,
                                                           reference=reference),
    }
    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    scores, errors = {}, {}
    for name, result in zip(calls, results):
        if isinstance(result, BaseException):
            scores[name], errors[name] = None, f"{type(result).__name__}: {result}"[:300]
        else:
            value = float(result.value)
            scores[name] = None if math.isnan(value) else value
    return scores | ({"_errors": errors} if errors else {})


async def score(args) -> None:
    source, path = OUT / f"{args.config}.jsonl", OUT / f"{args.config}.scored.jsonl"
    records = read_jsonl(source)
    if not records:
        raise SystemExit(f"No answers in {source}: run `generate --config {args.config}` first.")
    scored = read_jsonl(path)
    if args.retry_errors:  # drop answers with a failed metric, so they are scored again in full
        kept = [r for r in scored if not r["scores"].get("_errors")]
        if len(kept) < len(scored):
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        scored = kept
    done_ids = {r["id"] for r in scored}
    todo = [r for r in records[: args.limit or None] if r["id"] not in done_ids]
    print(f"Scoring {len(todo)} answers from {source.name} ({len(done_ids)} already scored)")
    judge = make_judge(get_settings())
    metrics = make_metrics(judge, make_embeddings())
    gate = asyncio.Semaphore(args.concurrency)

    async def run(record: dict) -> None:
        async with gate:
            usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            JUDGE_USAGE.set(usage)  # this task's context only (gather runs each answer as its own task)
            started = time.perf_counter()
            scores = await score_one(metrics, record)
            append_jsonl(path, {"id": record["id"], "config": args.config, "scores": scores,
                                "judge": usage | {"per_answer": True},
                                "seconds": round(time.perf_counter() - started, 1)})
            shown = "  ".join(f"{m} {scores[m]:.2f}" if scores.get(m) is not None else f"{m} -" for m in METRICS)
            print(f"  {record['id']}  {shown}{'  ERRORS' if scores.get('_errors') else ''}")

    await asyncio.gather(*(run(r) for r in todo))


# --- router ------------------------------------------------------------------

async def router(args) -> None:
    runtime = await build_runtime(get_settings())
    routers = [("gemini", runtime.routes["supervisor"], runtime.models["supervisor"]),
               ("nemotron-lightning", runtime.fallbacks["supervisor"], runtime.fallback_models["supervisor"])]
    path = OUT / "router.jsonl"
    done = {(r["id"], r["router"]) for r in read_jsonl(path)}
    items = read_jsonl(ROUTING_SET)[: args.limit or None]
    print(f"Routing {len(items)} messages with {len(routers)} routers")
    for item in items:
        history = [HumanMessage(h["text"]) if h["role"] == "user" else AIMessage(h["text"]) for h in item["history"]]
        messages = [SystemMessage(prompts.SUPERVISOR),
                    planner_request([*history, HumanMessage(item["message"])], item["message"])]
        for name, route, llm in routers:
            if (item["id"], name) in done:
                continue
            started = time.perf_counter()
            record = {"id": item["id"], "router": name, "model": route.model, "message": item["message"],
                      "history": bool(item["history"]), "expected": item["expected"]}
            try:
                plan, use = await asyncio.wait_for(request_plan(route, llm, messages), args.timeout)
                record |= {"agents": plan.agents, "reason": plan.reason, "standalone": plan.standalone,
                           "correct": set(plan.agents) == set(item["expected"]),
                           "exact": plan.agents == item["expected"], **use, "error": None}
            except Exception as error:  # a 429 or timeout counts against the router, as in production
                record |= {"agents": None, "correct": False, "exact": False,
                           "error": f"{type(error).__name__}: {error}"[:200]}
            record["ms"] = round((time.perf_counter() - started) * 1000)
            append_jsonl(path, record)
            mark = "ok " if record["correct"] else "MISS"
            print(f"  {item['id']}  {name:<19} {mark} {record['ms']:>6} ms  {record['agents']}  "
                  f"(expected {item['expected']}){'  ' + record['error'] if record['error'] else ''}")
        await asyncio.sleep(args.pause)


# --- report ------------------------------------------------------------------

def stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
            "min": ordered[0], "max": ordered[-1]}


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model.startswith(FREE_MODELS):
        return 0.0
    price = TOKEN_PRICES.get(model)
    return None if price is None else (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def query_cost(record: dict) -> dict:
    """Marginal cost of one query at pay-as-you-go prices, ignoring free tiers."""
    b = record["breakdown"]
    models = 0.0
    for model, use in b["tokens_by_model"].items():
        cost = token_cost(model, use["input_tokens"], use["output_tokens"])
        models += cost or 0.0
    web = b["web_searches"] * TAVILY_PER_SEARCH
    compute = (record["total_ms"] or 0) / 1000 * CLOUD_RUN_PER_SECOND + CLOUD_RUN_PER_REQUEST
    return {"models": models, "web": web, "compute": compute, "total": models + web + compute}


def fmt(s: dict, scale: float = 1.0, digits: int = 2) -> str:
    if not s.get("n"):
        return "n/a"
    return f"{s['mean'] * scale:.{digits}f} ± {s['sd'] * scale:.{digits}f}"


def report(args) -> None:
    def med_p90(values) -> str:
        st = stats([v / 1000 for v in values if v is not None])
        return f"{st['median']:.1f} / {st['p90']:.1f}" if st.get("n") else "n/a"

    print(f"# Evaluation results (prices as of {PRICES_AS_OF})\n")
    print("## RAGAS (mean ± sd; n = questions scored)\n")
    print("| Configuration | " + " | ".join(METRICS) + " | Judge calls | Judge cost |")
    print("|---" * (len(METRICS) + 3) + "|")
    notes = []
    for config, label in CONFIGS.items():
        scored = read_jsonl(OUT / f"{config}.scored.jsonl")
        if not scored:
            continue
        cells = []
        for metric in METRICS:
            st = stats([r["scores"].get(metric) for r in scored])
            cells.append(f"{fmt(st)} (n={st['n']})")
        # Answers scored before judge usage was tracked per answer recorded a share of every
        # concurrent answer's calls; their totals overstate, so they're left out.
        exact = [r for r in scored if r["judge"].get("per_answer")]
        if len(exact) == len(scored):
            cost = token_cost(get_settings().gemini_model, sum(r["judge"]["input_tokens"] for r in exact),
                              sum(r["judge"]["output_tokens"] for r in exact))
            judge = f"{sum(r['judge']['calls'] for r in exact)} | ${cost or 0:.3f}"
        else:
            judge = "not measured | not measured"
            notes.append(f"- {config}: judge usage not shown; {len(scored) - len(exact)} answers were scored "
                         "before it was tracked per answer, and their counts overlap.")
        print(f"| {label} | " + " | ".join(cells) + f" | {judge} |")
        for r in scored:
            for metric, error in (r["scores"].get("_errors") or {}).items():
                notes.append(f"- {config} {r['id']}: {metric} not scored ({error[:90]}...)")
    if notes:
        print("\nNotes (missing scores are excluded from the means; re-score them with "
              "`score <config> --retry-errors`):\n")
        print("\n".join(notes))

    print("\n## Local runs: latency in seconds (median / p90) and usage per query\n")
    print("| Configuration | n | First words | Total | Routing | Retrieval | Web search | Model calls (sum) |"
          " Model calls | Web searches | Gemini planned | Cost / query |")
    print("|---" * 12 + "|")
    details = []
    for config, label in CONFIGS.items():
        records = [r for r in read_jsonl(OUT / f"{config}.jsonl") if r.get("total_ms")]
        if not records:
            continue
        b = [r["breakdown"] for r in records]
        n = len(records)
        gemini_share = sum(1 for x in b if x["planner"] and x["planner"].startswith("google")) / n
        backups = [plan_backup(x) for x in b]
        parts = [query_cost(r) for r in records]
        print(f"| {label} | {n} | {med_p90([r['ttft_ms'] for r in records])} | "
              f"{med_p90([r['total_ms'] for r in records])} | {med_p90([plan_wall_ms(x) for x in b])} | "
              f"{med_p90([x['retrieval_ms'] for x in b])} | {med_p90([x['web_ms'] for x in b])} | "
              f"{med_p90([x['llm_ms'] for x in b])} | {statistics.fmean(x['llm_calls'] for x in b):.1f} | "
              f"{statistics.fmean(x['web_searches'] for x in b):.2f} | {gemini_share:.0%} | "
              f"${statistics.fmean(p['total'] for p in parts):.5f} |")
        stage_ms: dict[str, list[float]] = {}
        for x in b:
            for stage, st in x["llm_by_stage"].items():
                stage_ms.setdefault(stage, []).append(st["ms"] / 1000)
        details.append(f"\n**{config}**: the backup router ran because Gemini was slow (over 4 s) in "
                       f"{backups.count('slow')} of {n} queries and because it failed in {backups.count('failed')}.")
        details.append("Model time by stage, median seconds in the queries that had it: "
                       + ", ".join(f"{k} {statistics.median(v):.1f} (n={len(v)})" for k, v in sorted(stage_ms.items())))
        for model in sorted({m for x in b for m in x["tokens_by_model"]}):
            uses = [x["tokens_by_model"].get(model) for x in b]
            ran = [u for u in uses if u]
            unreported = sum(u["unreported"] for u in ran)
            details.append(f"- {model}: ran in {len(ran)} of {n} queries; per query overall "
                           f"{sum(u['calls'] for u in ran) / n:.1f} calls, "
                           f"{sum(u['input_tokens'] for u in ran) / n:,.0f} input and "
                           f"{sum(u['output_tokens'] for u in ran) / n:,.0f} output tokens"
                           f"{f' ({unreported} calls without token counts)' if unreported else ''}")
        details.append(f"- cost per query: models ${statistics.fmean(p['models'] for p in parts):.5f}, "
                       f"web search ${statistics.fmean(p['web'] for p in parts):.5f}, "
                       f"Cloud Run ${statistics.fmean(p['compute'] for p in parts):.5f}")
    print("\n".join(details))

    routed = read_jsonl(OUT / "router.jsonl")
    if routed:
        print("\n## Router comparison\n")
        print("| Router | Model | n | Correct agents | Exact order | Errors | Latency median / p90 (s) |"
              " Tokens per call (in / out) |")
        print("|---" * 8 + "|")
        for name in ("gemini", "nemotron-lightning"):
            rows = [r for r in routed if r["router"] == name]
            if not rows:
                continue
            ok = [r for r in rows if not r["error"]]
            tokens_in = [r["input_tokens"] for r in ok if r.get("input_tokens") is not None]
            tokens_out = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]
            tokens = (f"{statistics.fmean(tokens_in):,.0f} / {statistics.fmean(tokens_out):,.0f}"
                      if tokens_in else "not reported")
            print(f"| {name} | {rows[0]['model']} | {len(rows)} | {sum(r['correct'] for r in rows)}/{len(rows)} | "
                  f"{sum(r['exact'] for r in rows)}/{len(rows)} | {len(rows) - len(ok)} | "
                  f"{med_p90([r['ms'] for r in ok])} | {tokens} |")
        misses = [r for r in routed if not r["correct"]]
        for r in misses:
            print(f"- {r['router']} {r['id']} \"{r['message']}\": got {r['agents']}, expected {r['expected']}"
                  f"{' (' + r['error'] + ')' if r['error'] else ''}")

    latency = read_jsonl(OUT / "latency.jsonl")
    if latency:
        print("\n## Deployed service (client-side seconds, median / p90; n)\n")
        print("| Category | Warm first words | Warm total | Cold first words | Cold total |")
        print("|---|---|---|---|---|")
        ok = [r for r in latency if not r.get("errors")]
        for category in sorted({r["category"] for r in ok}):
            rows = [r for r in ok if r["category"] == category]

            def cell(cold: bool, key: str) -> str:
                values = [r[key] for r in rows if r["cold"] == cold and r.get(key) is not None]
                return f"{med_p90(values)} (n={len(values)})" if values else "n/a"

            print(f"| {category} | {cell(False, 'client_ttft_ms')} | {cell(False, 'client_total_ms')} | "
                  f"{cell(True, 'client_ttft_ms')} | {cell(True, 'client_total_ms')} |")
        # Startup wait: what the client waited before the server started on the request.
        for cold in (True, False):
            waits = [(r["client_ttft_ms"] - r["server_ttft_ms"]) for r in ok if r["cold"] == cold
                     and r.get("client_ttft_ms") is not None and r.get("server_ttft_ms") is not None]
            if waits:
                print(f"\n{'Cold' if cold else 'Warm'} requests: client first words minus server first words "
                      f"(startup and network), median / p90 {med_p90(waits)} s over {len(waits)} requests.")
        b = [r["breakdown"] for r in ok if r.get("breakdown")]
        if b:
            backups = [plan_backup(x) for x in b]
            print(f"\nRouting on the deployed service: Gemini planned "
                  f"{sum(1 for x in b if (x['planner'] or '').startswith('google'))} of {len(b)} requests; the "
                  f"backup router ran because Gemini was slow in {backups.count('slow')} and because it failed "
                  f"in {backups.count('failed')}. Routing wall time median / p90 "
                  f"{med_p90([plan_wall_ms(x) for x in b])} s.")
            problems = sorted({m for x in b for m in x.get("error_messages", [])})
            if problems:
                print("Errors in the traces: " + "; ".join(problems))
        failed = [r for r in latency if r.get("errors")]
        if failed:
            print(f"\n{len(failed)} deployed requests returned errors: "
                  + "; ".join(f"{r['category']}: {r['errors']}" for r in failed[:5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="answer the RAGAS questions with the full system")
    gen.add_argument("--config", choices=CONFIGS, default="default")
    gen.add_argument("--limit", type=int, default=0, help="only the first N questions")
    gen.add_argument("--pause", type=float, default=2.0, help="seconds between questions (NVIDIA rate limit)")
    sc = commands.add_parser("score", help="grade saved answers with RAGAS")
    sc.add_argument("config", choices=CONFIGS)
    sc.add_argument("--limit", type=int, default=0)
    sc.add_argument("--concurrency", type=int, default=3, help="answers graded at once")
    sc.add_argument("--retry-errors", action="store_true", help="re-score answers that had a metric error")
    rt = commands.add_parser("router", help="compare the two routers on the routing set")
    rt.add_argument("--limit", type=int, default=0)
    rt.add_argument("--pause", type=float, default=1.0)
    rt.add_argument("--timeout", type=float, default=30.0)
    rp = commands.add_parser("report", help="print markdown tables from the saved results")
    rp.add_argument("--out", help="also write the report to this file (UTF-8)")
    args = parser.parse_args()
    if args.command == "report":
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report(args)
        print(buffer.getvalue(), end="")
        if args.out:  # written by Python, so characters like ± survive any console code page
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(buffer.getvalue(), encoding="utf-8")
    else:
        asyncio.run({"generate": generate, "score": score, "router": router}[args.command](args))


if __name__ == "__main__":
    main()
