"""Shared by scripts/evaluate.py and scripts/measure_latency.py.

`breakdown` turns one turn's trace events into where the time went (routing,
retrieval, web search, SQL, each kind of model call) and what it used (tokens
per model, web searches), which is what docs/evaluation.md reports.
`SSEReader` parses the API's Server-Sent Events, including a stream that
arrives in arbitrary chunks over the network.

Tool calls in one round run concurrently, and chapter digests run in parallel,
so summed durations can exceed wall-clock time; totals are reported separately.
"""

import json
from collections import defaultdict


class SSEReader:
    """Incremental parser for `event: <name>` / `data: <json>` frames separated by blank lines."""

    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str) -> list[tuple[str, dict]]:
        self.buffer += chunk.replace("\r\n", "\n")
        frames = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            event, data = None, []
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: "):
                    data.append(line[len("data: "):])
            if event:
                frames.append((event, json.loads("\n".join(data)) if data else {}))
        return frames


def parse_sse(text: str) -> list[tuple[str, dict]]:
    return SSEReader().feed(text if text.endswith("\n\n") else text + "\n\n")


def breakdown(events: list[dict]) -> dict:
    """Latency and usage for one turn, from its trace events."""
    stages: dict[str, dict] = defaultdict(lambda: {"calls": 0, "ms": 0})
    models: dict[str, dict] = defaultdict(lambda: {"provider": None, "calls": 0, "input_tokens": 0,
                                                  "output_tokens": 0, "unreported": 0, "output_chars": 0})
    out = {"plan_ms": None, "plan_wall_ms": None, "planner": None, "hedged": False, "plan_backup": None,
           "retrieval_ms": 0, "retrievals": 0, "web_ms": 0, "web_searches": 0, "web_extracts": 0,
           "sql_ms": 0, "sql_queries": 0, "llm_ms": 0, "llm_calls": 0, "errors": 0, "error_messages": [],
           "agent_fallbacks": []}
    plan_failed = False

    def count_usage(event: dict, provider: str | None) -> None:
        model = models[event.get("model") or "unknown"]
        model["provider"] = model["provider"] or provider
        model["calls"] += 1
        model["output_chars"] += event.get("output_chars") or 0
        if event.get("input_tokens") is None:
            model["unreported"] += 1
        else:
            model["input_tokens"] += event["input_tokens"]
            model["output_tokens"] += event.get("output_tokens") or 0

    for event in events:
        kind = event.get("type")
        if kind == "agent_plan":
            provider = event["planner"].split()[0]
            # Why the backup router ran. Traces from before the server reported it set `hedged` whenever
            # the backup ran; a failed primary plan (an error event first) tells the two cases apart.
            backup = event.get("backup", ("failed" if plan_failed else "slow") if event.get("hedged") else None)
            out.update(plan_ms=event.get("ms"), plan_wall_ms=event.get("wall_ms"), planner=event["planner"],
                       hedged=backup == "slow", plan_backup=backup)
            count_usage(event, provider)
        elif kind == "llm_call":
            stage = stages[event["stage"]]
            stage["calls"] += 1
            stage["ms"] += event.get("ms") or 0
            out["llm_ms"] += event.get("ms") or 0
            out["llm_calls"] += 1
            count_usage(event, event.get("provider"))
        elif kind == "rag_retrieval":
            out["retrieval_ms"] += event.get("ms") or 0
            out["retrievals"] += 1
        elif kind == "mcp_tool_call":
            out["web_ms"] += event.get("ms") or 0
            out["web_extracts" if "extract" in event.get("tool", "") else "web_searches"] += 1
        elif kind == "sql_query":
            out["sql_ms"] += event.get("ms") or 0
            out["sql_queries"] += 1
        elif kind == "error":
            out["errors"] += 1
            out["error_messages"].append(f"{event.get('agent')} {event.get('stage')}: {event.get('message')}")
            plan_failed = plan_failed or (event.get("agent") == "supervisor" and event.get("stage") == "plan")
        elif kind == "llm_route" and event.get("fallback") and event.get("agent") != "supervisor":
            out["agent_fallbacks"].append(event["agent"])
    out["llm_by_stage"] = dict(stages)
    out["tokens_by_model"] = dict(models)
    return out


def plan_backup(b: dict) -> str | None:
    """Why the backup router ran ("slow" or "failed"), from a breakdown; None if it didn't.

    Breakdowns saved before this was recorded set `hedged` in both cases. For those: if the primary
    router's plan was used, it was only slow; if the backup's was, an error not explained by an
    agent failing over means the primary's plan failed."""
    if "plan_backup" in b:
        return b["plan_backup"]
    if not b.get("hedged"):
        return None
    if (b.get("planner") or "").endswith("primary"):
        return "slow"
    return "failed" if (b.get("errors") or 0) > len(b.get("agent_fallbacks") or []) else "slow"


def plan_wall_ms(b: dict) -> int | None:
    """How long the planning step took, from a breakdown. Newer traces report it; for older ones it
    is the winning call's time, plus the wait before the backup started when Gemini was slow (a
    failed primary usually fails fast, so that case adds nothing and slightly understates)."""
    from src.agents.graph import HEDGE_AFTER_S

    if b.get("plan_wall_ms") is not None:
        return b["plan_wall_ms"]
    if b.get("plan_ms") is None:
        return None
    return b["plan_ms"] + (round(HEDGE_AFTER_S * 1000) if plan_backup(b) == "slow" else 0)

