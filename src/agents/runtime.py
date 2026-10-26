"""Everything the graph needs at run time, built once at startup."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents import prompts
from src.agents.plan import RoutePlan
from src.agents.tools import describe_database
from src.agents.web import TavilyMCP
from src.config import Settings
from src.llm import Route, call_options, chat_model, describe_failure, fallback_table, routing_table
from src.rag.retriever import default_store

log = logging.getLogger(__name__)
WARM_UP_TIMEOUT_S = 10.0


@dataclass
class Runtime:
    settings: Settings
    routes: dict[str, Route]  # primary route per agent
    fallbacks: dict[str, Route]  # backup route per agent, on the other provider
    models: dict[str, BaseChatModel]
    fallback_models: dict[str, BaseChatModel]
    store: Chroma
    sections: dict[str, dict]
    web: TavilyMCP
    db_schema: str

    def attempts(self, agent: str) -> list[tuple[Route, BaseChatModel]]:
        """Primary first, then the cross-provider fallback."""
        return [(self.routes[agent], self.models[agent]), (self.fallbacks[agent], self.fallback_models[agent])]


def load_sections(settings: Settings) -> dict[str, dict]:
    with open(settings.corpus_path, encoding="utf-8") as f:
        return {record["source_id"]: record for record in map(json.loads, f)}


async def warm_up(runtime: Runtime) -> dict[str, str]:
    """One small call per distinct model client, in parallel, before the first user turn.

    A fresh process pays for credential refresh and connection setup on its first
    call to each provider (about +0.7 s measured for Gemini on Vertex AI), which can
    push the first turn's routing call past the hedge. The Gemini router is warmed with
    the real routing request (same prompt and response schema), because a plain ping
    still left the first real plan at 5.3 s against 1.7 s for the next. Failures are
    logged, never raised.
    """
    clients: dict[int, tuple[str, Route, object]] = {}
    for agent in runtime.routes:
        for route, llm in runtime.attempts(agent):
            if agent == "supervisor" or not route.fallback:  # skip clients used only on failover
                clients.setdefault(id(llm), (f"{route.provider} {route.model}", route, llm))

    async def ping(route: Route, llm) -> str:
        if route.agent == "supervisor" and route.provider == "google":  # the planner's exact call shape
            call = llm.with_structured_output(RoutePlan).ainvoke([
                SystemMessage(prompts.SUPERVISOR), HumanMessage("Latest message, the only one to route:\nHello")])
        else:
            call = llm.bind(**call_options(route)).ainvoke("Reply with the word OK.")
        started = time.perf_counter()
        try:
            await asyncio.wait_for(call, WARM_UP_TIMEOUT_S)
        except Exception as error:
            return f"failed ({describe_failure(error, route.provider, WARM_UP_TIMEOUT_S)})"
        return f"{time.perf_counter() - started:.1f} s"

    results = await asyncio.gather(*(ping(route, llm) for _, route, llm in clients.values()))
    timings = {name: result for (name, _, _), result in zip(clients.values(), results)}
    log.info("Warm-up: %s", "; ".join(f"{name} {result}" for name, result in timings.items()))
    return timings


async def build_runtime(settings: Settings) -> Runtime:
    routes, fallbacks = routing_table(settings), fallback_table(settings)
    web = TavilyMCP(settings.tavily_api_key)
    await web.connect()  # failure is logged; agents fall back to the textbook

    clients: dict[tuple, object] = {}  # agents on the same model share one client (and its connections)

    def client(route: Route):
        key = (route.provider, route.model, route.agent == "supervisor")  # the router's client has no retries
        if key not in clients:
            clients[key] = chat_model(route, settings)
        return clients[key]

    runtime = Runtime(
        settings=settings,
        routes=routes,
        fallbacks=fallbacks,
        models={agent: client(route) for agent, route in routes.items()},
        fallback_models={agent: client(route) for agent, route in fallbacks.items()},
        store=default_store(),
        sections=load_sections(settings),
        web=web,
        db_schema=describe_database(settings.patient_db_path),
    )
    if settings.warm_up:
        await warm_up(runtime)
    return runtime
