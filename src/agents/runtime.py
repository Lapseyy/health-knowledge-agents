"""Everything the graph needs at run time, built once at startup."""

import json
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from src.agents.tools import describe_database
from src.agents.web import TavilyMCP
from src.config import Settings
from src.llm import Route, chat_model, fallback_table, routing_table
from src.rag.retriever import default_store


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


async def build_runtime(settings: Settings) -> Runtime:
    routes, fallbacks = routing_table(settings), fallback_table(settings)
    web = TavilyMCP(settings.tavily_api_key)
    await web.connect()  # failure is logged; agents fall back to the textbook
    return Runtime(
        settings=settings,
        routes=routes,
        fallbacks=fallbacks,
        models={agent: chat_model(route, settings) for agent, route in routes.items()},
        fallback_models={agent: chat_model(route, settings) for agent, route in fallbacks.items()},
        store=default_store(),
        sections=load_sections(settings),
        web=web,
        db_schema=describe_database(settings.patient_db_path),
    )
