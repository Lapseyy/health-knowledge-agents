"""Test doubles shared by the agent and API tests: scripted LLMs and a fake MCP server."""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import Field

from src.agents.runtime import Runtime
from src.agents.tools import describe_database
from src.agents.web import TavilyMCP
from src.llm import fallback_table, routing_table


class ScriptedModel(BaseChatModel):
    """Returns queued replies in order; `plan` answers structured-output calls.

    `delay` (seconds) makes async calls slow, to stage a slow or hung provider.
    """

    replies: list = Field(default_factory=list)
    plan: object = None
    delay: float = 0
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

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        await asyncio.sleep(self.delay)
        return self._generate(messages, stop, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self.bind(**kwargs)

    def with_structured_output(self, schema, **kwargs):
        def plan(messages):
            self.seen.append(messages)
            if isinstance(self.plan, Exception):
                raise self.plan
            return self.plan

        async def delayed_plan(messages):
            await asyncio.sleep(self.delay)
            return plan(messages)
        return RunnableLambda(plan, afunc=delayed_plan)


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
