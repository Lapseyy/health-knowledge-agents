"""FastAPI app: the streaming chat endpoint, a health check, and the static UI."""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.agents.graph import build_graph
from src.agents.runtime import build_runtime
from src.api.stream import stream_turn
from src.config import ROOT, get_settings

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO, and the Tavily MCP URL carries the API key as a
    # query parameter, so those lines would put the key in local and Cloud Run logs. httpx2 (its
    # successor, already used by google-genai and langsmith) logs the same way.
    for client_logger in ("httpx", "httpx2"):
        logging.getLogger(client_logger).setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)  # per-call "AFC is enabled" noise


def create_app(runtime_factory=build_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        runtime = await runtime_factory(get_settings())
        app.state.runtime = runtime
        # Conversation memory per session_id. In-process: lost when the instance
        # restarts, so the deployment must say so or use a durable checkpointer.
        app.state.graph = build_graph(runtime, InMemorySaver())
        app.state.session_locks = defaultdict(asyncio.Lock)
        yield

    app = FastAPI(title="Health knowledge agents", lifespan=lifespan)

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        runtime = request.app.state.runtime
        return {
            "status": "ok",
            "web_search": bool(runtime.web.tools),
            "routes": {agent: asdict(route) for agent, route in runtime.routes.items()},
            "fallbacks": {agent: asdict(route) for agent, route in runtime.fallbacks.items()},
        }

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        state = request.app.state

        async def frames():
            # One turn at a time per session, so two tabs can't interleave a conversation.
            async with state.session_locks[body.session_id]:
                async for frame in stream_turn(state.graph, body.query.strip(), body.session_id):
                    yield frame

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
    return app
