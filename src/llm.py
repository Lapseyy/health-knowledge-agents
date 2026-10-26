"""Per-agent LLM routing across two providers, with cross-provider failover.

Each agent has a primary provider chosen for its job (cost- or
capability-based) and a fallback on the *other* provider, so one provider's
outage or rate limit never takes a turn down. Every Route carries its reason,
so the trace shows why a provider got the call, not just which one did.

Measured on Oct 25, 2026 (see docs/evaluation.md): Nemotron Ultra answered
short prompts in ~0.7-1.0 s warm; Gemini 3.8 Flash on Vertex AI in ~2-4 s,
with occasional 429s and one 504 deadline on a long summary.
"""

from dataclasses import asdict, dataclass

from langchain_core.language_models import BaseChatModel

from src.config import Settings

FAILOVER_REASON = ("Cross-provider failover: the primary provider errored or timed out, so the other "
                   "provider completes the turn instead of failing it.")


@dataclass(frozen=True)
class Route:
    agent: str
    provider: str  # "google" or "nvidia"
    model: str
    dimension: str  # "cost", "capability", "hybrid", or "failover"
    reason: str
    fallback: bool = False

    def as_event(self) -> dict:
        return {"type": "llm_route", **asdict(self)}


def routing_table(settings: Settings) -> dict[str, Route]:
    """Primary route per agent."""
    gemini, nemotron = settings.gemini_model, settings.nvidia_model
    return {
        "supervisor": Route(
            "supervisor", "google", gemini, "cost",
            "Routing is one short structured-output call per turn: a low-cost model is enough, and it "
            "keeps NVIDIA's free-tier request budget for the agents' heavier generation.",
        ),
        "research": Route(
            "research", "nvidia", nemotron, "capability",
            "Multi-source synthesis (textbook + web) with citations needs the most capable model, "
            "and Nemotron Ultra was also the faster provider per call.",
        ),
        "summary": Route(
            "summary", "nvidia", nemotron, "capability",
            "Summaries read whole sections (up to ~7k tokens) and must stay faithful to them; Nemotron "
            "handled long inputs reliably, while Gemini on Vertex hit a 504 deadline on a long summary.",
        ),
        "data": Route(
            "data", "nvidia", nemotron, "capability",
            "Writing correct SQL over an unfamiliar schema and recovering from query errors rewards "
            "the stronger tool-calling model.",
        ),
    }


def fallback_table(settings: Settings) -> dict[str, Route]:
    """Backup route per agent, always on the other provider."""
    gemini = settings.gemini_model
    return {
        "supervisor": Route(
            "supervisor", "nvidia", settings.nvidia_fast_model, "failover",
            "Gemini was rate-limited or slower than the routing timeout, so a small, fast Nemotron "
            "model plans the turn instead.", fallback=True,
        ),
        **{agent: Route(agent, "google", gemini, "failover", FAILOVER_REASON, fallback=True)
           for agent in ("research", "summary", "data")},
    }


def chat_model(route: Route, settings: Settings) -> BaseChatModel:
    if route.provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        # Nemotron's reasoning mode is turned off per call (see call_options).
        return ChatNVIDIA(model=route.model, temperature=0.2, max_completion_tokens=2500)

    if route.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Vertex AI authenticates with Google Cloud credentials (no API key);
        # with GOOGLE_GENAI_USE_VERTEXAI=false the client uses GOOGLE_API_KEY.
        vertex = {"vertexai": True, "project": settings.google_cloud_project,
                  "location": settings.google_cloud_location} if settings.gemini_use_vertex else {}
        return ChatGoogleGenerativeAI(
            model=route.model,
            temperature=0.2,
            max_output_tokens=4096,  # includes Gemini's reasoning tokens
            reasoning_effort=settings.gemini_reasoning_effort,
            max_retries=1,
            timeout=120,  # also sent to the server as the request deadline
            **vertex,
        )

    raise ValueError(f"Unknown provider {route.provider!r}")


def call_options(route: Route) -> dict:
    """Extra per-call arguments: Nemotron's reasoning mode is off for latency."""
    return {"thinking_mode": False} if route.provider == "nvidia" else {}
