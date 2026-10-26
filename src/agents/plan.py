"""The supervisor's structured output: which agents run this turn, and why."""

from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["research", "summary", "data"]


class RoutePlan(BaseModel):
    agents: list[AgentName] = Field(default_factory=list, description="Agents to run, in order (0 to 3).")
    reason: str = Field(description="One sentence on why these agents fit the message.")
    standalone: str | None = Field(default=None, description=(
        "The user's latest message rewritten as a self-contained question, resolving references to "
        "earlier turns (for example 'which one' -> 'which heart chamber'). Omit if already self-contained."))
    reply: str | None = Field(default=None, description="Only when agents is empty: a short direct reply.")
