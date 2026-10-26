"""Confirm both generation providers answer, and how fast.

Sends the same short prompt to each model the app uses (Nemotron Ultra, the
fast Nemotron fallback router, and Gemini) three times each. The first call
includes connection setup; the later ones show steady-state latency. Gemini's automatic retries are off so a slow call shows
up as slow (or as the error behind it), not hidden behind silent backoff.

    python -m scripts.check_providers
"""

import time

from src.config import get_settings

PROMPT = "Reply with one short sentence: what organ pumps blood?"
CALLS = 3


def ping(label: str, model) -> None:
    for call in range(1, CALLS + 1):
        started = time.perf_counter()
        try:
            text = model.invoke(PROMPT).text  # .text flattens Gemini's content blocks
            status = f"OK      {text.strip()[:70]}"
        except Exception as error:
            status = f"FAILED  {str(error).splitlines()[0][:120]}"
        print(f"{label:7} call {call}  {time.perf_counter() - started:6.1f}s  {status}")


def main() -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    settings = get_settings()
    print(f"NVIDIA: {settings.nvidia_model}, fallback router {settings.nvidia_fast_model}   "
          f"Gemini: {settings.gemini_model}\n")
    for label, model in (("NVIDIA", settings.nvidia_model), ("NV-fast", settings.nvidia_fast_model)):
        ping(label, ChatNVIDIA(model=model, max_completion_tokens=64).with_thinking_mode(enabled=False))

    vertex = {"vertexai": True, "project": settings.google_cloud_project,
              "location": settings.google_cloud_location} if settings.gemini_use_vertex else {}
    gemini = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        max_output_tokens=256,
        # Gemini 3 Flash thinks at "high" effort by default; the app uses GEMINI_REASONING_EFFORT.
        reasoning_effort=settings.gemini_reasoning_effort,
        max_retries=0,
        timeout=60,
        **vertex,
    )
    ping("Gemini", gemini)


if __name__ == "__main__":
    main()
