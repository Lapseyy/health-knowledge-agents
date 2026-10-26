"""Confirm both generation providers answer, and how fast, before building agents.

Sends the same short prompt to the NVIDIA and Gemini models named in .env three
times each. The first call includes connection setup; the later ones show
steady-state latency. Gemini's automatic retries are off so a slow call shows
up as slow (or as the error behind it), not hidden behind silent backoff.

    python -m scripts.check_providers
"""

import os
import time

from dotenv import load_dotenv

from src.config import ROOT

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
    load_dotenv(ROOT / ".env")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    print(f"NVIDIA: {os.getenv('NVIDIA_MODEL')}   Gemini: {os.getenv('GEMINI_MODEL')}\n")
    nvidia = ChatNVIDIA(model=os.environ["NVIDIA_MODEL"], max_completion_tokens=64)
    ping("NVIDIA", nvidia.with_thinking_mode(enabled=False))

    gemini = ChatGoogleGenerativeAI(
        model=os.environ["GEMINI_MODEL"],
        max_output_tokens=256,
        # Gemini 3 Flash thinks at "high" effort by default, which is wasted on
        # short classification calls like routing.
        reasoning_effort=os.getenv("GEMINI_REASONING_EFFORT", "minimal"),
        max_retries=0,
        timeout=60,
    )
    ping("Gemini", gemini)


if __name__ == "__main__":
    main()
