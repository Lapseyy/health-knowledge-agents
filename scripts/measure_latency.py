"""Measure latency on the deployed service, as a user sees it.

    python -m scripts.measure_latency --url https://...run.app              # warm: 7 questions x 2
    python -m scripts.measure_latency --url https://...run.app --cold 3     # 3 cold starts, 20 min apart

Every request uses a fresh session, so no answer benefits from conversation memory. First
words and total time are measured on this machine (network and any cold start included);
the server's own timings and trace breakdown are saved alongside. A request is labeled a
cold start when the server reports it as the instance's first turn with under 60 s of
uptime, so a cold sample is never guessed from timing alone.

--cold N waits --idle-minutes (default 20) before each request so Cloud Run can scale the
service to zero; leave the machine awake and don't use the site meanwhile. A warm run first
sends one priming request (saved, and labeled cold if it was) so the measured requests hit a
running instance. Results append to docs/eval/latency.jsonl; summarize them with
`python -m scripts.evaluate report`. Needs httpx (in requirements-dev.txt).
"""

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

import httpx

from src.config import ROOT
from src.evaluation import SSEReader, breakdown

OUT = ROOT / "docs" / "eval" / "latency.jsonl"
COLD_UPTIME_S = 60
LATENCY_SET = [
    ("textbook", "What does the left ventricle do?"),
    ("textbook", "How do gap junctions and intercalated disks aid contraction of the heart?"),
    ("web", "What are the latest guidelines for treating high blood pressure?"),
    ("web", "Has the FDA approved any new Alzheimer's drugs recently?"),
    ("data", "How many patients have essential hypertension, and what is their average BMI?"),
    ("summary", "Summarize the textbook section on the cardiac cycle."),
    ("chapter", "Summarize chapter 2."),
]


async def ask(client: httpx.AsyncClient, url: str, category: str, question: str) -> dict:
    session = f"latency-{uuid.uuid4().hex[:12]}"
    reader, frames = SSEReader(), []
    first_words = None
    started = time.perf_counter()
    error = None
    try:
        async with client.stream("POST", f"{url.rstrip('/')}/api/chat",
                                 json={"query": question, "session_id": session}) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                for event, data in reader.feed(chunk):
                    frames.append((event, data))
                    if first_words is None and event in ("token", "final"):
                        first_words = time.perf_counter()
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    done = next((data for event, data in reversed(frames) if event == "done"), {})
    instance = done.get("instance") or {}
    events = [data for event, data in frames if event == "trace"]
    errors = [data.get("message") for event, data in frames if event == "error"] + ([error] if error else [])
    return {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url, "category": category, "question": question,
        "client_ttft_ms": round((first_words - started) * 1000) if first_words else None,
        "client_total_ms": round(total * 1000),
        "server_ttft_ms": done.get("ttft_ms"), "server_total_ms": done.get("total_ms"),
        "instance": instance,
        "cold": instance.get("turn") == 1 and (instance.get("uptime_s") or 1e9) < COLD_UPTIME_S,
        "plan": done.get("plan"),
        "breakdown": breakdown(events),
        "errors": errors,
        "events": events,  # the full trace, so later analysis isn't limited to the breakdown
    }


def save(record: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    ttft = f"{record['client_ttft_ms'] / 1000:5.1f} s" if record["client_ttft_ms"] else "    - "
    b = record["breakdown"]
    backup = {"slow": " (primary slow)", "failed": " (primary failed)"}.get(b.get("plan_backup"), "")
    problems = [*map(str, record["errors"]), *b.get("error_messages", [])]
    print(f"  {'COLD' if record['cold'] else 'warm'}  {record['category']:<9} first words {ttft}"
          f"  total {record['client_total_ms'] / 1000:5.1f} s  {str(b['planner']) + backup:<31}  plan {record['plan']}"
          f"  (instance turn {record['instance'].get('turn')}, up {record['instance'].get('uptime_s')} s)"
          f"{'  ERRORS: ' + '; '.join(problems) if problems else ''}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the Cloud Run service URL")
    parser.add_argument("--repeats", type=int, default=2, help="warm runs of the question set")
    parser.add_argument("--cold", type=int, default=0, help="number of cold-start samples to take instead")
    parser.add_argument("--idle-minutes", type=float, default=20.0, help="idle time before each cold sample")
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between warm requests")
    args = parser.parse_args()
    timeout = httpx.Timeout(300.0, connect=60.0)  # a cold start can hold the connection for a while
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.cold:
            for i in range(args.cold):
                print(f"Cold sample {i + 1}/{args.cold}: waiting {args.idle_minutes:g} min for the service to "
                      f"scale to zero (started {datetime.now().strftime('%H:%M')})")
                await asyncio.sleep(args.idle_minutes * 60)
                save(await ask(client, args.url, *LATENCY_SET[0]))
            return
        print("Priming request (saved; labeled cold if it started an instance)")
        save(await ask(client, args.url, *LATENCY_SET[0]))
        for run in range(args.repeats):
            print(f"Warm run {run + 1}/{args.repeats}")
            for category, question in LATENCY_SET:
                await asyncio.sleep(args.pause)
                save(await ask(client, args.url, category, question))


if __name__ == "__main__":
    asyncio.run(main())
