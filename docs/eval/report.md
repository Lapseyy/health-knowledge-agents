# Evaluation results (prices as of 2026-10-26)

## RAGAS (mean ± sd; n = questions scored)

| Configuration | faithfulness | answer_relevancy | context_precision | context_recall | Judge calls | Judge cost |
|---|---|---|---|---|---|---|
| As deployed: Research on Nemotron 3 Ultra (NVIDIA) | 0.80 ± 0.19 (n=30) | 0.89 ± 0.05 (n=30) | 0.73 ± 0.27 (n=30) | 0.88 ± 0.26 (n=30) | not measured | not measured |
| Provider comparison: Research on Gemini 3.8 Flash (Google) | 0.92 ± 0.13 (n=30) | 0.89 ± 0.07 (n=30) | 0.78 ± 0.28 (n=30) | 0.87 ± 0.29 (n=30) | not measured | not measured |

Notes (missing scores are excluded from the means; re-score them with `score <config> --retry-errors`):

- default: judge usage not shown; 30 answers were scored before it was tracked per answer, and their counts overlap.
- research-gemini: judge usage not shown; 30 answers were scored before it was tracked per answer, and their counts overlap.

## Local runs: latency in seconds (median / p90) and usage per query

| Configuration | n | First words | Total | Routing | Retrieval | Web search | Model calls (sum) | Model calls | Web searches | Gemini planned | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| As deployed: Research on Nemotron 3 Ultra (NVIDIA) | 30 | 10.7 / 25.0 | 14.8 / 31.9 | 3.4 / 8.2 | 0.0 / 0.0 | 0.0 / 4.8 | 9.2 / 25.6 | 3.6 | 0.57 | 73% | $0.00557 |
| Provider comparison: Research on Gemini 3.8 Flash (Google) | 30 | 8.7 / 45.3 | 10.8 / 47.9 | 3.6 / 6.5 | 0.0 / 0.0 | 0.0 / 0.0 | 6.8 / 30.9 | 2.5 | 0.10 | 63% | $0.00609 |

**default**: the backup router ran because Gemini was slow (over 4 s) in 12 of 30 queries and because it failed in 0.
Model time by stage, median seconds in the queries that had it: answer 4.9 (n=30), tools 4.7 (n=30)
- gemini-3.8-flash: ran in 22 of 30 queries; per query overall 0.7 calls, 455 input and 39 output tokens
- nvidia/nemotron-3-ultra-550b-a55b: ran in 30 of 30 queries; per query overall 3.6 calls, 9,750 input and 351 output tokens
- nvidia/nemotron-3.5-lightning-30b-a3b: ran in 8 of 30 queries; per query overall 0.3 calls, 104 input and 19 output tokens
- cost per query: models $0.00049, web search $0.00453, Cloud Run $0.00055

**research-gemini**: the backup router ran because Gemini was slow (over 4 s) in 14 of 30 queries and because it failed in 0.
Model time by stage, median seconds in the queries that had it: answer 4.0 (n=30), tools 2.5 (n=30)
- gemini-3.8-flash: ran in 30 of 30 queries; per query overall 2.9 calls, 4,594 input and 343 output tokens
- nvidia/nemotron-3-ultra-550b-a55b: ran in 3 of 30 queries; per query overall 0.2 calls, 461 input and 23 output tokens
- nvidia/nemotron-3.5-lightning-30b-a3b: ran in 11 of 30 queries; per query overall 0.4 calls, 142 input and 28 output tokens
- cost per query: models $0.00473, web search $0.00080, Cloud Run $0.00056

## Router comparison

| Router | Model | n | Correct agents | Exact order | Errors | Latency median / p90 (s) | Tokens per call (in / out) |
|---|---|---|---|---|---|---|---|
| gemini | gemini-3.8-flash | 26 | 26/26 | 26/26 | 0 | 2.8 / 10.1 | 617 / 55 |
| nemotron-lightning | nvidia/nemotron-3.5-lightning-30b-a3b | 26 | 26/26 | 26/26 | 0 | 1.0 / 1.7 | 386 / 56 |

## Deployed service (client-side seconds, median / p90; n)

| Category | Warm first words | Warm total | Cold first words | Cold total |
|---|---|---|---|---|
| chapter | 28.2 / 33.3 (n=2) | 37.2 / 45.9 (n=2) | n/a | n/a |
| data | 8.2 / 9.5 (n=2) | 11.1 / 14.7 (n=2) | n/a | n/a |
| summary | 6.2 / 6.8 (n=2) | 18.4 / 21.2 (n=2) | n/a | n/a |
| textbook | 7.2 / 12.0 (n=4) | 10.1 / 18.8 (n=4) | 34.2 / 55.9 (n=4) | 35.5 / 79.0 (n=4) |
| web | 18.0 / 30.0 (n=4) | 23.3 / 31.1 (n=4) | n/a | n/a |

Cold requests: client first words minus server first words (startup and network), median / p90 20.2 / 21.1 s over 4 requests.

Warm requests: client first words minus server first words (startup and network), median / p90 0.0 / 0.1 s over 14 requests.

Routing on the deployed service: Gemini planned 5 of 18 requests; the backup router ran because Gemini was slow in 6 and because it failed in 7. Routing wall time median / p90 1.9 / 5.2 s.
