# Cost Analysis

Commercial model prices, free tiers, and student quotas change frequently. This file therefore defines the **measurement method** instead of hard-coding an evergreen dollar total.

## Reference routing configuration

| Role | Provider | Current reference model |
|---|---|---|
| Supervisor | OpenAI | `gpt-6-sol` |
| Research | Anthropic | `claude-sonnet-5` |
| Summary | Google | `gemini-3.8-flash` |
| Data | OpenAI | `gpt-6-sol` |
| RAGAS judge | OpenAI | `gpt-6-sol` |
| Embedding | OpenAI | `text-embedding-3-small` |

NVIDIA NIM may replace one or more roles while at least one other provider remains active. Current example: `nvidia/nemotron-3-ultra-550b-a55b`.

## What to record for a verification pass

For each query or evaluation batch, record:

1. provider and exact model ID;
2. input/output tokens when the SDK exposes them;
3. current provider price used for the estimate;
4. embedding tokens/cost for corpus ingestion;
5. Tavily usage under the currently available student/free plan;
6. Cloud Run compute estimate;
7. persistence cost when material.

## Why there is no NVIDIA-only "free path"

The old implementation allowed `LLM_BACKEND=nvidia` to route every role to a single provider. That no longer exists because Assignment 2 requires routing across at least two different providers.

A low-cost configuration may pair NVIDIA with another provider, for example Research on NVIDIA and Summary on Gemini, provided both are genuinely exercised.

## Persistence

The canonical reference uses Firestore. GCS/GCSFuse and Cloud SQL are valid alternatives. For coursework-scale verification, persistence charges are normally small, but current pricing should be checked rather than assumed.

## Release-time action

Before publishing a dollar estimate, re-check each provider's official pricing page and date-stamp the estimate in this file.
