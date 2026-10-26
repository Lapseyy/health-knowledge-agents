# Evaluation

Measured on October 26, 2026, 00:18–02:09 PDT, against Cloud Run revision
`health-knowledge-agents-00003-kbb`. Two RAGAS metric calls that hit a Vertex AI rate limit were
re-scored the same morning. Every number here comes from the raw results in
[`docs/eval/`](eval/), one JSON line per answer, score, router call, and deployed request;
[`docs/eval/report.md`](eval/report.md) holds the generated tables. This document explains them.

## Summary

- **Quality, as deployed** (RAGAS, 30 textbook questions): faithfulness 0.80, answer relevancy
  0.89, context precision 0.73, context recall 0.88.
- **Latency, deployed.** For a warm textbook question, first words arrive after a median of 7.2 s
  and the answer completes at 10.1 s. A web question takes 18.0 s to first words and a chapter
  summary 28.2 s. A cold request reaches first words after a median of 34.2 s; about 20 s of that
  is the instance starting.
- **Cost:** at list prices, a textbook question costs $0.0003 and a question that searches the
  web costs $0.013, most of it for the Tavily search. Per query, only the Gemini tokens were
  billed; NVIDIA, Tavily, and Cloud Run usage stayed within their free tiers.
- **Providers.** With Research on Gemini, faithfulness was 0.13 higher (95% CI +0.06 to +0.19),
  but the grader was Gemini itself. Research on Nemotron had the shorter tail (p90 first words
  25 s against 45 s) and never failed over; Research on Gemini failed over on 3 of 30. Both
  routers chose the right agents for 26 of 26 messages, and Lightning was about 3× faster.
- **Findings:**
  - Research searched the web on 10 of 30 textbook questions. That doubled their time to first
    words and lowered their context precision from 0.81 to 0.56.
  - On the deployed service, Gemini's plan was used for only 5 of 18 requests, because Vertex AI
    was slow or failing that night. The hedged router kept the cost of that to a few seconds.

## 1. Setup

| Set | Size | What it tests |
|---|---|---|
| RAGAS questions (`data/eval_ragas.jsonl`) | 30 | OpenStax end-of-chapter critical-thinking questions from 28 chapters, with the book's own written answers as references. Questions that depend on a figure, or whose answer is too short to grade, are excluded (seed 599). |
| Routing messages (`data/eval_routing.jsonl`) | 26 | 22 single messages covering every agent, two-part questions, greetings, and an off-topic question; 4 follow-ups that need the conversation history |
| Deployed requests (`scripts/measure_latency.py`) | 18 | 7 questions (2 textbook, 2 web, 1 data, 1 section summary, 1 chapter summary), sent as 1 priming request, 2 warm passes, and 3 cold starts after 20 idle minutes each |

`scripts/build_eval_sets.py` writes both question sets, and both are committed, so every run
asks the same questions.

**Configurations.** Both configurations run the same code:

- **As deployed:** Gemini 3.8 Flash routes, with Nemotron 3.5 Lightning as the backup router.
  Research, Summary, and Data run on Nemotron 3 Ultra.
- **Research on Gemini:** the same, except that Research runs on Gemini 3.8 Flash, with Nemotron
  as its backup. This configuration is the provider comparison.

**How each part was measured.**

- **Local runs.** `scripts/evaluate.py generate` sends each question through the full graph
  in-process, using the API's own streaming code. It ran on a desktop calling the same model
  endpoints as the service. Stage times come from the trace: every routing decision, model call,
  and tool call carries its duration.
- **Deployed service.** `scripts/measure_latency.py` streams from the public URL. It records
  first words and total time on the client side, and the server's own timings from the final
  frame. A request counts as cold when the server reports it as the instance's first turn, on an
  instance under 60 s old.
- **RAGAS 0.4.3 metrics:** faithfulness, answer relevancy (strictness 3), context precision with
  reference, and context recall.
  - The judge is Gemini 3.8 Flash at temperature 0, called through structured output. Answer
    relevancy embeds with the same local bge-small model that retrieval uses.
  - The contexts are exactly what the answer call saw: the prefetched textbook chunks plus any
    chunks or web results the agent added.

## 2. Answer quality (RAGAS)

Mean ± standard deviation, n = 30 per configuration:

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| As deployed: Research on Nemotron 3 Ultra | 0.80 ± 0.19 | 0.89 ± 0.05 | 0.73 ± 0.27 | 0.88 ± 0.26 |
| Research on Gemini 3.8 Flash | 0.92 ± 0.13 | 0.89 ± 0.07 | 0.78 ± 0.28 | 0.87 ± 0.29 |

Paired differences (Gemini minus Nemotron) on the same 30 questions, with 95% bootstrap
confidence intervals from 10,000 resamples:

| Metric | Difference (95% CI) | Gemini higher / lower / equal |
|---|---|---|
| Faithfulness | +0.13 (+0.06 to +0.19) | 20 / 5 / 5 |
| Answer relevancy | −0.00 (−0.03 to +0.02) | 13 / 15 / 2 |
| Context precision | +0.05 (+0.02 to +0.10) | 13 / 3 / 14 |
| Context recall | −0.01 (−0.08 to +0.08) | 1 / 3 / 26 |

Retrieval on its own was checked separately; see the retrieval check in
[architecture.md](architecture.md#rag-pipeline): the source section was in the top 5 for 88% of
50 questions.

### What drives the scores

**Web searches lower context precision.** The contexts include any web results an agent
fetched, and precision is graded against the textbook's reference answer.

- Research on Nemotron searched the web for 10 of the 30 questions, although every question is a
  textbook question.
- Those 10 scored 0.56 on context precision, against 0.81 for the other 20. Faithfulness (0.79
  against 0.80) and recall (0.90 against 0.87) barely differed.
- Research on Gemini searched on 3 of 30, which explains most of its precision edge.
- The searches also cost time: the 10 questions had a median of 21.2 s to first words, against
  8.4 s for the rest.

The Research prompt already says to call no tools when the prefetched passages answer the
question. For these open-ended questions, Nemotron judged the passages insufficient.

**Faithfulness: some real additions, some grading errors.** The three lowest-scoring Nemotron
answers were checked against their contexts:

- **q26 (anaphylaxis, 0.32).** The answer adds symptoms that its sources don't mention (hives,
  nausea). It also ends with the two-sentence safety note required by the prompt's "not medical
  advice" rule. RAGAS treats every sentence as a claim, so the note itself counts as
  unsupported. q26 is the only one of the 30 answers that includes the note.
- **q24 (ultrasound, 0.41).** The answer adds specifics that no source contains: head and
  abdominal circumference, femur length, and MRI being more expensive and less accessible.
- **q06 (smooth muscle, 0.36).** Its specific claims are in the sources: "80 percent to 120
  percent of its resting length" (§10.4), and latch-bridges and the stress-relaxation response
  (§10.8). On this check the answer is supported, so the low score looks like a grading error.

**The judge is noisy.** Context precision and context recall don't depend on the answer, so two
runs with identical contexts should score identically. 9 questions had identical contexts in
both configurations:

- 7 of the 9 scored the same in both.
- q06's precision differed (0.70 against 0.76).
- q04's recall was 0.00 in one run and 1.00 in the other. Its contexts contain the reference's
  gap-junction explanation, so 1.00 is the right score.

A single score can therefore move by more than the gaps between the two configurations. Means
over 30 questions are steadier. The paired intervals above capture spread across questions, but
not the judge's own variation.

### Caveats

- **The judge graded its own answers.** It is Gemini 3.8 Flash, the same model that wrote the
  Research-on-Gemini answers. LLM judges tend to rate their own outputs higher (Panickssery et
  al., 2024), so the faithfulness gap may be overstated. Re-judging with a second model would
  separate the two effects.
- **Small sample:** 30 questions, one answer per question per configuration, and one scoring
  pass.
- **Recall is narrow.** The references are the book's short answers to open-ended questions, so
  recall measures whether the retrieved text covers them, not whether the answer is complete.

## 3. Latency

### 3.1 Deployed service, warm

The table gives medians in seconds. First words and total are measured on the client; the
stages come from the server's trace. Warm requests are the representative case.

| Request | n | First words | Total | Routing | Tool-round model calls | Web search | Chapter digests* | Answer call |
|---|---|---|---|---|---|---|---|---|
| Textbook question | 4 | 7.2 | 10.1 | 3.3 | 1.8 | – | – | 4.0 |
| Web question | 4 | 18.0 | 23.3 | 3.0 | 9.5 | 1.9 | – | 5.7 |
| Data (SQL) | 2 | 8.2 | 11.1 | 1.9 | 5.6 | – | – | 3.4 |
| Section summary | 2 | 6.2 | 18.4 | 0.6† | 3.8 | – | – | 13.0 |
| Chapter summary | 2 | 28.2 | 37.2 | 3.3 | 10.8 | – | 47.2 | 9.6 |

\* Summed over 6 digest calls that run in parallel, so this is longer than the time the user
waits. † Both section-summary requests failed over from Gemini; their routing time leaves out
the failed Gemini call, which this run didn't record.

Routing, the tool rounds, and the answer call run in sequence. The answer call streams, so
first words arrive early in it. A section summary shows its first words after 6 s and then
streams for another 12 s.

### 3.2 Cold and warm first words

| | n | First words (client) | First words (server) | Startup wait (client minus server) |
|---|---|---|---|---|
| Cold: first request to a new instance | 4 | median 34.2 (30.4–55.9) | median 14.2 (9.3–36.5) | median 20.2 (19.1–21.1) |
| Warm: textbook question | 4 | median 7.2 (5.5–12.0) | same as client | under 0.1 |

**Startup wait.** A cold request waits about 20 s while Cloud Run starts an instance. In that
time the instance:

- starts the container and Python, and loads the embedding model and the Chroma index;
- connects to Tavily;
- makes one small warm-up call to each model client.

**First turn after startup.** The first turn then ran 9–37 s on the server, against 5.5–12 s
warm. The extra time was in Nemotron calls, not in the app's own startup, which finishes before
the request is served:

- In the slowest sample, one tool round took 14.9 s and the answer call took 42 s.
- In two others, three tool rounds took about 9 s in total.

Four samples are too few to say whether NVIDIA's endpoint is consistently slower after a quiet
period.

**Against the reference deployment.** The assignment's reference deployment shows about
20–25 s cold and 1.5–2 s warm.

- The cold-start penalty here, about 20 s, is in line with it.
- Warm first words are slower: 7 s for a textbook question. The reason is that three model calls
  run before the first answer token:
  1. routing (1–5 s);
  2. a tool round, in which Research decides whether it needs more evidence (0.6–2.4 s);
  3. the answer call's own first token.

The page shows the routing decision and every tool call as they happen, so it isn't blank
during that time. §6 lists ways to remove one of those calls.

### 3.3 Local runs by stage

The 30 RAGAS questions, median / p90 seconds:

| Stage | As deployed | Research on Gemini |
|---|---|---|
| Routing (wall time, including any hedge wait) | 3.4 / 8.2 | 3.6 / 6.5 |
| Textbook retrieval (local Chroma) | 0.01 / 0.01 | 0.01 / 0.01 |
| Tool-round model calls | 4.7 / 13.8 | 2.5 / 18.7 |
| Web search, in the turns that searched | 4.1 (10 turns) | 2.2 (3 turns) |
| Answer call | 4.9 / 11.0 | 4.0 / 9.1 |
| **First words** | **10.7 / 25.0** | **8.7 / 45.3** |
| **Total** | **14.8 / 31.9** | **10.8 / 47.9** |
| Model calls per question, not counting routing | 3.6 | 2.5 |

**First words, as deployed.** Medians don't add exactly, but routing (3.4 s) and the tool rounds
(4.7 s) make up most of the 10.7 s. What remains, mostly the answer call's time to its first token,
was a median of about 0.7 s.

**Research on Gemini's tail** comes from four turns:

- q06 hit the 90 s agent limit and reran on Nemotron, reaching first words at 96 s.
- q07, q08, and q20 reached first words after 45–52 s. In each, a single Gemini call took
  19–40 s, although its output was short.

### 3.4 Routing

| | Local, as deployed (30) | Local, Research on Gemini (30) | Deployed (18) |
|---|---|---|---|
| Gemini's plan used | 22 (73%) | 19 (63%) | 5 (28%) |
| Backup router started because Gemini was slow (over 4 s) | 12 (Gemini still won 4) | 14 (Gemini still won 3) | 6 |
| Backup router started because Gemini failed | 0 | 0 | 7 |
| Routing wall time, median / p90 | 3.4 / 8.2 | 3.6 / 6.5 | 1.9 / 5.2 |

**The hedge works as designed.** A slow Gemini costs at most about 4 s before Lightning's plan
is used. A failing Gemini costs only the failed call, because Lightning starts immediately.

**Gemini on the deployed service.** During this run, Gemini's plan was used for only 5 of 18
deployed requests. In the 7 failures, the trace logged an error and Lightning's plan followed.

- The latency script saved error counts but not the messages, so the cause wasn't recorded.
- The RAGAS judge got 429 RESOURCE_EXHAUSTED responses from the same Gemini endpoint that night,
  which makes Vertex AI's shared capacity the likely cause.
- The scripts now keep the full trace, and the server logs failed plans, so the next run will
  record the cause.

## 4. Cost per query

Costs are estimated from the token counts in each trace, at the prices below, which were
checked on October 26, 2026:

| Item | Price |
|---|---|
| Gemini 3.8 Flash, Vertex AI global endpoint ([pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)) | $0.75 per 1M input tokens and $3.75 per 1M output tokens, including reasoning. This introductory price runs through Dec 31, 2026; from Jan 1, 2027 it is $1.50 / $7.50. |
| Nemotron 3 Ultra and Nemotron 3.5 Lightning, NVIDIA API catalog ([build.nvidia.com](https://build.nvidia.com)) | $0 per token on the free developer tier, which is rate-limited per minute |
| Tavily search ([API credits](https://docs.tavily.com/documentation/api-credits)) | 1 credit per basic search, at $0.008 per credit pay-as-you-go; 1,000 free credits a month |
| Cloud Run, us-west1 (Tier 1), request-based billing ([pricing](https://cloud.google.com/run/pricing)) | $0.000024 per vCPU-second, $0.0000025 per GiB-second, and $0.40 per million requests. Free each month: 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million requests. |
| Embeddings | $0: the bge-small model runs locally |
| Storage | None: the index is baked into the image, and there is no database |

**Method.** For each query, the cost is the sum of:

- the tokens for each model, times that model's price;
- the number of web searches, times $0.008;
- the request time, times the rate for 1 vCPU and 2 GiB, plus the per-request fee.

These are list prices with no free tiers, so they give the cost of one more query once the
free allowances are used up. For the local runs, the local turn duration stands in for Cloud Run
time.

| Measured on | Mean per query | Models | Web search | Cloud Run |
|---|---|---|---|---|
| Local, as deployed (30 questions) | $0.0056 | $0.0005 | $0.0045 | $0.0006 |
| Local, Research on Gemini (30 questions) | $0.0061 | $0.0047 | $0.0008 | $0.0006 |

As deployed, the median query cost $0.0011. The mean is higher because the 10 questions that
searched the web cost up to $0.018 each.

Deployed service, warm, mean per request:

| Textbook | Web | Data | Section summary | Chapter summary |
|---|---|---|---|---|
| $0.0003 | $0.0126 | $0.0009 | $0.0005 | $0.0014 |

**A web search dominates any turn it appears in.** One Tavily search ($0.008) costs more than
all of a typical turn's model calls combined: the Nemotron calls are free, and Gemini only
routes.

**Chapter summaries are token-heavy.** One makes 9 Nemotron calls with about 26,000 input
tokens. That is free here, but on a paid model it would be the most expensive request.

**From January 2027,** Gemini's price doubles. The as-deployed model cost rises to about $0.001
per query, and Research on Gemini to about $0.0095.

**What was actually paid.** NVIDIA (free tier), Tavily (within the 1,000 free monthly credits),
and Cloud Run (within its free tier) cost nothing per query. The Gemini tokens are billed to the
Google Cloud project, where trial credit covered them. Deploying has its own costs, which are not
per query: Cloud Build bills each build by the minute, and Artifact Registry stores the image.

**Not priced:**

- **Paid Nemotron.** A production deployment wouldn't rely on NVIDIA's rate-limited free
  endpoints. Self-hosting Nemotron or using a paid provider would add a per-token cost that this
  estimate leaves out.
- **The RAGAS judge.** Its own cost isn't reported because this run misattributed judge calls
  when it scored answers concurrently. `scripts/evaluate.py` now tracks them per answer, and
  `report.md` marks this run's judge usage as not measured.

## 5. Provider comparison

### Research agent: Nemotron 3 Ultra or Gemini 3.8 Flash

| | Nemotron 3 Ultra (deployed) | Gemini 3.8 Flash |
|---|---|---|
| Faithfulness | 0.80 | 0.92 (graded by Gemini) |
| Answer relevancy | 0.89 | 0.89 |
| Context precision | 0.73 | 0.78 |
| Context recall | 0.88 | 0.87 |
| First words, median / p90 | 10.7 / 25.0 s | 8.7 / 45.3 s |
| Total, median / p90 | 14.8 / 31.9 s | 10.8 / 47.9 s |
| Questions that searched the web | 10 of 30 | 3 of 30 |
| Failed over to the other provider | 0 of 30 | 3 of 30 (one after the 90 s limit) |
| Answer length, median | 178 words | 220 words |
| Model cost per query | $0.0005 (routing only) | $0.0047 |

**Decision: keep Research on Nemotron.**

- **Quality.** Gemini's faithfulness lead holds across this data, but Gemini graded it. Most of
  its precision lead comes from Nemotron's unneeded web searches, a prompt issue that can be
  fixed on Nemotron (§6).
- **Reliability.** Research on Gemini failed over on 3 of 30 questions and took 45–52 s on 3
  more. Research on Nemotron never failed over. On the deployed service, the same Vertex AI
  endpoint was slow or failed on 13 of 18 routing calls that night. During routing, the hedge
  holds a failure's cost to a few seconds; during a long answer, a failure loses the whole call.
- **Cost.** Research on Gemini costs about 10× as much in model fees per query, though still
  under half a cent.

### Router: Gemini 3.8 Flash or Nemotron 3.5 Lightning

Each of the 26 routing messages went to both routers, with the production prompt and input
format.

| | Gemini 3.8 Flash | Nemotron 3.5 Lightning |
|---|---|---|
| Correct agents | 26 / 26 | 26 / 26 |
| Correct agents in the expected order | 26 / 26 | 26 / 26 |
| Latency, median / p90 | 2.8 / 10.1 s | 1.0 / 1.7 s |
| Calls over 4 s (the hedge threshold) | 8 of 26 (longest 17.2 s) | 2 of 26 |
| Tokens per call, input / output | 617 / 55 | 386 / 56 |
| Cost per plan | $0.00067 | $0 (free tier) |

**Both were accurate.** Both routers handled the two-part questions and the follow-ups. For
example, both rewrote "And the right one?" as "What does the right ventricle do?".

**Lightning is the better router on speed alone,** and it is free. Gemini stays the primary
router for two reasons:

- It keeps NVIDIA's per-minute request budget for the agents, which make 2–9 NVIDIA calls per
  turn.
- The assignment requires both providers to do real work in normal use, and routing is where
  Gemini's slowness costs least, because the hedge caps it at about 4 s.

## 6. Findings and next changes

1. **Unneeded web searches.** Research searched the web on 10 of 30 textbook questions. Each of
   those costs about 13 s more to first words and $0.008 for the search, and lowers context
   precision by 0.25.
   - **Change:** let the router's plan say whether a question needs current information, and
     skip Research's tool round when it doesn't. Alternatively, tighten the tool rule.
   - **Check:** re-run `generate` and `score` for the deployed configuration.
2. **Gemini routing on the deployed service** was slow or failing on 13 of 18 requests.
   - **Next:** re-measure first. The scripts now record why each plan failed.
   - **Levers left:** less thinking isn't one. `low` is already set, `minimal` returns 400 for
     this model, and its regional endpoint returned 404, so it runs on the global endpoint.
     What remains is the hedge threshold: a shorter wait makes routing faster, but Gemini's plan
     is then used less often. The other option is Provisioned Throughput, which reserves Vertex
     AI capacity for a fixed fee.
3. **Warm first words** for a textbook question take 7 s, because three model calls run before
   the first token.
   - **Change:** a textbook-only fast path that answers straight from the prefetched passages
     would remove the tool round, which took 0.6–2.4 s warm, and up to 15 s in the cold
     samples.
4. **Faithfulness.** Answers sometimes add detail that is not in their sources (q24, q26).
   - **Change:** state the "use only these sources" rule more strongly in the Research answer
     prompt.
   - **Change:** drop the medical-advice sentence from answers. The page already shows that
     note below the question box, and in an answer RAGAS counts it as an unsupported claim.
5. **Evaluation noise.** Judge scores vary between identical inputs (q04's recall was 0.00 in
   one run and 1.00 in the other).
   - **Change:** average two judges, one from each provider; that also removes the
     self-preference question.
   - **Change:** add more questions.

## 7. Reproducing

The commands are in the [README](../README.md#evaluation). On Windows,
`.\scripts\run_evaluation.ps1 -Cold 3` runs everything in order. Results:

| File | Contents |
|---|---|
| `docs/eval/default.jsonl`, `docs/eval/research-gemini.jsonl` | Answers, the contexts each answer call saw, the plan, and a timing and token breakdown per question |
| `docs/eval/default.scored.jsonl`, `docs/eval/research-gemini.scored.jsonl` | RAGAS scores per question |
| `docs/eval/router.jsonl` | Each router's plan, latency, and tokens per message |
| `docs/eval/latency.jsonl` | Deployed requests: client and server timings, instance turn and uptime, trace breakdown |
| `docs/eval/report.md`, `docs/eval/run.log` | Generated tables, and the run's console output |

## Sources

- [Vertex AI generative AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [Tavily API credits](https://docs.tavily.com/documentation/api-credits)
- [NVIDIA API catalog](https://build.nvidia.com)
- Panickssery, Bowman, and Feng, [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076) (2024)
