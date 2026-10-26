# Health knowledge agents

A multi-agent assistant that answers health questions from three kinds of evidence: an anatomy
and physiology textbook (retrieval-augmented generation over Chroma), the live web (Tavily
Remote MCP), and a synthetic patient database (read-only SQL). A LangGraph supervisor plans
each turn and routes the work across two LLM providers, Google Gemini on Vertex AI and NVIDIA
Nemotron, with failover between them. A FastAPI backend streams answers over Server-Sent Events
to a plain HTML/CSS/JS page that shows, for every answer, which agent and model did each step
and why, a latency timeline of every model call and tool call, and the sources behind each
citation.

Built for CSCI 599 (Agentic AI) at USC, Fall 2026.

**Live:** https://health-knowledge-agents-1082060235449.us-west1.run.app. It scales to zero when idle, so the first request after a quiet period waits while an instance starts.

## How it works

| Agent | Model (provider) | Why this provider | Tools |
|---|---|---|---|
| Supervisor | Gemini 3.8 Flash (Google); backup Nemotron 3.5 Lightning (NVIDIA) | Cost: one short structured call per turn | none; plans the agents, rewrites follow-ups |
| Research | Nemotron 3 Ultra (NVIDIA); backup Gemini | Capability: multi-source synthesis with citations | `search_textbook`, `search_web` |
| Summary | Nemotron 3 Ultra (NVIDIA); backup Gemini | Capability: long, faithful inputs | `find_sections`, `read_section`, `summarize_chapter`, `search_web`, `extract_webpage` |
| Data | Nemotron 3 Ultra (NVIDIA); backup Gemini | Capability: SQL tool calling | `run_sql` |

- **Routing and failover.** If Gemini errors (for example a 429 from Vertex AI's shared
  capacity), the backup router plans immediately; if it is merely slow, the backup starts after
  4 s and the first plan back wins. Any agent whose provider fails reruns on the other provider.
  Every decision, fallback, and cancelled call is in the trace with its reason.
- **RAG.** The OpenStax *Anatomy & Physiology 2e* textbook (197 sections, ~415k words) is split
  by headings into 2,679 chunks and embedded locally with `BAAI/bge-small-en-v1.5`. Research
  retrieves the top 5 chunks for every question; answers cite them as numbered sources.
- **MCP tool sharing.** Tavily's `tavily_search` reaches Research and Summary as the same
  `search_web` tool: Research uses it for current information, Summary to find an article.
- **Chapter summaries.** `summarize_chapter` reads every section of a chapter in full, condenses
  each one in parallel, then writes the summary from those digests, citing each section.

## Run locally

Requirements: Python 3.12, the [gcloud CLI](https://cloud.google.com/sdk/docs/install) (Gemini
runs on Vertex AI with your Google Cloud credentials), an
[NVIDIA API key](https://build.nvidia.com), and a [Tavily API key](https://tavily.com).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env              # then fill in the keys and GOOGLE_CLOUD_PROJECT
gcloud auth application-default login

python -m src.rag.ingest          # builds chroma_index/ (downloads the 70 MB embedding model once)
python -m pytest -q
python main.py                    # open http://localhost:8080
```

`python scripts/ask.py "What does the left ventricle do?"` asks from the command line and prints
the trace as it happens.

## Deploy to Cloud Run

Requirements: a Google Cloud project with billing enabled, the gcloud CLI signed in
(`gcloud auth login`) as a project Owner, and `.env` filled in, including `GCP_PROJECT_ID`
(the defaults deploy service `health-knowledge-agents` to `us-west1`). From the repository root:

```bash
bash scripts/deploy.sh            # Cloud Shell, macOS, or Linux
```
```powershell
.\scripts\deploy.ps1              # Windows PowerShell
```

The script is safe to re-run and prints the live URL at the end. It:

1. Enables Cloud Run, Cloud Build, Artifact Registry, Vertex AI, and Secret Manager.
2. Creates an Artifact Registry repository for the image.
3. Creates a runtime service account (`hka-runtime`) with the Vertex AI User role, so the
   service calls Gemini with Google Cloud credentials instead of an API key.
4. Stores `NVIDIA_API_KEY` and `TAVILY_API_KEY` in Secret Manager, readable only by that account.
5. Builds the image with Cloud Build on an 8-vCPU machine (e2-highcpu-8, billed per build-minute;
   embedding the textbook dominates the build). The Dockerfile installs the pinned packages and
   builds the Chroma index inside the image, so no local artifacts are needed.
6. Deploys a public service with the settings below.

<details>
<summary>The equivalent commands</summary>

```bash
set -a; source .env; set +a
export CLOUDSDK_CORE_PROJECT=$GCP_PROJECT_ID
REGION=us-west1; SERVICE=health-knowledge-agents
IMAGE=$REGION-docker.pkg.dev/$GCP_PROJECT_ID/health-agents/$SERVICE
SA=hka-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com secretmanager.googleapis.com
gcloud artifacts repositories create health-agents --repository-format=docker --location=$REGION
gcloud iam service-accounts create hka-runtime
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID --member=serviceAccount:$SA \
  --role=roles/aiplatform.user --condition=None
for pair in nvidia-api-key:NVIDIA_API_KEY tavily-api-key:TAVILY_API_KEY; do
  gcloud secrets create ${pair%%:*} --replication-policy=automatic
  var=${pair##*:}; printf '%s' "${!var}" | gcloud secrets versions add ${pair%%:*} --data-file=-
  gcloud secrets add-iam-policy-binding ${pair%%:*} --member=serviceAccount:$SA \
    --role=roles/secretmanager.secretAccessor
done
gcloud builds submit --tag $IMAGE --machine-type=e2-highcpu-8 --timeout=30m .
gcloud run deploy $SERVICE --image=$IMAGE --region=$REGION --platform=managed \
  --allow-unauthenticated --service-account=$SA \
  --memory=2Gi --cpu=1 --cpu-boost --concurrency=10 --timeout=300 \
  --min-instances=0 --max-instances=1 \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=$GCP_PROJECT_ID,GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=true \
  --set-secrets=NVIDIA_API_KEY=nvidia-api-key:latest,TAVILY_API_KEY=tavily-api-key:latest
```
</details>

| Setting | Value | Why |
|---|---|---|
| Memory, CPU | 2 GiB, 1 vCPU with startup boost | The embedding model and index load at startup |
| Concurrency | 10 | Streaming requests are I/O-bound; NVIDIA's free-tier rate limit is the real ceiling |
| Request timeout | 300 s | Streams finish well under a minute |
| Instances | min 0, max 1 | Scales to zero when idle; one instance holds every conversation (below) |

**Conversation memory is not durable.** Follow-up questions work because each session's
history is kept in the memory of the running instance. With one instance, every request in a
session reaches it, but the history is lost when Cloud Run scales the service to zero after a
period without traffic or replaces the instance. The app does not claim memory beyond that.
The vector index is separate: it is baked into the image and read-only.

**Cold starts.** The first request to an idle service waits while an instance starts: it loads
the embedding model and index, connects to Tavily, and makes one small call to each model so
the first real question isn't slowed by connection setup. Requests wait rather than fail. To
keep an instance warm (billed while idle):
`gcloud run services update health-knowledge-agents --region=us-west1 --min-instances=1`.

**Troubleshooting.**
- Windows says running scripts is disabled:
  `powershell -ExecutionPolicy Bypass -File .\scripts\deploy.ps1`.
- `gcloud builds submit` fails with `PERMISSION_DENIED: The caller does not have permission`
  on the first run: the script had just enabled Cloud Build, which can take a minute or two to
  finish setting up. Wait, then re-run the script; it skips the steps already done.
- The build itself fails with a permission error naming a service account: grant that account
  the Cloud Build Service Account role (`roles/cloudbuild.builds.builder`), then re-run.

## Configuration

Copy `.env.example` to `.env` for local runs; on Cloud Run, the deploy script sets these.

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY`, `TAVILY_API_KEY` | API keys (Secret Manager on Cloud Run) |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | Vertex AI project and location (`global`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` for Vertex AI; `false` plus `GOOGLE_API_KEY` for an AI Studio key |
| `NVIDIA_MODEL`, `NVIDIA_FAST_MODEL`, `GEMINI_MODEL`, `GEMINI_REASONING_EFFORT` | Model overrides |
| `WARM_UP` | `true` to warm the model clients at startup; `false` to measure a fully cold first turn |
| `GCP_PROJECT_ID`, `GCP_REGION`, `CLOUD_RUN_SERVICE` | Deploy target |

## Repository layout

```
main.py              FastAPI app entry point (reads PORT)
src/agents/          LangGraph graph, agents, tools, prompts, Tavily MCP client
src/rag/             chunking, local embeddings, Chroma ingest and retrieval
src/api/             SSE streaming endpoint and app setup
src/llm.py           routing table, fallbacks, model clients
static/              the web UI (no build step)
data/                textbook sections, evaluation questions, synthetic patient database
scripts/             deploy, CLI, data extraction, and checks
tests/               pytest suite (scripted models, no network)
```

## Data and licenses

- Textbook: *Anatomy and Physiology 2e*, OpenStax, Rice University, CC BY-NC-SA 4.0. The files
  in `data/` derived from it are shared under the same license.
- Patients: generated with [Synthea](https://github.com/synthetichealth/synthea) (Apache 2.0).
  Every patient is fictional; no real patient data is included.

See `NOTICE` for full attribution. Answers are educational information, not medical advice.
