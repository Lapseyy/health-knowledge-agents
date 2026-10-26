#!/usr/bin/env bash
# Deploy to Cloud Run from the repository root:   bash scripts/deploy.sh
#
# Works in Cloud Shell or any bash with the gcloud CLI, against a fresh project. Reads
# settings and API keys from .env (copy .env.example). Safe to re-run: existing
# resources are reused, and secrets get a new version only when the key changed.
#
#   1. Enable the APIs (Cloud Run, Cloud Build, Artifact Registry, Vertex AI, Secret Manager)
#   2. Create an Artifact Registry repository for the image
#   3. Create the service's runtime account and grant it Vertex AI access (Gemini via
#      Google Cloud credentials, no API key)
#   4. Store NVIDIA_API_KEY and TAVILY_API_KEY in Secret Manager, readable by that account
#   5. Build the image with Cloud Build (the Chroma index is built inside the image)
#   6. Deploy: public URL, 1 instance max (conversation memory lives in that instance)
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] || { echo "Missing .env: copy .env.example to .env and fill it in." >&2; exit 1; }
set -a; source .env; set +a

PROJECT="${GCP_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${CLOUD_RUN_SERVICE:-health-knowledge-agents}"
: "${PROJECT:?Set GCP_PROJECT_ID in .env}"
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY in .env}"
: "${TAVILY_API_KEY:?Set TAVILY_API_KEY in .env}"

export CLOUDSDK_CORE_PROJECT="$PROJECT"  # every gcloud call below targets this project
REPO="health-agents"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE"
SA_NAME="hka-runtime"
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"

echo "==> Project $PROJECT, region $REGION, service $SERVICE"

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com secretmanager.googleapis.com

echo "==> Artifact Registry repository"
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"

echo "==> Runtime service account with Vertex AI access"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create "$SA_NAME" --display-name="Health knowledge agents (Cloud Run)"
for attempt in 1 2 3 4 5; do  # a brand-new account can take a few seconds to become visible to IAM
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
    --role=roles/aiplatform.user --condition=None --quiet >/dev/null && break
  [ "$attempt" = 5 ] && exit 1
  sleep 5
done

echo "==> Secrets"
put_secret() {  # name, value
  local name="$1" value="$2"
  gcloud secrets describe "$name" >/dev/null 2>&1 ||
    gcloud secrets create "$name" --replication-policy=automatic --quiet
  if [ "$(gcloud secrets versions access latest --secret="$name" 2>/dev/null || true)" != "$value" ]; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- --quiet
  fi
  gcloud secrets add-iam-policy-binding "$name" --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor --quiet >/dev/null
}
put_secret nvidia-api-key "$NVIDIA_API_KEY"
put_secret tavily-api-key "$TAVILY_API_KEY"

echo "==> Building the image (installs packages and builds the Chroma index)"
# 8 vCPUs: embedding the textbook dominates the build and took ~15 min on the default 2-vCPU machine.
gcloud builds submit --tag "$IMAGE" --machine-type=e2-highcpu-8 --timeout=30m .

ENV_VARS="GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION:-global},GOOGLE_GENAI_USE_VERTEXAI=true"
for var in NVIDIA_MODEL NVIDIA_FAST_MODEL GEMINI_MODEL GEMINI_REASONING_EFFORT; do
  if [ -n "${!var:-}" ]; then ENV_VARS+=",$var=${!var}"; fi
done

echo "==> Deploying"
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --service-account="$SA" \
  --memory=2Gi \
  --cpu=1 \
  --cpu-boost \
  --concurrency=10 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=1 \
  --set-env-vars="$ENV_VARS" \
  --set-secrets="NVIDIA_API_KEY=nvidia-api-key:latest,TAVILY_API_KEY=tavily-api-key:latest" \
  --quiet

echo "==> Live at $(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
