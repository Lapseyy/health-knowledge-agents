# Deploy to Cloud Run from Windows PowerShell:   .\scripts\deploy.ps1
#
# Same steps as scripts/deploy.sh (see that file for the full list). Reads settings and
# API keys from .env. Safe to re-run: existing resources are reused, and secrets get a
# new version only when the key changed.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) { throw "Missing .env: copy .env.example to .env and fill it in." }

$cfg = @{}
foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $cfg[$Matches[1]] = $Matches[2].Trim('"', "'") }
}
function Setting([string]$name, [string]$default = "") { if ($cfg[$name]) { $cfg[$name] } else { $default } }

$project = Setting "GCP_PROJECT_ID" (Setting "GOOGLE_CLOUD_PROJECT")
$region = Setting "GCP_REGION" "us-west1"
$service = Setting "CLOUD_RUN_SERVICE" "health-knowledge-agents"
$required = [ordered]@{ GCP_PROJECT_ID = $project; NVIDIA_API_KEY = (Setting "NVIDIA_API_KEY"); TAVILY_API_KEY = (Setting "TAVILY_API_KEY") }
foreach ($name in $required.Keys) { if (-not $required[$name]) { throw "Set $name in .env" } }

$env:CLOUDSDK_CORE_PROJECT = $project  # every gcloud call below targets this project
$repo = "health-agents"
$image = "$region-docker.pkg.dev/$project/$repo/$service"
$saName = "hka-runtime"
$sa = "$saName@$project.iam.gserviceaccount.com"

# Helpers are not named "gcloud": PowerShell resolves functions before programs, ignoring case,
# so a function called Gcloud would call itself instead of the gcloud CLI.
function Invoke-Gcloud([string[]]$GcloudArgs) {
    & gcloud @GcloudArgs
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($GcloudArgs -join ' ') failed" }
}
function Succeeds([string[]]$GcloudArgs) {  # runs gcloud quietly; true when it exits 0
    $previous = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & gcloud @GcloudArgs *> $null
    $ok = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previous
    return $ok
}

Write-Host "==> Project $project, region $region, service $service"

Write-Host "==> Enabling APIs"
Invoke-Gcloud @("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com",
         "aiplatform.googleapis.com", "secretmanager.googleapis.com")

Write-Host "==> Artifact Registry repository"
if (-not (Succeeds @("artifacts", "repositories", "describe", $repo, "--location=$region"))) {
    Invoke-Gcloud @("artifacts", "repositories", "create", $repo, "--repository-format=docker", "--location=$region")
}

Write-Host "==> Runtime service account with Vertex AI access"
if (-not (Succeeds @("iam", "service-accounts", "describe", $sa))) {
    Invoke-Gcloud @("iam", "service-accounts", "create", $saName, "--display-name=Health knowledge agents (Cloud Run)")
}
foreach ($attempt in 1..5) {  # a brand-new account can take a few seconds to become visible to IAM
    if (Succeeds @("projects", "add-iam-policy-binding", $project, "--member=serviceAccount:$sa",
                 "--role=roles/aiplatform.user", "--condition=None", "--quiet")) { break }
    if ($attempt -eq 5) { throw "Could not grant Vertex AI User to $sa" }
    Start-Sleep -Seconds 5
}

Write-Host "==> Secrets"
function Put-Secret([string]$name, [string]$value) {
    if (-not (Succeeds @("secrets", "describe", $name))) {
        Invoke-Gcloud @("secrets", "create", $name, "--replication-policy=automatic", "--quiet")
    }
    $previous = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $current = & gcloud secrets versions access latest "--secret=$name" 2> $null
    $ErrorActionPreference = $previous
    if ($current -ne $value) {
        # Write the key to a file with no trailing newline: piping a string in PowerShell adds one.
        $tmp = New-TemporaryFile
        try {
            [IO.File]::WriteAllText($tmp.FullName, $value)
            Invoke-Gcloud @("secrets", "versions", "add", $name, "--data-file=$($tmp.FullName)", "--quiet")
        } finally { Remove-Item $tmp.FullName -Force }
    }
    Invoke-Gcloud @("secrets", "add-iam-policy-binding", $name, "--member=serviceAccount:$sa",
             "--role=roles/secretmanager.secretAccessor", "--quiet") | Out-Null
}
Put-Secret "nvidia-api-key" (Setting "NVIDIA_API_KEY")
Put-Secret "tavily-api-key" (Setting "TAVILY_API_KEY")

Write-Host "==> Building the image (installs packages and builds the Chroma index)"
# 8 vCPUs: embedding the textbook dominates the build and took ~15 min on the default 2-vCPU machine.
Invoke-Gcloud @("builds", "submit", "--tag", $image, "--machine-type=e2-highcpu-8", "--timeout=30m", ".")

$envVars = "GOOGLE_CLOUD_PROJECT=$project,GOOGLE_CLOUD_LOCATION=$(Setting 'GOOGLE_CLOUD_LOCATION' 'global'),GOOGLE_GENAI_USE_VERTEXAI=true"
foreach ($name in @("NVIDIA_MODEL", "NVIDIA_FAST_MODEL", "GEMINI_MODEL", "GEMINI_REASONING_EFFORT")) {
    if (Setting $name) { $envVars += ",$name=$(Setting $name)" }
}

Write-Host "==> Deploying"
Invoke-Gcloud @("run", "deploy", $service,
    "--image=$image",
    "--region=$region",
    "--platform=managed",
    "--allow-unauthenticated",
    "--service-account=$sa",
    "--memory=2Gi",
    "--cpu=1",
    "--cpu-boost",
    "--concurrency=10",
    "--timeout=300",
    "--min-instances=0",
    "--max-instances=1",
    "--set-env-vars=$envVars",
    "--set-secrets=NVIDIA_API_KEY=nvidia-api-key:latest,TAVILY_API_KEY=tavily-api-key:latest",
    "--quiet")

$url = & gcloud run services describe $service "--region=$region" '--format=value(status.url)'
Write-Host "==> Live at $url"
