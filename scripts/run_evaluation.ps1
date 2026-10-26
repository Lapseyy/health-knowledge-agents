# Run the whole evaluation in order (Windows PowerShell), after the quick check passes:
#
#   .\scripts\run_evaluation.ps1             # local runs, then the deployed site warm (~1 hour)
#   .\scripts\run_evaluation.ps1 -Cold 3     # also 3 cold starts at the end (+~1 hour; good overnight)
#   .\scripts\run_evaluation.ps1 -SkipDeployed   # local steps and the report only
#
# Steps: answer the 30 questions (as deployed, then with Research on Gemini), score both with
# RAGAS, compare the routers, measure the live site, and write docs/eval/report.md. Everything
# printed is also appended to docs/eval/run.log. It stops at the first failed step; fix it and
# run the script again: generate, score, and router skip work already saved. The live-site
# measurements append new samples each time they run, so add -SkipDeployed when re-running
# only to finish the local steps. Cold starts need the site left idle: don't use it meanwhile.

param(
    [string]$Url = "https://health-knowledge-agents-1082060235449.us-west1.run.app",
    [int]$Cold = 0,
    [switch]$SkipDeployed
)

Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force "docs\eval" | Out-Null
$log = "docs\eval\run.log"
$env:PYTHONUNBUFFERED = "1"     # show progress lines as they happen, through the log pipe
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$runStarted = Get-Date

function Step([string]$title, [string[]]$arguments) {
    $line = "==> $title    ($(Get-Date -Format 'HH:mm'))"
    Write-Host "`n$line" -ForegroundColor Cyan
    Add-Content -Path $log -Value "`n$line"
    $started = Get-Date
    & python @arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n$title failed (exit code $LASTEXITCODE). Fix it, then run this script again:" -ForegroundColor Red
        Write-Host "finished work is skipped. Full output is in $log." -ForegroundColor Red
        exit 1
    }
    Write-Host ("    done in {0:n1} min" -f ((Get-Date) - $started).TotalMinutes) -ForegroundColor DarkGray
}

Step "Answer the 30 questions (as deployed: Research on Nemotron)" @("-m", "scripts.evaluate", "generate")
Step "Answer them again with Research on Gemini" @("-m", "scripts.evaluate", "generate", "--config", "research-gemini")
Step "Score the deployed configuration with RAGAS" @("-m", "scripts.evaluate", "score", "default")
Step "Score the Research-on-Gemini configuration with RAGAS" @("-m", "scripts.evaluate", "score", "research-gemini")
Step "Compare the two routers" @("-m", "scripts.evaluate", "router")
if (-not $SkipDeployed) {
    Step "Measure the live site, warm" @("-m", "scripts.measure_latency", "--url", $Url)
    if ($Cold -gt 0) {
        Step "Measure $Cold cold starts (20 min idle before each)" @("-m", "scripts.measure_latency", "--url", $Url, "--cold", "$Cold")
    }
}
Step "Write the report" @("-m", "scripts.evaluate", "report", "--out", "docs\eval\report.md")

Write-Host ("`nAll steps finished in {0:n0} min. Report: docs\eval\report.md" -f ((Get-Date) - $runStarted).TotalMinutes) -ForegroundColor Green
