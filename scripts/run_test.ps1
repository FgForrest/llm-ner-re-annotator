$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

function Write-Log {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Err {
    param([string]$Message)
    Write-Host ""
    Write-Host "!! $Message" -ForegroundColor Red
}

function Run-Step {
    param(
        [string]$Command,
        [string]$ErrorMessage,
        [string]$DownCommand
    )

    Invoke-Expression $Command

    if ($LASTEXITCODE -ne 0) {
        Write-Err "$ErrorMessage Exit code: $LASTEXITCODE"

        if ($DownCommand -and $DownCommand.Trim().Length -gt 0) {
            Invoke-Expression $DownCommand
        }

        exit $LASTEXITCODE
    }
}


if (-not (Test-Path "data")) {
    Write-Err "Missing data\ directory"
    exit 1
}

# Akceptujeme libovolný *.json (Stage export `export-*-latest.json` i genericky
# JSON). Stačí, aby byl alespoň jeden JSON soubor.
$jsonFiles = Get-ChildItem -Path "data" -Filter "*.json" -ErrorAction SilentlyContinue
if (-not $jsonFiles) {
    Write-Err "No *.json files found in data\"
    Write-Err "Copy your input export(s) into data\, e.g.:"
    Write-Err "  data\export-mydataset-latest.json"
    Write-Err "(Any other *.json is handled by the generic reader.)"
    exit 1
}
Write-Log ("Detected {0} JSON file(s) in data\: {1}" -f $jsonFiles.Count, ($jsonFiles.Name -join ", "))

try {
    docker ps | Out-Null
} catch {
    Write-Err "Docker does not accept commands. Is Docker Desktop running?"
    exit 1
}

try {
    docker compose version | Out-Null
} catch {
    Write-Err "docker compose is not available."
    exit 1
}

Write-Log "Checking Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi | Out-Host

if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker GPU check failed. Check Docker Desktop, WSL2 integration and NVIDIA support."
    exit $LASTEXITCODE
}


Write-Log "Cleaning old test checkpoints..."

$files = @(
    "checkpoints\extractions.jsonl",
    "checkpoints\extractions_synthetic.jsonl",
    "checkpoints\critic_results.jsonl",
    "checkpoints\critic_results_synthetic.jsonl",
    "checkpoints\final_labels.jsonl",
    "checkpoints\final_labels_synthetic.jsonl",
    "checkpoints\final_labels.json",
    "checkpoints\final_labels_synthetic.json",
    "checkpoints\stats.json"
)

foreach ($f in $files) {
    if (Test-Path $f) {
        Remove-Item $f -Force
    }
}

if (-not (Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
}

$ComposeTest = "docker compose -f docker-compose.yml -f docker-compose.test.yml"
$ComposeMain = "docker compose -f docker-compose.yml"

Write-Log "PHASE 1 - mini extractor Qwen3-0.6B, limit=3 per dataset"

Run-Step `
    -Command "$ComposeTest --profile test-extract up --build --abort-on-container-exit --exit-code-from labeler-extract-sglang" `
    -ErrorMessage "Extract phase failed." `
    -DownCommand "$ComposeTest --profile test-extract down"

Invoke-Expression "$ComposeTest --profile test-extract down"

Write-Log "PHASE 2 - mini critic Gemma-3-1b (runs alone on GPU)"

Run-Step `
    -Command "$ComposeTest --profile test-critic up --build --abort-on-container-exit --exit-code-from labeler-critic-sglang" `
    -ErrorMessage "Critic phase failed." `
    -DownCommand "$ComposeTest --profile test-critic down"

Invoke-Expression "$ComposeTest --profile test-critic down"

Write-Log "PHASE 2b - revisor (Qwen3-0.6B) walks non-ok critics and fills extractor_revised"

Run-Step `
    -Command "$ComposeTest --profile test-revise up --build --abort-on-container-exit --exit-code-from labeler-revise-sglang" `
    -ErrorMessage "Revise phase failed." `
    -DownCommand "$ComposeTest --profile test-revise down"

Invoke-Expression "$ComposeTest --profile test-revise down"

Write-Log "PHASE 3 - finalize"

Run-Step `
    -Command "$ComposeMain --profile finalize up --build --abort-on-container-exit --exit-code-from labeler-finalize" `
    -ErrorMessage "Finalize phase failed." `
    -DownCommand "$ComposeMain --profile finalize down"

Invoke-Expression "$ComposeMain --profile finalize down"

Write-Log "RESULTS"

$finals = @(
    "final_labels.json",
    "final_labels_synthetic.json",
    "final_labels.jsonl",
    "final_labels_synthetic.jsonl",
    "stats.json"
)

foreach ($name in $finals) {
    $path = "checkpoints\$name"

    if (Test-Path $path) {
        if ($name.EndsWith(".jsonl")) {
            $lineCount = (Get-Content $path | Measure-Object -Line).Lines
            Write-Host "  OK  $path  ($lineCount lines)"
        } else {
            $size = (Get-Item $path).Length
            Write-Host "  OK  $path  ($size B)"
        }
    } else {
        Write-Host "  MISSING  $path"
    }
}

Write-Host ""
Write-Host "Intermediate files after finalize should be deleted:"

$intermediates = @(
    "extractions.jsonl",
    "extractions_synthetic.jsonl",
    "critic_results.jsonl",
    "critic_results_synthetic.jsonl"
)

foreach ($name in $intermediates) {
    $path = "checkpoints\$name"

    if (Test-Path $path) {
        Write-Host "  WARNING  $path still exists"
    } else {
        Write-Host "  OK       $path deleted"
    }
}

Write-Log "stats.json"

if (Test-Path "checkpoints\stats.json") {
    Get-Content "checkpoints\stats.json" | Out-Host
} else {
    Write-Err "checkpoints\stats.json is missing"
}

Write-Log "DONE"
Write-Host "Send the content of checkpoints folder, especially:"
Write-Host "  - final_labels.json"
Write-Host "  - final_labels_synthetic.json"
Write-Host "  - final_labels.jsonl"
Write-Host "  - final_labels_synthetic.jsonl"
Write-Host "  - stats.json"