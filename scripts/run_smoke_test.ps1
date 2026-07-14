# Production smoke test — empiricky odhad doby behu pipeline na cilovem HW.
#
# Co dela:
#   1. Z kazdeho datasetu v data/ vezme N items (default 50)
#   2. Postupne spusti vsechny 4 faze (extract/critic/revise/finalize) na
#      smoke vzorku — vzdy 1 model na GPU, jako v produkci
#   3. Meri wall-clock per faze
#   4. Linearne extrapoluje na full produkcni rozsah (pure empirical —
#      zadne teoreticke scaling factory, zadne GPU-class projekce)
#   5. Vygeneruje report (Markdown + JSON) v checkpoints/smoke/
#
# Pouziti (DOCKER runtime — default):
#   .\scripts\run_smoke_test.ps1                                       # interaktivni prompt
#   .\scripts\run_smoke_test.ps1 -GpuCount 2                           # preskoci prompt
#   .\scripts\run_smoke_test.ps1 -SamplesPerDataset 100
#   .\scripts\run_smoke_test.ps1 -SmokePhases "extract","critic","finalize"
#
# Pouziti (NATIVE runtime — bez dockeru):
#   # 1) Sam si spust LLM server, napr. v jinem terminalu:
#   #    python -m sglang.launch_server --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 ...
#   # 2) Spust skript v native modu:
#   .\scripts\run_smoke_test.ps1 -Runtime native
#   # 3) Pokud mas LLM server jinde nez localhost:8000:
#   $env:EXTRACTOR_BASE_URL = "http://10.0.0.5:8000/v1"
#   $env:CRITIC_BASE_URL = "http://10.0.0.5:8001/v1"
#   .\scripts\run_smoke_test.ps1 -Runtime native
#
# Skript se na zacatku zepta kolik B200 GPU mas (1, 2 nebo 4 - power-of-2
# pro tensor parallelism MoE). Tento pocet pak nastavi EXTRACTOR_TP_SIZE
# a HW_LABEL pro report. Pokud -GpuCount je predany, prompt se preskoci.

param(
    [int]$GpuCount = 0,                       # 0 = interactive prompt
    [string]$HwLabel = "",                    # auto-derive from GpuCount if empty
    [int]$SamplesPerDataset = 100,
    [string[]]$SmokePhases = @("extract", "critic", "revise", "finalize"),
    [double]$DockerStartupS = 0.0,
    [ValidateSet("docker", "native")]
    [string]$Runtime = "docker"               # docker (default) / native (host Python proti bezicimu LLM serveru)
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

# ---------------------------------------------------------------------------
# Preflight checks + Interactive GPU prompt
# ---------------------------------------------------------------------------
function Get-DetectedGpuCount {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return 0
    }
    $out = & nvidia-smi --list-gpus 2>$null | Measure-Object | Select-Object -ExpandProperty Count
    if ($out -eq $null) { return 0 }
    return [int]$out
}

function Test-QuantizationPreflight {
    $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
    $quant = $env:EXTRACTOR_QUANTIZATION
    if ($quant -and $model -match "(NVFP4|FP4|nvfp4|fp4)") {
        Write-Err "Quantization mismatch detected:"
        Write-Err "  EXTRACTOR_MODEL=$model (NVFP4 model)"
        Write-Err "  EXTRACTOR_QUANTIZATION=$quant (manualni override)"
        Write-Err "NVFP4 modely maji quantization v config.json - auto-detect."
        Write-Err "Fix: zakomentuj EXTRACTOR_QUANTIZATION v .env"
        throw "Quantization mismatch (EXTRACTOR)"
    }
    $cmodel = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
    $cquant = $env:CRITIC_QUANTIZATION
    if ($cquant -and $cmodel -match "(NVFP4|FP4|nvfp4|fp4)") {
        Write-Err "Quantization mismatch detected:"
        Write-Err "  CRITIC_MODEL=$cmodel (NVFP4 model)"
        Write-Err "  CRITIC_QUANTIZATION=$cquant (manualni override)"
        Write-Err "Fix: zakomentuj CRITIC_QUANTIZATION v .env"
        throw "Quantization mismatch (CRITIC)"
    }
}

function Test-HfTokenPreflight {
    $t = $env:HF_TOKEN
    if (-not $t -or $t -eq "hf_TADY_DOPLN_TOKEN") {
        Write-Warn "HF_TOKEN neni nastaveny nebo ma placeholder hodnotu."
        Write-Warn "Gated modely (Gemma, restricted Llama) NEPUJDOU stahnout."
    }
}

# ---------------------------------------------------------------------------
# .env auto-config — po prompt podle GpuCount
# ---------------------------------------------------------------------------
function Update-EnvVar {
    param([string]$File, [string]$Key, [string]$Value)
    if (-not (Test-Path $File)) { return }
    $lines = Get-Content $File
    $found = $false
    $newLines = foreach ($line in $lines) {
        if ($line -match "^[ \t]*${Key}=") {
            $found = $true
            "${Key}=${Value}"
        } else {
            $line
        }
    }
    if (-not $found) {
        $newLines = @($newLines) + "${Key}=${Value}"
    }
    Set-Content -Path $File -Value $newLines -Encoding utf8
}

function Set-EnvForGpuCount {
    param([int]$Count)
    $envFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { ".env" }
    if (-not (Test-Path $envFile)) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" $envFile
            Write-Log "Vytvoreno $envFile z .env.example (auto-config)"
        } else {
            Write-Warn "Neexistuje $envFile ani .env.example — auto-config preskocen."
            return
        }
    }
    Update-EnvVar -File $envFile -Key "GPU_COUNT" -Value "$Count"
    Update-EnvVar -File $envFile -Key "EXTRACTOR_TP_SIZE" -Value "$Count"
    Update-EnvVar -File $envFile -Key "CRITIC_TP_SIZE" -Value "1"
    Write-Log "Auto-config ${envFile}: GPU_COUNT=$Count EXTRACTOR_TP_SIZE=$Count CRITIC_TP_SIZE=1"
}

function Read-GpuCount {
    param([int]$Detected)
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════"
    Write-Host "  Kolik NVIDIA B200 GPU mas k dispozici pro pipeline?"
    Write-Host "  Podporovane hodnoty: 1, 2, 4 nebo 8"
    Write-Host "  (jen power-of-2 - TP=3/5/6/7 SGLang/vLLM nepodporuji)"
    if ($Detected -gt 0) {
        Write-Host "  -> Detekovano $Detected GPU na hostu"
    }
    Write-Host "════════════════════════════════════════════════════════"
    while ($true) {
        $n = Read-Host "Zadej pocet B200 GPU [1/2/4/8]"
        if ($n -in @("1","2","4","8")) {
            $nInt = [int]$n
            if ($Detected -gt 0 -and $nInt -gt $Detected) {
                Write-Warning "Host ma jen $Detected GPU, zadano $nInt. Vyber max $Detected."
                continue
            }
            return $nInt
        }
        Write-Warning "Neplatna hodnota '$n'. Zadej 1, 2, 4 nebo 8."
    }
}

Test-QuantizationPreflight
Test-HfTokenPreflight

$detected = Get-DetectedGpuCount
if ($detected -gt 0) {
    Write-Log "Detekovano $detected GPU na hostu"
} else {
    Write-Warn "nvidia-smi nedostupne - nemuzu detekovat GPU count automaticky"
}

if ($GpuCount -eq 0) {
    $GpuCount = Read-GpuCount -Detected $detected
} elseif ($GpuCount -notin @(1, 2, 4, 8)) {
    Write-Error "-GpuCount musi byt 1, 2, 4 nebo 8 (dostal $GpuCount)"
    exit 1
} elseif ($detected -gt 0 -and $GpuCount -gt $detected) {
    Write-Err "GpuCount=$GpuCount, ale host ma jen $detected GPU."
    Write-Err "Docker by selhal s 'nvidia-container-cli: device error'."
    throw "GPU count mismatch"
}

# Auto-config .env podle GpuCount — pro docker compose i native
Set-EnvForGpuCount -Count $GpuCount

# Propagace do env pro docker compose
$env:EXTRACTOR_TP_SIZE = "$GpuCount"
if (-not $env:CRITIC_TP_SIZE) { $env:CRITIC_TP_SIZE = "1" }

if ([string]::IsNullOrWhiteSpace($HwLabel)) {
    $HwLabel = "${GpuCount}x B200"
}

$ExtractorBackends = if ($env:EXTRACTOR_BACKENDS) { $env:EXTRACTOR_BACKENDS -split ' ' } else { @("sglang", "vllm") }
$CriticBackends    = if ($env:CRITIC_BACKENDS)    { $env:CRITIC_BACKENDS    -split ' ' } else { @("sglang", "vllm") }
$ReviseBackends    = if ($env:REVISE_BACKENDS)    { $env:REVISE_BACKENDS    -split ' ' } else { @("sglang", "vllm") }

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
function Test-CudaPreflight {
    if ($env:SKIP_CUDA_CHECK -eq "1") {
        Write-Warn "SKIP_CUDA_CHECK=1 — preskakuji CUDA verifikaci."
        return
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warn "nvidia-smi nenalezen — nemuzu verifikovat CUDA verzi."
        return
    }
    $smiOutput = & nvidia-smi 2>$null | Out-String
    $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $match.Success) { return }
    $major = [int]$match.Groups[1].Value
    $minor = [int]$match.Groups[2].Value
    if ($major -lt 13) {
        Write-Err "Detekovana CUDA $major.$minor — pipeline VYZADUJE CUDA 13+ pro NVFP4."
        Write-Err "Upgrade driveru >=580 a CUDA toolkit >=13.0."
        throw "CUDA 13+ required (detected $major.$minor)"
    }
    Write-Log "Preflight OK — CUDA $major.$minor detekovana (>=13.0)."
}

function Test-DataDir {
    if (-not (Test-Path "data")) {
        Write-Err "Chybi slozka data/ — nakopiruj tam vstupni JSON soubory"
        throw "data/ missing"
    }
    $jsonCount = (Get-ChildItem -Path "data" -Filter "*.json" -File | Measure-Object).Count
    if ($jsonCount -eq 0) {
        Write-Err "V data/ nenalezeny zadne *.json soubory"
        throw "no JSON files"
    }
    Write-Log "Detekovano $jsonCount JSON soubor(u) v data/"
}

# ---------------------------------------------------------------------------
# Per-phase smoke runner
# ---------------------------------------------------------------------------
function Invoke-PhaseViaProfile {
    param(
        [string]$Phase,
        [string]$Profile,
        [string]$LabelerService
    )
    Write-Log "Smoke FAZE [$Phase] — profile=$Profile (detached -d na pozadi)"
    # docker compose run -d prebehne up LLM dependency + spusti override command
    # v detached modu. Vrati container ID, ktery pak cekame pres docker wait.
    $cid = (docker compose --profile $Profile run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        $LabelerService `
        smoke-test --phase $Phase `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "[$Phase] docker compose run -d selhal na profilu $Profile"
        docker compose --profile $Profile down 2>&1 | Out-Null
        return $false
    }
    Write-Log "Container $cid bezi na pozadi. Sleduj: docker logs -f $cid"
    docker wait $cid 2>&1 | Out-Null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    docker compose --profile $Profile down 2>&1 | Out-Null
    if ($rc -eq 0) {
        Write-Log "[$Phase] hotov pres $Profile"
        return $true
    }
    Write-Err "[$Phase] selhal na profilu $Profile (rc=$rc)"
    return $false
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers — bez dockeru, Python proti bezicimu LLM serveru
# ---------------------------------------------------------------------------
# $NativeAutoLlm: 1 (default) = auto-install + auto-start/stop SGLang
#                 0 = manual (user si LLM zapne sam)
$NativeAutoLlm = if ($env:NATIVE_AUTO_LLM) { $env:NATIVE_AUTO_LLM } else { "1" }

# Setup env vars 1:1 jako Docker (parita s docker-compose.yml LLM service)
function Set-NativeEnvParity {
    if ($Runtime -ne "native") { return }
    if (-not $env:HF_XET_HIGH_PERFORMANCE)           { $env:HF_XET_HIGH_PERFORMANCE = "1" }
    if (-not $env:PYTHONWARNINGS)                    { $env:PYTHONWARNINGS = "ignore::DeprecationWarning,ignore::FutureWarning" }
    if (-not $env:TRANSFORMERS_VERBOSITY)            { $env:TRANSFORMERS_VERBOSITY = "error" }
    if (-not $env:TRANSFORMERS_NO_ADVISORY_WARNINGS) { $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1" }
    if (-not $env:EXTRACTOR_BASE_URL)                { $env:EXTRACTOR_BASE_URL = "http://localhost:8000/v1" }
    if (-not $env:CRITIC_BASE_URL)                   { $env:CRITIC_BASE_URL = "http://localhost:8001/v1" }
    if (-not $env:REVISOR_BASE_URL)                  { $env:REVISOR_BASE_URL = $env:EXTRACTOR_BASE_URL }
    Write-Log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress,"
    Write-Log "  EXTRACTOR_BASE_URL=$($env:EXTRACTOR_BASE_URL)"
    Write-Log "  CRITIC_BASE_URL=$($env:CRITIC_BASE_URL)"
}

# Sestavi SGLang args 1:1 jako docker-compose.yml command block.
function Get-SglangArgsForPhase {
    param([string]$Phase)
    $args = @()
    switch ($Phase) {
        { $_ -in "extract", "revise" } {
            $batch = if ($env:EXTRACTOR_BATCH_SIZE) { $env:EXTRACTOR_BATCH_SIZE } else { "32" }
            $ctx   = if ($env:EXTRACTOR_CONTEXT_LEN) { $env:EXTRACTOR_CONTEXT_LEN } else { "65536" }
            $args  = @("--max-running-requests=$batch", "--context-length=$ctx", "--reasoning-parser=qwen3")
            if ($env:EXTRACTOR_QUANTIZATION) { $args += "--quantization=$($env:EXTRACTOR_QUANTIZATION)" }
            $eff = if ($env:EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH) { $env:EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH } else { $env:DISABLE_PIECEWISE_CUDA_GRAPH }
            if ($eff -eq "1" -or $eff -eq "true") { $args += "--disable-piecewise-cuda-graph" }
        }
        "critic" {
            $batch = if ($env:CRITIC_BATCH_SIZE) { $env:CRITIC_BATCH_SIZE } else { "64" }
            $ctx   = if ($env:CRITIC_CONTEXT_LEN) { $env:CRITIC_CONTEXT_LEN } else { "65536" }
            $args  = @("--max-running-requests=$batch", "--context-length=$ctx")
            if ($env:CRITIC_QUANTIZATION) { $args += "--quantization=$($env:CRITIC_QUANTIZATION)" }
            $eff = if ($env:CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH) { $env:CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH } else { $env:DISABLE_PIECEWISE_CUDA_GRAPH }
            if ($eff -eq "1" -or $eff -eq "true") { $args += "--disable-piecewise-cuda-graph" }
        }
    }
    return ($args -join ' ')
}

Set-NativeEnvParity

function Test-NativeLlmEndpoint {
    param([string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri "$($Url.TrimEnd('/'))/models" `
            -Method Get -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        return ($resp.StatusCode -lt 400)
    } catch { return $false }
}

function Install-SgLangIfMissing {
    try {
        python -c "import sglang" 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    } catch {}
    Write-Warn "SGLang nenalezen."
    if ($NativeAutoLlm -ne "1") {
        Write-Err "NATIVE_AUTO_LLM=0 - rucne: .\scripts\install_sglang.ps1"
        return $false
    }
    Write-Log "Auto-install SGLang..."
    & "$PSScriptRoot\install_sglang.ps1"
    return ($LASTEXITCODE -eq 0)
}

function Start-NativeLlmForPhase {
    param([string]$Phase, [int]$Port)
    if (Test-NativeLlmEndpoint -Url "http://localhost:$Port/v1") {
        Write-Log "LLM server na portu $Port uz bezi - pouzivam ho"
        return $true
    }
    if ($NativeAutoLlm -ne "1") {
        Write-Err "Endpoint http://localhost:$Port/v1 NEODPOVIDA. Manual mode: spust sam:"
        Write-Err "  .\scripts\start_native_llm.ps1 <model> $Port $GpuCount"
        return $false
    }
    $model = ""
    $tpSize = 1
    switch ($Phase) {
        { $_ -in "extract", "revise" } {
            $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
            $tpSize = $GpuCount
        }
        "critic" {
            $model = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
            $tpSize = if ($env:CRITIC_TP_SIZE) { [int]$env:CRITIC_TP_SIZE } else { 1 }
        }
        default { return $true }
    }
    # Args 1:1 jako Docker
    $extra = Get-SglangArgsForPhase -Phase $Phase
    Write-Log "Auto-start SGLang pro '$Phase': $model na portu $Port (tp=$tpSize)"
    Write-Log "  Args: $extra"
    & "$PSScriptRoot\start_native_llm.ps1" -ModelPath $model -Port $Port -TpSize $tpSize -ExtraArgs $extra
    return ($LASTEXITCODE -eq 0)
}

function Stop-NativeLlmForPhase {
    param([int]$Port)
    if ($NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    if (Test-Path (Join-Path $LogDir "sglang_$Port.pid")) {
        Write-Log "Auto-stop SGLang na portu $Port"
        & "$PSScriptRoot\stop_native_llm.ps1" -Port $Port
    }
}

# Cleanup hook — pri exit (Ctrl+C, fail, normal) zastav VSECHNY SGLang
# servery, ktere jsme my spustili. Iteruje pres PID files misto hardcoded
# ports — robustni pro budouci rozsireni.
function Invoke-NativeCleanup {
    if ($Runtime -ne "native" -or $NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    $pidFiles = Get-ChildItem -Path $LogDir -Filter "sglang_*.pid" -ErrorAction SilentlyContinue
    foreach ($pf in $pidFiles) {
        $port = $pf.BaseName -replace '^sglang_', ''
        Write-Warn "Cleanup: stop SGLang na portu $port (script exit)"
        try {
            & "$PSScriptRoot\stop_native_llm.ps1" -Port ([int]$port) 2>&1 | Out-Null
        } catch {}
    }
}
# PowerShell ekvivalent bash trap EXIT — exit handler pres engine event
Register-EngineEvent PowerShell.Exiting -Action { Invoke-NativeCleanup } | Out-Null

function Invoke-PhaseNative {
    param([string]$Phase)
    Write-Log "Smoke FAZE [$Phase] — NATIVE (Python primo, zadny docker)"

    if ($Phase -ne "finalize" -and $Phase -ne "report") {
        if (-not (Install-SgLangIfMissing)) { return $false }
        $port = switch ($Phase) {
            "extract" { 8000 }
            "critic"  { 8001 }
            "revise"  { 8000 }
        }
        if (-not (Start-NativeLlmForPhase -Phase $Phase -Port $port)) { return $false }
    }

    python -m labeler.main smoke-test --phase $Phase `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS
    $rc = $LASTEXITCODE

    if ($Phase -ne "finalize" -and $Phase -ne "report") {
        $port = switch ($Phase) { "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        Stop-NativeLlmForPhase -Port $port
    }

    if ($rc -eq 0) {
        Write-Log "[$Phase] hotov (native)"
        return $true
    }
    Write-Err "[$Phase] selhal (native, exit $rc)"
    return $false
}

function Invoke-SmokeExtract {
    Write-Log "FAZE 1 (smoke) — EXTRACT"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "extract")) { throw "extract failed (native)" }
        return
    }
    foreach ($backend in $ExtractorBackends) {
        if (Invoke-PhaseViaProfile -Phase "extract" -Profile "extract-$backend" -LabelerService "labeler-extract-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro extract, zkousim dalsi…"
    }
    Write-Err "Vsechny extract backendy selhaly."
    throw "extract failed"
}

function Invoke-SmokeCritic {
    Write-Log "FAZE 2 (smoke) — CRITIC"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "critic")) { throw "critic failed (native)" }
        return
    }
    foreach ($backend in $CriticBackends) {
        if (Invoke-PhaseViaProfile -Phase "critic" -Profile "critic-$backend" -LabelerService "labeler-critic-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro critic, zkousim dalsi…"
    }
    Write-Err "Vsechny critic backendy selhaly."
    throw "critic failed"
}

function Invoke-SmokeRevise {
    Write-Log "FAZE 2b (smoke) — REVISE"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "revise")) { throw "revise failed (native)" }
        return
    }
    foreach ($backend in $ReviseBackends) {
        if (Invoke-PhaseViaProfile -Phase "revise" -Profile "revise-$backend" -LabelerService "labeler-revise-$backend") {
            return $true
        }
        Write-Warn "Backend $backend selhal pro revise, zkousim dalsi…"
    }
    Write-Err "Vsechny revise backendy selhaly."
    throw "revise failed"
}

function Invoke-SmokeFinalize {
    Write-Log "FAZE 3 (smoke) — FINALIZE (bez LLM, detached -d)"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "finalize")) { throw "finalize failed (native)" }
        return
    }
    $cid = (docker compose --profile finalize run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        labeler-finalize `
        smoke-test --phase finalize `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "Finalize smoke: docker compose run -d selhal"
        throw "finalize failed"
    }
    Write-Log "Container $cid bezi na pozadi. Sleduj: docker logs -f $cid"
    docker wait $cid 2>&1 | Out-Null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    if ($rc -ne 0) {
        Write-Err "Finalize smoke selhal (rc=$rc)"
        throw "finalize failed"
    }
    Write-Log "Finalize smoke hotov"
}

function Invoke-SmokeReport {
    Write-Log "FAZE 4 (smoke) — REPORT (extrapolace + +/-2h konfidence)"
    if ($Runtime -eq "native") {
        if (-not (Invoke-PhaseNative -Phase "report")) { throw "report failed (native)" }
        return
    }
    $cid = (docker compose --profile finalize run -d --rm `
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke `
        labeler-finalize `
        smoke-test --phase report `
        --hw-label "$HwLabel" `
        --samples-per-dataset $SamplesPerDataset `
        --docker-startup-s $DockerStartupS 2>$null | Select-Object -Last 1)
    if (-not $cid) {
        Write-Err "Report: docker compose run -d selhal"
        throw "report failed"
    }
    docker wait $cid 2>&1 | Out-Null
    docker logs $cid 2>$null
    $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
    if ($rc -ne 0) { Write-Err "Report selhal (rc=$rc)"; throw "report failed" }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Test-CudaPreflight
Test-DataDir

New-Item -ItemType Directory -Force -Path "checkpoints/smoke" | Out-Null

Write-Log "Smoke test config:"
Write-Host "  Runtime=$Runtime"
Write-Host "  GpuCount=$($GpuCount)x B200"
Write-Host "  EXTRACTOR_TP_SIZE=$($env:EXTRACTOR_TP_SIZE)"
Write-Host "  CRITIC_TP_SIZE=$($env:CRITIC_TP_SIZE)"
Write-Host "  HwLabel=$HwLabel"
Write-Host "  SamplesPerDataset=$SamplesPerDataset"
Write-Host "  SmokePhases=$($SmokePhases -join ',')"
Write-Host "  DockerStartupS=$DockerStartupS"
if ($Runtime -eq "native") {
    $extUrl = if ($env:EXTRACTOR_BASE_URL) { $env:EXTRACTOR_BASE_URL } else { "http://localhost:8000/v1" }
    $critUrl = if ($env:CRITIC_BASE_URL)   { $env:CRITIC_BASE_URL }   else { "http://localhost:8001/v1" }
    Write-Host "  EXTRACTOR_BASE_URL=$extUrl (musi uz bezet)"
    Write-Host "  CRITIC_BASE_URL=$critUrl (musi uz bezet)"
}
Write-Host "  Output: checkpoints/smoke/"

# V native modu chceme aby smoke_test.py psal do checkpoints/smoke/
if ($Runtime -eq "native") {
    if (-not $env:SMOKE_CHECKPOINT_DIR) {
        $env:SMOKE_CHECKPOINT_DIR = Join-Path (Get-Location) "checkpoints/smoke"
    }
}

foreach ($phase in $SmokePhases) {
    switch ($phase) {
        "extract"  { Invoke-SmokeExtract  | Out-Null }
        "critic"   { Invoke-SmokeCritic   | Out-Null }
        "revise"   { Invoke-SmokeRevise   | Out-Null }
        "finalize" { Invoke-SmokeFinalize }
        default {
            Write-Err "Neznama faze: $phase"
            throw "unknown phase"
        }
    }
}

Invoke-SmokeReport

Write-Log "Smoke test hotov."
Write-Host ""
Write-Host "Artefakty:"
Write-Host "  checkpoints/smoke/smoke_timings.json     — raw per-fazova mereni"
Write-Host "  checkpoints/smoke/smoke_test_report.md   — human-readable report"
Write-Host "  checkpoints/smoke/smoke_test_report.json — strojovy export"
