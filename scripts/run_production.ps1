param(
    [string]$Phase = "all",
    [int]$GpuCount = 0,                       # 0 = interactive prompt (preskoci se pro 'finalize')
    [ValidateSet("docker", "native")]
    [string]$Runtime = "docker"               # docker (default) / native (host Python proti bezicimu LLM)
)

# Pouziti (native runtime):
#   # 1) Sam si spust LLM server, napr.:
#   #    python -m sglang.launch_server --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 ...
#   # 2) Spust skript v native modu:
#   .\scripts\run_production.ps1 all -Runtime native

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

$ExtractorBackends = if ($env:EXTRACTOR_BACKENDS) { $env:EXTRACTOR_BACKENDS -split ' ' } else { @("sglang", "vllm") }
$CriticBackends    = if ($env:CRITIC_BACKENDS)    { $env:CRITIC_BACKENDS    -split ' ' } else { @("sglang", "vllm") }
$ReviseBackends    = if ($env:REVISE_BACKENDS)    { $env:REVISE_BACKENDS    -split ' ' } else { @("sglang", "vllm") }

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Preflight checks — chyti bezne konfiguracni problemy PRED docker compose
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
        Write-Err ""
        Write-Err "NVFP4 modely maji quantization v config.json (compressed-tensors)"
        Write-Err "- SGLang/vLLM si ji auto-detect. Rucni override = ValueError."
        Write-Err ""
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
        Write-Warn "Doplnte HF_TOKEN v .env."
    }
}

# ---------------------------------------------------------------------------
# Interactive GPU prompt (s GPU count validation)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# .env auto-config
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

# Phase=finalize je CPU only — preflighty + prompt nepotrebujeme.
if ($Phase -ne "finalize") {
    Test-QuantizationPreflight
    Test-HfTokenPreflight

    $detected = Get-DetectedGpuCount
    if ($detected -gt 0) {
        Write-Log "Detekovano $detected GPU na hostu (pres nvidia-smi)"
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
    if (-not $env:EXTRACTOR_TP_SIZE) { $env:EXTRACTOR_TP_SIZE = "$GpuCount" }
    if (-not $env:CRITIC_TP_SIZE)    { $env:CRITIC_TP_SIZE    = "1" }
    Write-Host ""
    Write-Host "Docker GPU config: EXTRACTOR_TP_SIZE=$($env:EXTRACTOR_TP_SIZE) CRITIC_TP_SIZE=$($env:CRITIC_TP_SIZE)"
}

function Test-CudaPreflight {
    if ($env:SKIP_CUDA_CHECK -eq "1") {
        Write-Warn "SKIP_CUDA_CHECK=1 — preskakuji CUDA verifikaci."
        return
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warn "nvidia-smi nenalezen — nemuzu verifikovat CUDA verzi."
        Write-Warn "Pokud kontejnery selhou s 'no kernel image' chybou, ujisti se,"
        Write-Warn "ze mas NVIDIA driver >= 580 (CUDA 13.0 capable)."
        return
    }
    $smiOutput = & nvidia-smi 2>$null | Out-String
    $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $match.Success) {
        Write-Warn "Nelze ziskat CUDA verzi z nvidia-smi vystupu — pokracuju."
        return
    }
    $major = [int]$match.Groups[1].Value
    $minor = [int]$match.Groups[2].Value
    if ($major -lt 13) {
        Write-Err "Detekovana CUDA $major.$minor — pipeline VYZADUJE CUDA 13+ pro NVFP4"
        Write-Err "na Blackwell SM120. Upgrade NVIDIA driveru na >=580 a CUDA toolkit >=13.0."
        Write-Err "Pokud chces pokracovat na vlastni riziko (napr. testujes FP8/BF16 fallback),"
        Write-Err "spust: `$env:SKIP_CUDA_CHECK='1'; .\scripts\run_production.ps1 $Phase"
        throw "CUDA 13+ required (detected $major.$minor)"
    }
    Write-Log "Preflight OK — CUDA $major.$minor detekovana (>=13.0)."
}

if ($Phase -ne "finalize") {
    Test-CudaPreflight
}

function Try-Profile {
    param([string]$Profile, [string]$LabelerService)
    Write-Log "Zkousim profil: $Profile (detached -d, na pozadi)"
    docker compose --profile $Profile up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Profile ${Profile}: docker compose up -d selhal"
        docker compose --profile $Profile down 2>&1 | Out-Null
        return $false
    }
    Write-Log "Containery bezi na pozadi. Sleduj progress v separatnim terminalu:"
    Write-Log "  docker compose --profile $Profile logs -f --no-log-prefix $LabelerService"
    # Cekej na labeler exit code
    docker compose --profile $Profile wait $LabelerService 2>&1 | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        # Fallback: ps -q + docker inspect
        $cid = (docker compose --profile $Profile ps -q $LabelerService 2>$null | Select-Object -First 1)
        if ($cid) {
            docker wait $cid 2>&1 | Out-Null
            $rc = [int](docker inspect --format='{{.State.ExitCode}}' $cid 2>$null)
        }
    }
    docker compose --profile $Profile down 2>&1 | Out-Null
    if ($rc -eq 0) {
        Write-Log "Profile $Profile hotov +"
        return $true
    }
    Write-Err "Profile $Profile selhal (rc=$rc)"
    return $false
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers — bez dockeru, Python proti bezicimu LLM serveru
# ---------------------------------------------------------------------------
$NativeAutoLlm = if ($env:NATIVE_AUTO_LLM) { $env:NATIVE_AUTO_LLM } else { "1" }

function Set-NativeEnvParity {
    if ($Runtime -ne "native") { return }
    if (-not $env:HF_XET_HIGH_PERFORMANCE)           { $env:HF_XET_HIGH_PERFORMANCE = "1" }
    if (-not $env:PYTHONWARNINGS)                    { $env:PYTHONWARNINGS = "ignore::DeprecationWarning,ignore::FutureWarning" }
    if (-not $env:TRANSFORMERS_VERBOSITY)            { $env:TRANSFORMERS_VERBOSITY = "error" }
    if (-not $env:TRANSFORMERS_NO_ADVISORY_WARNINGS) { $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1" }
    if (-not $env:EXTRACTOR_BASE_URL)                { $env:EXTRACTOR_BASE_URL = "http://localhost:8000/v1" }
    if (-not $env:CRITIC_BASE_URL)                   { $env:CRITIC_BASE_URL = "http://localhost:8001/v1" }
    if (-not $env:REVISOR_BASE_URL)                  { $env:REVISOR_BASE_URL = $env:EXTRACTOR_BASE_URL }
    Write-Log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress"
    Write-Host "  EXTRACTOR_BASE_URL=$($env:EXTRACTOR_BASE_URL)"
    Write-Host "  CRITIC_BASE_URL=$($env:CRITIC_BASE_URL)"
}

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
        Write-Log "LLM server na portu $Port uz bezi"
        return $true
    }
    if ($NativeAutoLlm -ne "1") {
        Write-Err "Endpoint http://localhost:$Port/v1 NEODPOVIDA. Manual: spust LLM:"
        Write-Err "  .\scripts\start_native_llm.ps1 <model> $Port $($env:EXTRACTOR_TP_SIZE)"
        return $false
    }
    $model = ""
    $tp = 1
    switch ($Phase) {
        { $_ -in "extract", "revise" } {
            $model = if ($env:EXTRACTOR_MODEL) { $env:EXTRACTOR_MODEL } else { "RedHatAI/Qwen3.5-122B-A10B-NVFP4" }
            $tp = if ($env:EXTRACTOR_TP_SIZE) { [int]$env:EXTRACTOR_TP_SIZE } else { 1 }
        }
        "critic" {
            $model = if ($env:CRITIC_MODEL) { $env:CRITIC_MODEL } else { "nvidia/Gemma-4-31B-IT-NVFP4" }
            $tp = if ($env:CRITIC_TP_SIZE) { [int]$env:CRITIC_TP_SIZE } else { 1 }
        }
        default { return $true }
    }
    # Args 1:1 jako Docker
    $extra = Get-SglangArgsForPhase -Phase $Phase
    Write-Log "Auto-start SGLang pro '$Phase': $model na portu $Port (tp=$tp)"
    Write-Log "  Args: $extra"
    & "$PSScriptRoot\start_native_llm.ps1" -ModelPath $model -Port $Port -TpSize $tp -ExtraArgs $extra
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

# Cleanup hook — Ctrl+C / fail / normal exit → zastav vsechny SGLang servery.
# Iteruje pres PID files (sglang_*.pid), ne hardcoded ports.
function Invoke-NativeCleanup {
    if ($Runtime -ne "native" -or $NativeAutoLlm -ne "1") { return }
    $LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
    $pidFiles = Get-ChildItem -Path $LogDir -Filter "sglang_*.pid" -ErrorAction SilentlyContinue
    foreach ($pf in $pidFiles) {
        $port = $pf.BaseName -replace '^sglang_', ''
        Write-Warn "Cleanup: stop SGLang na portu $port (exit)"
        try {
            & "$PSScriptRoot\stop_native_llm.ps1" -Port ([int]$port) 2>&1 | Out-Null
        } catch {}
    }
}
Register-EngineEvent PowerShell.Exiting -Action { Invoke-NativeCleanup } | Out-Null

function Run-NativePhase {
    param([string]$Ph)
    Write-Log "FAZE [$Ph] — NATIVE (Python primo, bez dockeru)"

    if ($Ph -ne "finalize") {
        if (-not (Install-SgLangIfMissing)) { throw "$($Ph): SGLang missing/install failed" }
        $port = switch ($Ph) { "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        if (-not (Start-NativeLlmForPhase -Phase $Ph -Port $port)) {
            throw "$($Ph): LLM endpoint nedostupny / auto-start selhal"
        }
    }

    python -m labeler.main $Ph
    $rc = $LASTEXITCODE

    if ($Ph -ne "finalize") {
        $port = switch ($Ph) { "extract" { 8000 } "critic" { 8001 } "revise" { 8000 } }
        Stop-NativeLlmForPhase -Port $port
    }

    if ($rc -ne 0) { throw "$Ph selhal (native, rc=$rc)" }
    Write-Log "OK $Ph hotov (native)"
}

function Run-Extract {
    Write-Log "FÁZE 1 — EXTRACT (Qwen3.5-122B-A10B)"
    if ($Runtime -eq "native") { Run-NativePhase "extract"; return }
    foreach ($backend in $ExtractorBackends) {
        if (Try-Profile "extract-$backend" "labeler-extract-$backend") {
            Write-Log "OK Extract hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY extract backendy selhaly."
}

function Run-Critic {
    Write-Log "FÁZE 2 — CRITIC (Gemma 4 31B, sám na GPU)"
    $ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
    if (-not (Test-Path "$ckpt\extractions.jsonl")) {
        throw "Chybí $ckpt\extractions.jsonl — nejdřív pusť extract fázi"
    }
    if ($Runtime -eq "native") { Run-NativePhase "critic"; return }
    foreach ($backend in $CriticBackends) {
        if (Try-Profile "critic-$backend" "labeler-critic-$backend") {
            Write-Log "OK Critic hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY critic backendy selhaly."
}

function Run-Revise {
    Write-Log "FÁZE 2b — REVISE (Qwen revisor projde non-ok kritiky, sám na GPU)"
    $ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
    if (-not (Test-Path "$ckpt\critic_results.jsonl")) {
        throw "Chybí $ckpt\critic_results.jsonl — nejdřív pusť critic fázi"
    }
    if ($Runtime -eq "native") { Run-NativePhase "revise"; return }
    foreach ($backend in $ReviseBackends) {
        if (Try-Profile "revise-$backend" "labeler-revise-$backend") {
            Write-Log "OK Revise hotov přes backend: $backend"
            return
        }
        Write-Err "Backend $backend selhal, zkouším další..."
    }
    throw "VŠECHNY revise backendy selhaly."
}

function Run-Finalize {
    Write-Log "FAZE 3 — FINALIZE (merge + stats, bez LLM, detached -d)"
    if ($Runtime -eq "native") { Run-NativePhase "finalize"; return }
    docker compose --profile finalize up -d --build
    if ($LASTEXITCODE -ne 0) {
        docker compose --profile finalize down 2>&1 | Out-Null
        throw "Finalize: docker compose up -d selhal"
    }
    Write-Log "Container bezi na pozadi. Sleduj: docker compose --profile finalize logs -f labeler-finalize"
    docker compose --profile finalize wait labeler-finalize 2>&1 | Out-Null
    $rc = $LASTEXITCODE
    docker compose --profile finalize down 2>&1 | Out-Null
    if ($rc -ne 0) { throw "Finalize selhal (rc=$rc)" }
    Write-Log "OK Finalize hotov"
}

switch ($Phase) {
    "extract"  { Run-Extract }
    "critic"   { Run-Critic }
    "revise"   { Run-Revise }
    "finalize" { Run-Finalize }
    "all"      {
        Run-Extract
        Run-Critic
        Run-Revise
        Run-Finalize
        $ckpt = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { ".\checkpoints" }
        Write-Log "*** VŠE HOTOVO. Výsledky v $ckpt\ ***"
        Write-Log "   - final_labels.json + .jsonl            (reálné popisky)"
        Write-Log "   - final_labels_synthetic.json + .jsonl  (syntetické texty)"
        Write-Log "   - stats.json                            (souhrn)"
    }
    default {
        Write-Err "Neznámá fáze: $Phase. Použij: extract | critic | revise | finalize | all"
        exit 1
    }
}

