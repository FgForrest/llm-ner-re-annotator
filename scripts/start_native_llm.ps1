# Spusti SGLang server v background modu, pocka az odpovi /v1/models.
# PID zaloguje do %TEMP%\sglang_<port>.pid pro pozdejsi stop_native_llm.ps1.
#
# Pouziti:
#   .\scripts\start_native_llm.ps1 <model_path> <port> [<tp_size>] [-ExtraArgs ...]

param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [Parameter(Mandatory=$true)][int]$Port,
    [int]$TpSize = 1,
    [string]$ExtraArgs = "",
    [double]$MemFractionStatic = 0.85,
    [int]$MaxRunningRequests = 32,
    [int]$ContextLen = 65536,
    [int]$ReadyTimeoutS = 1200
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

$LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
$PidFile = Join-Path $LogDir "sglang_$Port.pid"
$LogFile = Join-Path $LogDir "sglang_$Port.log"

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# HF_TOKEN preflight - gated modely (Gemma, Llama) bez tokenu nepojedou
# ---------------------------------------------------------------------------
if ($ModelPath -match "(Gemma|gemma|Llama|llama|meta-llama|google)") {
    if (-not $env:HF_TOKEN -or $env:HF_TOKEN -eq "hf_TADY_DOPLN_TOKEN") {
        Write-Warn "HF_TOKEN neni nastaveny - model '$ModelPath' je pravdepodobne gated."
        Write-Warn "Pokud SGLang start selze: `$env:HF_TOKEN = 'hf_xxxxx...'"
    }
}

# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------
$HfHome = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $env:USERPROFILE ".cache\huggingface" }
try {
    $drive = (Get-Item (Split-Path $HfHome -Qualifier) -ErrorAction SilentlyContinue).PSDrive
    if ($drive) {
        $availGb = [math]::Round($drive.Free / 1GB)
        $needed = if ($ModelPath -match "122B|-122B-") { 90 } `
                  elseif ($ModelPath -match "31B|-31B-") { 20 } `
                  else { 10 }
        if ($availGb -lt $needed) {
            Write-Warn "$HfHome ma jen $availGb GB volnych, model potrebuje ~$needed GB."
        }
    }
} catch {}

# Pokud uz existuje PID file s zivym procesem
if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile
    $proc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Warn "SGLang uz bezi na portu $Port (PID $existingPid). Nech ho byt."
        exit 0
    }
    Remove-Item $PidFile -Force
}

# Health check pokud nekdo uz SGLang spustil rucne
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:$Port/v1/models" `
        -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    if ($resp.StatusCode -lt 400) {
        Write-Warn "Port $Port uz odpovida - nejaky LLM tam bezi. Nech ho byt."
        exit 0
    }
} catch {
    # Endpoint neodpovida, OK, spustime ho
}

# Pokud caller poslal --max-running-requests / --context-length pres
# -ExtraArgs (typicky run_*.ps1 Get-SglangArgsForPhase), pouziji se jeho
# hodnoty. Pokud ne (standalone use), doplnime defaulty z parametru.
$finalExtra = $ExtraArgs
if ($finalExtra -notmatch "--max-running-requests") {
    $finalExtra = "--max-running-requests=$MaxRunningRequests $finalExtra"
}
if ($finalExtra -notmatch "--context-length") {
    $finalExtra = "--context-length=$ContextLen $finalExtra"
}

Write-Log "Spoustim SGLang server na portu $Port"
Write-Host "  model:               $ModelPath"
Write-Host "  tp_size:             $TpSize"
Write-Host "  mem_fraction_static: $MemFractionStatic"
Write-Host "  extra args:          $finalExtra"
Write-Host "  log file:            $LogFile"

# Background launch via Start-Process.
# `--max-running-requests` a `--context-length` jsou v $finalExtra
# (caller-supplied NEBO param-default).
$argList = @(
    "-m", "sglang.launch_server",
    "--model-path", $ModelPath,
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--tp", "$TpSize",
    "--mem-fraction-static", "$MemFractionStatic",
    "--trust-remote-code"
)
if ($finalExtra) {
    # Split na whitespace (ne jen single space) + filtrace prázdných pro
    # robustní handling více mezer / tab characters.
    $argList += ($finalExtra -split '\s+' | Where-Object { $_ })
}

$proc = Start-Process -FilePath "python" -ArgumentList $argList `
    -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
    -PassThru -NoNewWindow

Set-Content -Path $PidFile -Value $proc.Id
Write-Log "SGLang server PID: $($proc.Id) (zalogovano do $PidFile)"

# Wait for ready
Write-Log "Cekam na ready (max ${ReadyTimeoutS}s)..."
$startTs = [int][double]::Parse((Get-Date -UFormat %s))
while ($true) {
    # Spadl proces?
    $procCheck = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if (-not $procCheck) {
        Write-Err "SGLang proces (PID $($proc.Id)) skoncil. Posledni log:"
        if (Test-Path $LogFile) { Get-Content $LogFile -Tail 50 }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        exit 1
    }
    # Health check
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$Port/v1/models" `
            -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -lt 400) {
            $elapsed = [int][double]::Parse((Get-Date -UFormat %s)) - $startTs
            Write-Log "+ SGLang ready (po ${elapsed}s) na http://localhost:$Port/v1"
            exit 0
        }
    } catch {}
    # Timeout
    $elapsed = [int][double]::Parse((Get-Date -UFormat %s)) - $startTs
    if ($elapsed -gt $ReadyTimeoutS) {
        Write-Err "Timeout po ${elapsed}s. SGLang nedobehl. Log:"
        if (Test-Path $LogFile) { Get-Content $LogFile -Tail 50 }
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        exit 1
    }
    if (($elapsed % 30) -eq 0 -and $elapsed -gt 0) {
        Write-Host "  ... ${elapsed}s, stale cekam..."
    }
    Start-Sleep -Seconds 1
}
