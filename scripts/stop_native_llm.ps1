# Zastavi SGLang server bezici z start_native_llm.ps1.
# Pouziti: .\scripts\stop_native_llm.ps1 <port>

param([Parameter(Mandatory=$true)][int]$Port)

$LogDir = if ($env:SGLANG_LOG_DIR) { $env:SGLANG_LOG_DIR } else { $env:TEMP }
$PidFile = Join-Path $LogDir "sglang_$Port.pid"

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

if (-not (Test-Path $PidFile)) {
    Write-Warn "PID file $PidFile neexistuje. SGLang nebyl spusten nasim skriptem."
    exit 0
}

$llmPid = Get-Content $PidFile
$proc = Get-Process -Id $llmPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Warn "PID $llmPid uz nebezi. Cleanup PID file."
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    exit 0
}

# ---------------------------------------------------------------------------
# Posbiraj VSECHNY procesy ve stromu (parent + descendants).
# SGLang TP>1 forkuje workery - bez kill -children drzi GPU pamet po smrti
# parent procesu (OOM pri swapu Qwen<->Gemma mezi fazama).
# ---------------------------------------------------------------------------
function Get-Descendants {
    param([int]$ParentId)
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" `
        -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessId
    foreach ($k in $kids) {
        $k
        Get-Descendants -ParentId $k
    }
}

$descendants = @(Get-Descendants -ParentId $llmPid)
$allPids = @($llmPid) + $descendants

Write-Log "Zastavuji SGLang (PID $llmPid) na portu $Port - graceful stop celeho stromu..."
Write-Log "  Strom procesu: $($allPids -join ' ')"

# SIGTERM ekvivalent (Stop-Process bez -Force) - children PRVNI
foreach ($p in $descendants) {
    Stop-Process -Id $p -ErrorAction SilentlyContinue
}
Stop-Process -Id $llmPid -ErrorAction SilentlyContinue

# Wait up to 10s for graceful shutdown of WHOLE TREE
for ($i = 1; $i -le 10; $i++) {
    $stillAlive = @()
    foreach ($p in $allPids) {
        if (Get-Process -Id $p -ErrorAction SilentlyContinue) {
            $stillAlive += $p
        }
    }
    if ($stillAlive.Count -eq 0) {
        Write-Log "+ SGLang graceful stop (cely strom $($allPids -join ' '))"
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Warn "SGLang neodpovida na graceful stop. Force kill celeho stromu..."
foreach ($p in $descendants) {
    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
}
Stop-Process -Id $llmPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Verify
$remaining = @()
foreach ($p in $allPids) {
    if (Get-Process -Id $p -ErrorAction SilentlyContinue) {
        $remaining += $p
    }
}
if ($remaining.Count -gt 0) {
    Write-Warn "Tyto PIDs se odmitaji ukoncit: $($remaining -join ' ')"
    Write-Warn "Zkontroluj rucne: Get-Process -Id $($remaining -join ',') ; nvidia-smi"
} else {
    Write-Log "SGLang killed (cely strom)"
}
Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
