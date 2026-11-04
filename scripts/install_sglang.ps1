# Auto-install SGLang pro native mode pipeline (Windows / PowerShell).
#
# Co dela:
#   1. Verifikuje CUDA 13+ (NVFP4 na B200 to vyzaduje)
#   2. Verifikuje Python verzi (3.10-3.12 typicky podporovane)
#   3. Instaluje SGLang s CUDA 13 wheels + flashinfer-python pro NVFP4 MoE
#   4. Nainstaluje labeler python deps (requirements.txt)
#   5. Verifikuje ze vse importuje
#
# Pouziti:
#   .\scripts\install_sglang.ps1
#   .\scripts\install_sglang.ps1 -SglangVersion 0.4.5
#   .\scripts\install_sglang.ps1 -UseVenv
#
# POZN.: SGLang nativne bezi NEJLIP na Linuxu. Windows uzivate doporucene
# WSL2 + bash variantu (./scripts/install_sglang.sh). Tato PS verze
# funguje pro vyvoj/testovani; produkce by mela bezet na Linuxu.

param(
    [string]$SglangVersion = "",                  # prazdne = latest stable z PyPI
    # FlashInfer wheel index pro CUDA 13 + PyTorch 2.7 (B200 + NVFP4).
    # Pro starsi torch (2.5): https://flashinfer.ai/whl/cu130/torch2.5/
    [string]$FlashinferIndexUrl = "https://flashinfer.ai/whl/cu130/torch2.7/",
    [switch]$SkipFlashinfer = $false,
    [switch]$UseVenv = $false,
    [switch]$SkipCudaCheck = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

function Write-Log  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Err  { param($m) Write-Host "`n!! $m" -ForegroundColor Red }
function Write-Warn { param($m) Write-Host "`n!! $m" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# 1) CUDA verifikace
# ---------------------------------------------------------------------------
function Test-Cuda {
    if ($SkipCudaCheck -or $env:SKIP_CUDA_CHECK -eq "1") {
        Write-Warn "SkipCudaCheck=true - preskakuji CUDA verifikaci."
        return
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Err "nvidia-smi nenalezen. SGLang vyzaduje NVIDIA GPU + driver."
        throw "CUDA missing"
    }
    $smiOutput = & nvidia-smi 2>$null | Out-String
    $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $match.Success) {
        Write-Warn "Nepodarilo se zjistit CUDA verzi"
        return
    }
    $major = [int]$match.Groups[1].Value
    $minor = [int]$match.Groups[2].Value
    if ($major -lt 13) {
        Write-Err "CUDA $major.$minor detekovano - pipeline vyzaduje CUDA 13+ pro NVFP4."
        throw "CUDA <13"
    }
    Write-Log "CUDA $major.$minor detekovano +"
}

# ---------------------------------------------------------------------------
# 2) Python verifikace
# ---------------------------------------------------------------------------
function Get-Python {
    $py = Get-Command python3 -ErrorAction SilentlyContinue
    if (-not $py) {
        $py = Get-Command python -ErrorAction SilentlyContinue
    }
    if (-not $py) {
        Write-Err "Python nenalezen v PATH."
        throw "no python"
    }
    return $py.Source
}

function Test-PythonVersion {
    param([string]$Py)
    $ver = & $Py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Log "Python $ver ($Py)"
    $parts = $ver -split '\.'
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -ne 3 -or $minor -lt 10 -or $minor -gt 12) {
        Write-Warn "Python $ver - SGLang typicky podporuje 3.10-3.12. Muzou byt compat issues."
    }
}

# ---------------------------------------------------------------------------
# 3) Virtualenv (volitelne)
# ---------------------------------------------------------------------------
function Initialize-Venv {
    param([string]$Py)
    if (-not $UseVenv) { return $Py }
    if (-not (Test-Path ".venv")) {
        Write-Log "Vytvarim .venv..."
        & $Py -m venv .venv
    }
    $venvPy = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        ".\.venv\Scripts\python.exe"
    } else {
        ".\.venv\bin\python"
    }
    Write-Log "Aktivovan .venv: $venvPy"
    return $venvPy
}

# ---------------------------------------------------------------------------
# 4) Install
# ---------------------------------------------------------------------------
function Install-PipDeps {
    param([string]$Py)
    Write-Log "Upgraduje pip + wheel + setuptools..."
    & $Py -m pip install --upgrade pip wheel setuptools
    if (Test-Path "requirements.txt") {
        Write-Log "Instaluje labeler deps (requirements.txt)..."
        & $Py -m pip install -r requirements.txt
    }
}

function Install-SgLang {
    param([string]$Py)
    $pkgSpec = "sglang[all]"
    if ($SglangVersion) {
        $pkgSpec = "sglang[all]==$SglangVersion"
    }
    Write-Log "Instaluje SGLang ($pkgSpec) z PyPI..."
    # SGLang z PyPI. flashinfer dep prepsan v dalsim kroku.
    & $Py -m pip install $pkgSpec
}

function Install-FlashInfer {
    param([string]$Py)
    if ($SkipFlashinfer) {
        Write-Warn "SkipFlashinfer=true - flashinfer-python nenainstalovan."
        Write-Warn "NVFP4 MoE (Qwen3.5-122B-A10B) bez flashinfer NEPOJEDE."
        return
    }
    Write-Log "Instaluje flashinfer-python s CUDA 13 wheels (override sglang[all] dep)..."
    Write-Host "  Index: $FlashinferIndexUrl"
    # KRITICKE: --index-url misto --extra-index-url + --force-reinstall
    # zaruci ze prepiseme generic PyPI build cu130 buildem.
    try {
        & $Py -m pip install --force-reinstall --no-deps flashinfer-python `
            --index-url $FlashinferIndexUrl
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Log "OK flashinfer-python nainstalovan"
    } catch {
        Write-Warn "flashinfer z $FlashinferIndexUrl selhal."
        Write-Warn "Mozne priciny: chybi wheel pro Python/CUDA combo, firewall, spatny URL."
        Write-Warn "Fallback: PyPI flashinfer-python (omezena NVFP4 podpora)..."
        & $Py -m pip install --force-reinstall --no-deps flashinfer-python
    }
}

# ---------------------------------------------------------------------------
# 5) Verify
# ---------------------------------------------------------------------------
function Test-Install {
    param([string]$Py)
    Write-Log "Verifikuje instalaci..."
    & $Py -c "import sglang; print(f'  sglang {sglang.__version__}')"
    if (-not $SkipFlashinfer) {
        try {
            & $Py -c "import flashinfer; print(f'  flashinfer {flashinfer.__version__}')" 2>$null
        } catch {
            Write-Warn "flashinfer neni dostupny (NVFP4 MoE NEPOJEDE)"
        }
    }
    try {
        & $Py -c "import labeler" 2>$null
        Write-Host "  labeler module +"
    } catch {
        Write-Warn "labeler module nelze importovat - jsi v root adresari projektu?"
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Log "SGLang auto-install pro Mudrc Labeler pipeline"
Write-Host "  Config:"
Write-Host "    SglangVersion=$(if ($SglangVersion) { $SglangVersion } else { 'latest stable' })"
Write-Host "    FlashinferIndexUrl=$FlashinferIndexUrl"
Write-Host "    SkipFlashinfer=$SkipFlashinfer"
Write-Host "    UseVenv=$UseVenv"

Test-Cuda

$py = Get-Python
Test-PythonVersion -Py $py
$py = Initialize-Venv -Py $py

Install-PipDeps -Py $py
Install-SgLang -Py $py
Install-FlashInfer -Py $py
Test-Install -Py $py

Write-Log "Instalace hotova."
Write-Host ""
Write-Host "Dalsi kroky:"
Write-Host "  1) Spust SGLang server:"
Write-Host "     .\scripts\start_native_llm.ps1 <model> <port> [<tp_size>]"
Write-Host "  2) Nebo pust cely pipeline v native modu:"
Write-Host "     .\scripts\run_smoke_test.ps1 -Runtime native"
