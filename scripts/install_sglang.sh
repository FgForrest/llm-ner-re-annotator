#!/usr/bin/env bash
# Auto-install SGLang pro native mode pipeline.
#
# Co dělá:
#   1. Verifikuje CUDA 13+ (NVFP4 na B200 to vyžaduje)
#   2. Verifikuje Python verzi (3.10-3.12 typicky podporované SGLangem)
#   3. Instaluje SGLang s CUDA 13 wheels + flashinfer-python pro NVFP4 MoE
#   4. Nainstaluje labeler python deps (requirements.txt)
#   5. Verifikuje že vše importuje
#
# Spouští se:
#   - Automaticky z run_smoke_test.sh / run_production.sh v native módu,
#     pokud `python -c "import sglang"` selže
#   - Manuálně: ./scripts/install_sglang.sh
#
# Override env vars:
#   SGLANG_VERSION=0.4.5     # konkrétní SGLang verze (default = latest stable z PyPI)
#   FLASHINFER_INDEX_URL=... # flashinfer wheel index pro tvoji CUDA/torch verzi
#                            # default: https://flashinfer.ai/whl/cu130/torch2.7/
#                            # fallback varianty:
#                            #   https://flashinfer.ai/whl/cu130/torch2.5/   (starší torch)
#                            #   https://flashinfer.ai/whl/cu128/torch2.7/   (CUDA 12.8)
#   SKIP_FLASHINFER=1        # přeskoč flashinfer (pokud nepotřebuješ NVFP4 MoE)
#   USE_VENV=1               # vytvoří/aktivuje .venv místo system Python
#   SKIP_CUDA_CHECK=1        # přeskoč CUDA 13 verifikaci

set -euo pipefail

cd "$(dirname "$0")/.."

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
err()  { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

SGLANG_VERSION="${SGLANG_VERSION:-}"            # prázdné = latest stable
# FlashInfer wheel index pro CUDA 13 + PyTorch 2.7 (B200 + NVFP4 MoE).
# Pro starší torch (2.5) přenastav přes env var.
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/cu130/torch2.7/}"
SKIP_FLASHINFER="${SKIP_FLASHINFER:-0}"
USE_VENV="${USE_VENV:-0}"

# ---------------------------------------------------------------------------
# 1) CUDA verifikace
# ---------------------------------------------------------------------------
check_cuda() {
    if [ "${SKIP_CUDA_CHECK:-}" = "1" ]; then
        warn "SKIP_CUDA_CHECK=1 — přeskakuji CUDA verifikaci."
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi nenalezen. SGLang vyžaduje NVIDIA GPU + driver."
        return 1
    fi
    local cuda_line cuda_ver major
    cuda_line=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | head -1 || true)
    if [ -z "$cuda_line" ]; then
        warn "Nepodařilo se zjistit CUDA verzi z nvidia-smi"
        return 0
    fi
    cuda_ver="${cuda_line#CUDA Version: }"
    major="${cuda_ver%%.*}"
    if [ "$major" -lt 13 ]; then
        err "CUDA $cuda_ver detekováno — pipeline vyžaduje CUDA 13+ pro NVFP4."
        err "Upgrade NVIDIA driveru na ≥580 (CUDA 13.0 capable)."
        err "Nebo SKIP_CUDA_CHECK=1 pro FP8/BF16 fallback (může mít omezenou kompatibilitu)."
        return 1
    fi
    log "CUDA $cuda_ver detekováno ✓"
}

# ---------------------------------------------------------------------------
# 2) Python verifikace
# ---------------------------------------------------------------------------
check_python() {
    if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
        err "Python nenalezen v PATH."
        return 1
    fi
    PY="$(command -v python3 || command -v python)"
    local ver
    ver=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python $ver ($PY)"
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -ne 3 ] || [ "$minor" -lt 10 ] || [ "$minor" -gt 12 ]; then
        warn "Python $ver — SGLang typicky podporuje 3.10-3.12. Můžou být kompat issues."
    fi
}

# ---------------------------------------------------------------------------
# 3) Virtualenv (volitelně)
# ---------------------------------------------------------------------------
setup_venv() {
    if [ "$USE_VENV" != "1" ]; then
        return 0
    fi
    if [ ! -d ".venv" ]; then
        log "Vytvářím .venv (USE_VENV=1)..."
        "$PY" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="$(command -v python)"
    log "Aktivován .venv: $PY"
}

# ---------------------------------------------------------------------------
# 4) Pip upgrade + SGLang install
# ---------------------------------------------------------------------------
install_pip_deps() {
    log "Upgraduje pip + wheel + setuptools..."
    "$PY" -m pip install --upgrade pip wheel setuptools

    # Labeler runtime deps (httpx, pydantic, rdflib)
    if [ -f requirements.txt ]; then
        log "Instaluje labeler deps (requirements.txt)..."
        "$PY" -m pip install -r requirements.txt
    fi
}

install_sglang() {
    local pkg_spec="sglang[all]"
    if [ -n "$SGLANG_VERSION" ]; then
        pkg_spec="sglang[all]==${SGLANG_VERSION}"
    fi
    log "Instaluje SGLang ($pkg_spec) z PyPI..."

    # SGLang z PyPI. Dependencies (vč. flashinfer-python ze sglang[all]
    # extras) z PyPI defaultem — ale ten je pro generic CUDA build.
    # Po tomto kroku přepíšeme flashinfer správnou cu130 verzí níže.
    "$PY" -m pip install "$pkg_spec"
}

install_flashinfer() {
    if [ "$SKIP_FLASHINFER" = "1" ]; then
        warn "SKIP_FLASHINFER=1 — flashinfer-python NENAINSTALOVÁN."
        warn "NVFP4 MoE (Qwen3.5-122B-A10B) bez flashinfer NEPOJEDE."
        return 0
    fi
    log "Instaluje flashinfer-python s CUDA 13 wheels (override sglang[all] dep)..."
    log "  Index: $FLASHINFER_INDEX_URL"

    # KRITICKÉ: --index-url místo --extra-index-url. Pip by jinak
    # preferoval PyPI verzi (generic CUDA build), což na B200 NVFP4
    # nefunguje. --force-reinstall pak zaručí přepsání toho, co sglang[all]
    # případně nainstaloval.
    if ! "$PY" -m pip install --force-reinstall --no-deps flashinfer-python \
            --index-url "$FLASHINFER_INDEX_URL"; then
        warn "flashinfer-python z $FLASHINFER_INDEX_URL selhal."
        warn "Možné příčiny:"
        warn "  - Žádný wheel pro tvoji Python/CUDA/torch verzi v cu130 indexu"
        warn "  - Nedostupný flashinfer.ai (firewall, sítě)"
        warn "  - Špatný FLASHINFER_INDEX_URL (zkus torch2.5/ místo torch2.7/)"
        warn ""
        warn "Fallback: zkusím PyPI flashinfer-python (může mít omezenou NVFP4 podporu)..."
        if ! "$PY" -m pip install --force-reinstall --no-deps flashinfer-python; then
            warn "I PyPI fallback selhal. SGLang spadne při startu NVFP4 modelu."
            return 1
        fi
    fi
    log "✅ flashinfer-python nainstalován"
}

# ---------------------------------------------------------------------------
# 5) Verify
# ---------------------------------------------------------------------------
verify_install() {
    log "Verifikuje instalaci..."
    if ! "$PY" -c "import sglang; print(f'  sglang {sglang.__version__}')"; then
        err "import sglang selhal."
        return 1
    fi
    if [ "$SKIP_FLASHINFER" != "1" ]; then
        if "$PY" -c "import flashinfer; print(f'  flashinfer {flashinfer.__version__}')" 2>/dev/null; then
            :
        else
            warn "flashinfer není dostupný (NVFP4 MoE NEPOJEDE)"
        fi
    fi
    if "$PY" -c "import labeler" 2>/dev/null; then
        echo "  labeler module ✓"
    else
        warn "labeler module nelze importovat — jsi v root adresáři projektu?"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "🚀 SGLang auto-install pro Mudrc Labeler pipeline"
echo "  Config:"
echo "    SGLANG_VERSION=${SGLANG_VERSION:-latest stable}"
echo "    FLASHINFER_INDEX_URL=$FLASHINFER_INDEX_URL"
echo "    SKIP_FLASHINFER=$SKIP_FLASHINFER"
echo "    USE_VENV=$USE_VENV"

check_cuda
check_python
setup_venv
install_pip_deps
install_sglang
install_flashinfer
verify_install

log "✅ Instalace hotová."
echo ""
echo "Další kroky:"
echo "  1) Spusť SGLang server:"
echo "     ./scripts/start_native_llm.sh <model> <port> [<tp_size>]"
echo "  2) Nebo pusť celý pipeline v native módu:"
echo "     RUNTIME=native ./scripts/run_smoke_test.sh"
echo ""
echo "Pokud používáš venv:"
echo "     source .venv/bin/activate    # aktivuj před každým během"
