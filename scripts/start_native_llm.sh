#!/usr/bin/env bash
# Spustí SGLang server v background módu na zadaném portu a počká, dokud
# server neodpoví na /v1/models. PID se zaloguje do /tmp/sglang_<port>.pid
# pro pozdější `stop_native_llm.sh`.
#
# Použití:
#   ./scripts/start_native_llm.sh <model_path> <port> [<tp_size>] [<extra_args>...]
#
# Příklady:
#   ./scripts/start_native_llm.sh RedHatAI/Qwen3.5-122B-A10B-NVFP4 8000 1
#   ./scripts/start_native_llm.sh nvidia/Gemma-4-31B-IT-NVFP4 8001 1
#
# Env override:
#   SGLANG_MEM_FRACTION_STATIC=0.85
#   SGLANG_MAX_RUNNING_REQUESTS=32
#   SGLANG_CONTEXT_LEN=65536
#   SGLANG_EXTRA_ARGS="--reasoning-parser=qwen3"
#   SGLANG_LOG_DIR=/tmp
#   SGLANG_READY_TIMEOUT_S=1200   # max čekání BEZ PROGRESU (default 20 min);
#                                 # stahování vah z HF se NEzapočítává — deadline
#                                 # se resetuje, dokud log roste

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <port> [<tp_size>] [<extra_args>...]" >&2
    exit 2
fi

MODEL_PATH="$1"
PORT="$2"
TP_SIZE="${3:-1}"
shift 3 2>/dev/null || shift $#

SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-65536}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/tmp}"
SGLANG_READY_TIMEOUT_S="${SGLANG_READY_TIMEOUT_S:-1200}"

PID_FILE="${SGLANG_LOG_DIR}/sglang_${PORT}.pid"
LOG_FILE="${SGLANG_LOG_DIR}/sglang_${PORT}.log"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
err()  { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

# ---------------------------------------------------------------------------
# HF_TOKEN preflight — gated modely (Gemma, Llama) bez tokenu nepojdou
# ---------------------------------------------------------------------------
case "$MODEL_PATH" in
    *Gemma*|*gemma*|*Llama*|*llama*|*meta-llama*|*google*)
        if [ -z "${HF_TOKEN:-}" ] || [ "${HF_TOKEN}" = "hf_TADY_DOPLN_TOKEN" ]; then
            warn "HF_TOKEN není nastavený — model '$MODEL_PATH' je pravděpodobně gated."
            warn "Pokud SGLang start selže s 'Repository not found' nebo '401 Unauthorized':"
            warn "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx"
            warn "  Token musí mít přístup k cílovému modelu (https://huggingface.co/$MODEL_PATH)"
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# Disk space check — Qwen3.5-122B-NVFP4 ~80 GB, Gemma-4-31B-NVFP4 ~16 GB
# ---------------------------------------------------------------------------
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if command -v df >/dev/null 2>&1; then
    avail_kb=$(df -k "$(dirname "$HF_HOME")" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$avail_kb" ]; then
        avail_gb=$((avail_kb / 1024 / 1024))
        case "$MODEL_PATH" in
            *122B*|*-122B-*)  needed=90 ;;   # Qwen 122B NVFP4 + KV cache
            *31B*|*-31B-*)    needed=20 ;;   # Gemma 31B NVFP4 + KV cache
            *)                needed=10 ;;   # menší modely
        esac
        if [ "$avail_gb" -lt "$needed" ]; then
            warn "$HF_HOME má jen ${avail_gb} GB volných, model potřebuje ~${needed} GB."
            warn "Pokud model ještě není stažený, download selže s 'No space left'."
        fi
    fi
fi

# Pokud už server běží na tomto portu, varuj a vyber
if [ -f "$PID_FILE" ]; then
    local_pid=$(cat "$PID_FILE")
    if kill -0 "$local_pid" 2>/dev/null; then
        warn "SGLang už běží na portu $PORT (PID $local_pid). Nech ho být a používej."
        warn "Pro restart: ./scripts/stop_native_llm.sh $PORT && ./scripts/start_native_llm.sh ..."
        exit 0
    fi
    # PID file ale proces mrtvý → cleanup
    rm -f "$PID_FILE"
fi

# Health check — pokud někdo už SGLang spustil ručně, použij ho
if command -v curl >/dev/null 2>&1; then
    if curl -fs --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        warn "Port $PORT už odpovídá — nějaký LLM server tam běží (možná ne náš)."
        warn "Nech ho být, skript dál. PID se nepamatuje, stop tedy musíš manuálně."
        exit 0
    fi
fi

# --- Resolve final SGLang args ---------------------------------------------
# Pokud caller poslal --max-running-requests / --context-length přes
# SGLANG_EXTRA_ARGS (typicky run_*.sh::build_sglang_args_for_phase), použijí
# se jeho hodnoty. Pokud ne (standalone use), doplníme defaulty z env vars.
SGLANG_FINAL_EXTRA="$SGLANG_EXTRA_ARGS"
case " $SGLANG_FINAL_EXTRA " in
    *" --max-running-requests"*) ;;  # caller už zadal
    *) SGLANG_FINAL_EXTRA="--max-running-requests=$SGLANG_MAX_RUNNING_REQUESTS $SGLANG_FINAL_EXTRA" ;;
esac
case " $SGLANG_FINAL_EXTRA " in
    *" --context-length"*) ;;
    *) SGLANG_FINAL_EXTRA="--context-length=$SGLANG_CONTEXT_LEN $SGLANG_FINAL_EXTRA" ;;
esac

log "Spouštím SGLang server na portu $PORT"
echo "  model:               $MODEL_PATH"
echo "  tp_size:             $TP_SIZE"
echo "  mem_fraction_static: $SGLANG_MEM_FRACTION_STATIC"
echo "  extra args:          $SGLANG_FINAL_EXTRA $*"
echo "  log file:            $LOG_FILE"

# Background launch — nohup pro detach od terminálu.
# `--max-running-requests` a `--context-length` jsou součástí
# SGLANG_FINAL_EXTRA výše (caller-supplied OR env-var default).
nohup python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp "$TP_SIZE" \
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --trust-remote-code \
    $SGLANG_FINAL_EXTRA "$@" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
log "SGLang server PID: $SERVER_PID (zalogováno do $PID_FILE)"

# ---------------------------------------------------------------------------
# Wait for ready — poll /v1/models každou sekundu.
#
# SGLANG_READY_TIMEOUT_S je "NO-PROGRESS" timeout, NE celkový wall-clock.
# Dokud log roste (stahování vah z HF + load běží), deadline se resetuje —
# legitimně pomalý cold download (Qwen 122B ~80 GB) se NIKDY nezabije. Kill
# přijde jen když server fakt VISÍ (žádný nový output po SGLANG_READY_TIMEOUT_S)
# nebo když proces umře. Dřív to byl celkový-elapsed timeout, který počítal
# i download → kill uprostřed stahování.
# ---------------------------------------------------------------------------
log "Čekám na ready (no-progress timeout ${SGLANG_READY_TIMEOUT_S}s; download se NEzapočítává)..."
start_ts=$(date +%s)
last_progress_ts=$start_ts
last_log_size=0
last_tick=0
while true; do
    # Spadl proces?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        err "SGLang proces (PID $SERVER_PID) skončil. Posledních 50 řádků logu:"
        tail -n 50 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    # Health check
    if command -v curl >/dev/null 2>&1; then
        if curl -fs --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            elapsed=$(( $(date +%s) - start_ts ))
            log "✅ SGLang ready (po ${elapsed}s) na http://localhost:${PORT}/v1"
            exit 0
        fi
    fi
    now=$(date +%s)
    # Progress detekce — log roste (download/load běží) → resetuj no-progress hodiny.
    cur_log_size=$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "${cur_log_size:-0}" -gt "$last_log_size" ]; then
        last_log_size="$cur_log_size"
        last_progress_ts="$now"
    fi
    # No-progress timeout — server VISÍ (žádný nový output), ne legitimně stahuje.
    stalled=$(( now - last_progress_ts ))
    if [ "$stalled" -gt "$SGLANG_READY_TIMEOUT_S" ]; then
        elapsed=$(( now - start_ts ))
        err "Žádný progress ${stalled}s (celkem ${elapsed}s) — SGLang visí, ne stahuje. Kill."
        err "Posledních 50 řádků logu:"
        tail -n 50 "$LOG_FILE" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
    fi
    # Progress log ~každých 30s (nezávislé na délce curl/wc iterace).
    elapsed=$(( now - start_ts ))
    if [ "$elapsed" -gt 0 ] && [ $(( elapsed - last_tick )) -ge 30 ]; then
        last_tick="$elapsed"
        echo "  ... ${elapsed}s (poslední progress logu před ${stalled}s), stále čekám..."
    fi
    sleep 1
done
