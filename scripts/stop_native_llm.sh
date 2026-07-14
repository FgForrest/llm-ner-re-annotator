#!/usr/bin/env bash
# Zastaví SGLang server běžící z `start_native_llm.sh`. Najde PID podle
# /tmp/sglang_<port>.pid, pošle SIGTERM (graceful), počká 10s, fallback SIGKILL.
#
# Použití:
#   ./scripts/stop_native_llm.sh <port>
#
# Příklady:
#   ./scripts/stop_native_llm.sh 8000   # extractor (Qwen)
#   ./scripts/stop_native_llm.sh 8001   # critic (Gemma)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <port>" >&2
    exit 2
fi

PORT="$1"
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/tmp}"
PID_FILE="${SGLANG_LOG_DIR}/sglang_${PORT}.pid"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

if [ ! -f "$PID_FILE" ]; then
    warn "PID file $PID_FILE neexistuje. SGLang na portu $PORT nebyl spuštěn naším skriptem."
    warn "Pokud tam něco běží, stop ho manuálně: lsof -i :$PORT"
    exit 0
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    warn "PID $PID už neběží. Cleanup PID file."
    rm -f "$PID_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Posbírej VŠECHNY procesy v procesním stromě (parent + descendants).
#
# SGLang při TP>1 forkuje worker procesy. Pokud killneme jen parent přes
# SIGKILL, workers zůstanou jako orphans krátkou dobu a DRŽÍ GPU paměť →
# OOM při swapu Qwen→Gemma nebo Gemma→Qwen mezi fázema.
#
# Strategie: kill SIGTERM celý strom (parent + children + grandchildren),
# graceful 10s wait, fallback SIGKILL na zbylé.
# ---------------------------------------------------------------------------
collect_descendants() {
    local parent="$1"
    local kids
    # `pgrep -P` najde přímé children. Rekurze pro celý strom.
    if command -v pgrep >/dev/null 2>&1; then
        kids=$(pgrep -P "$parent" 2>/dev/null || true)
        for k in $kids; do
            echo "$k"
            collect_descendants "$k"
        done
    else
        # Fallback: ps + awk
        ps -eo pid,ppid 2>/dev/null | awk -v p="$parent" '$2==p {print $1}' | while read -r k; do
            [ -n "$k" ] && echo "$k" && collect_descendants "$k"
        done
    fi
}

DESCENDANTS=$(collect_descendants "$PID" | sort -u | tr '\n' ' ')
ALL_PIDS="$PID $DESCENDANTS"

log "Zastavuji SGLang (PID $PID) na portu $PORT — SIGTERM celý strom..."
log "  Stromě procesů: $ALL_PIDS"

# SIGTERM workers PRVNÍ (před parentem) — jinak SGLang main může restartovat workery
for p in $DESCENDANTS; do
    kill -TERM "$p" 2>/dev/null || true
done
# Pak SIGTERM main
kill -TERM "$PID" 2>/dev/null || true

# Wait up to 10s for graceful shutdown WHOLE TREE
for i in $(seq 1 10); do
    still_alive=""
    for p in $ALL_PIDS; do
        if kill -0 "$p" 2>/dev/null; then
            still_alive="$still_alive $p"
        fi
    done
    if [ -z "$still_alive" ]; then
        log "✅ SGLang graceful stop (celý strom $ALL_PIDS)"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

warn "SGLang neodpovídá na SIGTERM. SIGKILL celý strom..."
# SIGKILL workers PRVNÍ (kernel-level kill, žádný handler)
for p in $DESCENDANTS; do
    kill -KILL "$p" 2>/dev/null || true
done
kill -KILL "$PID" 2>/dev/null || true
sleep 2

# Verify nothing left
remaining=""
for p in $ALL_PIDS; do
    if kill -0 "$p" 2>/dev/null; then
        remaining="$remaining $p"
    fi
done
if [ -n "$remaining" ]; then
    warn "Tyto PIDs se odmítají ukončit (zombie?): $remaining"
    warn "Zkontroluj ručně: ps -p $remaining ; nvidia-smi"
else
    log "SGLang killed (celý strom $ALL_PIDS)"
fi
rm -f "$PID_FILE"
