#!/usr/bin/env bash
# Production smoke test — empirický odhad doby běhu pipeline na cílovém HW.
#
# Co dělá:
#   1. Z každého datasetu v data/ vezme N items (default 50)
#   2. Postupně spustí všechny 4 fáze (extract/critic/revise/finalize) na
#      smoke vzorku — vždy 1 model na GPU, jako v produkci
#   3. Měří wall-clock per fáze
#   4. Lineárně extrapoluje na full produkční rozsah (pure empirical —
#      žádné teoretické scaling factory, žádné GPU-class projekce)
#   5. Vygeneruje report (Markdown + JSON) v checkpoints/smoke/
#
# Použití (DOCKER runtime — default):
#   ./scripts/run_smoke_test.sh              # interaktivní prompt (1/2/4× B200)
#   GPU_COUNT=2 ./scripts/run_smoke_test.sh  # přeskočí prompt, použije 2× B200
#   SAMPLES_PER_DATASET=100 ./scripts/run_smoke_test.sh
#   SMOKE_PHASES="extract critic finalize" ./scripts/run_smoke_test.sh   # přeskoč revise
#
# Použití (NATIVE runtime — bez dockeru):
#   # 1) Sám si spusť LLM server, např. v jiném terminálu:
#   #    python -m sglang.launch_server \
#   #        --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 \
#   #        --host 0.0.0.0 --port 8000 --tp 1
#   # 2) Spusť skript v native módu:
#   RUNTIME=native ./scripts/run_smoke_test.sh
#   # 3) Pokud máš LLM server jinde než localhost:8000:
#   RUNTIME=native EXTRACTOR_BASE_URL=http://10.0.0.5:8000/v1 \
#       CRITIC_BASE_URL=http://10.0.0.5:8001/v1 \
#       ./scripts/run_smoke_test.sh
#
# Skript se na začátku zeptá kolik B200 GPU máš (1, 2 nebo 4 — power-of-2
# pro tensor parallelism MoE). Tento počet pak nastaví EXTRACTOR_TP_SIZE
# a HW_LABEL pro report. Pokud GPU_COUNT je v env, prompt se přeskočí.

set -euo pipefail

cd "$(dirname "$0")/.."

SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-100}"
SMOKE_PHASES="${SMOKE_PHASES:-taxonomy extract critic revise finalize}"
DOCKER_STARTUP_S="${DOCKER_STARTUP_S:-0}"
EXTRACTOR_BACKENDS="${EXTRACTOR_BACKENDS:-sglang vllm}"
CRITIC_BACKENDS="${CRITIC_BACKENDS:-sglang vllm}"
REVISE_BACKENDS="${REVISE_BACKENDS:-sglang vllm}"

# RUNTIME: kde pipeline poběží:
#   docker (default) — orchestrace přes docker-compose (LLM servery + labeler v kontejnerech)
#   native           — Python labeler spuštěný přímo na hostu proti běžícímu
#                      LLM serveru (uživatel si SGLang/vLLM zapne sám, např.
#                      `python -m sglang.launch_server --model-path=...`)
RUNTIME="${RUNTIME:-docker}"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
err()  { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

# ---------------------------------------------------------------------------
# Preflight checks — chytá běžné konfiguraní problémy PŘED docker compose
# ---------------------------------------------------------------------------
detect_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return 0
    fi
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
}

# ---------------------------------------------------------------------------
# .env auto-config — po prompt podle GPU_COUNT
# ---------------------------------------------------------------------------
# Aktualizuje GPU_COUNT, EXTRACTOR_TP_SIZE, CRITIC_TP_SIZE v .env. Pokud
# .env neexistuje, zkopíruje .env.example → .env a pak ho updatne.
# Tím se konfigurace dostane jak do docker compose (čte .env), tak do
# native runu (skript ji navíc exportuje pro current shell).
_update_env_var() {
    local file="$1" key="$2" value="$3"
    if [ ! -f "$file" ]; then return 1; fi
    if grep -qE "^[[:space:]]*${key}=" "$file"; then
        # Update existing line (POSIX-friendly sed: -i s backup pak smaž)
        sed -i.bak -E "s|^[[:space:]]*${key}=.*|${key}=${value}|" "$file" && rm -f "${file}.bak"
    else
        # Append nový řádek
        printf "%s=%s\n" "$key" "$value" >> "$file"
    fi
}

# Přečte hodnotu jedné proměnné z .env BEZ sourcingu. `source .env` by spadl
# na hodnotách s mezerami (např. `EXTRACTOR_BACKENDS=sglang vllm`), proto
# čteme cíleně grepem — 1:1 jako docker compose: ignoruje zakomentované řádky
# (`# KEY=`), poslední výskyt vyhrává, ořízne obklopující uvozovky.
_read_env_var() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 0
    local line
    line=$(grep -E "^[[:space:]]*${key}=" "$file" | tail -n1 || true)
    [ -n "$line" ] || return 0
    local val="${line#*=}"
    case "$val" in
        \"*\") val="${val#\"}"; val="${val%\"}" ;;
        \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
    printf '%s' "$val"
}

configure_env_for_gpu_count() {
    local n="$1"
    local env_file="${ENV_FILE:-.env}"
    if [ ! -f "$env_file" ]; then
        if [ -f .env.example ]; then
            cp .env.example "$env_file"
            log "Vytvořeno $env_file z .env.example (auto-config)"
        else
            warn "Neexistuje $env_file ani .env.example — auto-config přeskočen."
            warn "Vytvoř $env_file manuálně nebo zkopíruj z .env.example."
            return 0
        fi
    fi
    _update_env_var "$env_file" GPU_COUNT "$n"
    _update_env_var "$env_file" EXTRACTOR_TP_SIZE "$n"
    _update_env_var "$env_file" CRITIC_TP_SIZE "1"
    log "Auto-config $env_file: GPU_COUNT=$n EXTRACTOR_TP_SIZE=$n CRITIC_TP_SIZE=1"
}

preflight_quantization() {
    local model="${EXTRACTOR_MODEL:-RedHatAI/Qwen3.5-122B-A10B-NVFP4}"
    local quant="${EXTRACTOR_QUANTIZATION:-}"
    if [ -n "$quant" ]; then
        case "$model" in
            *NVFP4*|*-FP4*|*nvfp4*|*-fp4*)
                err "Quantization mismatch detected:"
                err "  EXTRACTOR_MODEL=$model (NVFP4 model)"
                err "  EXTRACTOR_QUANTIZATION=$quant (manuální override)"
                err ""
                err "NVFP4 modely mají quantization v config.json"
                err "(compressed-tensors) — SGLang/vLLM si ji auto-detect."
                err "Ruční override způsobí ValueError při startu."
                err ""
                err "Fix: zakomentuj EXTRACTOR_QUANTIZATION v .env"
                err "  sed -i 's/^EXTRACTOR_QUANTIZATION=/# &/' .env"
                exit 1
                ;;
        esac
    fi
    local cmodel="${CRITIC_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
    local cquant="${CRITIC_QUANTIZATION:-}"
    if [ -n "$cquant" ]; then
        case "$cmodel" in
            *NVFP4*|*-FP4*|*nvfp4*|*-fp4*)
                err "Quantization mismatch detected:"
                err "  CRITIC_MODEL=$cmodel (NVFP4 model)"
                err "  CRITIC_QUANTIZATION=$cquant (manuální override)"
                err ""
                err "Fix: zakomentuj CRITIC_QUANTIZATION v .env"
                err "  sed -i 's/^CRITIC_QUANTIZATION=/# &/' .env"
                exit 1
                ;;
        esac
    fi
}

preflight_hf_token() {
    local env_file="${ENV_FILE:-.env}"
    local token="${HF_TOKEN:-}"
    # Token nemusí být v shellu — docker compose ho čte přímo z .env. Když
    # v env není, sáhni do .env (tam ho user reálně vyplňuje), ať preflight
    # nehází falešný poplach jen proto, že skript .env nesourcuje.
    if [ -z "$token" ]; then
        token=$(_read_env_var "$env_file" HF_TOKEN)
    fi
    if [ -z "$token" ] || [ "$token" = "hf_TADY_DOPLN_TOKEN" ]; then
        warn "HF_TOKEN není nastavený nebo má placeholder hodnotu."
        warn "Gated modely (Gemma, restricted Llama) NEPŮJDOU stáhnout."
        warn "Doplň HF_TOKEN v .env (musí mít přístup k cílovým modelům)."
    fi
}

# ---------------------------------------------------------------------------
# Interactive GPU prompt (s GPU count validation)
# ---------------------------------------------------------------------------
# Pokud GPU_COUNT není v env (např. CI), interaktivně vyzve uživatele.
# Validní hodnoty: 1, 2, 4 (power-of-2 pro tensor parallelism MoE).
# Min 1, max 4 — produkční nasazení této pipeline je single-node B200.
prompt_gpu_count() {
    local detected
    detected=$(detect_gpu_count)
    if [ "$detected" -gt 0 ]; then
        log "Detekováno $detected GPU na hostu (přes nvidia-smi)"
    else
        warn "nvidia-smi nedostupné — nemůžu detekovat GPU count automaticky"
    fi

    if [ -n "${GPU_COUNT:-}" ]; then
        case "$GPU_COUNT" in
            1|2|4|8) ;;
            *)
                err "GPU_COUNT='$GPU_COUNT' není podporovaná. Validní: 1, 2, 4, 8."
                exit 1
                ;;
        esac
        if [ "$detected" -gt 0 ] && [ "$GPU_COUNT" -gt "$detected" ]; then
            err "GPU_COUNT=$GPU_COUNT, ale host má jen $detected GPU."
            err "Docker by selhal s 'nvidia-container-cli: device error: $detected: unknown device'."
            err "Sniž GPU_COUNT na max $detected nebo přidej víc GPU."
            exit 1
        fi
        log "GPU_COUNT z env: ${GPU_COUNT}× B200"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Kolik NVIDIA B200 GPU máš k dispozici pro pipeline?"
    echo "  Podporované hodnoty: 1, 2, 4 nebo 8"
    echo "  (jen power-of-2 — TP=3/5/6/7 SGLang/vLLM nepodporují)"
    if [ "$detected" -gt 0 ]; then
        echo "  → Detekováno $detected GPU na hostu"
    fi
    echo "════════════════════════════════════════════════════════"
    while true; do
        read -r -p "Zadej počet B200 GPU [1/2/4/8]: " n
        case "$n" in
            1|2|4|8)
                if [ "$detected" -gt 0 ] && [ "$n" -gt "$detected" ]; then
                    warn "Host má jen $detected GPU, zadáno $n. Vyber max $detected."
                    continue
                fi
                GPU_COUNT="$n"
                break
                ;;
            *)
                warn "Neplatná hodnota '$n'. Zadej 1, 2, 4 nebo 8."
                ;;
        esac
    done
    log "OK: ${GPU_COUNT}× B200"
}

# Spustit preflighty před prompt
preflight_quantization
preflight_hf_token
prompt_gpu_count

# Auto-config .env podle vybraného GPU_COUNT — pro docker (čte .env) i native
configure_env_for_gpu_count "$GPU_COUNT"

# Pozn.: setup_native_env_parity se volá až v sekci Main (níž), protože jeho
# definice je v "NATIVE runtime helpers" pod tímhle blokem. Bash funkce musí
# být definovaná PŘED voláním — volání tady by spadlo na `command not found`.

# Propagace GPU_COUNT do environment proměnných pro docker compose
export EXTRACTOR_TP_SIZE="$GPU_COUNT"
# Critic (Gemma 4 31B NVFP4) ~16 GB → vždy stačí 1 GPU
export CRITIC_TP_SIZE="${CRITIC_TP_SIZE:-1}"
# HW label propaguje do labeler containeru přes argument (viz invoke_*)
HW_LABEL="${HW_LABEL:-${GPU_COUNT}x B200}"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight_cuda() {
    if [ "${SKIP_CUDA_CHECK:-}" = "1" ]; then
        warn "SKIP_CUDA_CHECK=1 — přeskakuji CUDA verifikaci."
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi nenalezen — nemůžu verifikovat CUDA verzi."
        return 0
    fi
    local cuda_line
    cuda_line=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | head -1 || true)
    [ -z "$cuda_line" ] && return 0
    local cuda_ver major
    cuda_ver="${cuda_line#CUDA Version: }"
    major="${cuda_ver%%.*}"
    if [ "$major" -lt 13 ]; then
        err "Detekováno CUDA $cuda_ver — pipeline vyžaduje CUDA 13+ pro NVFP4."
        err "Upgrade driveru ≥580. Nebo SKIP_CUDA_CHECK=1 pro test FP8/BF16 fallback."
        return 1
    fi
    log "Preflight OK — CUDA $cuda_ver detekována (≥13.0)."
}

check_data() {
    if [ ! -d "data" ]; then
        err "Chybí složka data/ — nakopíruj tam vstupní JSON soubory"
        exit 1
    fi
    local json_count
    json_count=$(find data -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
    if [ "$json_count" -eq 0 ]; then
        err "V data/ nenalezeny žádné *.json soubory"
        exit 1
    fi
    log "Detekováno $json_count JSON soubor(ů) v data/"
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers — žádný docker, Python přímo proti běžícímu LLM
# ---------------------------------------------------------------------------
# NATIVE_AUTO_LLM:
#   1 (default) — skript SÁM nainstaluje SGLang (pokud chybí), spustí
#                 server pro každou fázi, počká na ready, a po fázi
#                 server zastaví. Plně automatický native workflow.
#   0           — manuální mode: skript jen ověří endpoint health check,
#                 user si server spouští sám.
NATIVE_AUTO_LLM="${NATIVE_AUTO_LLM:-1}"

# Setup env vars 1:1 jako Docker (parita s docker-compose.yml LLM service):
#   - HF caching + xet performance
#   - PYTHONWARNINGS + TRANSFORMERS_VERBOSITY (suppress noise)
#   - BASE_URL labeler → localhost (config.py default je Docker hostname)
setup_native_env_parity() {
    if [ "$RUNTIME" != "native" ]; then
        return 0
    fi
    # === HuggingFace ===
    # Default cache na ~/.cache/huggingface; Docker mapuje na /models —
    # tu volbu necháme uživateli (kdo chce sdílet model cache mezi
    # Docker/native, nastaví HF_HOME=$MODEL_CACHE_DIR explicit).
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

    # === Suppress noisy warnings (1:1 s Docker LLM containerem) ===
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning,ignore::FutureWarning}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"

    # === Labeler BASE URLs ===
    # Config.py defaultně použije Docker hostnames (`extractor:8000` /
    # `critic:8001`), které na hostu neexistují. Pro native nastavíme
    # localhost defaults — uživatel může override přes env.
    export EXTRACTOR_BASE_URL="${EXTRACTOR_BASE_URL:-http://localhost:8000/v1}"
    export CRITIC_BASE_URL="${CRITIC_BASE_URL:-http://localhost:8001/v1}"
    # REVISOR_BASE_URL defaultně = EXTRACTOR_BASE_URL (config.py logika)
    export REVISOR_BASE_URL="${REVISOR_BASE_URL:-$EXTRACTOR_BASE_URL}"

    log "Native env parita: HF_XET_HIGH_PERFORMANCE=1, PYTHONWARNINGS suppress,"
    log "                   EXTRACTOR_BASE_URL=$EXTRACTOR_BASE_URL"
    log "                   CRITIC_BASE_URL=$CRITIC_BASE_URL"
}

# Sestavi seznam SGLang argumentů 1:1 jako docker-compose.yml command bloky.
# Args: $1=phase (taxonomy|extract|critic|revise) — taxonomy sdílí extractor model
# Outputs (přes echo): args řetězec pro `--key value` páry
build_sglang_args_for_phase() {
    local phase="$1"
    local args=""
    local quant_var batch_var ctx_var piecewise_var quant_val

    case "$phase" in
        taxonomy|extract|revise)
            quant_var="EXTRACTOR_QUANTIZATION"
            batch_var="EXTRACTOR_BATCH_SIZE"
            ctx_var="EXTRACTOR_CONTEXT_LEN"
            piecewise_var="EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH"
            # Default batch / context per Docker compose extractor
            local batch="${EXTRACTOR_BATCH_SIZE:-32}"
            local ctx="${EXTRACTOR_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx --reasoning-parser=qwen3"
            ;;
        critic)
            quant_var="CRITIC_QUANTIZATION"
            batch_var="CRITIC_BATCH_SIZE"
            ctx_var="CRITIC_CONTEXT_LEN"
            piecewise_var="CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH"
            # Default batch / context per Docker compose critic
            local batch="${CRITIC_BATCH_SIZE:-64}"
            local ctx="${CRITIC_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx"
            ;;
        *) return 0 ;;
    esac

    # QUANT_FLAG (1:1 logika z docker-compose.yml)
    quant_val=$(eval "echo \${$quant_var:-}")
    if [ -n "$quant_val" ]; then
        args="$args --quantization=$quant_val"
    fi

    # PIECEWISE_FLAG (per-container > globální fallback)
    local piecewise_val global_piecewise
    piecewise_val=$(eval "echo \${$piecewise_var:-}")
    global_piecewise="${DISABLE_PIECEWISE_CUDA_GRAPH:-}"
    local eff_piecewise="${piecewise_val:-$global_piecewise}"
    if [ "$eff_piecewise" = "1" ] || [ "$eff_piecewise" = "true" ]; then
        args="$args --disable-piecewise-cuda-graph"
    fi

    echo "$args"
}

# Auto-install SGLang pokud chybí (NATIVE_AUTO_LLM=1)
auto_install_sglang_if_missing() {
    if python -c "import sglang" 2>/dev/null; then
        return 0
    fi
    warn "SGLang nenalezen v Python prostředí."
    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        err "NATIVE_AUTO_LLM=0 → instalaci přeskakuji. Nainstaluj ručně:"
        err "  ./scripts/install_sglang.sh"
        return 1
    fi
    log "NATIVE_AUTO_LLM=1 → spouštím auto-install (./scripts/install_sglang.sh)"
    if ./scripts/install_sglang.sh; then
        log "✅ SGLang nainstalován"
        return 0
    fi
    err "Auto-install SGLang selhal."
    return 1
}

# Health check LLM endpointu
verify_native_llm_endpoint() {
    local url="$1"
    local label="$2"
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl není nainstalovaný, přeskakuji health check $label endpointu"
        return 0
    fi
    if curl -fs --max-time 5 "${url%/}/models" >/dev/null 2>&1; then
        log "✓ $label endpoint $url odpovídá"
        return 0
    fi
    return 1
}

# Auto-start SGLang server pro danou fázi
# Args: phase (taxonomy|extract|critic|revise), port
# Returns: 0 = server běží (buď naším skriptem nebo už předtím), 1 = fail
auto_start_native_llm() {
    local phase="$1"
    local port="$2"
    local check_url="http://localhost:${port}/v1"

    # Pokud server už odpovídá (user ho spustil ručně nebo zůstal z minulé fáze)
    if verify_native_llm_endpoint "$check_url" "$phase"; then
        log "LLM server na portu $port už běží — nepouštím nový"
        return 0
    fi

    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        err "$phase endpoint $check_url NEODPOVÍDÁ."
        err "V manual native módu (NATIVE_AUTO_LLM=0) musíš PŘED skriptem"
        err "sám spustit LLM server. Příklad:"
        err "  ./scripts/start_native_llm.sh <model> $port $GPU_COUNT"
        return 1
    fi

    # Auto-start — per-fáze model + TP size + args 1:1 jako Docker
    local model tp_size
    case "$phase" in
        taxonomy|extract|revise)
            # Qwen3.5-122B-A10B-NVFP4 — TP=EXTRACTOR_TP_SIZE
            model="${EXTRACTOR_MODEL:-RedHatAI/Qwen3.5-122B-A10B-NVFP4}"
            tp_size="${EXTRACTOR_TP_SIZE:-${GPU_COUNT:-1}}"
            ;;
        critic)
            # Gemma-4-31B-IT-NVFP4 (~16 GB) — VŽDY TP=1
            model="${CRITIC_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
            tp_size="${CRITIC_TP_SIZE:-1}"
            ;;
        *)
            return 0  # finalize / report → bez LLM
            ;;
    esac

    # Sestavit args 1:1 jako Docker (--max-running-requests, --context-length,
    # --reasoning-parser=qwen3 pro extract, --quantization, --disable-piecewise-cuda-graph)
    local sglang_args
    sglang_args=$(build_sglang_args_for_phase "$phase")

    log "Auto-start SGLang pro fázi '$phase': $model na portu $port (tp=$tp_size)"
    log "  Args: $sglang_args"
    if SGLANG_EXTRA_ARGS="$sglang_args" \
        ./scripts/start_native_llm.sh "$model" "$port" "$tp_size"; then
        return 0
    fi
    err "Auto-start SGLang selhal pro fázi '$phase'."
    return 1
}

# Auto-stop SGLang server po fázi (jen pokud jsme ho my zapnuli)
auto_stop_native_llm() {
    local port="$1"
    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        return 0
    fi
    if [ -f "/tmp/sglang_${port}.pid" ]; then
        log "Auto-stop SGLang na portu $port (cleanup před další fází)"
        ./scripts/stop_native_llm.sh "$port" || true
    fi
}

# Spustí Python labeler smoke-test --phase X přímo na hostu, bez docker compose.
run_phase_native() {
    local phase="$1"
    log "Smoke FÁZE [$phase] — NATIVE (Python přímo, žádný docker)"

    # 0) Auto-install SGLang (jen pokud LLM fáze)
    if [ "$phase" != "finalize" ] && [ "$phase" != "report" ]; then
        auto_install_sglang_if_missing || return 1
    fi

    # 1) Auto-start LLM server pro tuto fázi
    local port=""
    case "$phase" in
        taxonomy|extract|revise) port="8000" ;;
        critic)         port="8001" ;;
    esac
    if [ -n "$port" ]; then
        auto_start_native_llm "$phase" "$port" || return 1
    fi

    # 2) Spustit Python labeler smoke-test --phase X
    local rc=0
    python -m labeler.main smoke-test --phase "$phase" \
        --hw-label "$HW_LABEL" \
        --samples-per-dataset "$SAMPLES_PER_DATASET" \
        --docker-startup-s "$DOCKER_STARTUP_S" || rc=$?

    # 3) Auto-stop LLM server (pokud jsme ho my zapnuli)
    if [ -n "$port" ]; then
        auto_stop_native_llm "$port"
    fi

    if [ "$rc" -eq 0 ]; then
        log "[$phase] hotov (native)"
        return 0
    fi
    err "[$phase] selhal (native, rc=$rc)"
    return $rc
}

# Cleanup hook — při exit (i KILL) zastav VŠECHNY SGLang servery, které
# jsme my spustili. Iterujeme přes PID files místo hardcoded ports — kdyby
# se v budoucnu přidaly další porty (REVISOR samostatný atd.), trap je
# zachytí automaticky.
cleanup_native_servers_on_exit() {
    if [ "$RUNTIME" != "native" ] || [ "$NATIVE_AUTO_LLM" != "1" ]; then
        return 0
    fi
    local log_dir="${SGLANG_LOG_DIR:-/tmp}"
    # shellcheck disable=SC2144
    for pid_file in "$log_dir"/sglang_*.pid; do
        [ -e "$pid_file" ] || continue   # glob nematchne → exit loop
        # Extract port z file name: sglang_8000.pid → 8000
        local port
        port=$(basename "$pid_file" .pid)
        port="${port#sglang_}"
        warn "Cleanup: stopping SGLang na portu $port (script exit)"
        ./scripts/stop_native_llm.sh "$port" 2>/dev/null || true
    done
}
trap cleanup_native_servers_on_exit EXIT

# ---------------------------------------------------------------------------
# Per-phase smoke runner (DOCKER mód)
#   $1 = phase (extract | critic | revise)
#   $2 = profile prefix (extract-sglang, extract-vllm, ...)
#   $3 = labeler service name (labeler-extract-sglang, ...)
# ---------------------------------------------------------------------------
run_phase_via_profile() {
    local phase="$1"
    local profile="$2"
    local labeler_service="$3"
    log "Smoke FÁZE [$phase] — profile=$profile (detached -d na pozadí)"

    # `docker compose run -d --rm` spustí labeler v background módu.
    # Vrátí container ID; přes `docker wait` čekáme na exit code.
    # `--rm` smaže container po skončení; LLM server zůstane, smažeme přes
    # `docker compose --profile X down` po dokončení wait.
    local cid
    cid=$(docker compose --profile "$profile" run -d --rm \
           -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke \
           "$labeler_service" \
           smoke-test --phase "$phase" \
           --hw-label "$HW_LABEL" \
           --samples-per-dataset "$SAMPLES_PER_DATASET" \
           --docker-startup-s "$DOCKER_STARTUP_S" 2>/dev/null || true)

    if [ -z "$cid" ]; then
        err "[$phase] docker compose run -d selhal (profile $profile)"
        docker compose --profile "$profile" down 2>/dev/null || true
        return 1
    fi

    log "Container $cid běží na pozadí. Sleduj progress:"
    log "  docker logs -f $cid"

    # Čekej na exit kódu containeru
    local rc=0
    docker wait "$cid" >/dev/null 2>&1 || true
    rc=$(docker inspect --format='{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 1)

    docker compose --profile "$profile" down 2>/dev/null || true

    if [ "$rc" -eq 0 ]; then
        log "[$phase] hotov přes $profile"
        return 0
    fi
    err "[$phase] selhal na profilu $profile (rc=$rc). Cleanup hotov."
    return 1
}

run_smoke_taxonomy() {
    log "FÁZE 0 (smoke) — TAXONOMY (auto-discovery, jednorázová konstanta)"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "taxonomy"
        return $?
    fi
    # Taxonomy auto-discovery používá EXTRACTOR LLM → běží přes extractor
    # profily (stejný model i container jako extract).
    for backend in $EXTRACTOR_BACKENDS; do
        if run_phase_via_profile "taxonomy" "extract-${backend}" "labeler-extract-${backend}"; then
            return 0
        fi
        warn "Backend $backend selhal pro taxonomy, zkouším další…"
    done
    err "Všechny taxonomy backendy selhaly."
    return 1
}

run_smoke_extract() {
    log "FÁZE 1 (smoke) — EXTRACT"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "extract"
        return $?
    fi
    for backend in $EXTRACTOR_BACKENDS; do
        if run_phase_via_profile "extract" "extract-${backend}" "labeler-extract-${backend}"; then
            return 0
        fi
        warn "Backend $backend selhal pro extract, zkouším další…"
    done
    err "Všechny extract backendy selhaly."
    return 1
}

run_smoke_critic() {
    log "FÁZE 2 (smoke) — CRITIC"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "critic"
        return $?
    fi
    for backend in $CRITIC_BACKENDS; do
        if run_phase_via_profile "critic" "critic-${backend}" "labeler-critic-${backend}"; then
            return 0
        fi
        warn "Backend $backend selhal pro critic, zkouším další…"
    done
    err "Všechny critic backendy selhaly."
    return 1
}

run_smoke_revise() {
    log "FÁZE 2b (smoke) — REVISE"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "revise"
        return $?
    fi
    for backend in $REVISE_BACKENDS; do
        if run_phase_via_profile "revise" "revise-${backend}" "labeler-revise-${backend}"; then
            return 0
        fi
        warn "Backend $backend selhal pro revise, zkouším další…"
    done
    err "Všechny revise backendy selhaly."
    return 1
}

run_smoke_finalize() {
    log "FÁZE 3 (smoke) — FINALIZE (bez LLM, čistě CPU, detached -d)"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "finalize"
        return $?
    fi
    local cid
    cid=$(docker compose --profile finalize run -d --rm \
           -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke \
           labeler-finalize \
           smoke-test --phase finalize \
           --hw-label "$HW_LABEL" \
           --samples-per-dataset "$SAMPLES_PER_DATASET" \
           --docker-startup-s "$DOCKER_STARTUP_S" 2>/dev/null || true)
    if [ -z "$cid" ]; then
        err "Finalize smoke: docker compose run -d selhal"
        return 1
    fi
    log "Container $cid běží na pozadí. Sleduj: docker logs -f $cid"
    docker wait "$cid" >/dev/null 2>&1 || true
    local rc
    rc=$(docker inspect --format='{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 1)
    if [ "$rc" -eq 0 ]; then
        log "Finalize smoke hotov"
        return 0
    fi
    err "Finalize smoke selhal (rc=$rc)"
    return 1
}

run_smoke_report() {
    log "FÁZE 4 (smoke) — REPORT (extrapolace + ±2h konfidence)"
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "report"
        return $?
    fi
    # Report je rychlý (jen čte JSON timings), spustíme ho v background
    # módu kvůli konzistenci, ale výsledek je hned dostupný v checkpoints/smoke/.
    local cid
    cid=$(docker compose --profile finalize run -d --rm \
        -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke \
        labeler-finalize \
        smoke-test --phase report \
        --hw-label "$HW_LABEL" \
        --samples-per-dataset "$SAMPLES_PER_DATASET" \
        --docker-startup-s "$DOCKER_STARTUP_S" 2>/dev/null || true)
    if [ -z "$cid" ]; then
        err "Report: docker compose run -d selhal"
        return 1
    fi
    docker wait "$cid" >/dev/null 2>&1 || true
    docker logs "$cid" 2>/dev/null || true
    local rc
    rc=$(docker inspect --format='{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 1)
    [ "$rc" -eq 0 ] || err "Report selhal (rc=$rc)"
    return "$rc"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
preflight_cuda
check_data

# Setup env vars 1:1 jako Docker (jen v native módu — no-op v dockeru).
# Musí běžet až tady: definice je v sekci "NATIVE runtime helpers" výš, ale
# nad původním místem volání (před GPU promptem) ještě neexistovala, takže to
# padalo na `setup_native_env_parity: command not found`.
setup_native_env_parity

mkdir -p checkpoints/smoke

log "Smoke test config:"
echo "  RUNTIME=$RUNTIME"
echo "  GPU_COUNT=${GPU_COUNT}× B200"
echo "  EXTRACTOR_TP_SIZE=$EXTRACTOR_TP_SIZE"
echo "  CRITIC_TP_SIZE=$CRITIC_TP_SIZE"
echo "  HW_LABEL=$HW_LABEL"
echo "  SAMPLES_PER_DATASET=$SAMPLES_PER_DATASET"
echo "  SMOKE_PHASES=$SMOKE_PHASES"
echo "  DOCKER_STARTUP_S=$DOCKER_STARTUP_S"
if [ "$RUNTIME" = "native" ]; then
    echo "  EXTRACTOR_BASE_URL=${EXTRACTOR_BASE_URL:-http://localhost:8000/v1} (musí už běžet)"
    echo "  CRITIC_BASE_URL=${CRITIC_BASE_URL:-http://localhost:8001/v1} (musí už běžet)"
fi
echo "  Output: checkpoints/smoke/"

# V native módu chceme aby smoke_test.py psal do checkpoints/smoke/, ne do
# čisté CHECKPOINT_DIR. Toto env var smoke_test.py respektuje.
if [ "$RUNTIME" = "native" ]; then
    export SMOKE_CHECKPOINT_DIR="${SMOKE_CHECKPOINT_DIR:-$(pwd)/checkpoints/smoke}"
fi

for phase in $SMOKE_PHASES; do
    case "$phase" in
        taxonomy) run_smoke_taxonomy ;;
        extract)  run_smoke_extract ;;
        critic)   run_smoke_critic ;;
        revise)   run_smoke_revise ;;
        finalize) run_smoke_finalize ;;
        *)
            err "Neznámá fáze: $phase. Podporované: taxonomy extract critic revise finalize"
            exit 1
            ;;
    esac
done

run_smoke_report

log "🎉 Smoke test hotov."
echo ""
echo "Artefakty:"
echo "  checkpoints/smoke/smoke_timings.json     — raw per-fázová měření"
echo "  checkpoints/smoke/smoke_test_report.md   — human-readable report"
echo "  checkpoints/smoke/smoke_test_report.json — strojový export"
