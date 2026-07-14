#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
cd "$(dirname "$0")/.."

EXTRACTOR_BACKENDS="${EXTRACTOR_BACKENDS:-sglang vllm}"
CRITIC_BACKENDS="${CRITIC_BACKENDS:-sglang vllm}"
REVISE_BACKENDS="${REVISE_BACKENDS:-sglang vllm}"

# RUNTIME: kde pipeline poběží:
#   docker (default) — orchestrace přes docker-compose (LLM servery + labeler v kontejnerech)
#   native           — Python labeler spuštěný přímo na hostu proti běžícímu
#                      LLM serveru (uživatel si SGLang/vLLM zapne sám)
#
# Použití (native):
#   # 1) Sám si spusť SGLang server:
#   #    python -m sglang.launch_server --model-path RedHatAI/Qwen3.5-122B-A10B-NVFP4 \
#   #        --host 0.0.0.0 --port 8000 --tp 1
#   # 2) Pak skript v native módu:
#   RUNTIME=native ./scripts/run_production.sh all
RUNTIME="${RUNTIME:-docker}"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
err()  { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

# ---------------------------------------------------------------------------
# Preflight checks — chytá běžné konfiguraní problémy PŘED docker compose
# ---------------------------------------------------------------------------
# Bez těchto checků skončíš s krypitckými Docker errory typu:
#   - "nvidia-container-cli: device error: 1: unknown device"
#   - "Quantization method specified in model config does not match"
#
# Tady detekujeme tyto situace a vyhodíme čitelný error s návodem na fix.

# Detekuje skutečný počet GPU na hostu via nvidia-smi. Vrátí 0 pokud
# nvidia-smi není dostupné (běžně CI nebo CPU-only host).
detect_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return 0
    fi
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
}

# Quantization preflight — chytí mismatch mezi modelem a env varem.
# ---------------------------------------------------------------------------
# .env auto-config — po prompt podle GPU_COUNT
# ---------------------------------------------------------------------------
_update_env_var() {
    local file="$1" key="$2" value="$3"
    if [ ! -f "$file" ]; then return 1; fi
    if grep -qE "^[[:space:]]*${key}=" "$file"; then
        sed -i.bak -E "s|^[[:space:]]*${key}=.*|${key}=${value}|" "$file" && rm -f "${file}.bak"
    else
        printf "%s=%s\n" "$key" "$value" >> "$file"
    fi
}

# Přečte hodnotu jedné proměnné z .env BEZ sourcingu (1:1 jako run_smoke_test.sh).
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
            return 0
        fi
    fi
    _update_env_var "$env_file" GPU_COUNT "$n"
    _update_env_var "$env_file" EXTRACTOR_TP_SIZE "$n"
    _update_env_var "$env_file" CRITIC_TP_SIZE "1"
    log "Auto-config $env_file: GPU_COUNT=$n EXTRACTOR_TP_SIZE=$n CRITIC_TP_SIZE=1"
}

preflight_quantization() {
    # Pokud uživatel ručně nastavil EXTRACTOR_QUANTIZATION, validuj že
    # nemiří proti NVFP4 modelu (ten má quantization v config.json
    # a SGLang/vLLM si ji auto-detect).
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
    # Stejná logika pro critic
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

# HF_TOKEN preflight — některé modely jsou gated.
preflight_hf_token() {
    local env_file="${ENV_FILE:-.env}"
    local token="${HF_TOKEN:-}"
    # Token nemusí být v shellu — docker compose ho čte přímo z .env. Když
    # v shellu není, načti ho z .env (tam ho user reálně vyplňuje), ať
    # nehlásíme falešně "HF_TOKEN není nastavený".
    if [ -z "$token" ]; then
        token=$(_read_env_var "$env_file" HF_TOKEN)
    fi
    if [ -z "$token" ] || [ "$token" = "hf_TADY_DOPLN_TOKEN" ]; then
        warn "HF_TOKEN není nastavený nebo má placeholder hodnotu."
        warn "Gated modely (Gemma, restricted Llama atd.) NEPŮJDOU stáhnout."
        warn "Pokud potřebuješ, doplň HF_TOKEN v .env:"
        warn "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    fi
}

# ---------------------------------------------------------------------------
# Interactive GPU prompt (s GPU count validation)
# ---------------------------------------------------------------------------
# Pokud GPU_COUNT není v env (např. CI), interaktivně vyzve uživatele.
# Validní hodnoty: 1, 2, 4 (power-of-2 pro tensor parallelism MoE).
# Min 1, max 4 — produkční nasazení této pipeline je single-node B200.
#
# Pro fázi 'finalize' (CPU only, bez LLM) prompt přeskočíme.
prompt_gpu_count() {
    if [ "$PHASE" = "finalize" ]; then
        log "Phase=finalize → bez LLM, GPU prompt přeskočen."
        return 0
    fi

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
        # Validace proti reálnému počtu GPU
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

# Spustit preflighty pro fáze, které spouštějí LLM
if [ "$PHASE" != "finalize" ]; then
    preflight_quantization
    preflight_hf_token
fi

prompt_gpu_count

# Auto-config .env podle GPU_COUNT (pro docker compose i native; jen pokud
# prompt_gpu_count nastavil GPU_COUNT — finalize fáze ho přeskočí).
if [ -n "${GPU_COUNT:-}" ]; then
    configure_env_for_gpu_count "$GPU_COUNT"
fi

# Propagace GPU_COUNT do environment proměnných pro docker compose.
# EXTRACTOR_TP_SIZE = GPU_COUNT; CRITIC_TP_SIZE = 1 (Gemma 4 31B NVFP4 ~16 GB
# se vejde na 1 GPU vždy — TP > 1 by jen přidal overhead).
if [ "${PHASE:-all}" != "finalize" ]; then
    export EXTRACTOR_TP_SIZE="${EXTRACTOR_TP_SIZE:-$GPU_COUNT}"
    export CRITIC_TP_SIZE="${CRITIC_TP_SIZE:-1}"
    log "Docker GPU config: EXTRACTOR_TP_SIZE=$EXTRACTOR_TP_SIZE CRITIC_TP_SIZE=$CRITIC_TP_SIZE"
fi

# ---------------------------------------------------------------------------
# Preflight: CUDA 13+ check
# ---------------------------------------------------------------------------
preflight_cuda() {
    if [ "${SKIP_CUDA_CHECK:-}" = "1" ]; then
        warn "SKIP_CUDA_CHECK=1 — preskakuji CUDA verifikaci."
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi nenalezen — nemůžu verifikovat CUDA verzi."
        warn "Pokud kontejnery selhou s 'no kernel image' chybou, ujisti se,"
        warn "že máš NVIDIA driver ≥ 580 (CUDA 13.0 capable)."
        return 0
    fi
    # nvidia-smi vypise "CUDA Version: 13.0" v hlavicce.
    local cuda_line
    cuda_line=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | head -1 || true)
    if [ -z "$cuda_line" ]; then
        warn "Nelze získat CUDA verzi z nvidia-smi výstupu — pokračuji."
        return 0
    fi
    local cuda_ver major
    cuda_ver="${cuda_line#CUDA Version: }"
    major="${cuda_ver%%.*}"
    if [ "$major" -lt 13 ]; then
        err "Detekováno CUDA $cuda_ver — pipeline VYŽADUJE CUDA 13+ pro NVFP4"
        err "na Blackwell SM120. Upgrade NVIDIA driveru na ≥ 580 a CUDA toolkit ≥ 13.0."
        err "Pokud chceš pokračovat na vlastní riziko (např. testuješ FP8/BF16 fallback),"
        err "spusť: SKIP_CUDA_CHECK=1 ./scripts/run_production.sh $PHASE"
        return 1
    fi
    log "Preflight OK — CUDA $cuda_ver detekována (≥13.0)."
}

if [ "$PHASE" != "finalize" ]; then
    preflight_cuda
fi

try_profile() {
    local profile="$1"
    local labeler_service="$2"
    log "Zkouším profil: $profile (detached -d, containery běží na pozadí)"

    # Start v detached módu — pokračuje hned po startu containerů.
    if ! docker compose --profile "$profile" up -d; then
        err "Profile $profile: docker compose up -d selhal"
        docker compose --profile "$profile" down 2>/dev/null || true
        return 1
    fi

    log "Containery běží na pozadí. Sleduj progress v separátním terminálu:"
    log "  docker compose --profile $profile logs -f --no-log-prefix $labeler_service"

    # Čekej, až labeler container skončí, použij jeho exit code.
    # `docker compose wait` (compose v2.21+) vrátí přesný exit code.
    local rc=0
    if ! docker compose --profile "$profile" wait "$labeler_service" >/dev/null 2>&1; then
        # Fallback pro starší compose: získej container ID a `docker wait`
        local cid
        cid=$(docker compose --profile "$profile" ps -q "$labeler_service" 2>/dev/null | head -1)
        if [ -n "$cid" ]; then
            docker wait "$cid" >/dev/null 2>&1
            rc=$(docker inspect --format='{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 1)
        else
            err "Nelze najít $labeler_service container"
            rc=1
        fi
    else
        rc=$(docker compose --profile "$profile" wait "$labeler_service" 2>/dev/null || echo 1)
    fi

    docker compose --profile "$profile" down 2>/dev/null || true
    if [ "$rc" -eq 0 ]; then
        log "Profile $profile hotov ✓"
        return 0
    fi
    err "Profile $profile selhal (rc=$rc)"
    return 1
}

# ---------------------------------------------------------------------------
# NATIVE runtime helpers — bez dockeru, Python přímo proti běžícímu LLM
# ---------------------------------------------------------------------------
# NATIVE_AUTO_LLM: 1 (default) = auto-install + auto-start/stop SGLang
#                  0 = manual (user si LLM zapne sám, skript jen čeká)
NATIVE_AUTO_LLM="${NATIVE_AUTO_LLM:-1}"

# Setup env vars 1:1 jako Docker (parita)
setup_native_env_parity() {
    if [ "$RUNTIME" != "native" ]; then
        return 0
    fi
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning,ignore::FutureWarning}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
    export EXTRACTOR_BASE_URL="${EXTRACTOR_BASE_URL:-http://localhost:8000/v1}"
    export CRITIC_BASE_URL="${CRITIC_BASE_URL:-http://localhost:8001/v1}"
    export REVISOR_BASE_URL="${REVISOR_BASE_URL:-$EXTRACTOR_BASE_URL}"
    log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress,"
    log "  EXTRACTOR_BASE_URL=$EXTRACTOR_BASE_URL"
    log "  CRITIC_BASE_URL=$CRITIC_BASE_URL"
}

# Sestaví SGLang args 1:1 jako docker-compose.yml command block.
# Args: $1 = phase (extract|critic|revise)
build_sglang_args_for_phase() {
    local phase="$1"
    local args=""
    case "$phase" in
        taxonomy|extract|revise)
            local batch="${EXTRACTOR_BATCH_SIZE:-32}"
            local ctx="${EXTRACTOR_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx --reasoning-parser=qwen3"
            [ -n "${EXTRACTOR_QUANTIZATION:-}" ] && args="$args --quantization=${EXTRACTOR_QUANTIZATION}"
            local eff="${EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH:-${DISABLE_PIECEWISE_CUDA_GRAPH:-}}"
            [ "$eff" = "1" ] || [ "$eff" = "true" ] && args="$args --disable-piecewise-cuda-graph"
            ;;
        critic)
            local batch="${CRITIC_BATCH_SIZE:-64}"
            local ctx="${CRITIC_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx"
            [ -n "${CRITIC_QUANTIZATION:-}" ] && args="$args --quantization=${CRITIC_QUANTIZATION}"
            local eff="${CRITIC_DISABLE_PIECEWISE_CUDA_GRAPH:-${DISABLE_PIECEWISE_CUDA_GRAPH:-}}"
            [ "$eff" = "1" ] || [ "$eff" = "true" ] && args="$args --disable-piecewise-cuda-graph"
            ;;
    esac
    echo "$args"
}

setup_native_env_parity

auto_install_sglang_if_missing() {
    if python -c "import sglang" 2>/dev/null; then
        return 0
    fi
    warn "SGLang nenalezen."
    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        err "NATIVE_AUTO_LLM=0 → ručně: ./scripts/install_sglang.sh"
        return 1
    fi
    log "Auto-install SGLang..."
    ./scripts/install_sglang.sh
}

verify_native_llm_endpoint() {
    local url="$1"
    if ! command -v curl >/dev/null 2>&1; then
        return 0
    fi
    curl -fs --max-time 5 "${url%/}/models" >/dev/null 2>&1
}

auto_start_native_llm() {
    local phase="$1"
    local port="$2"
    if verify_native_llm_endpoint "http://localhost:${port}/v1"; then
        log "LLM server na portu $port už běží — používám ho"
        return 0
    fi
    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        err "Endpoint http://localhost:${port}/v1 NEODPOVÍDÁ."
        err "V manual módu (NATIVE_AUTO_LLM=0) spusť LLM server ručně:"
        err "  ./scripts/start_native_llm.sh <model> $port ${EXTRACTOR_TP_SIZE:-1}"
        return 1
    fi
    local model tp_size
    case "$phase" in
        taxonomy|extract|revise)
            model="${EXTRACTOR_MODEL:-RedHatAI/Qwen3.5-122B-A10B-NVFP4}"
            tp_size="${EXTRACTOR_TP_SIZE:-${GPU_COUNT:-1}}"
            ;;
        critic)
            model="${CRITIC_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
            tp_size="${CRITIC_TP_SIZE:-1}"
            ;;
        *) return 0 ;;
    esac
    # Args 1:1 jako Docker (batch, context, quantization, piecewise atd.)
    local sglang_args
    sglang_args=$(build_sglang_args_for_phase "$phase")
    log "Auto-start SGLang pro '$phase': $model na portu $port (tp=$tp_size)"
    log "  Args: $sglang_args"
    SGLANG_EXTRA_ARGS="$sglang_args" \
        ./scripts/start_native_llm.sh "$model" "$port" "$tp_size"
}

auto_stop_native_llm() {
    local port="$1"
    if [ "$NATIVE_AUTO_LLM" != "1" ]; then
        return 0
    fi
    if [ -f "/tmp/sglang_${port}.pid" ]; then
        log "Auto-stop SGLang na portu $port"
        ./scripts/stop_native_llm.sh "$port" || true
    fi
}

cleanup_native_servers_on_exit() {
    if [ "$RUNTIME" != "native" ] || [ "$NATIVE_AUTO_LLM" != "1" ]; then
        return 0
    fi
    # Iterate over PID files místo hardcoded portů — robustní pro budoucí
    # rozšíření (např. samostatný REVISOR endpoint).
    local log_dir="${SGLANG_LOG_DIR:-/tmp}"
    # shellcheck disable=SC2144
    for pid_file in "$log_dir"/sglang_*.pid; do
        [ -e "$pid_file" ] || continue
        local port
        port=$(basename "$pid_file" .pid)
        port="${port#sglang_}"
        warn "Cleanup: stop SGLang na portu $port (exit)"
        ./scripts/stop_native_llm.sh "$port" 2>/dev/null || true
    done
}
trap cleanup_native_servers_on_exit EXIT

run_native_phase() {
    local phase="$1"  # extract|critic|revise|finalize
    log "FÁZE [$phase] — NATIVE (Python přímo, bez dockeru)"

    if [ "$phase" != "finalize" ]; then
        auto_install_sglang_if_missing || return 1
        local port=""
        case "$phase" in
            taxonomy|extract|revise) port="8000" ;;
            critic)                  port="8001" ;;
        esac
        if [ -n "$port" ]; then
            auto_start_native_llm "$phase" "$port" || return 1
        fi
    fi

    # extract dostane EXTRACT_EXTRA_ARGS (mj. --taxonomy-mode off, když fázi 0
    # spustil samostatný taxonomy krok v `all` — ať se discovery nedělá dvakrát).
    local extra=""
    [ "$phase" = "extract" ] && extra="${EXTRACT_EXTRA_ARGS:-}"
    local rc=0
    python -m labeler.main "$phase" $extra || rc=$?

    # Auto-stop po fázi (mezi fázema swapujeme Qwen ↔ Gemma)
    if [ "$phase" != "finalize" ]; then
        local port=""
        case "$phase" in
            taxonomy|extract|revise) port="8000" ;;
            critic)                  port="8001" ;;
        esac
        [ -n "$port" ] && auto_stop_native_llm "$port"
    fi

    if [ "$rc" -eq 0 ]; then
        log "✅ $phase hotov (native)"
        return 0
    fi
    err "$phase selhal (native, rc=$rc)"
    return $rc
}

run_taxonomy() {
    log "FÁZE 0 — TAXONOMY (auto-discovery, jednotná taxonomie napříč datasety)"
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "taxonomy"
        return $?
    fi
    # Taxonomy běží na EXTRACTOR LLM (Qwen). Spustíme přes `run` (override
    # commandu na `taxonomy`), foreground → přímý exit code (žádný wait/inspect).
    for backend in $EXTRACTOR_BACKENDS; do
        log "Zkouším profil: extract-${backend} (taxonomy)"
        local rc=0
        docker compose --profile "extract-${backend}" run --rm \
            "labeler-extract-${backend}" taxonomy --mode auto || rc=$?
        docker compose --profile "extract-${backend}" down 2>/dev/null || true
        if [ "$rc" -eq 0 ]; then
            log "✅ Taxonomy hotova přes backend: $backend"
            return 0
        fi
        warn "Backend $backend selhal pro taxonomy (rc=$rc), zkouším další…"
    done
    err "Všechny taxonomy backendy selhaly."
    return 1
}

run_extract() {
    log "FÁZE 1 — EXTRACT (Qwen3.5-122B-A10B)"
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "extract"
        return $?
    fi
    for backend in $EXTRACTOR_BACKENDS; do
        if try_profile "extract-${backend}" "labeler-extract-${backend}"; then
            log "✅ Extract hotov přes backend: $backend"
            return 0
        fi
        err "Backend $backend selhal, zkouším další…"
    done
    err "VŠECHNY extract backendy selhaly. Zkontroluj logy."
    return 1
}

run_critic() {
    log "FÁZE 2 — CRITIC (Gemma 4 31B, sám na GPU)"
    if [ ! -f "${CHECKPOINT_DIR:-./checkpoints}/extractions.jsonl" ]; then
        err "Chybí checkpoints/extractions.jsonl — nejdřív spusť extract fázi"
        return 1
    fi
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "critic"
        return $?
    fi
    for backend in $CRITIC_BACKENDS; do
        if try_profile "critic-${backend}" "labeler-critic-${backend}"; then
            log "✅ Critic hotov přes backend: $backend"
            return 0
        fi
        err "Backend $backend selhal, zkouším další…"
    done
    err "VŠECHNY critic backendy selhaly."
    return 1
}

run_revise() {
    log "FÁZE 2b — REVISE (Qwen revisor projde non-ok kritiky, sám na GPU)"
    if [ ! -f "${CHECKPOINT_DIR:-./checkpoints}/critic_results.jsonl" ]; then
        err "Chybí checkpoints/critic_results.jsonl — nejdřív spusť critic fázi"
        return 1
    fi
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "revise"
        return $?
    fi
    for backend in $REVISE_BACKENDS; do
        if try_profile "revise-${backend}" "labeler-revise-${backend}"; then
            log "✅ Revise hotov přes backend: $backend"
            return 0
        fi
        err "Backend $backend selhal, zkouším další…"
    done
    err "VŠECHNY revise backendy selhaly."
    return 1
}

run_finalize() {
    log "FÁZE 3 — FINALIZE (merge + stats, bez LLM)"
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "finalize"
        return $?
    fi
    log "Spouštím finalize v detached módu (-d)"
    if ! docker compose --profile finalize up -d; then
        err "Finalize: docker compose up -d selhal"
        docker compose --profile finalize down 2>/dev/null || true
        return 1
    fi
    log "Container běží na pozadí. Sleduj progress:"
    log "  docker compose --profile finalize logs -f --no-log-prefix labeler-finalize"
    local rc=0
    rc=$(docker compose --profile finalize wait labeler-finalize 2>/dev/null || echo 1)
    docker compose --profile finalize down 2>/dev/null || true
    if [ "$rc" -eq 0 ]; then
        log "✅ Finalize hotov"
        return 0
    fi
    err "Finalize selhal (rc=$rc)"
    return 1
}

# POZN.: build je MANUÁLNÍ — tento script už image NEbuildí (žádné --build).
# Po změně kódu spusť nejdřív:  ./scripts/build.sh
warn "Build je manuální: tento script NEbuildí image. Pokud jsi měnil kód, spusť nejdřív ./scripts/build.sh"

case "$PHASE" in
    taxonomy) run_taxonomy ;;
    extract)  run_extract ;;
    critic)   run_critic ;;
    revise)   run_revise ;;
    finalize) run_finalize ;;
    all)
        # 1:1 jako smoke test (taxonomy → extract → critic → revise → finalize),
        # ale na CELÉM datasetu (žádné --samples-per-dataset).
        run_taxonomy
        # Fáze 0 proběhla samostatně → extract ji jen načte, nediscoveruje znovu.
        export EXTRACT_EXTRA_ARGS="--taxonomy-mode off${EXTRACT_EXTRA_ARGS:+ $EXTRACT_EXTRA_ARGS}"
        run_extract
        run_critic
        run_revise
        run_finalize
        log "🎉 VŠE HOTOVO. Výsledky v ${CHECKPOINT_DIR:-./checkpoints}/"
        log "   - final_labels.json + .jsonl  (hlavní výstup)"
        log "   - stats.json                  (souhrnné statistiky)"
        ;;
    *)
        err "Neznámá fáze: $PHASE. Použij: taxonomy|extract|critic|revise|finalize|all"
        exit 1
        ;;
esac
