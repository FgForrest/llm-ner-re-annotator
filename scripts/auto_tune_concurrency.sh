#!/usr/bin/env bash
# auto_tune_concurrency.sh — PLNĚ AUTOMATICKÉ naladění souběžnosti EXTRACTORU,
# CRITICU i REVISE fáze. Stačí spustit, nic dalšího.
#
# Co to udělá (pro každou zvolenou roli):
#   1. Nahodí přes docker compose (samostatný docker-compose.tune.yml) DVA
#      kontejnery: LLM server (extractor-sglang / critic-sglang) + Python
#      kontejner s scripts/probe_concurrency.py. Server se pro tuning pustí s
#      velkorysým --max-running-requests (= nejvyšší testovaná úroveň).
#   2. Probe pošle dávky reálných requestů (parita s produkcí té role — vč.
#      správných promptů NER/RE té fáze) a změří runaway (finish_reason=length).
#   3. Najde NEJVYŠŠÍ STABILNÍ souběžnost.
#   4. Tu hodnotu SÁM zapíše do .env (a .env.example, pokud existuje):
#        extractor → EXTRACTOR_CONCURRENCY + EXTRACTOR_BATCH_SIZE
#        critic    → CRITIC_CONCURRENCY   + CRITIC_BATCH_SIZE
#        revise    → REVISOR_CONCURRENCY  (revisor = defaultně extractor server;
#                                          server batch sdílí s extractem)
#   5. Kontejnery zase shodí (GPU se uvolní).
#
# Použití (z kořene repa, Docker + NVIDIA runtime):
#   ./scripts/auto_tune_concurrency.sh                  # DEFAULT = all (extractor+critic+revise)
#   ROLE=extractor ./scripts/auto_tune_concurrency.sh   # jen extractor
#   ROLE=critic   ./scripts/auto_tune_concurrency.sh    # jen critic
#   ROLE=revise   ./scripts/auto_tune_concurrency.sh    # jen revise (revizní prompty)
#   ROLE=both     ./scripts/auto_tune_concurrency.sh    # extractor + critic (zpětná kompat.)
#   GPU_COUNT=4   ./scripts/auto_tune_concurrency.sh    # 4× B200 (TP=4, vyšší levels)
#   LEVELS=1,32,64,96,128 ./scripts/auto_tune_concurrency.sh   # vlastní rozsah
#   APPLY=0 ./scripts/auto_tune_concurrency.sh          # jen změř, .env nepiš
#
# MULTI-GPU: script počítá s počtem B200. Extractor (Qwen3.5-122B MoE) poběží
# s tensor parallelism TP = největší mocnina 2 ≤ GPU_COUNT (1/2/4/8). Víc GPU =
# vyšší throughput + větší KV cache → vyšší stabilní strop, takže DEFAULT LEVELS
# se škálují s počtem GPU (1 GPU → do 64; 4 GPU → do 256; 8 GPU → do 512) a
# probe najde skutečný strop. Critic (Gemma 31B) jede TP=1 (vejde se na 1 GPU).
#
# Konfigurace přes env (defaulty v závorkách):
#   ROLE (all)  GPU_COUNT (auto-detekce/1)  LEVELS (auto dle GPU)
#   EXTRACTOR_LEVELS / REVISE_LEVELS (= LEVELS)  CRITIC_LEVELS (1,16,32,48,64,96)
#   MODE (ner)  MAX_TOKENS (dle role: extractor/revise 4096, critic 16384)
#   THINKING (dle role)  PER_LEVEL (auto)  TIMEOUT (900)
#   ENV_FILE (.env)  APPLY (1)  KEEP_SERVER (0)
#
# Pozn.: critic běží s thinking ON (generuje thinking trace + JSON), takže
# probe critic je POMALEJŠÍ a používá vyšší max_tokens — to je v pořádku.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Konfigurace ------------------------------------------------------------
ROLE="${ROLE:-all}"                # all (default) | extractor | critic | revise | both
LEVELS="${LEVELS:-}"               # prázdné → default dle počtu GPU (viz níže)
MODE="${MODE:-ner}"
MAX_TOKENS="${MAX_TOKENS:-}"       # prázdné → probe zvolí dle role
THINKING="${THINKING:-}"           # prázdné → probe zvolí dle role
PER_LEVEL="${PER_LEVEL:-}"
TIMEOUT="${TIMEOUT:-900}"
ENV_FILE="${ENV_FILE:-.env}"
APPLY="${APPLY:-1}"
KEEP_SERVER="${KEEP_SERVER:-0}"
# Benchmark je ZÁMĚRNĚ v samostatném compose (mimo produkční docker-compose.yml).
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.tune.yml}"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
err()  { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }
warn() { echo -e "\n\033[1;33m!!\033[0m $*" >&2; }

# --- .env helpers (1:1 jako run_smoke_test.sh) ------------------------------
_update_env_var() {
    local file="$1" key="$2" value="$3"
    [ -f "$file" ] || return 1
    if grep -qE "^[[:space:]]*${key}=" "$file"; then
        sed -i.bak -E "s|^[[:space:]]*${key}=.*|${key}=${value}|" "$file" && rm -f "${file}.bak"
    else
        printf "%s=%s\n" "$key" "$value" >> "$file"
    fi
}

_read_env_var() {
    local file="$1" key="$2" line val
    [ -f "$file" ] || return 0
    line=$(grep -E "^[[:space:]]*${key}=" "$file" | tail -n1 || true)
    [ -n "$line" ] || return 0
    val="${line#*=}"
    case "$val" in
        \"*\") val="${val#\"}"; val="${val%\"}" ;;
        \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
    printf '%s' "$val"
}

# --- Preflight --------------------------------------------------------------
command -v docker >/dev/null 2>&1 || { err "docker nenalezen v PATH."; exit 1; }
docker compose version >/dev/null 2>&1 || { err "'docker compose' (v2) nenalezen."; exit 1; }

case "$ROLE" in
    extractor|critic|revise|both|all) ;;
    *) err "ROLE='$ROLE' není podporováno. Validní: extractor | critic | revise | both | all."; exit 1 ;;
esac

if [ ! -f "$ENV_FILE" ]; then
    if [ -f .env.example ]; then
        cp .env.example "$ENV_FILE"; log "Vytvořeno $ENV_FILE z .env.example"
    else
        : > "$ENV_FILE"; warn "Neexistuje .env ani .env.example — vytvořen prázdný $ENV_FILE."
    fi
fi

HF_TOKEN_VAL="${HF_TOKEN:-$(_read_env_var "$ENV_FILE" HF_TOKEN)}"
if [ -z "$HF_TOKEN_VAL" ] || [ "$HF_TOKEN_VAL" = "hf_TADY_DOPLN_TOKEN" ]; then
    warn "HF_TOKEN není nastaven/má placeholder — pokud je model gated, stažení selže."
fi

detect_gpu_count() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return 0; }
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
}
if [ -z "${GPU_COUNT:-}" ]; then
    GPU_COUNT="$(detect_gpu_count)"
    [ "$GPU_COUNT" -ge 1 ] 2>/dev/null || GPU_COUNT=1
fi

# Extractor (Qwen3.5-122B MoE) tensor parallelism MUSÍ být mocnina 2 (1/2/4/8).
# Když má host např. 3 GPU, použijeme největší mocninu 2 ≤ count (zbylé volné).
_largest_pow2_le() {
    local n="$1" p=1
    while [ $((p * 2)) -le "$n" ] && [ $((p * 2)) -le 8 ]; do p=$((p * 2)); done
    echo "$p"
}
TP_EXTRACTOR="$(_largest_pow2_le "$GPU_COUNT")"
if [ "$TP_EXTRACTOR" != "$GPU_COUNT" ]; then
    warn "GPU_COUNT=$GPU_COUNT není mocnina 2 — extractor poběží s TP=$TP_EXTRACTOR (MoE TP musí být 1/2/4/8; zbylé GPU zůstanou volné)."
fi

# Default LEVELS škáluje s počtem GPU: víc GPU (vyšší TP) = vyšší throughput a
# větší KV cache → vyšší stabilní strop, takže testujeme výš. Probe pak najde
# skutečný strop. Override kdykoli přes LEVELS= / EXTRACTOR_LEVELS= / CRITIC_LEVELS=.
default_levels_for_gpu() {
    case "$1" in
        1) echo "32,48,64,96,128,160,192" ;;
        2) echo "32,64,96,128,192,256,320" ;;
        4) echo "32,64,128,192,256,384,512" ;;
        8) echo "64,128,256,384,512,768,1024" ;;
        *) echo "32,48,64,96,128,160,192" ;;
    esac
}
if [ -z "$LEVELS" ]; then
    LEVELS="$(default_levels_for_gpu "$TP_EXTRACTOR")"
    log "Default LEVELS pro ${TP_EXTRACTOR}× GPU: $LEVELS (override přes LEVELS=…)"
fi

CHECKPOINT_DIR="$(_read_env_var "$ENV_FILE" CHECKPOINT_DIR)"; CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
mkdir -p "${CHECKPOINT_DIR%/}/tune"

compose_down() {
    docker compose --env-file "$ENV_FILE" -f docker-compose.yml -f "$COMPOSE_FILE" down 2>/dev/null || true
}
teardown() {
    if [ "$KEEP_SERVER" = "1" ]; then
        warn "KEEP_SERVER=1 — nechávám servery běžet (shoď ručně: docker compose -f $COMPOSE_FILE down)."
        return 0
    fi
    log "Cleanup: shazuji tune kontejnery"
    compose_down
}
trap teardown EXIT

# --- Tuning jedné role ------------------------------------------------------
# $1 = role (extractor | critic | revise). Vrací 0 = zapsáno/OK, 2 = žádná
# stabilní úroveň.
tune_one() {
    local role="$1"
    local tune_service role_levels max_level conc_key batch_key result_host result_cont

    case "$role" in
        extractor)
            tune_service="labeler-tune-sglang"
            tune_profile="extract-sglang"
            role_levels="${EXTRACTOR_LEVELS:-$LEVELS}"
            conc_key="EXTRACTOR_CONCURRENCY"; batch_key="EXTRACTOR_BATCH_SIZE"
            ;;
        revise)
            # Revisor = defaultně tentýž server/model jako extractor (jen jiná
            # fáze + revizní prompty). Reuse extractor serveru; píšeme JEN
            # REVISOR_CONCURRENCY — server batch sdílí s extractem
            # (EXTRACTOR_BATCH_SIZE), proto batch_key prázdný.
            tune_service="labeler-tune-sglang"
            tune_profile="extract-sglang"
            role_levels="${REVISE_LEVELS:-$LEVELS}"
            conc_key="REVISOR_CONCURRENCY"; batch_key=""
            ;;
        critic)
            tune_service="labeler-tune-critic"
            tune_profile="critic-sglang"
            # Critic = Gemma 31B na 1 GPU (TP=1) → strop neškáluje s počtem GPU,
            # vlastni default. Override pres CRITIC_LEVELS=.
            role_levels="${CRITIC_LEVELS:-32,48,64,96,128,160,192}"
            conc_key="CRITIC_CONCURRENCY"; batch_key="CRITIC_BATCH_SIZE"
            ;;
    esac

    local current_levels="$role_levels"
    local best_safe=0

    while true; do
        max_level="$(printf '%s' "$current_levels" | tr ',' '\n' | sort -n | tail -n1)"
        [ -n "$max_level" ] || { err "[$role] LEVELS prázdné/neplatné: '$current_levels'"; return 1; }
        result_host="${CHECKPOINT_DIR%/}/tune/result_${role}.env"
        result_cont="/app/checkpoints/tune/result_${role}.env"
        rm -f "$result_host"

        # Server batch = nejvyšší úroveň (víc prostoru pro měření); TP dle role.
        # Extractor I revise jedou na EXTRACTOR serveru (revisor = extractor);
        # critic na svém.
        if [ "$role" = "critic" ]; then
            export CRITIC_TP_SIZE="${CRITIC_TP_SIZE:-1}"   # Gemma 31B se vejde na 1 GPU
            export CRITIC_BATCH_SIZE="$max_level"
        else
            export EXTRACTOR_TP_SIZE="$TP_EXTRACTOR"   # mocnina 2 ≤ GPU_COUNT
            export EXTRACTOR_BATCH_SIZE="$max_level"
        fi

        log "[$role] tuning — LEVELS=$current_levels (server --max-running-requests=$max_level)  MODE=$MODE"

        local probe_args=(
            scripts/probe_concurrency.py
            --role "$role" --mode "$MODE" --levels "$current_levels"
            --timeout "$TIMEOUT" --result-file "$result_cont"
        )
        [ -n "$MAX_TOKENS" ] && probe_args+=(--max-tokens "$MAX_TOKENS")
        [ -n "$THINKING" ]   && probe_args+=(--thinking "$THINKING")
        [ -n "$PER_LEVEL" ]  && probe_args+=(--per-level "$PER_LEVEL")

        log "[$role] rebuild image + spouštím SGLang + probe (první běh stahuje model z HF — desítky minut)…"
        docker compose --profile "$tune_profile" --env-file "$ENV_FILE" -f docker-compose.yml -f "$COMPOSE_FILE" build "$tune_service"

        local rc=0
        docker compose --profile "$tune_profile" --env-file "$ENV_FILE" -f docker-compose.yml -f "$COMPOSE_FILE" run --rm "$tune_service" \
            "${probe_args[@]}" || rc=$?

        # Server po měření shoď (aby se při další iteraci projevil nový batch_size);
        # KEEP_SERVER to přeskočí (při dynamickém zvedání to ale může bránit alokaci).
        [ "$KEEP_SERVER" = "1" ] || docker compose --profile "$tune_profile" --env-file "$ENV_FILE" -f docker-compose.yml -f "$COMPOSE_FILE" down 2>/dev/null || true

        if [ "$rc" -ne 0 ]; then
            err "[$role] probe kontejner skončil s rc=$rc (viz výpis výše)."
            return "$rc"
        fi
        if [ ! -f "$result_host" ]; then
            err "[$role] probe nezapsal výsledek ($result_host)."
            return 1
        fi

        local safe any_stable can_higher
        safe="$(_read_env_var "$result_host" safe_concurrency)"
        any_stable="$(_read_env_var "$result_host" any_stable)"
        can_higher="$(_read_env_var "$result_host" can_go_higher)"
        log "[$role] výsledek: safe_concurrency=$safe  any_stable=$any_stable  can_go_higher=$can_higher"

        if [ "${any_stable:-0}" = "1" ] && [ "${safe:-0}" -gt "$best_safe" ]; then
            best_safe="$safe"
        fi

        if [ "${any_stable:-0}" != "1" ] || [ "${safe:-0}" -lt 1 ] 2>/dev/null; then
            if [ "$best_safe" -gt 0 ]; then
                log "[$role] další úroveň už padla. Zastavuji hledání. (Nejvyšší stabilní dosud = $best_safe)"
                break
            else
                err "[$role] žádná stabilní úroveň (runaway už na nejnižší). .env NEMĚNÍM."
                return 2
            fi
        fi

        if [ "${can_higher:-0}" = "1" ]; then
            log "[$role] stabilní i na nejvyšší úrovni ($max_level) — dynamicky navyšuji zátěž a zkouším znovu!"
            local step=$(( max_level / 2 ))
            [ "$step" -lt 16 ] && step=16
            current_levels="$((max_level + step)),$((max_level + 2*step)),$((max_level + 3*step))"
            continue
        fi

        break
    done

    local final_safe=$(( best_safe * 2 / 3 ))
    [ "$final_safe" -lt 1 ] && final_safe=1

    echo ""
    log "[$role] TESTY DOKONČENY!"
    echo "=========================================================="
    echo "  [1] MAXIMÁLNÍ STABILNÍ: $best_safe"
    echo "      (Nejvyšší zátěž, která ještě prošla bez chyb)"
    echo "  [2] SUPER STABILNÍ:     $final_safe"
    echo "      (S velkou 33% rezervou pro absolutní jistotu)"
    echo "=========================================================="

    if [ "$APPLY" != "1" ]; then
        log "[$role] APPLY=0 → doporučené hodnoty (nezapsáno)."
        return 0
    fi

    local chosen_safe=""
    while true; do
        read -r -p "Vyberte variantu pro zápis do .env (1 nebo 2): " choice
        case "$choice" in
            1) chosen_safe="$best_safe"; break ;;
            2) chosen_safe="$final_safe"; break ;;
            *) echo " Neplatná volba. Napište prosím 1 nebo 2." ;;
        esac
    done

    log "[$role] Vybrána hodnota: $chosen_safe"

    local targets=("$ENV_FILE")
    [ -f .env.example ] && [ ".env.example" != "$ENV_FILE" ] && targets+=(".env.example")
    local f
    for f in "${targets[@]}"; do
        _update_env_var "$f" "$conc_key" "$chosen_safe"
        [ -n "$batch_key" ] && _update_env_var "$f" "$batch_key" "$chosen_safe"
        log "[$role] zapsáno do $f: $conc_key=$chosen_safe${batch_key:+  $batch_key=$chosen_safe}"
    done

    return 0
}

# --- Main -------------------------------------------------------------------
log "Auto-tune souběžnosti — ROLE=$ROLE  GPU_COUNT=$GPU_COUNT (extractor TP=$TP_EXTRACTOR)  ENV_FILE=$ENV_FILE  APPLY=$APPLY"

roles="$ROLE"
case "$ROLE" in
    both) roles="extractor critic" ;;          # zpětná kompatibilita
    all)  roles="extractor critic revise" ;;
esac

rc_total=0
for r in $roles; do
    tune_one "$r" || rc_total=$?
done

if [ "$rc_total" -eq 0 ]; then
    log "🎉 Hotovo (ROLE=$ROLE). Ostrý běh (rebuild labeleru): docker compose --profile extract-sglang up -d --build"
else
    warn "Tuning skončil s rc=$rc_total (některá role neměla stabilní úroveň — viz výše)."
fi
exit "$rc_total"
