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

# ---------------------------------------------------------------------------
# Použití
# ---------------------------------------------------------------------------
#   ./scripts/run_production.sh all              celá pipeline (0a → finalize)
#   ./scripts/run_production.sh taxonomy         jedna fáze
#   ./scripts/run_production.sh from-extract     od extract dál (extract → … → finalize)
#
# Fáze: sample | taxonomy | extract | critic | revise | finalize | all
#        from-sample | from-taxonomy | from-extract | from-critic | from-revise
#
# Když běh začíná na/za `extract`, fáze 0 se NESPOUŠTÍ — extract si načte
# existující checkpoints/entity_taxonomy.json. Vlastní režim si vynutíš přes
# EXTRACT_EXTRA_ARGS="--taxonomy-mode auto".
#
# Skript se před startem ptá na typ/počet GPU, počet ukázek pro taxonomii a
# (volitelně) na souběžnost a dávky. Dotaz na souběžnost vypneš
# THROUGHPUT_PROMPT=0; neinteraktivní běh se neptá vūbec.

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

# ---------------------------------------------------------------------------
# HW presety per typ GPU (modely, kvantizace, backendy, VRAM, kontext)
# ---------------------------------------------------------------------------
# Blackwell (b200 / rtxpro6000) → NVFP4 modely, SGLang primárně.
# Ampere (a100 40 GB)           → SGLang (stejně jako Blackwell), ale INT4
#   kvantizace místo NVFP4/FP8:
#   - NVFP4 vyžaduje Blackwell.
#   - FP8 na Ampere nejde u MoE v žádném enginu: SM80 nemá fp8e4nv, v SGLangu
#     je Marlin FP8 teprve feature request (sglang#12887, #7715) a Triton
#     fused_moe ve vLLM chce stejný dtype (vllm#18008, vllm#54318).
#   - INT4 weight-only (W4A16) přes Marlin MoE ale SGLang na Ampere umí
#     (ověřeno na SM80/SM86), takže extraktor jede
#     Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 (~65 GB vah, TP=4 → ~17 GB/GPU).
#   - Critic je dense: oficiální QAT W4A16 google/gemma-4-31B-it-qat-w4a16-ct
#     (~20 GB vah) jede na Ampere taky přes Marlin → TP=2.
#   Backendy zůstávají "sglang vllm" — SGLang primárně, vLLM jako záloha.
#   Menší VRAM a W4A16 (dequant do BF16) ⇒ nižší kontext i server-side dávky
#   než na Blackwellu. Pokud SGLang Marlin MoE spadne na dtype aktivací,
#   nastav EXTRACTOR_DTYPE=float16 (scales checkpointu v fp16).
#
# Modely/parametry se přepíšou JEN když v .env drží některou ze spravovaných
# hodnot (nebo nic) — vlastní model uživatele zůstane.
_MANAGED_EXTRACTOR_MODELS="RedHatAI/Qwen3.5-122B-A10B-NVFP4 Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 QuantTrio/Qwen3.5-122B-A10B-AWQ Intel/Qwen3.5-122B-A10B-int4-AutoRound RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic Qwen/Qwen3.5-122B-A10B-FP8 Qwen/Qwen3.5-122B-A10B"
_MANAGED_CRITIC_MODELS="nvidia/Gemma-4-31B-IT-NVFP4 google/gemma-4-31B-it-qat-w4a16-ct RedHatAI/gemma-4-31B-it-FP8-block google/gemma-4-31B-it"
# Hodnoty *_BATCH_SIZE, které nastavují samotné presety — jen ty se přepisují
# při změně HW. Vlastní číslo (např. z auto_tune nebo z dotazu při startu)
# zůstává.
_MANAGED_EXTRACTOR_BATCH="8 32"
_MANAGED_CRITIC_BATCH="16 64"

# Zapíše hodnotu jen pokud je aktuální prázdná nebo patří do spravovaného seznamu.
_update_env_var_if_managed() {
    local file="$1" key="$2" value="$3" managed="$4"
    local cur
    cur=$(_read_env_var "$file" "$key")
    if [ -z "$cur" ]; then
        _update_env_var "$file" "$key" "$value"
        return 0
    fi
    case " $managed " in
        *" $cur "*)
            [ "$cur" = "$value" ] || _update_env_var "$file" "$key" "$value"
            return 0
            ;;
    esac
    warn "$key='$cur' je vlastní hodnota — nechávám ji (HW preset by nastavil '$value')."
    return 0
}

# Zvedne hodnotu na minimum (nižší přepíše, vyšší uživatelovu nechá).
_raise_env_var() {
    local file="$1" key="$2" min="$3" cur
    cur=$(_read_env_var "$file" "$key")
    cur="${cur%%.*}"
    if ! [ "${cur:-0}" -ge "$min" ] 2>/dev/null; then
        _update_env_var "$file" "$key" "$min"
        log "Auto-config: $key=$min ($4)"
    fi
}

_apply_gpu_preset() {
    local file="$1" gpu_type="$2" count="$3"
    case "$gpu_type" in
        a100)
            # INT4 W4A16 — to SGLang i vLLM na Ampere umí přes Marlin
            # (FP8 ne, viz hlavička). Extraktor MoE + dense critic (QAT).
            _update_env_var_if_managed "$file" EXTRACTOR_MODEL \
                "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" "$_MANAGED_EXTRACTOR_MODELS"
            _update_env_var_if_managed "$file" CRITIC_MODEL \
                "google/gemma-4-31B-it" "$_MANAGED_CRITIC_MODELS"
            # Qwen3.5 MoE INT4 v SGLangu: Triton w4a16 cesta (moe_wna16), NE
            # Marlin. Marlin MoE si při prepacku vyrobí fp16 scales (checkpoint
            # sám žádné fp16 tensory nemá — jen BF16/F32/I32), takže:
            #   - v bf16 spadne moe_wna16_marlin_gemm na
            #     "assumes hidden_states.dtype == w1_scale.dtype",
            #   - a v fp16 (--dtype float16) spadne hybridní lineární attention
            #     (GDN), jejíž conv cache zůstává bf16 (sglang#30178).
            # moe_wna16 používá bf16 aktivace i scales → ani jeden assert.
            _update_env_var "$file" EXTRACTOR_QUANTIZATION "moe_wna16"
            _update_env_var "$file" CRITIC_QUANTIZATION ""
            # Dtype necháváme na auto-detectu (bf16) — fp16 už není potřeba
            _update_env_var "$file" EXTRACTOR_DTYPE ""
            _update_env_var "$file" CRITIC_DTYPE ""
            # GDN conv cache MUSÍ mít stejný dtype jako model: když tu z
            # předchozího pokusu zůstane float16 a model jede bf16, Triton
            # kernel causal_conv1d spadne na "Mismatched type for
            # new_conv_state between then block (fp16) and else block (bf16)".
            _update_env_var "$file" SGLANG_MAMBA_CONV_DTYPE ""
            # Všechny fáze na SGLangu, vLLM jen jako záloha.
            _update_env_var "$file" EXTRACTOR_BACKENDS "sglang vllm"
            _update_env_var "$file" CRITIC_BACKENDS "sglang vllm"
            _update_env_var "$file" REVISE_BACKENDS "sglang vllm"
            # Critic bf16 Gemma 4 31B ≈ 62 GB vah → TP=4 (~15,5 GB/GPU).
            # Kvantizované varianty (QAT w4a16-ct / FP8-block) by se vlezly na
            # 2 karty, ale SGLang je na Ampere nenaloží — QAT checkpoint má
            # kvantizovaný jen jazykový model a SGLang pro vision tower vyrobí
            # parametry, které v něm nejsou (padá v process_weights_after_loading).
            # Fáze jedou po sobě, takže 4 karty pro critic nic neblokují.
            _update_env_var "$file" CRITIC_TP_SIZE "4"
            # Triton moe_wna16 potřebuje workspace + prefill CUDA grafy se
            # na 40GB kartě nevejdou při 0.92 (dokumentovaný příklad pro tenhle
            # model jede 0.8–0.85).
            _update_env_var "$file" EXTRACTOR_MEM_FRACTION "0.85"
            _update_env_var "$file" CRITIC_MEM_FRACTION "0.88"
            # Extraktor 64k: prompt fáze 0 (všechny domény × ukázky) měl přes
            # 31k tokenů a do 32k kontextu se s odpovědí nevejde. Max kontext
            # nealokuje paměť (KV pool řídí *_MEM_FRACTION), jen strop délky
            # jednoho requestu. Critic dostává per-item prompty → 32k stačí.
            _update_env_var "$file" EXTRACTOR_CONTEXT_LEN "65536"
            _update_env_var "$file" CRITIC_CONTEXT_LEN "32768"
            _update_env_var "$file" EXTRACTOR_MAX_TOKENS "16384"
            _update_env_var "$file" CRITIC_MAX_TOKENS "24576"
            _update_env_var "$file" EXTRACTOR_NER_MAX_TOKENS "8192"
            # Server-side dávky (max-running-requests). Nealokují paměť — KV pool
            # je dán *_MEM_FRACTION a SGLang v něm sekvence plánuje (při plném
            # poolu requesty frontí / retrahuje). Malá dávka = dequant vah
            # (moe_wna16) se rozpustí do mála tokenů → karty běží na ~40 % TDP
            # a propustnost je zbytečně nízká. Optimum změří
            # ./scripts/auto_tune_concurrency.sh.
            # *_CONCURRENCY (in-flight HTTP requesty z labeleru) záměrně
            # NEPŘEPISUJEME — to si drží .env / auto_tune.
            _update_env_var_if_managed "$file" EXTRACTOR_BATCH_SIZE "32" "$_MANAGED_EXTRACTOR_BATCH"
            _update_env_var_if_managed "$file" CRITIC_BATCH_SIZE "64" "$_MANAGED_CRITIC_BATCH"
            # Marlin dequant do BF16 + malé dávky = pomalejší per-request
            _raise_env_var "$file" LLM_TIMEOUT_S 7200 "A100: Marlin dequant, menší dávky"
            log "Auto-config $file: A100 preset — extraktor GPTQ-Int4 (moe_wna16, TP=$count, kontext 65536), critic bf16 Gemma (TP=4, kontext 32768), backend sglang+vllm"
            ;;
        b200|rtxpro6000)
            # Blackwell → NVFP4 (plný NVFP4 stack, SGLang primárně)
            _update_env_var_if_managed "$file" EXTRACTOR_MODEL \
                "RedHatAI/Qwen3.5-122B-A10B-NVFP4" "$_MANAGED_EXTRACTOR_MODELS"
            _update_env_var_if_managed "$file" CRITIC_MODEL \
                "nvidia/Gemma-4-31B-IT-NVFP4" "$_MANAGED_CRITIC_MODELS"
            _update_env_var "$file" EXTRACTOR_QUANTIZATION ""
            _update_env_var "$file" CRITIC_QUANTIZATION ""
            # NVFP4 jede v bf16 → dtype necháváme na auto-detectu
            _update_env_var "$file" EXTRACTOR_DTYPE ""
            _update_env_var "$file" CRITIC_DTYPE ""
            # conv cache v sync s modelem (viz A100 výše)
            _update_env_var "$file" SGLANG_MAMBA_CONV_DTYPE ""
            _update_env_var "$file" EXTRACTOR_BACKENDS "sglang vllm"
            _update_env_var "$file" CRITIC_BACKENDS "sglang vllm"
            _update_env_var "$file" REVISE_BACKENDS "sglang vllm"
            _update_env_var "$file" CRITIC_TP_SIZE "1"
            _update_env_var "$file" EXTRACTOR_MEM_FRACTION "0.85"
            _update_env_var "$file" CRITIC_MEM_FRACTION "0.85"
            _update_env_var "$file" EXTRACTOR_CONTEXT_LEN "65536"
            _update_env_var "$file" CRITIC_CONTEXT_LEN "65536"
            _update_env_var "$file" EXTRACTOR_MAX_TOKENS "24576"
            _update_env_var "$file" CRITIC_MAX_TOKENS "49152"
            _update_env_var "$file" EXTRACTOR_NER_MAX_TOKENS "8192"
            # *_CONCURRENCY necháváme na .env / auto_tune_concurrency.sh
            _update_env_var_if_managed "$file" EXTRACTOR_BATCH_SIZE "32" "$_MANAGED_EXTRACTOR_BATCH"
            _update_env_var_if_managed "$file" CRITIC_BATCH_SIZE "64" "$_MANAGED_CRITIC_BATCH"
            if [ "$gpu_type" = "rtxpro6000" ]; then
                # GDDR7 + TP přes PCIe → delší worst-case request
                _raise_env_var "$file" LLM_TIMEOUT_S 3600 "RTX PRO 6000"
            fi
            ;;
    esac
}

configure_env_for_gpu_count() {
    local n="$1" gpu_type="${2:-b200}"
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
    _update_env_var "$env_file" GPU_TYPE "$gpu_type"
    _update_env_var "$env_file" GPU_COUNT "$n"
    _update_env_var "$env_file" EXTRACTOR_TP_SIZE "$n"
    _update_env_var "$env_file" CRITIC_TP_SIZE "1"
    # HW preset: modely + kvantizace + backendy + VRAM + kontext + dávky
    # (NVFP4 na Blackwellu, FP8 + vLLM na Ampere — viz _apply_gpu_preset).
    _apply_gpu_preset "$env_file" "$gpu_type" "$n"
    log "Auto-config $env_file: GPU_TYPE=$gpu_type GPU_COUNT=$n EXTRACTOR_TP_SIZE=$n CRITIC_TP_SIZE=1"
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
# Interactive GPU prompt — typ GPU + počet GPU (s validací)
# ---------------------------------------------------------------------------
# Podporované typy (GPU_TYPE):
#   b200        NVIDIA B200, 180 GB HBM3e — Qwen3.5-122B-NVFP4 (~80 GB vah)
#               se vejde na 1 GPU → 1, 2, 4 nebo 8 GPU
#   rtxpro6000  NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 — Qwen se na jednu
#               kartu nevejde (váhy + KV cache) → extractor TP ≥ 2 → 2, 4 nebo 8
# Počty jen power-of-2 (TP=3/5/6/7 SGLang/vLLM pro MoE nepodporují).
#
# GPU_TYPE / GPU_COUNT v env prompt přeskočí (CI). Když je v env jen
# GPU_COUNT, typ se vezme z nvidia-smi, jinak b200 (dřívější chování).
# Pro fázi 'finalize' (CPU only, bez LLM) prompt přeskočíme.

gpu_type_label() {
    case "$1" in
        b200)       echo "B200" ;;
        rtxpro6000) echo "RTX PRO 6000" ;;
        a100)       echo "A100 40GB" ;;
        *)          echo "$1" ;;
    esac
}

# Povolené počty GPU pro daný typ (mezerou oddělené).
gpu_type_counts() {
    case "$1" in
        rtxpro6000) echo "2 4 8" ;;
        a100)       echo "4 8" ;;
        *)          echo "1 2 4 8" ;;
    esac
}

# Typ GPU podle názvu první karty v nvidia-smi; prázdné = nerozpoznáno.
detect_gpu_type() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
    case "$name" in
        *B200*)           echo b200 ;;
        *"RTX PRO 6000"*) echo rtxpro6000 ;;
        *A100*)           echo a100 ;;
    esac
}

# ---------------------------------------------------------------------------
# Pořadí fází + spuštění od konkrétní fáze (from-<fáze>)
# ---------------------------------------------------------------------------
# `./scripts/run_production.sh from-extract` pustí extract → critic → revise →
# finalize. Fáze 0 (taxonomy) se v takovém běhu NEspouští — extract si načte
# existující checkpoints/entity_taxonomy.json (viz EXTRACT_EXTRA_ARGS níže).
PIPELINE_ORDER="sample taxonomy extract critic revise finalize"

_phase_index() {
    case "$1" in
        sample)   echo 0 ;;
        taxonomy) echo 1 ;;
        extract)  echo 2 ;;
        critic)   echo 3 ;;
        revise)   echo 4 ;;
        finalize) echo 5 ;;
        *)        echo -1 ;;
    esac
}

# Fáze od zadané dál (včetně).
_phases_from() {
    local want="$1" out="" seen=0 p
    for p in $PIPELINE_ORDER; do
        [ "$p" = "$want" ] && seen=1
        [ "$seen" = "1" ] && out="$out $p"
    done
    printf '%s' "${out# }"
}

FROM_PHASE=""
case "$PHASE" in
    from-*)
        FROM_PHASE="${PHASE#from-}"
        if [ "$(_phase_index "$FROM_PHASE")" -lt 0 ]; then
            err "from-$FROM_PHASE: neznámá fáze. Validní: ${PIPELINE_ORDER// /, }."
            exit 1
        fi
        ;;
esac

# První fáze běhu — podle ní se rozhoduje, co se ptát a co přeskočit.
START_PHASE="$PHASE"
[ "$PHASE" = "all" ] && START_PHASE="sample"
[ -n "$FROM_PHASE" ] && START_PHASE="$FROM_PHASE"

# Běží v tomhle běhu daná fáze?
_run_has_phase() {
    local want="$1"
    [ "$PHASE" = "$want" ] && return 0
    local si wi
    si="$(_phase_index "$START_PHASE")"
    wi="$(_phase_index "$want")"
    if [ "$PHASE" = "all" ] || [ -n "$FROM_PHASE" ]; then
        [ "$wi" -ge "$si" ] && return 0
    fi
    return 1
}

# Volitelny dotaz na soubeznost a davky. Enter = hodnoty z .env (resp. HW
# preset). Vypnout: THROUGHPUT_PROMPT=0. Neinteraktivni beh se neptá.
#
#   *_CONCURRENCY = kolik requestu posila labeler soucasne
#   *_BATCH_SIZE  = kolik jich server pocita soucasne (max-running-requests)
# Soubeznost > batch znamena, ze cast requestu stoji ve fronte serveru, coz je
# v poradku (server ma vzdy co delat). Batch pamet nealokuje — KV pool urcuje
# *_MEM_FRACTION. Optimum zmeri ./scripts/auto_tune_concurrency.sh.
_ask_positive_int() {
    local label="$1" key="$2" cur="$3" n
    while true; do
        read -r -p "  $label [$cur]: " n
        n="${n:-$cur}"
        case "$n" in
            ''|*[!0-9]*) warn "Zadej kladne cele cislo." ; continue ;;
        esac
        [ "$n" -gt 0 ] || { warn "Zadej kladne cele cislo."; continue; }
        _update_env_var "${ENV_FILE:-.env}" "$key" "$n"
        export "$key=$n"
        return 0
    done
}

prompt_throughput() {
    [ "${THROUGHPUT_PROMPT:-1}" = "0" ] && return 0
    [ ! -t 0 ] && return 0
    local env_file="${ENV_FILE:-.env}" ec eb cc cb yn
    ec="${EXTRACTOR_CONCURRENCY:-$(_read_env_var "$env_file" EXTRACTOR_CONCURRENCY)}"
    eb="${EXTRACTOR_BATCH_SIZE:-$(_read_env_var "$env_file" EXTRACTOR_BATCH_SIZE)}"
    cc="${CRITIC_CONCURRENCY:-$(_read_env_var "$env_file" CRITIC_CONCURRENCY)}"
    cb="${CRITIC_BATCH_SIZE:-$(_read_env_var "$env_file" CRITIC_BATCH_SIZE)}"
    ec="${ec:-32}"; eb="${eb:-32}"; cc="${cc:-64}"; cb="${cb:-64}"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Soubeznost a davky (aktualne z .env):"
    echo "    extractor: concurrency $ec / server batch $eb"
    echo "    critic:    concurrency $cc / server batch $cb"
    echo "  concurrency = kolik requestu posila labeler soucasne"
    echo "  batch       = kolik jich server pocita soucasne"
    echo "  (optimum zmeri ./scripts/auto_tune_concurrency.sh)"
    echo "════════════════════════════════════════════════════════"
    read -r -p "Upravit tyhle hodnoty? [a/N] (Enter = nechat): " yn
    case "$yn" in
        a|A|y|Y|ano|Ano|yes|Yes) ;;
        *) log "Soubeznost zustava: extractor $ec/$eb, critic $cc/$cb"; return 0 ;;
    esac
    _ask_positive_int "EXTRACTOR_CONCURRENCY" EXTRACTOR_CONCURRENCY "$ec"
    _ask_positive_int "EXTRACTOR_BATCH_SIZE " EXTRACTOR_BATCH_SIZE "$eb"
    _ask_positive_int "CRITIC_CONCURRENCY   " CRITIC_CONCURRENCY "$cc"
    _ask_positive_int "CRITIC_BATCH_SIZE    " CRITIC_BATCH_SIZE "$cb"
    log "OK: extractor $EXTRACTOR_CONCURRENCY/$EXTRACTOR_BATCH_SIZE, critic $CRITIC_CONCURRENCY/$CRITIC_BATCH_SIZE"
}

prompt_gpu_type() {
    if [ "$PHASE" = "finalize" ]; then
        return 0
    fi

    local detected_type
    detected_type=$(detect_gpu_type)

    if [ -n "${GPU_TYPE:-}" ]; then
        case "$GPU_TYPE" in
            b200|rtxpro6000|a100) ;;
            *)
                err "GPU_TYPE='$GPU_TYPE' není podporovaný. Validní: b200, rtxpro6000, a100."
                exit 1
                ;;
        esac
        log "GPU_TYPE z env: $(gpu_type_label "$GPU_TYPE")"
        return 0
    fi
    if [ -n "${GPU_COUNT:-}" ]; then
        GPU_TYPE="${detected_type:-b200}"
        if [ -n "$detected_type" ]; then
            log "GPU_TYPE: $(gpu_type_label "$GPU_TYPE") (detekováno přes nvidia-smi)"
        else
            log "GPU_TYPE: B200 (default; pro RTX PRO 6000 nastav GPU_TYPE=rtxpro6000)"
        fi
        return 0
    fi

    local default=""
    case "$detected_type" in
        b200)       default=1 ;;
        rtxpro6000) default=2 ;;
        a100)       default=3 ;;
    esac
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Jaký typ GPU máš k dispozici pro pipeline?"
    echo "    1) NVIDIA B200          (180 GB) — NVFP4, Qwen 122B na 1 GPU"
    echo "    2) NVIDIA RTX PRO 6000  (96 GB)  — NVFP4, Qwen 122B min. 2 GPU"
    echo "    3) NVIDIA A100          (40 GB)  — INT4 (W4A16), Qwen 122B min. 4 GPU"
    if [ -n "$detected_type" ]; then
        echo "  → Detekováno: $(gpu_type_label "$detected_type")"
    fi
    echo "════════════════════════════════════════════════════════"
    local t
    while true; do
        read -r -p "Vyber typ GPU [1/2/3]${default:+ (Enter = $default)}: " t
        t="${t:-$default}"
        case "$t" in
            1) GPU_TYPE=b200; break ;;
            2) GPU_TYPE=rtxpro6000; break ;;
            3) GPU_TYPE=a100; break ;;
            *) warn "Neplatná volba '$t'. Zadej 1, 2 nebo 3." ;;
        esac
    done
    log "OK: $(gpu_type_label "$GPU_TYPE")"
}

prompt_gpu_count() {
    if [ "$PHASE" = "finalize" ]; then
        log "Phase=finalize → bez LLM, GPU prompt přeskočen."
        return 0
    fi

    local detected label counts min_count
    detected=$(detect_gpu_count)
    label=$(gpu_type_label "$GPU_TYPE")
    counts=$(gpu_type_counts "$GPU_TYPE")
    min_count="${counts%% *}"
    if [ "$detected" -gt 0 ]; then
        log "Detekováno $detected GPU na hostu (přes nvidia-smi)"
        if [ "$detected" -lt "$min_count" ]; then
            err "Host má jen $detected GPU, ale $label potřebuje min. $min_count"
            err "(Qwen3.5-122B se nevejde: NVFP4 ~80 GB / INT4 ~65 GB vah + KV cache)."
            exit 1
        fi
    else
        warn "nvidia-smi nedostupné — nemůžu detekovat GPU count automaticky"
    fi

    if [ -n "${GPU_COUNT:-}" ]; then
        case " $counts " in
            *" $GPU_COUNT "*) ;;
            *)
                err "GPU_COUNT='$GPU_COUNT' není pro $label podporovaný. Validní: ${counts// /, }."
                case "$GPU_TYPE" in
                    rtxpro6000)
                        err "RTX PRO 6000 má 96 GB — Qwen3.5-122B-NVFP4 se na jednu kartu nevejde."
                        ;;
                    a100)
                        err "A100 má 40 GB — Qwen3.5-122B v INT4 (~65 GB) potřebuje min. 4 karty."
                        ;;
                esac
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
        log "GPU_COUNT z env: ${GPU_COUNT}× $label"
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Kolik NVIDIA $label GPU máš k dispozici pro pipeline?"
    echo "  Podporované hodnoty: ${counts// /, }"
    case "$GPU_TYPE" in
        rtxpro6000) echo "  (min. 2 — Qwen 122B v NVFP4 se na jednu 96GB kartu nevejde)" ;;
        a100)       echo "  (min. 4 — Qwen 122B v INT4 ~65 GB + KV cache; 4×40 GB = 160 GB)" ;;
    esac
    echo "  (jen power-of-2 — TP=3/5/6/7 SGLang/vLLM nepodporují)"
    if [ "$detected" -gt 0 ]; then
        echo "  → Detekováno $detected GPU na hostu"
    fi
    echo "════════════════════════════════════════════════════════"
    local n
    while true; do
        read -r -p "Zadej počet $label GPU [${counts// //}]: " n
        case " $counts " in
            *" $n "*)
                if ! [[ "$n" =~ ^[0-9]+$ ]]; then
                    warn "Neplatná hodnota '$n'. Zadej ${counts// /, }."
                    continue
                fi
                if [ "$detected" -gt 0 ] && [ "$n" -gt "$detected" ]; then
                    warn "Host má jen $detected GPU, zadáno $n. Vyber max $detected."
                    continue
                fi
                GPU_COUNT="$n"
                break
                ;;
            *)
                warn "Neplatná hodnota '$n'. Zadej ${counts// /, }."
                ;;
        esac
    done
    log "OK: ${GPU_COUNT}× $label"
}

# Spustit preflighty pro fáze, které spouštějí LLM
if [ "$PHASE" != "finalize" ]; then
    preflight_quantization
    preflight_hf_token
fi

# ---------------------------------------------------------------------------
# Interactive prompt — počet ukázek pro NÁVRH TAXONOMIE (fáze 0a/0)
# ---------------------------------------------------------------------------
# TAXONOMY_SAMPLES_PER_DATASET: auto (počet si každá doména spočítá sama
# z rozmanitosti obsahu přes embeddingy bge-m3) nebo číslo (přesně tolik
# ukázek per doména). Hodnota z env přeskočí dotaz; default = hodnota
# z .env, jinak auto. Bez terminálu se použije default.
_valid_taxonomy_samples() {
    [[ "$1" == "auto" || "$1" =~ ^[1-9][0-9]*$ ]]
}

prompt_taxonomy_samples() {
    local cur="${TAXONOMY_SAMPLES_PER_DATASET:-}"
    cur="$(printf '%s' "$cur" | tr '[:upper:]' '[:lower:]')"
    if [ -n "$cur" ]; then
        if ! _valid_taxonomy_samples "$cur"; then
            err "TAXONOMY_SAMPLES_PER_DATASET='$cur' musí být 'auto' nebo kladné celé číslo."
            exit 1
        fi
        export TAXONOMY_SAMPLES_PER_DATASET="$cur"
        log "TAXONOMY_SAMPLES_PER_DATASET z env: $cur"
        return 0
    fi
    local default
    default="$(_read_env_var "${ENV_FILE:-.env}" TAXONOMY_SAMPLES_PER_DATASET | tr '[:upper:]' '[:lower:]')"
    _valid_taxonomy_samples "$default" || default="auto"
    if [ ! -t 0 ]; then
        export TAXONOMY_SAMPLES_PER_DATASET="$default"
        log "Neinteraktivní běh → TAXONOMY_SAMPLES_PER_DATASET=$default (default)"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Kolik ukázek per doména má dostat LLM pro NÁVRH TAXONOMIE?"
    echo "    auto  = každá doména si počet spočítá sama podle rozmanitosti"
    echo "            obsahu (embeddingy BAAI/bge-m3)"
    echo "    číslo = přesně tolik ukázek per doména (co nejrůznější texty)"
    echo "════════════════════════════════════════════════════════"
    local n
    while true; do
        read -r -p "Ukázek pro taxonomii per doména [auto/číslo] (Enter = $default): " n
        n="$(printf '%s' "${n:-$default}" | tr '[:upper:]' '[:lower:]')"
        if _valid_taxonomy_samples "$n"; then
            export TAXONOMY_SAMPLES_PER_DATASET="$n"
            break
        fi
        warn "Neplatná hodnota '$n'. Zadej 'auto' nebo kladné celé číslo."
    done
    log "OK: ukázky pro taxonomii = $TAXONOMY_SAMPLES_PER_DATASET"
}

prompt_gpu_type
prompt_gpu_count
# Počet ukázek pro taxonomii se týká fází sample (0a) a taxonomy (0)
if _run_has_phase sample || _run_has_phase taxonomy; then
    prompt_taxonomy_samples
fi
# Auto-config .env podle GPU_TYPE + GPU_COUNT (pro docker compose i native;
# finalize fáze GPU nepoužívá → .env nechává být).
if [ "$PHASE" != "finalize" ] && [ -n "${GPU_COUNT:-}" ]; then
    configure_env_for_gpu_count "$GPU_COUNT" "$GPU_TYPE"
fi

# Souběžnost / dávky — jen když běží některá LLM fáze.
# MUSÍ být AŽ ZA HW presetem: preset šahá na *_BATCH_SIZE, takže kdyby se
# ptal dřív, uživatelovu hodnotu by pak preset přepsal.
if _run_has_phase extract || _run_has_phase critic || _run_has_phase revise; then
    prompt_throughput
fi

# Propagace GPU_COUNT do environment proměnných pro docker compose.
# EXTRACTOR_TP_SIZE = GPU_COUNT; CRITIC_TP_SIZE = 1 (Gemma 4 31B NVFP4 ~16 GB
# se vejde na 1 GPU vždy — B200 i RTX PRO 6000; TP > 1 by jen přidal overhead).
if [ "${PHASE:-all}" != "finalize" ]; then
    export EXTRACTOR_TP_SIZE="${EXTRACTOR_TP_SIZE:-$GPU_COUNT}"
    # CRITIC_TP_SIZE řídí HW preset v .env (A100 40 GB → 2; Blackwell → 1).
    # Ze shellu bereme jen explicitní override, jinak hodnotu z .env.
    if [ -z "${CRITIC_TP_SIZE:-}" ]; then
        CRITIC_TP_SIZE="$(_read_env_var "${ENV_FILE:-.env}" CRITIC_TP_SIZE)"
    fi
    export CRITIC_TP_SIZE="${CRITIC_TP_SIZE:-1}"
    _min_tp="$(gpu_type_counts "$GPU_TYPE")"
    _min_tp="${_min_tp%% *}"
    if [ "$EXTRACTOR_TP_SIZE" -lt "$_min_tp" ]; then
        err "EXTRACTOR_TP_SIZE=$EXTRACTOR_TP_SIZE (z env) — na $(gpu_type_label "$GPU_TYPE") musí být min. $_min_tp."
        exit 1
    fi
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
    # CUDA 13+ je potřeba jen pro NVFP4 na Blackwellu. A100 (Ampere) jede
    # INT4/FP8 přes Marlin, kde stačí CUDA 12.1+.
    local min_major=13
    if [ "${GPU_TYPE:-}" = "a100" ]; then
        min_major=12
    fi
    if [ "$major" -lt "$min_major" ]; then
        if [ "$min_major" -eq 13 ]; then
            err "Detekováno CUDA $cuda_ver — pipeline VYŽADUJE CUDA 13+ pro NVFP4"
            err "na Blackwell SM120. Upgrade NVIDIA driveru na ≥ 580 a CUDA toolkit ≥ 13.0."
        else
            err "Detekováno CUDA $cuda_ver — na A100 je potřeba CUDA 12.1+ (Marlin INT4/FP8)."
        fi
        err "Pokud chceš pokračovat na vlastní riziko (např. testuješ FP8/BF16 fallback),"
        err "spusť: SKIP_CUDA_CHECK=1 ./scripts/run_production.sh $PHASE"
        return 1
    fi
    log "Preflight OK — CUDA $cuda_ver detekována (min. ${min_major}.x pro $(gpu_type_label "${GPU_TYPE:-b200}"))."
}

if [ "$PHASE" != "finalize" ]; then
    preflight_cuda
fi

# Exit code labeler containeru — VŽDY číslo.
#
# `docker compose wait` vrací exit code kontejneru jako svůj vlastní status
# a na stdout píše větu `container "<id>" exited with status code 0`.
# Brát stdout jako rc tedy nešlo: `[ "$rc" -eq 0 ]` skončilo na
# "integer expected" a úspěšný běh se vyhodnotil jako selhání. Proto
# čekáme přes `wait` (status), ale číslo bereme z `docker inspect`, který
# vrací exit code strojově a nezávisle na verzi compose.
_labeler_exit_code() {
    local profile="$1" svc="$2" rc=0 cid code
    docker compose --profile "$profile" wait "$svc" >/dev/null 2>&1 || rc=$?
    # -a: už skončený container `ps -q` nevypisuje
    cid=$(docker compose --profile "$profile" ps -aq "$svc" 2>/dev/null | head -1)
    if [ -n "$cid" ]; then
        code=$(docker inspect --format='{{.State.ExitCode}}' "$cid" 2>/dev/null || echo "")
        case "$code" in
            ''|*[!0-9]*) ;;          # inspect nevyšel → nech rc ze `wait`
            *) rc="$code" ;;
        esac
    fi
    case "$rc" in
        ''|*[!0-9]*) rc=1 ;;         # pojistka: nikdy nevrať nečíslo
    esac
    printf '%s' "$rc"
}

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

    # Čekej, až labeler container skončí, a vezmi jeho exit code (viz
    # _labeler_exit_code — stdout `compose wait` je věta, ne číslo).
    local rc
    rc=$(_labeler_exit_code "$profile" "$labeler_service")

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
    # Embedding server fáze 0a (vLLM --runner pooling)
    export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8002/v1}"
    log "Native env parita: HF_XET=1, PYTHONWARNINGS=suppress,"
    log "  EXTRACTOR_BASE_URL=$EXTRACTOR_BASE_URL"
    log "  CRITIC_BASE_URL=$CRITIC_BASE_URL"
}

# Sestaví SGLang args 1:1 jako docker-compose.yml command block.
# Args: $1 = phase (extract|critic|revise)
# Backend pro fázi podle *_BACKENDS (A100 preset = jen vllm, protože SGLang
# na Ampere neumí FP8 ani Marlin MoE). Native runtime pak startuje vLLM.
native_engine_for_phase() {
    local phase="$1" backends=""
    case "$phase" in
        critic)        backends="${CRITIC_BACKENDS:-sglang vllm}" ;;
        revise)        backends="${REVISE_BACKENDS:-${EXTRACTOR_BACKENDS:-sglang vllm}}" ;;
        taxonomy|extract) backends="${EXTRACTOR_BACKENDS:-sglang vllm}" ;;
        *)             backends="sglang" ;;
    esac
    case " $backends " in
        *" sglang "*) echo sglang ;;
        *)            echo vllm ;;
    esac
}

# vLLM ekvivalent build_sglang_args_for_phase (jiné názvy flagů).
build_vllm_args_for_phase() {
    local phase="$1"
    case "$phase" in
        taxonomy|extract|revise)
            printf '%s\n' "--max-num-seqs=${EXTRACTOR_BATCH_SIZE:-32}" \
                "--max-model-len=${EXTRACTOR_CONTEXT_LEN:-65536}" \
                "--reasoning-parser=qwen3" "--language-model-only"
            [ -n "${EXTRACTOR_QUANTIZATION:-}" ] && printf '%s\n' "--quantization=${EXTRACTOR_QUANTIZATION}"
            [ -n "${EXTRACTOR_DTYPE:-}" ] && printf '%s\n' "--dtype=${EXTRACTOR_DTYPE}"
            ;;
        critic)
            printf '%s\n' "--max-num-seqs=${CRITIC_BATCH_SIZE:-64}" \
                "--max-model-len=${CRITIC_CONTEXT_LEN:-65536}"
            [ -n "${CRITIC_QUANTIZATION:-}" ] && printf '%s\n' "--quantization=${CRITIC_QUANTIZATION}"
            [ -n "${CRITIC_DTYPE:-}" ] && printf '%s\n' "--dtype=${CRITIC_DTYPE}"
            ;;
    esac
    return 0
}

build_sglang_args_for_phase() {
    local phase="$1"
    local args=""
    case "$phase" in
        taxonomy|extract|revise)
            local batch="${EXTRACTOR_BATCH_SIZE:-32}"
            local ctx="${EXTRACTOR_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx --reasoning-parser=qwen3"
            # --language-model-only umi SGLang jen pro par architektur
            # (Gemma4/Qwen3.5 tam nejsou) -> opt-in.
            [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] && args="$args --language-model-only"
            [ -n "${EXTRACTOR_QUANTIZATION:-}" ] && args="$args --quantization=${EXTRACTOR_QUANTIZATION}"
            # INT4 Marlin MoE na Ampere chce aktivace v dtype scales checkpointu
            [ -n "${EXTRACTOR_DTYPE:-}" ] && args="$args --dtype=${EXTRACTOR_DTYPE}"
            # Mamba state pool — u hybridniho Qwen3.5 dela strop souběžnosti on,
            # ne KV cache (viz komentar v docker-compose.yml u extractor-sglang).
            [ -n "${EXTRACTOR_MAX_MAMBA_CACHE:-}" ] && args="$args --max-mamba-cache-size=${EXTRACTOR_MAX_MAMBA_CACHE}"
            [ -n "${EXTRACTOR_MAMBA_SSM_DTYPE:-}" ] && args="$args --mamba-ssm-dtype=${EXTRACTOR_MAMBA_SSM_DTYPE}"
            [ -n "${EXTRACTOR_MAMBA_RADIX_STRATEGY:-}" ] && args="$args --mamba-radix-cache-strategy=${EXTRACTOR_MAMBA_RADIX_STRATEGY}"
            local eff="${EXTRACTOR_DISABLE_PIECEWISE_CUDA_GRAPH:-${DISABLE_PIECEWISE_CUDA_GRAPH:-}}"
            [ "$eff" = "1" ] || [ "$eff" = "true" ] && args="$args --disable-piecewise-cuda-graph"
            ;;
        critic)
            local batch="${CRITIC_BATCH_SIZE:-64}"
            local ctx="${CRITIC_CONTEXT_LEN:-65536}"
            args="--max-running-requests=$batch --context-length=$ctx"
            [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] && args="$args --language-model-only"
            [ -n "${CRITIC_QUANTIZATION:-}" ] && args="$args --quantization=${CRITIC_QUANTIZATION}"
            [ -n "${CRITIC_DTYPE:-}" ] && args="$args --dtype=${CRITIC_DTYPE}"
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
        sample)
            model="${TAXONOMY_EMBEDDING_MODEL:-BAAI/bge-m3}"
            tp_size=1
            ;;
        *) return 0 ;;
    esac
    # Fáze 0a: embedding server (bge-m3) — JEDINÁ část pipeline na vLLM.
    if [ "$phase" = "sample" ]; then
        local vargs=(--runner pooling --max-model-len 8192)
        case "$model" in
            *bge-m3*) vargs+=(--hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}' --pooler-config.task embed) ;;
        esac
        log "Auto-start vLLM (embedding) pro fázi 'sample': $model na portu $port"
        NATIVE_ENGINE=vllm SGLANG_MEM_FRACTION_STATIC="${EMBEDDING_MEM_FRACTION:-0.3}" \
            ./scripts/start_native_llm.sh "$model" "$port" 1 "${vargs[@]}"
        return $?
    fi
    # Engine podle *_BACKENDS: A100 jede vLLM (SGLang na Ampere neumí FP8
    # ani Marlin MoE), Blackwell SGLang.
    local engine
    engine=$(native_engine_for_phase "$phase")
    if [ "$engine" = "vllm" ]; then
        native_vllm_available || return 1
        local vargs=()
        while IFS= read -r _a; do
            [ -n "$_a" ] && vargs+=("$_a")
        done < <(build_vllm_args_for_phase "$phase")
        local mem="${EXTRACTOR_MEM_FRACTION:-0.85}"
        [ "$phase" = "critic" ] && mem="${CRITIC_MEM_FRACTION:-0.85}"
        log "Auto-start vLLM pro '$phase': $model na portu $port (tp=$tp_size)"
        log "  Args: ${vargs[*]}"
        NATIVE_ENGINE=vllm SGLANG_MEM_FRACTION_STATIC="$mem" \
            ./scripts/start_native_llm.sh "$model" "$port" "$tp_size" "${vargs[@]}"
        return $?
    fi
    # Args 1:1 jako Docker (batch, context, quantization, piecewise atd.)
    local sglang_args
    sglang_args=$(build_sglang_args_for_phase "$phase")
    # GDN conv cache musí mít stejný dtype jako model (sglang#30178) —
    # jinak první prefill spadne na index_put dtype mismatch.
    local mamba_conv="${SGLANG_MAMBA_CONV_DTYPE:-}"
    if [ -z "$mamba_conv" ] && [ "$phase" != "critic" ]; then
        mamba_conv="${EXTRACTOR_DTYPE:-}"
    fi
    log "Auto-start SGLang pro '$phase': $model na portu $port (tp=$tp_size)"
    log "  Args: $sglang_args"
    SGLANG_EXTRA_ARGS="$sglang_args" \
    SGLANG_MAMBA_CONV_DTYPE="$mamba_conv" \
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

# vLLM pro embedding server fáze 0a (nativně). Neinstaluje se automaticky —
# ve stejném prostředí jako SGLang se typicky perou verze torch; doporučený
# je vlastní venv a VLLM_BIN=/cesta/k/venv/bin/vllm.
native_vllm_available() {
    command -v "${VLLM_BIN:-vllm}" >/dev/null 2>&1 && return 0
    err "vLLM nenalezen (VLLM_BIN=${VLLM_BIN:-vllm}) — tato fáze potřebuje vLLM."
    err "  python -m venv ~/vllm-env && ~/vllm-env/bin/pip install vllm"
    err "  export VLLM_BIN=~/vllm-env/bin/vllm"
    return 1
}

run_native_phase() {
    local phase="$1"  # extract|critic|revise|finalize
    log "FÁZE [$phase] — NATIVE (Python přímo, bez dockeru)"

    if [ "$phase" != "finalize" ]; then
        if [ "$phase" = "sample" ]; then
            native_vllm_available || return 1
        elif [ "$(native_engine_for_phase "$phase")" = "vllm" ]; then
            # A100: backend je jen vllm → SGLang instalovat nemusíme
            native_vllm_available || return 1
        else
            auto_install_sglang_if_missing || return 1
        fi
        local port=""
        case "$phase" in
            sample)                  port="8002" ;;
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
    local cmd="$phase"
    [ "$phase" = "sample" ] && cmd="taxonomy-sample"
    python -m labeler.main "$cmd" $extra || rc=$?

    # Auto-stop po fázi (mezi fázema swapujeme Qwen ↔ Gemma)
    if [ "$phase" != "finalize" ]; then
        local port=""
        case "$phase" in
            sample)                  port="8002" ;;
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

run_sample() {
    log "FÁZE 0a — SAMPLE (výběr ukázek pro taxonomii, embeddingy bge-m3 na vLLM)"
    local rc=0
    if [ "$RUNTIME" = "native" ]; then
        run_native_phase "sample" || rc=$?
    else
        # Stejně jako ostatní fáze: detached (-d) a progress se čte z logu,
        # ať běh nedrží terminál (tmux, ssh) a dá se odpojit.
        log "Embedder na ${EMBEDDING_GPU_COUNT:-1}× GPU; progress:"
        log "  docker compose --profile embed-vllm logs -f --no-log-prefix labeler-embed-vllm"
        if docker compose --profile embed-vllm up -d; then
            rc=$(_labeler_exit_code embed-vllm labeler-embed-vllm)
        else
            err "Fáze 0a: docker compose up -d selhal"
            rc=1
        fi
        docker compose --profile embed-vllm down 2>/dev/null || true
    fi
    if [ "$rc" -eq 0 ]; then
        log "✅ Ukázky pro taxonomii vybrány (checkpoints/taxonomy_sampling.json)"
    else
        warn "Výběr ukázek přes embeddingy selhal (rc=$rc) — taxonomie použije náhodný výběr."
    fi
    return 0
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
    rc=$(_labeler_exit_code finalize labeler-finalize)
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
# ---------------------------------------------------------------------------
# Spuštění od extract dál → fáze 0 se NEspouští
# ---------------------------------------------------------------------------
# Default --taxonomy-mode v labeleru je `auto`, tj. extract by si taxonomii
# navrhl znovu. Když ale běh začíná na/za extract, uživatel fázi 0 vědomě
# přeskakuje → posláme `--taxonomy-mode off` a extract si načte existující
# checkpoints/entity_taxonomy.json (load_taxonomy_if_exists).
if [ "$(_phase_index "$START_PHASE")" -ge 2 ] && [ "$PHASE" != "finalize" ]; then
    case " ${EXTRACT_EXTRA_ARGS:-} " in
        *" --taxonomy-mode"*) ;;   # uživatel si režim určil sám
        *)
            export EXTRACT_EXTRA_ARGS="--taxonomy-mode off${EXTRACT_EXTRA_ARGS:+ $EXTRACT_EXTRA_ARGS}"
            _tax_file="${CHECKPOINT_DIR:-./checkpoints}/entity_taxonomy.json"
            if ! _run_has_phase extract; then
                :   # extract v tomhle běhu není — není co hlásit
            elif [ -f "$_tax_file" ]; then
                log "Fáze 0 se přeskakuje — extract použije existující $_tax_file"
            else
                warn "Fáze 0 se přeskakuje, ale $_tax_file neexistuje — extract pojede"
                warn "s defaultními 16 kategoriemi. Taxonomii navrhneš:"
                warn "  ./scripts/run_production.sh sample && ./scripts/run_production.sh taxonomy"
            fi
            ;;
    esac
fi

warn "Build je manuální: tento script NEbuildí image. Pokud jsi měnil kód, spusť nejdřív ./scripts/build.sh"

# Spustí zadané fáze v pořadí; před extract doplní --taxonomy-mode off,
# pokud v témže běhu proběhla fáze 0 (ať se nediscoveruje dvakrát).
_run_phases() {
    local p
    for p in "$@"; do
        case "$p" in
            extract)
                case " ${EXTRACT_EXTRA_ARGS:-} " in
                    *" --taxonomy-mode"*) ;;
                    *) export EXTRACT_EXTRA_ARGS="--taxonomy-mode off${EXTRACT_EXTRA_ARGS:+ $EXTRACT_EXTRA_ARGS}" ;;
                esac
                ;;
        esac
        "run_$p" || return $?
    done
}

if [ -n "$FROM_PHASE" ]; then
    log "Běžím od fáze '$FROM_PHASE' dál: $(_phases_from "$FROM_PHASE")"
    # shellcheck disable=SC2046
    _run_phases $(_phases_from "$FROM_PHASE")
    log "🎉 HOTOVO (od $FROM_PHASE). Výsledky v ${CHECKPOINT_DIR:-./checkpoints}/"
else
    case "$PHASE" in
        sample)   run_sample ;;
        taxonomy) run_taxonomy ;;
        extract)  run_extract ;;
        critic)   run_critic ;;
        revise)   run_revise ;;
        finalize) run_finalize ;;
        all)
            # 1:1 jako smoke test (sample → taxonomy → extract → critic → revise
            # → finalize), ale na CELÉM datasetu (žádné --samples-per-dataset).
            _run_phases sample taxonomy extract critic revise finalize
            log "🎉 VŠE HOTOVO. Výsledky v ${CHECKPOINT_DIR:-./checkpoints}/"
            log "   - final_labels.json + .jsonl  (hlavní výstup)"
            log "   - stats.json                  (souhrnné statistiky)"
            ;;
        *)
            err "Neznámá fáze: $PHASE."
            err "Jedna fáze:      ${PIPELINE_ORDER// /|}|all"
            err "Od fáze dál:     from-sample|from-taxonomy|from-extract|from-critic|from-revise"
            exit 1
            ;;
    esac
fi
