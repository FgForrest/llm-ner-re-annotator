#!/usr/bin/env bash
# Production smoke test — empirický odhad doby běhu pipeline na cílovém HW.
#
# Co dělá:
#   1. Z každého datasetu v data/ vezme N items (zeptá se; default 100)
#   2. Postupně spustí všechny 4 fáze (extract/critic/revise/finalize) na
#      smoke vzorku — vždy 1 model na GPU, jako v produkci
#   3. Měří wall-clock per fáze
#   4. Lineárně extrapoluje na full produkční rozsah (pure empirical —
#      žádné teoretické scaling factory, žádné GPU-class projekce)
#   5. Vygeneruje report (Markdown + JSON) v checkpoints/smoke/
#
# Použití (DOCKER runtime — default):
#   ./scripts/run_smoke_test.sh              # interaktivní prompt (typ + počet GPU)
#   GPU_COUNT=2 ./scripts/run_smoke_test.sh  # přeskočí prompt, 2× GPU (typ z nvidia-smi / B200)
#   GPU_TYPE=rtxpro6000 GPU_COUNT=2 ./scripts/run_smoke_test.sh   # 2× RTX PRO 6000
#   SAMPLES_PER_DATASET=100 ./scripts/run_smoke_test.sh   # přeskočí dotaz na počet položek
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
# Skript se na začátku zeptá na typ GPU (B200 / RTX PRO 6000) a kolik jich
# máš (power-of-2 pro tensor parallelism MoE; RTX PRO 6000 min. 2 — Qwen 122B
# se na jednu 96GB kartu nevejde). Podle toho nastaví EXTRACTOR_TP_SIZE
# a HW_LABEL pro report. GPU_TYPE / GPU_COUNT v env prompt přeskočí.

set -euo pipefail

cd "$(dirname "$0")/.."

# Počet položek per dataset: SAMPLES_PER_DATASET z env, jinak interaktivní dotaz
# (viz prompt_samples_per_dataset; Enter / neinteraktivní běh = default).
DEFAULT_SAMPLES_PER_DATASET=100
SMOKE_PHASES="${SMOKE_PHASES:-sample taxonomy extract critic revise finalize}"
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
# Hodnoty *_BATCH_SIZE, které nastavují samotné presety — vlastní číslo zůstává.
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
            warn "Vytvoř $env_file manuálně nebo zkopíruj z .env.example."
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

prompt_gpu_type() {
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

# ---------------------------------------------------------------------------
# Interactive prompt — počet položek per dataset
# ---------------------------------------------------------------------------
# Kolik itemů z každého datasetu projde smoke pipeline. Víc = přesnější
# extrapolace, ale delší smoke test. SAMPLES_PER_DATASET v env dotaz přeskočí;
# bez terminálu (nohup, CI) se použije default.
prompt_samples_per_dataset() {
    if [ -n "${SAMPLES_PER_DATASET:-}" ]; then
        if ! [[ "$SAMPLES_PER_DATASET" =~ ^[1-9][0-9]*$ ]]; then
            err "SAMPLES_PER_DATASET='$SAMPLES_PER_DATASET' musí být kladné celé číslo."
            exit 1
        fi
        log "SAMPLES_PER_DATASET z env: $SAMPLES_PER_DATASET"
        return 0
    fi
    if [ ! -t 0 ]; then
        SAMPLES_PER_DATASET="$DEFAULT_SAMPLES_PER_DATASET"
        log "Neinteraktivní běh → SAMPLES_PER_DATASET=$SAMPLES_PER_DATASET (default)"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Kolik položek z každého datasetu má smoke test zpracovat?"
    echo "  Víc = přesnější odhad doby běhu produkce, ale delší smoke test."
    echo "════════════════════════════════════════════════════════"
    local n
    while true; do
        read -r -p "Počet položek per dataset (Enter = $DEFAULT_SAMPLES_PER_DATASET): " n
        n="${n:-$DEFAULT_SAMPLES_PER_DATASET}"
        if [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
            SAMPLES_PER_DATASET="$n"
            break
        fi
        warn "Neplatná hodnota '$n'. Zadej kladné celé číslo."
    done
    log "OK: $SAMPLES_PER_DATASET položek per dataset"
}

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

# Spustit preflighty před prompt
preflight_quantization
preflight_hf_token
prompt_gpu_type
prompt_gpu_count
prompt_samples_per_dataset
prompt_taxonomy_samples

# Auto-config .env podle vybraného GPU_TYPE + GPU_COUNT — pro docker (čte .env) i native
configure_env_for_gpu_count "$GPU_COUNT" "$GPU_TYPE"

# Pozn.: setup_native_env_parity se volá až v sekci Main (níž), protože jeho
# definice je v "NATIVE runtime helpers" pod tímhle blokem. Bash funkce musí
# být definovaná PŘED voláním — volání tady by spadlo na `command not found`.

# Propagace GPU_COUNT do environment proměnných pro docker compose
export EXTRACTOR_TP_SIZE="$GPU_COUNT"
# Critic: na Blackwellu (NVFP4 ~16 GB) stačí 1 GPU, na A100 40 GB je FP8 Gemma
# ~31 GB → preset v .env nastaví 2.
    # CRITIC_TP_SIZE řídí HW preset v .env (A100 40 GB → 2; Blackwell → 1).
    # Ze shellu bereme jen explicitní override, jinak hodnotu z .env.
    if [ -z "${CRITIC_TP_SIZE:-}" ]; then
        CRITIC_TP_SIZE="$(_read_env_var "${ENV_FILE:-.env}" CRITIC_TP_SIZE)"
    fi
export CRITIC_TP_SIZE="${CRITIC_TP_SIZE:-1}"
# HW label propaguje do labeler containeru přes argument (viz invoke_*)
HW_LABEL="${HW_LABEL:-${GPU_COUNT}x $(gpu_type_label "$GPU_TYPE")}"

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
    # CUDA 13+ jen pro NVFP4 na Blackwellu; A100 jede INT4/FP8 přes Marlin.
    local min_major=13
    if [ "${GPU_TYPE:-}" = "a100" ]; then
        min_major=12
    fi
    if [ "$major" -lt "$min_major" ]; then
        if [ "$min_major" -eq 13 ]; then
            err "Detekováno CUDA $cuda_ver — pipeline vyžaduje CUDA 13+ pro NVFP4."
        else
            err "Detekováno CUDA $cuda_ver — na A100 je potřeba CUDA 12.1+ (Marlin INT4/FP8)."
        fi
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
    # Embedding server fáze 0a (vLLM --runner pooling)
    export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8002/v1}"

    log "Native env parita: HF_XET_HIGH_PERFORMANCE=1, PYTHONWARNINGS suppress,"
    log "                   EXTRACTOR_BASE_URL=$EXTRACTOR_BASE_URL"
    log "                   CRITIC_BASE_URL=$CRITIC_BASE_URL"
}

# Sestavi seznam SGLang argumentů 1:1 jako docker-compose.yml command bloky.
# Args: $1=phase (taxonomy|extract|critic|revise) — taxonomy sdílí extractor model
# Outputs (přes echo): args řetězec pro `--key value` páry
# Backend pro fázi podle *_BACKENDS (SGLang primárně, vLLM záloha). Když
# uživatel SGLang vyřadí, native runtime startuje vLLM.
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
            # Mamba state pool — u hybridniho Qwen3.5 dela strop souběžnosti on,
            # ne KV cache (viz komentar v docker-compose.yml u extractor-sglang).
            [ -n "${EXTRACTOR_MAX_MAMBA_CACHE:-}" ] && args="$args --max-mamba-cache-size=${EXTRACTOR_MAX_MAMBA_CACHE}"
            [ -n "${EXTRACTOR_MAMBA_SSM_DTYPE:-}" ] && args="$args --mamba-ssm-dtype=${EXTRACTOR_MAMBA_SSM_DTYPE}"
            [ -n "${EXTRACTOR_MAMBA_RADIX_STRATEGY:-}" ] && args="$args --mamba-radix-cache-strategy=${EXTRACTOR_MAMBA_RADIX_STRATEGY}"
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

    # --language-model-only umi SGLang jen pro par architektur (Gemma4
    # ani Qwen3.5 tam nejsou) → opt-in.
    if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
        args="$args --language-model-only"
    fi

    # DTYPE_FLAG — INT4 Marlin MoE na Ampere chce aktivace v dtype scales
    local dtype_var dtype_val
    case "$phase" in
        critic) dtype_var="CRITIC_DTYPE" ;;
        *)      dtype_var="EXTRACTOR_DTYPE" ;;
    esac
    dtype_val=$(eval "echo \${$dtype_var:-}")
    if [ -n "$dtype_val" ]; then
        args="$args --dtype=$dtype_val"
    fi

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
        sample)
            # BAAI/bge-m3 embedding server (fáze 0a, vLLM) — 1 GPU
            model="${TAXONOMY_EMBEDDING_MODEL:-BAAI/bge-m3}"
            tp_size=1
            ;;
        *)
            return 0  # finalize / report → bez LLM
            ;;
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
        log "Auto-start vLLM pro fázi '$phase': $model na portu $port (tp=$tp_size)"
        log "  Args: ${vargs[*]}"
        if NATIVE_ENGINE=vllm SGLANG_MEM_FRACTION_STATIC="$mem"             ./scripts/start_native_llm.sh "$model" "$port" "$tp_size" "${vargs[@]}"; then
            return 0
        fi
        err "Auto-start vLLM selhal pro fázi '$phase'."
        return 1
    fi

    # Sestavit args 1:1 jako Docker (--max-running-requests, --context-length,
    # --reasoning-parser=qwen3 pro extract, --quantization, --disable-piecewise-cuda-graph)
    local sglang_args
    sglang_args=$(build_sglang_args_for_phase "$phase")

    # GDN conv cache musí mít stejný dtype jako model (sglang#30178).
    local mamba_conv="${SGLANG_MAMBA_CONV_DTYPE:-}"
    if [ -z "$mamba_conv" ] && [ "$phase" != "critic" ]; then
        mamba_conv="${EXTRACTOR_DTYPE:-}"
    fi
    log "Auto-start SGLang pro fázi '$phase': $model na portu $port (tp=$tp_size)"
    log "  Args: $sglang_args"
    if SGLANG_EXTRA_ARGS="$sglang_args" \
        SGLANG_MAMBA_CONV_DTYPE="$mamba_conv" \
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

run_phase_native() {
    local phase="$1"
    log "Smoke FÁZE [$phase] — NATIVE (Python přímo, žádný docker)"

    # 0) Auto-install SGLang (jen pokud LLM fáze); embedding fáze jede na vLLM
    if [ "$phase" = "sample" ]; then
        native_vllm_available || return 1
    elif [ "$phase" != "finalize" ] && [ "$phase" != "report" ]; then
        if [ "$(native_engine_for_phase "$phase")" = "vllm" ]; then
            # A100: backend je jen vllm → SGLang instalovat nemusíme
            native_vllm_available || return 1
        else
            auto_install_sglang_if_missing || return 1
        fi
    fi

    # 1) Auto-start LLM server pro tuto fázi
    local port=""
    case "$phase" in
        sample)                  port="8002" ;;
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
compose_build() {
    local profile="$1" service="$2"
    # Explicitní VIDITELNÝ build: `docker compose run` sám od sebe nerebuiluje
    # existující image (jel by starý kód) a při prvním běhu by build probíhal
    # skrytě na pozadí (stderr je u run -d potlačený) — vypadalo by to zamrzle.
    log "Build image $service (první běh stahuje závislosti — může trvat pár minut)"
    if ! docker compose --profile "$profile" build "$service"; then
        err "Build image $service selhal."
        return 1
    fi
    return 0
}

run_phase_via_profile() {
    local phase="$1"
    local profile="$2"
    local labeler_service="$3"
    compose_build "$profile" "$labeler_service" || return 1
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

run_smoke_sample() {
    log "FÁZE 0a (smoke) — SAMPLE (výběr ukázek pro taxonomii, embeddingy bge-m3 na vLLM)"
    local rc=0
    if [ "$RUNTIME" = "native" ]; then
        run_phase_native "sample" || rc=$?
    else
        run_phase_via_profile "sample" "embed-vllm" "labeler-embed-vllm" || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        warn "Výběr ukázek přes embeddingy selhal (rc=$rc) — taxonomie použije náhodný výběr."
    fi
    return 0
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
    compose_build "finalize" "labeler-finalize" || return 1
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
    compose_build "finalize" "labeler-finalize" || return 1
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
echo "  GPU_TYPE=$GPU_TYPE"
echo "  GPU_COUNT=${GPU_COUNT}× $(gpu_type_label "$GPU_TYPE")"
echo "  EXTRACTOR_TP_SIZE=$EXTRACTOR_TP_SIZE"
echo "  CRITIC_TP_SIZE=$CRITIC_TP_SIZE"
echo "  HW_LABEL=$HW_LABEL"
echo "  SAMPLES_PER_DATASET=$SAMPLES_PER_DATASET"
echo "  TAXONOMY_SAMPLES_PER_DATASET=$TAXONOMY_SAMPLES_PER_DATASET"
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
        sample)   run_smoke_sample ;;
        taxonomy) run_smoke_taxonomy ;;
        extract)  run_smoke_extract ;;
        critic)   run_smoke_critic ;;
        revise)   run_smoke_revise ;;
        finalize) run_smoke_finalize ;;
        *)
            err "Neznámá fáze: $phase. Podporované: sample taxonomy extract critic revise finalize"
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
