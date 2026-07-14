set -euo pipefail

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL_DIR="${MODEL_CACHE_DIR:-./models}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
CRITIC_MODEL="${CRITIC_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"

mkdir -p "$MODEL_DIR"

echo "============================================================"
echo "Stáhnu modely do: $MODEL_DIR"
echo "  Extractor: $EXTRACTOR_MODEL"
echo "  Critic:    $CRITIC_MODEL"
echo "============================================================"
echo ""

# Odhad velikosti
case "$EXTRACTOR_MODEL" in
    *FP8*)   ext_size="~125 GB" ;;
    *NVFP4*) ext_size="~65 GB" ;;
    *)       ext_size="varies" ;;
esac
case "$CRITIC_MODEL" in
    *NVFP4*) crit_size="~16 GB" ;;
    *FP8*)   crit_size="~31 GB" ;;
    *)       crit_size="~62 GB (BF16)" ;;
esac
echo "Odhadovaná velikost: extractor $ext_size + critic $crit_size"
echo ""

if ! command -v huggingface-cli &> /dev/null; then
    echo "Instaluju huggingface_hub..."
    pip install --upgrade "huggingface_hub[cli,hf_transfer]"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

slug() {
    echo "$1" | sed 's|/|--|g'
}

echo "[1/2] Stahuji extractor: $EXTRACTOR_MODEL"
huggingface-cli download "$EXTRACTOR_MODEL" \
    --local-dir "$MODEL_DIR/$(slug "$EXTRACTOR_MODEL")"

echo ""
echo "[2/2] Stahuji critic: $CRITIC_MODEL"
huggingface-cli download "$CRITIC_MODEL" \
    --local-dir "$MODEL_DIR/$(slug "$CRITIC_MODEL")"

echo ""
echo "✅ Modely stažené do $MODEL_DIR"
du -sh "$MODEL_DIR"/* 2>/dev/null || true
