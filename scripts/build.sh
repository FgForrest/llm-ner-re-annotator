#!/usr/bin/env bash
# build.sh — MANUÁLNÍ build labeler image(s). Spouštěj sám, ručně, PŘED během
# pipeline. run_production.sh i run_smoke_test.sh už image NEbuildí — build je
# všude manuální přes tenhle script. Běží v POPŘEDÍ (ne na pozadí), takže vidíš
# průběh a počkáš na dokončení.
#
# Použití (z kořene repa):
#   ./scripts/build.sh                 # postaví labeler image (stačí na VŠE)
#   ./scripts/build.sh all             # projde všechny labeler služby (ověření)
#   ./scripts/build.sh labeler-critic-sglang   # konkrétní služba/y
#
# JEDNA IMAGE PRO VŠECHNY FÁZE: labeler služby (extract / critic / revise /
# finalize / embed) sdílejí image ${LABELER_IMAGE:-llm-ner-re-annotator:latest} ze
# stejného Dockerfile.labeler — postaví se jednou a používá ji produkční
# i testovací compose.
#
# POZOR: holé `docker compose build` NIC nepostaví ("No services to build") —
# labeler služby mají profily (extract-sglang, finalize, embed-vllm…) a Compose
# bez aktivního profilu nebo jména služby žádnou nevidí. Buildi VŽDY přes
# tenhle script (nebo `docker compose build <služba>`).
#
# Pozn.: LLM servery (extractor-sglang / critic-sglang / embedder-vllm) se
# NEbuildí — mají image: (stahují se přes pull), ne build:.

set -euo pipefail
cd "$(dirname "$0")/.."

# Sdílená image → stačí postavit přes jednu službu; ostatní ji jen používají.
DEFAULT_SVCS="labeler-finalize"
ALL_SVCS="labeler-extract-sglang labeler-extract-vllm labeler-critic-sglang labeler-critic-vllm labeler-revise-sglang labeler-revise-vllm labeler-embed-vllm labeler-finalize"

if [ "$#" -eq 0 ]; then
    SVCS="$DEFAULT_SVCS"
elif [ "$1" = "all" ]; then
    SVCS="$ALL_SVCS"
else
    SVCS="$*"
fi

command -v docker >/dev/null 2>&1 || { echo "!! docker nenalezen v PATH." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "!! 'docker compose' (v2) nenalezen." >&2; exit 1; }

IMAGE="${LABELER_IMAGE:-llm-ner-re-annotator:latest}"
echo "==> Building (popředí): $SVCS → image $IMAGE"
docker compose build $SVCS

# Pojistka: kdyby build kvůli profilům nic nepostavil, ať to víme TEĎ a ne
# až uprostřed běhu pipeline.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "!! Image $IMAGE po buildu neexistuje — build nic nepostavil." >&2
    echo "!! Jsi v kořeni repa? Má služba v compose build: sekci?" >&2
    exit 1
fi
echo "==> Build hotov: $IMAGE ($(docker image inspect -f '{{.Created}}' "$IMAGE"))"
echo "    Image používají VŠECHNY labeler fáze (extract / critic / revise / finalize / embed)."
echo "    Teď můžeš spustit ./scripts/run_production.sh nebo ./scripts/run_smoke_test.sh."
