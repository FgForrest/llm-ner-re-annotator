#!/usr/bin/env bash
# build.sh — MANUÁLNÍ build labeler image(s). Spouštěj sám, ručně, PŘED během
# pipeline. run_production.sh i run_smoke_test.sh už image NEbuildí — build je
# všude manuální přes tenhle script. Běží v POPŘEDÍ (ne na pozadí), takže vidíš
# průběh a počkáš na dokončení.
#
# Použití (z kořene repa):
#   ./scripts/build.sh                 # default: sglang labeler služby + finalize
#   ./scripts/build.sh all             # všechny (sglang + vllm + finalize)
#   ./scripts/build.sh labeler-critic-sglang   # konkrétní služba/y
#
# Pozn.: LLM servery (extractor-sglang / critic-sglang) se NEbuildí — mají
# image: (stahují se přes pull), ne build:. Build se týká jen labeler image
# (Python pipeline kód z labeler/).

set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_SVCS="labeler-extract-sglang labeler-critic-sglang labeler-revise-sglang labeler-finalize"
ALL_SVCS="labeler-extract-sglang labeler-extract-vllm labeler-critic-sglang labeler-critic-vllm labeler-revise-sglang labeler-revise-vllm labeler-finalize"

if [ "$#" -eq 0 ]; then
    SVCS="$DEFAULT_SVCS"
elif [ "$1" = "all" ]; then
    SVCS="$ALL_SVCS"
else
    SVCS="$*"
fi

command -v docker >/dev/null 2>&1 || { echo "!! docker nenalezen v PATH." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "!! 'docker compose' (v2) nenalezen." >&2; exit 1; }

echo "==> Building (popředí): $SVCS"
docker compose build $SVCS
echo "==> Build hotov. Teď můžeš spustit ./scripts/run_production.sh nebo ./scripts/run_smoke_test.sh."
