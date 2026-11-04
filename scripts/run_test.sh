set -euo pipefail

cd "$(dirname "$0")/.."

log() { echo -e "\n\033[1;34m==>\033[0m $*"; }
err() { echo -e "\n\033[1;31m!!\033[0m $*" >&2; }

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.test.yml"

if [ ! -d "data" ]; then
    err "Chybí složka data/"
    exit 1
fi

# Akceptujeme libovolný *.json (Stage export `export-*-latest.json` i generický
# JSON). Stačí, aby byl alespoň jeden JSON soubor.
json_count=$(find data -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
if [ "$json_count" -eq 0 ]; then
    err "V data/ nenalezeny žádné *.json soubory"
    err "Nakopíruj tam vstupní exporty, např.:"
    err "  data/export-mujdataset-latest.json"
    err "(Libovolný jiný *.json zpracuje generický reader.)"
    exit 1
fi
log "Detekováno $json_count JSON soubor(ů) v data/: $(ls data/*.json | xargs -n1 basename | tr '\n' ' ')"

log "Čistím staré test checkpointy…"
rm -f checkpoints/extractions.jsonl \
      checkpoints/extractions_synthetic.jsonl \
      checkpoints/critic_results.jsonl \
      checkpoints/critic_results_synthetic.jsonl \
      checkpoints/final_labels.jsonl \
      checkpoints/final_labels_synthetic.jsonl \
      checkpoints/stats.json

log "FÁZE 1 — mini extractor (Qwen3-0.6B, limit=3/dataset)"
$COMPOSE --profile test-extract up \
    --build \
    --abort-on-container-exit \
    --exit-code-from labeler-extract-sglang
$COMPOSE --profile test-extract down

log "FÁZE 2 — mini critic (Gemma-3-1b), sám na GPU"
log "         Critic uloží verdikt; revisor přijde sólo ve fázi 2b."
$COMPOSE --profile test-critic up \
    --build \
    --abort-on-container-exit \
    --exit-code-from labeler-critic-sglang
$COMPOSE --profile test-critic down

log "FÁZE 2b — revisor (Qwen3-0.6B) projde non-ok kritiky a doplní extractor_revised"
log "          Modely nikdy neběží souběžně — extractor se znovu nahodí, critic už ne."
$COMPOSE --profile test-revise up \
    --build \
    --abort-on-container-exit \
    --exit-code-from labeler-revise-sglang
$COMPOSE --profile test-revise down

log "FÁZE 3 — finalize"
docker compose --profile finalize up \
    --build \
    --abort-on-container-exit \
    --exit-code-from labeler-finalize

log "VÝSLEDKY:"
for f in final_labels.jsonl final_labels_synthetic.jsonl stats.json; do
    path="checkpoints/$f"
    if [ -f "$path" ]; then
        if [[ "$f" == *.jsonl ]]; then
            n=$(wc -l < "$path")
            echo "  ✓ $path  ($n řádků)"
        else
            size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo "?")
            echo "  ✓ $path  ($size B)"
        fi
    else
        echo "  ✗ $path  CHYBÍ (očekáváno)"
    fi
done

echo ""
echo "Mezivýstupy (po finalize by měly být smazané):"
for f in extractions.jsonl extractions_synthetic.jsonl \
         critic_results.jsonl critic_results_synthetic.jsonl; do
    path="checkpoints/$f"
    if [ -f "$path" ]; then
        echo "  ?  $path  (stále existuje — finalize selhal?)"
    else
        echo "  ✓ $path  (smazán - OK)"
    fi
done

log "Posledních pár řádků final_labels.jsonl:"
if [ -f checkpoints/final_labels.jsonl ]; then
    tail -n 2 checkpoints/final_labels.jsonl | python3 -m json.tool --json-lines 2>/dev/null \
        || tail -n 2 checkpoints/final_labels.jsonl
fi

log "stats.json:"
if [ -f checkpoints/stats.json ]; then
    cat checkpoints/stats.json
fi

log "✅ TEST HOTOVÝ. Výsledky jsou v checkpoints/ (zejména final_labels.jsonl a stats.json)."
