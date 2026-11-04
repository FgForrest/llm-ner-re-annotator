# Examples

Everything in this folder is **fictitious**: the shop (ACME Outdoor), the
products (ALFA Trek 30, BETA Shell) and all texts were made up for the
documentation. No real customer data is used anywhere.

| File | What it is |
|---|---|
| [input/export-demo-outdoor-latest.json](input/export-demo-outdoor-latest.json) | Input — a Stage export (v2) with 2 products, 1 blog article and 1 system banner |
| [input/labeler_config.json](input/labeler_config.json) | Optional per-file config — domain description for the LLM prompt and the languages to read |
| [output/final_labels.json](output/final_labels.json) | Output of `finalize` for the input above (pretty-printed JSON array) |
| [output/final_labels.jsonl](output/final_labels.jsonl) | The same output as JSONL — one record per line, the format the pipeline writes |
| [entity_taxonomy.example.json](entity_taxonomy.example.json) | A user-defined taxonomy for `--taxonomy-mode user` |

## Input → output

The reader turns each record into one item to label (`item_id` =
`dataset:schema:record id`):

| Record | Item | Labeled text |
|---|---|---|
| product `1001` | `demo-outdoor:products:1001` | `description` (HTML stripped); brand and parameters go to the LLM as context |
| product `1002` | `demo-outdoor:products:1002` | `description` |
| blog page `edeePage_1` | `demo-outdoor:blog:edeePage_1` | `content`; `title` goes to the LLM as context |
| blog page `edeePage_2` | — | skipped: the `alertStrip` prototype is navigation, not content |

Things to look for in the output:

- **Positions are computed by code:** `start_char` / `end_char` always satisfy
  `input_text[start_char:end_char] == text`.
- **Label remapping:** `KATEGORIE` from the default taxonomy becomes
  `PRODUKT_TYP`; `PARAMETR`, `KOMPONENTA` and `POUZITI` become `KEYPHRASE_*`
  labels.
- **Keyphrases outside the NER graph:** `KEYPHRASE_*` entities and the relations
  that touch them are moved from `graph` into the `keyphrases` block; `graph`
  keeps only the NER entities.
- **Feedback loop:** for `demo-outdoor:products:1002` the critic pointed out the
  missing hood ("Kapuce"), the revisor accepted it, so the record has
  `critic_status: "revised"`, `final_graph_source: "extractor_revised"`,
  `was_revised: true` and the change listed in `changes_applied`.
- **Lemmas come from Stanza, token by token:** they are consistent but not
  always grammatical (`"1 , 1 kilogram"`, `"ramenní popruha"`). That is
  expected — the lemmas are keys for voting and deduplication.

## How the output was produced

The parts that normally come from the LLMs — entity spans, categories,
relations and the critic/revisor verdicts — were **written by hand** for these
three texts. Everything else is the real pipeline code: the Stage reader,
position computation, Stanza lemmatization, normalization (label remap,
overlap and offset checks), the keyphrase split, triples and the output
writer (`python -m labeler.main finalize`). The `extractor_model` /
`critic_model` fields show the default models a production run records. So
the output shows the exact format, not the quality of a real model run.

## Try it

```bash
mkdir -p data
cp examples/input/* data/
./scripts/run_test.sh          # mini models, Linux / WSL2
```

On Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\run_test.ps1`.
