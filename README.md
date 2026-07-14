# Mudrc Labeler

Automatizovaný labeling českých textů dvojicí LLM modelů (extractor + critic).
Výstup ve formátu **Mudrc SKG** (entities + relations + triples) —
ready-to-use jako training data pro deterministickou Stanza/spaCy pipeline.

## Klíčové vlastnosti

- **Univerzální taxonomie + discovery (default auto):** fáze 0 (`taxonomy`) běží **defaultně v auto módu i v produkci** — LLM analyzuje vzorky ze všech datasetů a navrhne jednotnou taxonomii (počet vzorků přes `TAXONOMY_SAMPLES_PER_DATASET`, default 10). Je **best-effort**: když auto selže (extractor nedostupný, prázdná data, přetečení kontextu…), spadne se na default 16 kategorií a běh POKRAČUJE — nikdy nezhodí pipeline. Alternativy: `--taxonomy-mode user` (uživatelský `entity_taxonomy.json`) nebo `--taxonomy-mode off` (16 kategorií, žádné LLM volání). Garantuje konzistenci — stejný koncept = stejný label napříč doménami; povinné labely (PRODUKT/ORGANIZACE/DATUM/MISTO/HODNOTA) doplní SW sám.
- **Normalizační vrstva pro silver dataset:** běží **automaticky v každém běhu** (produkce + smoke test). Fix offsetů, auto-expand všech výskytů entity, label remapping (16 → 11), drop překryvů, řešení konfliktů, drop strukturálních identifikátorů (URL/email/telefon regex). Output: `final_labels.{jsonl,json}` + `normalization_report.json`. Pro vypnutí (debugging) `--no-normalize`.
- **Dva oddělené datasety:** reálné popisky vs synteticky poskládané texty → každý má vlastní `final_labels*.{json,jsonl}`, aby se kvalita nemíchala
- **Cross-family validace:** extractor Qwen3.5 (Alibaba, MoE) + critic Gemma 4 (Google, dense) = různé tokenizery, různé trénovací data = skutečně nezávislá validace
- **Fáze 0 + 4 fáze sekvenčně, vždy 1 model na GPU:** taxonomy (default auto) → extract (Qwen) → critic (Gemma) → revise (Qwen revisor) → finalize. Žádné dva modely souběžně, ani v produkci, ani ve smoke testu — celá VRAM je vyhrazená pro jeden model najednou.
- **Split NER/RE (default ON, `SPLIT_NER_RE`):** extract, critic i revise běží jako **dvě samostatná LLM volání** — jedno na entity (NER), druhé na relace (RE, s už hotovými entitami jako kontextem) — místo jednoho volání nad celým grafem. Model se soustředí na jednu úlohu → lepší kvalita a méně malformed relací. Triplety se z relací dopočítají deterministicky (žádné extra volání). Cena: ~2× LLM volání na fázi. Pro původní chování (vše v jednom) `SPLIT_NER_RE=false`.
- **Automatická revisor smyčka Gemma → Qwen:** kritika (status `revised` / `reject`) se uloží do `critic_results.jsonl`, a fáze 2b ji projde — pošle revisorovi (defaultně tomu samému modelu, který popisek labeloval — Qwen3.5). Revisor sám rozhodne, zda kritiku zapracuje, nebo si stojí za svým výstupem. Konfigurovatelné přes `REVISOR_*` env (viz níže).
- **Pozice počítá aplikace, ne LLM:** LLM vrací jen surface form ("text") a lemma; `start_char`/`end_char` se dopočítají kódově nad původním textem, takže nemůžou halucinovat indexy. Garantuje se invariant `text[start:end] == entity.text` + hranice slova + bez překryvů.
- **Resume-safe:** každý JSON záznam fsync-nutý, crash = ztráta max 1 itemu; revise je idempotentní
- **Hash deduplikace:** identické texty se labelují jen jednou
- **Self-repair:** nevalidní JSON z extraktoru → retry s chybovou zprávou
- **Reasoning mode:** critic defaultně thinking-ON pro kvalitu, extractor + revisor OFF pro rychlost
- **Chunking:** dlouhé texty (>12000 znaků, `CHUNK_MAX_CHARS`) se dělí na věty; ve fázi finalize se chunky stejného dokumentu spojí zpět a entity se přemapují na merged text
- **Windows + Linux:** PowerShell + bash skripty, `.gitattributes` fixuje line endings
- **Smoke test profil:** mini modely (Qwen3-0.6B + Gemma-3-1b) na RTX 4070 (12 GB) nebo podobné — sdílí stejný 4-fázový design jako produkce
- **Production smoke test:** `scripts/run_smoke_test.{sh,ps1}` propustí 100 itemů per dataset celou pipeline na cílovém B200 HW a pure-empiricky extrapoluje na full produkční rozsah (lineární `T_full = T_smoke × full_chars/smoke_chars`, ±2h konfidence). Žádné teoretické GPU scaling factory — co naměříš, to dostaneš.
- **OWL ontology export (jen v smoke testu):** smoke-test `--phase finalize` vygeneruje plnou OWL 2 ontologii — `.owl` (RDF/XML) + `.ttl` (Turtle) + volitelně `.html` (přes `pylode`). Postaveno přes `rdflib` (žádné hand-written serializery). Obsahuje TBox (Document/Entity třídy + per-label subclasses + per-relation ObjectProperties) a ABox (Document a Entity individuals + relace mezi nimi). Combined + per-source split.

## Taxonomie (univerzální napříč doménami)

**Kategorie entit** (16 obecných tříd, názvy česky UPPERCASE — odpovídají Mudrc C# enumu):

| Kategorie | Popis | E-shop příklad | Pojištění příklad |
|---|---|---|---|
| `PRODUKT` | konkrétní pojmenovaný produkt/služba | Alpine Summit, ACME Basic | Pojištění auta OMEGA |
| `KATEGORIE` | typ produktu obecně | outdoorová bunda, sluchátka | povinné ručení, životní pojištění |
| `KOMPONENTA` | část produktu, dílčí složka | membrána, měnič, šev | klauzule o spoluúčasti |
| `PARAMETR` | měřitelná vlastnost | vodní sloupec, prodyšnost | limit plnění, spoluúčast |
| `HODNOTA` | konkrétní hodnota parametru | 20 000 mm, 40 mm, YKK | 100 000 Kč, 12 měsíců |
| `MATERIAL` | surový materiál | paměťová pěna, kov | (zřídka) |
| `VLASTNOST` | design/styl/featura | ergonomický střih | celosvětová platnost |
| `BENEFIT` | přínos pro uživatele | termoregulace, pohodlí | ochrana, klid v rodině |
| `POUZITI` | kontext / scénář / riziko | hory, město, kancelář | škoda na zdraví, krádež, požár |
| `TECHNOLOGIE` | technologie/metoda | ANC, GORE-TEX | online sjednání |
| `SPECIFIKACE` | technický standard | Bluetooth 5.3, USB-C | ISO 9001 |
| `ORGANIZACE` | firma, instituce | ACME, Apple | ACME Pojišťovna |
| `OSOBA` | konkrétní osoba | (zřídka) | (zřídka) |
| `MISTO` | geografické místo | (zřídka) | Praha, ČR |
| `DATUM` | datum, časový interval | (zřídka) | 1.1.2026, 12 měsíců |
| `SUBJEKT` | generický fallback | — | — |

**Klíčový princip:** pojišťovácké pojmy se NEMAPUJÍ na specifické kategorie typu `INSURANCE_PRODUCT` nebo `DEDUCTIBLE` — místo toho se mapují na obecné (`PRODUKT`, `PARAMETR`). LLM dostává oba doménové příklady v promptu, aby pochopil princip mapování.

**Relace:** 37 typů, anglické názvy v `snake_case` (`is_a`, `has_component`, `has_value`, `provides_benefit`, `covers`, `suitable_for`, ...). Plný seznam viz `labeler/schemas.py`.

## Taxonomy Discovery (fáze 0 — default auto)

Pipeline má **fázi 0** (`python -m labeler.main taxonomy`, nebo automaticky
před `extract`/`pipeline`), která ovlivní, jaké labely bude LLM používat ve
fázi extract. **Default je `auto` a běží i v produkci** (ne jen ve smoke testu):

| Mode | Co to dělá |
|---|---|
| `auto` (default) | LLM (Qwen3.5) si projde **vzorky ze všech datasetů v `DATA_DIR`** a navrhne **jednotnou taxonomii** pokrývající všechny domény. Garantuje konzistenci: stejný koncept = stejný label, žádné duplikáty napříč doménami. Kvalitativní brána + 1 retry. **Best-effort**: když auto selže (extractor nedostupný, prázdná data, přetečení kontextu…) NEBO LLM ani po reklamaci nedodá kvalitní taxonomii → spadne se na default 16 kategorií a běh POKRAČUJE. Počet vzorků per dataset: `TAXONOMY_SAMPLES_PER_DATASET` (default 10). |
| `off` | Žádná discovery, žádné LLM volání. Použije se default 16 kategorií z `EntityCategory` (`PRODUKT`, `KATEGORIE`, `POUZITI`, …). |
| `user` | Uživatel poskytne `entity_taxonomy.json` (viz [examples/entity_taxonomy.example.json](examples/entity_taxonomy.example.json)). Pipeline ho validuje a použije. |

Výstup (kromě `off`): `checkpoints/entity_taxonomy.json`. Fáze `extract` ho
automaticky načte (pokud existuje) a vloží do system promptu místo defaultu.

### Použití

```bash
# Default — extract sám pustí fázi 0 v auto módu (best-effort discovery)
python -m labeler.main extract

# Vypnout discovery (16 originálních kategorií, žádné LLM volání navíc)
python -m labeler.main extract --taxonomy-mode off

# Samostatná fáze 0 dopředu (volitelné — extract si ji jinak pustí sám)
python -m labeler.main taxonomy --mode auto --samples-per-dataset 15
python -m labeler.main extract       # použije už hotovou entity_taxonomy.json

# User mode (uživatelský JSON)
python -m labeler.main taxonomy --mode user \
    --user-taxonomy examples/entity_taxonomy.example.json
python -m labeler.main extract

# Integrace s pipeline (jediný příkaz) — auto je default, lze i off/user
python -m labeler.main pipeline --taxonomy-mode auto --samples-per-dataset 10
```

### Formát `entity_taxonomy.json`

```json
{
  "version": 1,
  "mode": "auto",
  "generated_at": "2026-05-14T10:00:00+00:00",
  "source_datasets": ["pojistovna-demo", "eshop-demo", "sperky-demo", "kosmetika-demo"],
  "labels": [
    {
      "code": "PRODUKT",
      "description": "Konkrétní pojmenovaný produkt nebo služba.",
      "examples": ["ACME Basic", "Pojištění auta OMEGA", "iPhone 15"],
      "domains": ["e-shop", "pojišťovnictví"],
      "aliases": ["PRODUCT", "ITEM"]
    },
    {
      "code": "ORGANIZACE",
      "description": "Firma, instituce, značka, výrobce.",
      "examples": ["ACME Pojišťovna", "ACME", "DEMO šperky"],
      "domains": ["e-shop", "pojišťovnictví"]
    }
  ]
}
```

Kompletní example: [examples/entity_taxonomy.example.json](examples/entity_taxonomy.example.json) — labely pokrývající e-shop, pojišťovnictví a kosmetiku.

> **Pozn.** URL, EMAIL a TELEFON do NER taxonomie nepatří — jsou to regex-detekovatelné struktury, ne pojmenované entity. Pokud je LLM přesto extrahuje, `_step_drop_structural_identifiers` v normalize je zahodí (counter `structural_identifiers_dropped`). Pokud je downstream kód potřebuje, pustí URL_RE / EMAIL_RE / PHONE_RE přímo na input_text — deterministické, model k tomu netřeba.

### Validační pravidla (auto + user)

- `code` musí být **UPPERCASE_SNAKE** (`^[A-Z][A-Z0-9_]{0,49}$`)
- `description` ≥ 10 znaků
- min 2 labely
- žádné duplicitní `code` (deduplikace přes `aliases`)
- LLM dostane explicit pravidlo: **PRODUKT vs ITEM = duplicitní, vrať jen PRODUKT**

### Kvalitativní brána + povinné labely (jen auto)

Auto mode navíc po parse + deduplikaci provede:

- **Doplnění povinných labelů** — pokud LLM nenavrhl některý z `PRODUKT`,
  `ORGANIZACE`, `DATUM`, `MISTO`, `HODNOTA`, doplní ho **SW sám** (kontrola
  case-insensitive vůči kódům i aliasům, takže se nepřidá duplicitně). Pipeline
  na tyto univerzální kategorie spoléhá → negarantuje se, že je „trefí" model.
- **Kvalitativní brána** — návrh musí mít aspoň **5 labelů od samotného LLM**
  (povinné doplněné SW se do toho nepočítají) a projít validací výše. Když
  neprojde, pošle se LLM **konkrétní reklamace** (co je špatně) a dostane
  **druhý pokus**. Když selže i ten, vyhodí se `TaxonomyQualityError` →
  `run_taxonomy_discovery` ji odchytí, **neukládá** `entity_taxonomy.json` a
  pipeline **spadne na default 16 kategorií** (jako `mode=off`). Běh se neshodí.

### Volitelná data-driven heuristiky (per label)

Každý `TaxonomyLabel` může obsahovat 3 volitelné fields pro řízení normalizace **bez hard-coding v kódu**:

| Field | Účel | Příklad (pojišťovácký dataset) |
|---|---|---|
| `force_text_match` | Texty, které VŽDY spadnou pod tento label, i když LLM dal jiný | `PRODUKT.force_text_match = ["ALFA", "BETA", "GAMA"]` |
| `unsafe_texts` | Texty, které se NIKDY nemají dostat do tohoto labelu (downgrade na KEYPHRASE_SUBJECT) | `PRODUKT.unsafe_texts = ["pojištění", "auto", "smlouva"]` |
| `case_sensitive` | Case sensitivity pro oba seznamy (default false) | `true` pro vlastní názvy (ALFA), `false` pro generická slova |

Pokud taxonomy tyto fields neobsahuje, `_step_apply_taxonomy_text_rules` se přeskočí — žádné hard-coded heuristiky se nepoužijí. Pipeline je **100% univerzální**: nemá zakódované žádné doménově-specifické termíny v Python kódu.

Pro **nové datasety** (libovolná doména, kde nikdy nebyl klíč pojišťovnictví) prostě vytvoříš `entity_taxonomy.json` (auto nebo user mode) a doplníš `force_text_match` / `unsafe_texts` podle potřeby — kód se nemění.

### Auto mode — jak funguje konzistence

LLM dostane jeden velký prompt:
- bullet list všech detekovaných **domén** (z `dataset.name`)
- ukázky z každé domény **rozprostřené po celém datasetu** — ne jen prvních N.
  Bere se 3 „kotvy" (první / prostřední / poslední položka) + `--samples-per-dataset`
  náhodných ze zbytku (deterministicky přes seed), takže výsledek je až `N+3`
  ukázek pokrývajících začátek, střed i konec datasetu. Default `N=10`.
- pravidla: stejný koncept = stejný label, žádné duplikáty (PRODUKT vs ITEM)
- povinné labely: PRODUKT, ORGANIZACE, DATUM, MISTO, HODNOTA (URL / EMAIL / TELEFON tu nepatří — regex-detekovatelné struktury, viz výše)

Vrátí jeden konsolidovaný JSON. Aplikace ho pak ještě dotřídí: `_deduplicate_aliases`
(sloučí 2 labely se stejným kódem/aliasem) → doplní chybějící **povinné labely**
→ projede **kvalitativní bránou** (min. 5 labelů od LLM + validace). Při
nedostatečné kvalitě jeden retry s reklamací, pak fallback na default 16
kategorií (viz [Kvalitativní brána + povinné labely](#kvalitativní-brána--povinné-labely-jen-auto) výše).

## Architektura

```
                      ┌───────────────────────────┐
                      │  FÁZE 0 (default auto):   │   model: Qwen3.5-122B
   *.json    ────────▶│  taxonomy discovery       │   (LLM analyzuje vzorky a
   v DATA_DIR         │  GPU obsazené: Qwen       │   navrhne jednotnou taxonomii;
                      │                           │   best-effort, vypni přes
                      │  --taxonomy-mode off/user │   --taxonomy-mode off)
                      └────────┬──────────────────┘
                               │ entity_taxonomy.json
                               │ (když auto selže → fallback na default 16 kategorií)
                               ▼
                      ┌───────────────────────────┐
                      │  FÁZE 1: extract          │   model: Qwen3.5-122B
                      │  split: 1a NER → 1b RE    │   (LLM vrací jen text+lemma,
                      │  GPU obsazené: Qwen       │   pozice doplní kód;
                      │                           │   SPLIT_NER_RE, default ON)
                      └────────┬──────────────────┘
                               │ extractions*.jsonl
                      ┌────────┴──────┐
                      │ Qwen down     │   ← VRAM uvolněna
                      └────────┬──────┘
                               ▼
                      ┌───────────────────────────┐
                      │  FÁZE 2: critic           │   model: Gemma 4 31B
                      │  GPU obsazené: Gemma      │   verdikt: ok/revised/reject
                      │  split: NER + RE zvlášť   │   + reasoning + corrected
                      └────────┬──────────────────┘
                               │ critic_results*.jsonl (bez extractor_revised)
                      ┌────────┴──────┐
                      │ Gemma down    │   ← VRAM uvolněna
                      └────────┬──────┘
                               ▼
                      ┌───────────────────────────┐
                      │  FÁZE 2b: revise          │   model: Qwen3.5 (revisor)
                      │  GPU obsazené: Qwen       │   defaultně = extractor;
                      │  split: NER + RE zvlášť   │   přepiš REVISOR_* v .env
                      │  Pro non-ok kritiky:      │   pro jiný model
                      │   - Qwen čte kritiku      │
                      │   - sám rozhodne          │
                      │   - vrátí opravený nebo   │
                      │     původní graf          │
                      │   - uloží do              │
                      │     extractor_revised     │
                      └────────┬──────────────────┘
                               │ critic_results*.jsonl (s extractor_revised)
                      ┌────────┴──────┐
                      │ Qwen down     │   ← VRAM uvolněna
                      └────────┬──────┘
                               ▼
                      ┌───────────────────────────┐
                      │  FÁZE 3: finalize         │   bez LLM
                      │  Merge + chunk join       │   preference:
                      │  + stats                  │   extractor_revised
                      └────────┬──────────────────┘   → corrected → original
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────────┐         ┌─────────────────────────┐
   │ final_labels.json    │         │ final_labels_synthetic  │
   │ + .jsonl             │         │  .json + .jsonl         │
   └──────────────────────┘         └─────────────────────────┘
                       + stats.json (souhrn obou)

    NIKDY 2 LLM modely zaráz — vždy jen 1 model na GPU.
    Mezi fázemi `compose down` → uvolnění VRAM → další fáze.

    Split NER/RE (default, SPLIT_NER_RE): extract/critic/revise = 2 LLM volání
    (entity, pak relace s entitami jako kontextem). Triplety se odvodí z relací.

    Mezivýstupy (extractions*, critic_results*) se po fázi 3 AUTOMATICKY SMAŽOU.
    Pro zachování pro debugging použij: finalize --keep-intermediate
```

**Proč se mezivýstupy mažou?**
Čistý výsledek má jen 3 soubory — dva finální JSONL a jeden stats.
Během běhu potřebujeme extraction + critic separátně kvůli resume/checkpoint
(kdyby to crashlo mezi fázemi, můžeš pokračovat odtamtud), ale jakmile je
finalize hotový, jsou zbytečné a jenom matou. Defaultně se smažou, přes
`--keep-intermediate` je zachováš.

**Proč dvě větve (real + synthetic)?**
Pokud item nemá `productDescription`, ale má jen `parameters` + `productName`,
poskládáme z toho syntetický text. Tohle je ale riskantnější (LLM může víc
halucinovat), tak to jde do samostatného datasetu — můžeš si rozhodnout,
jestli ho použít vedle reálného, nebo vyhodit.

**Proč 4 fáze, ne souběžně?**
- Žádné dva modely najednou na GPU = celá VRAM pro jeden model = větší batch
  / KV cache = vyšší throughput.
- Stejná architektura pro produkci (B200) i smoke test
  (1× 12 GB) — žádné rozdvojení mezi "rychlou produkci" a "úsporný test".
- Resume-safe na úrovni fází: pokud cokoliv crashne, navážeš odtamtud.
- Revise je idempotentní → bezpečné restartovat.

**Split NER/RE (default ON, `SPLIT_NER_RE=true`):**
Uvnitř fází extract, critic i revise se práce **rozloží na dvě samostatná LLM
volání** místo jednoho nad celým grafem:

1. **NER** — model řeší jen entity (`{"entities": [...]}`).
2. **RE** — model dostane už hotové entity jako **read-only kontext** a hledá
   jen relace mezi nimi (`subjectId`/`objectId` odkazují na jejich `id`).

Model se tak soustředí na jednu úlohu → měřitelně lepší kvalita a méně
malformed relací (model nevymýšlí relace mimo povolený seznam, když má před
sebou jen ten jeden úkol). Výsledné poloviny se deterministicky složí zpět do
jednoho grafu — osiřelé relace (na entitu, kterou druhá půlka zahodila),
self-loops a duplicity se ořežou. **Triplety nejsou samostatné volání** —
odvozují se z relací (`triple = (lemma subjektu, relation, lemma objektu)`).
`CriticResult` formát zůstává stejný, takže finalize ani revise nic nepoznají.

Cena: ~2× LLM volání na fázi (≈ 2× delší běh a 2× tokenů). Pro původní chování
(vše v jednom volání) nastav `SPLIT_NER_RE=false` v `.env` — žádný rebuild
netřeba pro přepnutí, je to jen runtime flag.

## Feedback smyčka (Gemma → revisor) — vždy automatická, vlastní fáze

Critic (Gemma, fáze 2) vrátí jeden ze tří verdiktů: `ok`, `revised` nebo `reject`,
a uloží `CriticResult` (BEZ `extractor_revised`). V navazující fázi 2b se
nahodí **revisor** — defaultně tentýž LLM, který popisek původně labeloval
(Qwen3.5). Revisor dostane:

1. původní text
2. původní výstup extraktoru
3. kritiku (status + zdůvodnění)
4. případně graf, který critic navrhl jako opravený

**Revisor pak sám rozhodne**, jestli kritiku zapracuje (vrátí opravený graf),
nebo jestli si stojí za svým (vrátí původní výstup beze změn). Toto rozhodnutí
se uloží do `extractor_revised` v `CriticResult` (fáze 2b atomicky přepíše
JSONL) a finalize (fáze 3) ho přednostně použije při merge.

| Critic verdict      | Co se uloží do final_labels                                      |
|---------------------|------------------------------------------------------------------|
| `ok`                | Původní extractor graf (revisor se nevolá)                       |
| `revised` + revisor zapracoval | `extractor_revised` graf (preferovaný)                |
| `revised` + revisor odmítl     | Vrácený graf revisora (Qwen vrátil v podstatě totéž)  |
| `revised` (revisor selhal)     | `corrected` od critic (fallback)                      |
| `reject` + revisor opravil     | `extractor_revised` graf                              |
| `reject` (revisor selhal)      | Původní extractor graf + flag `was_rejected`          |

### Konfigurace revisoru přes .env

Defaulty se odvodí od `EXTRACTOR_*` (revisor = stejný model i endpoint):

```bash
# .env — pokud chceš revisora MĚNIT na jiný model než extractor:
REVISOR_BASE_URL=http://my-other-llm:8000/v1
REVISOR_MODEL=meta-llama/Llama-3.3-70B-Instruct
REVISOR_REASONING=false
REVISOR_CONCURRENCY=8

# Pokud REVISOR_* NEnastavíš, použijí se EXTRACTOR_* (Qwen3.5).
```

### Vynechání revise fáze

Pokud nechceš revisor vůbec pouštět (např. rychlý test bez Qwen feedbacku),
prostě vynech fázi 2b — finalize si poradí i bez `extractor_revised` (použije
fallback `corrected` → původní extractor):

```bash
./scripts/run_production.sh extract
./scripts/run_production.sh critic
./scripts/run_production.sh finalize   # bez 'revise' fáze
```

Implementace: [labeler/revise.py](labeler/revise.py) — funkce `run_revise`
zavolá revisora pro každý non-ok záznam bez `extractor_revised`. Idempotentní
(opětovný běh přeskočí už zrevidované).

Implementace: [labeler/critic.py:_revise_with_revisor](labeler/critic.py) +
system prompt `EXTRACTOR_REVISION_SYSTEM` v [labeler/prompts.py](labeler/prompts.py).

## Pozice entit (start_char, end_char) — počítá se kódově, ne v LLM

LLM (extractor i critic) nikdy **neposílá** numerické indexy v textu. Místo toho
vrací jen `text` (přesný substring originálu) + `lemma`. Pozice doplní
deterministický kód v `labeler/positions.py::compute_entity_positions`:

1. Pro každou entitu se najde první výskyt `entity.text` v původním dokumentu
   (case-sensitive; pokud ne, fallback na case-insensitive a `text` se přepíše
   na přesný substring originálu).
2. Ověří se, že začátek i konec spanu leží na hranici slova (přechod alfanum
   ↔ ne-alfanum). To zaručí, že span půjde použít se spaCy tokenizátorem bez
   `Doc.char_span(...) == None`.
3. Greedy se vyřeší překryvy (delší span vyhrává, ties → dřívější výskyt).
4. Entity, které neprošly, se zahodí včetně relací, které je odkazují
   (kaskádové čištění).

**Proč:** LLM počítá indexy notoricky špatně — off-by-one, špatné skládání u
unicode (české háčky, ž/ě), halucinace u opakovaných slov. Server-side výpočet
je deterministický, levný a garantuje invariant, na který se spaCy/Stanza
trénink spoléhá.

Pokud LLM nějaké pozice vrátí (proti instrukci), `compute_entity_positions` je
nejdřív resetne na `None` a vypočítá vlastní hodnoty — LLM-provided indexy
nikdy neuniknou do výstupu.

## Požadavky na hostitele

| Komponenta | Minimální verze | Pozn. |
|---|---|---|
| **NVIDIA driver** | **≥ 580** | CUDA 13.0 capable |
| **CUDA toolkit / runtime** | **≥ 13.0** | NVFP4 vyžaduje 5th-gen Tensor Cores plně podporované od CUDA 13 |
| GPU | Blackwell SM120 (B200, …) | Pro NVFP4 native; pro FP8/BF16 fallback stačí Hopper SM90 |
| Docker | s NVIDIA Container Toolkit | `--gpus all` musí fungovat |
| Disk | ≥ 200 GB | Pro model cache (Qwen 122B + Gemma 31B v NVFP4 ≈ 91 GB; HF download s rezervou) |

`./scripts/run_production.{sh,ps1}` automaticky spustí preflight check pomocí
`nvidia-smi` a odmítne pokračovat, pokud detekuje CUDA < 13. Pro override
(např. testování na FP8/BF16 mimo Blackwell): `SKIP_CUDA_CHECK=1 ./scripts/run_production.sh`.

## Ověřené technické pozadí (duben 2026)

| Model | Backend | Status |
|---|---|---|
| Qwen3.5-122B-A10B-NVFP4 | **SGLang** (primární) | ✅ Komunitní NVFP4 checkpoint (RedHatAI), native MoE FP4 podpora |
| Qwen3.5-122B-A10B-NVFP4 | vLLM (fallback) | ✅ Funguje s `--moe_backend=flashinfer_cutlass` |
| Gemma 4 31B NVFP4 | **SGLang** (primární) | ✅ NVIDIA oficiální checkpoint |
| Gemma 4 31B NVFP4 | vLLM (fallback) | ✅ NVIDIA oficiální checkpoint |

Pipeline používá **SGLang jako primární backend**, vLLM jako fallback při selhání.

**Default Docker images (oba CUDA 13 build):**
- SGLang: `lmsysorg/sglang:latest-cu130`
- vLLM:   `vllm/vllm-openai:latest`

Pokud změníš tagy v `.env`, ujisti se, že obě image jsou explicitně postavené
proti CUDA 13 (typicky tag obsahuje `-cu130` nebo `cuda13`). Jinak NVFP4
inference selže s `no kernel image is available for execution on the device`.

## HW a odhady

Cílový HW: **NVIDIA Blackwell B200** (1×, 2×, 4× nebo 8× karta). Reálný runtime
zjistíš **production smoke testem** níže — žádné statické odhady, co naměříš,
to dostaneš.

### Production smoke test (pure empirický odhad)

⚠️ **Předchozí "orientační" tabulky byly v praxi výrazně mimo skutečný čas.**
Místo statického odhadu pipeline obsahuje **production smoke test**, který
provede pipeline na vzorku 100 itemů z každého datasetu na **přesně tom
hardware, kde poběží produkce**, a lineárně extrapoluje:

```
T_full = T_smoke × (full_chars / smoke_chars)
```

Žádné teoretické scaling factory, žádné GPU-class projekce, žádné
"konstantní model-load" odhady. Co naměříš, to dostaneš v odhadu.

#### Použití (one-liner script)

Default běh přes **docker** (LLM servery + labeler v kontejnerech):

```bash
# Linux / WSL2:
./scripts/run_smoke_test.sh                                     # default 1x B200, 100 items
HW_LABEL="2x B200" ./scripts/run_smoke_test.sh
SAMPLES_PER_DATASET=100 ./scripts/run_smoke_test.sh             # přesnější odhad

# Windows:
.\scripts\run_smoke_test.ps1
.\scripts\run_smoke_test.ps1 -HwLabel "4x B200" -SamplesPerDataset 100
```

Alternativně **native** mód (bez dockeru — Python labeler přímo na hostu).
Default chování: skript SÁM nainstaluje SGLang (pokud chybí), pro každou
fázi spustí LLM server, počká na ready, a po fázi server zastaví:

```bash
# Jednořádek: skript udělá VŠE (install + start + run + stop):
RUNTIME=native ./scripts/run_smoke_test.sh
.\scripts\run_smoke_test.ps1 -Runtime native
```

Co se v plně automatizovaném native módu (`NATIVE_AUTO_LLM=1` default) děje:

1. **Auto-install SGLang** přes `./scripts/install_sglang.sh` (`.ps1` na Windows)
   * Verifikace CUDA 13+, Python 3.10-3.12
   * `pip install sglang[all] flashinfer-python` s CUDA 13 wheels
   * `pip install -r requirements.txt` (labeler deps)
2. **Auto-start SGLang serveru** pro každou fázi přes `./scripts/start_native_llm.sh`
   * Extract / revise → `RedHatAI/Qwen3.5-122B-A10B-NVFP4` na portu 8000
   * Critic → `nvidia/Gemma-4-31B-IT-NVFP4` na portu 8001
   * Background process, PID v `/tmp/sglang_<port>.pid`
   * Polling `/v1/models` health check, max 20 min ready timeout
3. **Python labeler smoke-test --phase X** se spustí proti běžícímu LLM
4. **Auto-stop SGLang** po fázi přes `./scripts/stop_native_llm.sh`
   * Mezi fázemi se uvolní VRAM (Qwen → stop → Gemma start)
5. **Trap cleanup** na EXIT — i při Ctrl+C / fail se zbývající servery zastaví

Override:
```bash
# Manual mode — user si LLM zapne sám, skript jen čeká
NATIVE_AUTO_LLM=0 RUNTIME=native ./scripts/run_smoke_test.sh

# Vlastní LLM endpoint (např. server na jiném hostu)
RUNTIME=native \
    EXTRACTOR_BASE_URL=http://10.0.0.5:8000/v1 \
    CRITIC_BASE_URL=http://10.0.0.5:8001/v1 \
    ./scripts/run_smoke_test.sh
# Pokud endpoint odpoví health checku, skript nepouští vlastní server

# Pinned SGLang verze pro reproducibility
SGLANG_VERSION=0.4.5 ./scripts/install_sglang.sh

# Použít virtualenv místo system Python
USE_VENV=1 ./scripts/install_sglang.sh
```

Native mód předpokládá:
- **Python 3.10-3.12** v PATH (SGLang nepodporuje 3.13+)
- **NVIDIA driver ≥580** + CUDA 13.0 capable (B200 NVFP4 vyžaduje)
- **PyTorch 2.7+** s CUDA 13 build (auto-install zkusí najít; pokud selže, install ručně:
  `pip install torch --index-url https://download.pytorch.org/whl/cu130`)
- **HF_TOKEN** v env nebo `.env` (pro gated modely jako Gemma — `start_native_llm.sh`
  preflight check varuje pokud chybí)
- **Disk space**: ~90 GB pro Qwen3.5-122B-A10B-NVFP4, ~20 GB pro Gemma-4-31B-IT-NVFP4
  v `$HF_HOME` (default `~/.cache/huggingface`)
- **Síťový přístup** na PyPI + `flashinfer.ai/whl/cu130/` (auto-install)
- **Default endpoint**: `http://localhost:8000/v1` (extractor + revisor) a
  `http://localhost:8001/v1` (critic)

Troubleshooting native instalace:

| Symptom | Fix |
|---|---|
| `ImportError: flashinfer ... cu130` | `FLASHINFER_INDEX_URL=https://flashinfer.ai/whl/cu130/torch2.5/ ./scripts/install_sglang.sh` |
| `Repository not found` (Gemma) | `export HF_TOKEN=hf_xxx` (token musí mít přístup k modelu) |
| `No space left on device` | Uvolnit `~/.cache/huggingface` nebo `export HF_HOME=/větší/disk/hf` |
| `CUDA error: no kernel image` | Upgrade NVIDIA driveru ≥580 (CUDA 13.0 capable) |
| `port 8000 already in use` | `./scripts/stop_native_llm.sh 8000` nebo `lsof -i :8000` |

Docker mód orchestruje sekvenčně — pro každou fázi:

1. `docker compose --profile <extract|critic|revise>-<sglang|vllm> run --rm`
   spustí LLM server (Qwen / Gemma / Qwen revisor) + smoke-test labeler v jednom up.
2. Labeler vezme prvních N (default 100) itemů per dataset, pošle je na LLM,
   změří wall-clock + počet zpracovaných itemů + input chars.
3. Container down → další fáze začne s čerstvou VRAM (1 model najednou,
   stejný design jako produkce).

Pro finalize: bez LLM, jen `docker compose --profile finalize run`.
Na konci report.

`HW_LABEL` je pouze **dokumentační text** v reportu — nepoužívá se v žádném
výpočtu. Pro odhad na jiném HW prostě spusť smoke znovu na tom HW.

#### Manuální použití (per-fáze)

Pokud chceš jednotlivé fáze sám orchestrovat:

```bash
docker compose --profile extract-sglang run --rm \
  -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke \
  labeler-extract-sglang \
  smoke-test --phase extract --hw-label "1x B200" --samples-per-dataset 50
# (analogicky critic / revise / finalize)
docker compose --profile finalize run --rm \
  -e SMOKE_CHECKPOINT_DIR=/app/checkpoints/smoke \
  labeler-finalize \
  smoke-test --phase report --hw-label "1x B200"
```

#### Smoke test artefakty

- `checkpoints/smoke/smoke_timings.json` — raw per-fázové měření
- `checkpoints/smoke/smoke_test_report.md` — Markdown souhrn (čitelný)
- `checkpoints/smoke/smoke_test_report.json` — strojový export

Smoke output je v **subfolderu `checkpoints/smoke/`** — nezasáhne to
produkční `checkpoints/extractions.jsonl` apod. (override přes
`SMOKE_CHECKPOINT_DIR` env).

#### Jak je odhad přesný

Smoke test běží **na stejném hardware, na stejných modelech, na 50
reálných položkách z každého datasetu**. Extrapolační vzorec je čistě
lineární:

| Fáze | Vzorec | Důvod |
|------|--------|-------|
| extract / critic / revise | `T_smoke × (full_chars / smoke_chars)` | LLM cena škáluje s tokeny ≈ chars |
| revise | × `nonok_rate` | jen subset itemů potřebuje revizi |
| finalize | `T_smoke × (full_items / smoke_items)` | CPU only, per-item operace |

Co se NEpoužívá:
- ❌ Teoretické GPU scaling factory (1.8x, 3.4x …) — smoke běží na tom HW, na kterém poběží produkce, takže žádné projekce nejsou potřeba
- ❌ Konstantní "model load" overhead — to platí uživatel jednou při `docker compose up` MIMO smoke wall-clock; v reportu jen volitelně přes `--docker-startup-s`
- ❌ Hardcoded throughput tabulky pro různé GPU classes

Co JE měřeno:
- ✅ Wall-clock přímo na cílovém HW
- ✅ Critic non-ok rate (typicky 25–35 %) — neguessuje se
- ✅ Per-dataset items + chars (pro lineární extrapolaci)

Konfidence interval ±2h:
- Bootstrap z per-item timingů, 95 % CI
- 5 % floor pro neměřitelné produkční faktory (HW noise, batch scheduler, bias vzorku)
- sqrt-of-sum-of-squares přes fáze
- Cap na ±2h s warningem pokud target nesplněn (doporučí zvýšit `--samples-per-dataset`)

### Live progress + ETA v logu

Každá fáze loguje při startu, periodicky během běhu a na konci:

```
[extract] start: 60000 itemů ke zpracování (progress + ETA se loguje každých 1000 itemů a na konci)
[extract] 1000/60000 (1.7%) errors=0 skipped=12 elapsed=1min 15s ETA=1h 14min (47.4 it/min)
[extract] 2000/60000 (3.3%) errors=1 skipped=24 elapsed=2min 28s ETA=1h 12min (47.6 it/min)
...
[extract] 60000/60000 (100.0%) errors=12 skipped=140 elapsed=1h 22min ETA=0s (730.5 it/min)
[extract] FINISHED: 60000/60000 (errors=12, skipped=140) total time 1h 22min, 730.5 it/min
```

Co znamenají sloupce:
- `1000/60000 (1.7%)` — kolik itemů hotovo z celku, jako procento
- `errors=N` — kolik itemů selhalo i po self-repair retry
- `skipped=N` — kolik itemů využilo hash cache (identický text už byl labelován)
- `elapsed=Xh Ymin` — jak dlouho fáze běží
- `ETA=Xh Ymin` — odhad zbývajícího času (z průměrné rychlosti reálně zpracovaných itemů)
- `(47.4 it/min)` — aktuální throughput

Adaptivní logging — frekvence se ladí podle velikosti datasetu (cca 25 log řádků per fáze).

### Why NVFP4 a TP setup

**NVFP4 výhody na Blackwell SM120:**
- ~1.32× rychlejší inference (5th-gen Tensor Cores native FP4)
- 4× méně VRAM (Gemma 4 31B: 16 GB vs 62 GB BF16, Qwen3.5-122B: 75 GB vs 234 GB BF16)
- < 1 % ztráta přesnosti (validováno NVIDIA + community benchmarks)

**Rozdělení karet (192 GB celkem):**
- Qwen3.5-122B-NVFP4 (75 GB váhy) → TP=2 (rozprostře přes obě karty + obě KV cache)
- Gemma 4 31B-NVFP4 (16 GB váhy) → TP=1 (sám na 1 kartě, KV cache do volné rezervy)
- Mezi fázemi compose down → uvolnění VRAM → další model nahoře startuje s plnou kapacitou

## Rychlý start

### 1. Příprava

```bash
cp .env.example .env
# Uprav .env — zejména HF_TOKEN pro Gemma 4 (gated model)

mkdir -p data
# Nakopíruj své Stage export(y) — libovolný počet souborů, libovolná jména
# (`export-*-latest.json`). Jakýkoli jiný *.json zpracuje generický reader
# (auto-detekce textového pole).
#
# Doména je defaultně "produktový / obsahový katalog"; per soubor ji přepíšeš
# volitelným data/labeler_config.json:
#   { "export-mujdataset-latest.json": { "domain": "e-shop s elektronikou" } }
cp /cesta/k/exportu/export-mujdataset-latest.json data/
```

### 2. Lokální smoke test (RTX 4070 12 GB nebo podobné)

```bash
# Linux / WSL2:
./scripts/run_test.sh

# Windows PowerShell:
powershell -ExecutionPolicy Bypass -File .\scripts\run_test.ps1
```

Vytvoří 3 labely z každého datasetu pomocí mini modelů. Architektura je
**identická s produkcí** — 4 fáze, vždy 1 model na GPU:

| Fáze | Profil          | Model na GPU            | Co dělá                                                    |
|------|-----------------|-------------------------|------------------------------------------------------------|
| 1    | `test-extract`  | Qwen3-0.6B (extractor)  | Primární extrakce                                          |
| 2    | `test-critic`   | Gemma-3-1b (critic)     | Verdikt + reasoning + corrected                            |
| 2b   | `test-revise`   | Qwen3-0.6B (revisor)    | Projde non-ok kritiky a doplní `extractor_revised`         |
| 3    | `finalize`      | žádný                   | Merge + stats                                              |

Mezi fázemi `compose down` → uvolnění VRAM → `compose up` další fáze. Stejně
jako v produkci.

### 3. Produkční běh

#### Docker runtime (default)

```bash
# Stáhni modely (jednorázově, ~190 GB)
HF_TOKEN=hf_xxx ./scripts/download_models.sh

# Linux — všechny 4 fáze (extract → critic → revise → finalize):
./scripts/run_production.sh

# Single fáze:
./scripts/run_production.sh extract    # jen extract  (Qwen sám)
./scripts/run_production.sh critic     # jen critic   (Gemma sám)
./scripts/run_production.sh revise     # jen revise   (Qwen revisor sám)
./scripts/run_production.sh finalize   # jen finalize (bez LLM)

# Windows:
.\scripts\run_production.ps1
```

Mezi fázemi se vždy provede `compose down` → uvolní VRAM → další fáze.
Žádné dva modely na GPU souběžně.

#### Native runtime (bez dockeru, plně automatizovaný)

Pokud nechceš orchestraci přes docker compose, skript v native módu udělá
**všechno sám** — install SGLang + start/stop LLM server per fáze:

```bash
# Linux — celý pipeline jedním příkazem:
RUNTIME=native ./scripts/run_production.sh

# Single fáze:
RUNTIME=native ./scripts/run_production.sh extract
RUNTIME=native ./scripts/run_production.sh critic
RUNTIME=native ./scripts/run_production.sh revise
RUNTIME=native ./scripts/run_production.sh finalize

# Windows:
.\scripts\run_production.ps1 -Runtime native
```

Default `NATIVE_AUTO_LLM=1` chování:

| Krok | Co skript dělá |
|---|---|
| Pre-flight | `python -c "import sglang"` test |
| Auto-install (pokud chybí) | `./scripts/install_sglang.sh` — CUDA check, pip install SGLang + flashinfer + labeler deps |
| Per-phase start | `./scripts/start_native_llm.sh <model> <port>` — background, PID v `/tmp`, health check ready loop |
| Python labeler | `python -m labeler.main <phase>` přímo |
| Per-phase stop | `./scripts/stop_native_llm.sh <port>` — graceful SIGTERM, fallback SIGKILL |
| EXIT trap cleanup | Zbylé servery se zastaví i při Ctrl+C / fail |

Pokud chceš LLM server řídit ručně (vlastní deployment, k8s, Slurm),
nastav `NATIVE_AUTO_LLM=0`:

```bash
# 1) Sám si spusť LLM:
./scripts/start_native_llm.sh RedHatAI/Qwen3.5-122B-A10B-NVFP4 8000 1
# nebo přímo:
python -m sglang.launch_server --model-path ... --port 8000 --tp 1

# 2) Skript jen ověří endpoint + spustí labeler:
NATIVE_AUTO_LLM=0 RUNTIME=native ./scripts/run_production.sh extract
```

Standalone instalace SGLang (jednorázově, bez automatu):
```bash
./scripts/install_sglang.sh                                # default latest
SGLANG_VERSION=0.4.5 ./scripts/install_sglang.sh           # pinned
USE_VENV=1 ./scripts/install_sglang.sh                     # do .venv/
```

### 4. Taxonomy stage (fáze 0 — default auto)

Fázi 0 si `extract`/`pipeline` **pustí samy v auto módu** (best-effort —
viz výše). Spustit ji dopředu nebo změnit režim je volitelné:

```bash
# Vypnout discovery (16 originálních kategorií, žádné LLM volání navíc)
python -m labeler.main extract --taxonomy-mode off

# Spustit fázi 0 zvlášť dopředu (jinak ji extract pustí sám)
python -m labeler.main taxonomy --mode auto --samples-per-dataset 15

# User taxonomie z JSON
python -m labeler.main taxonomy --mode user \
    --user-taxonomy examples/entity_taxonomy.example.json

# Pak normální extract — automaticky použije entity_taxonomy.json
python -m labeler.main extract
```

### 5. Normalizační vrstva — běží AUTOMATICKY

Cleaning + normalizace běží **automaticky v každém běhu** (produkční
pipeline i smoke test). Nemusíš nic explicitně zapínat:

```bash
python -m labeler.main finalize
```

Default chování (default ON):
- ✅ Normalizace entit (label remap, fix offsetů, conflict resolve, drop překryvů)
- ✅ Strict offsets (drop entit s nedohledatelným offsetem)
- ✅ Auto-expand mentions (doplnění chybějících výskytů, jen safe entity)
- ✅ Drop strukturálních identifikátorů (URL/email/telefon dropnuté regexem)
- ✅ `normalization_report.json` s counters

Pro debugging můžeš jednotlivé kroky vypnout přes `--no-*` flagy:

```bash
python -m labeler.main finalize \
    --no-strict-offsets \      # ponechá entity s nedohledatelným offsetem
    --no-write-report          # přeskočí normalization_report.json
```

Vznikne `normalization_report.json` s counters (opravené offsety, doplněné výskyty, konflikty labelů, distribuce před/po).

## Výstup `finalize` (produkce)

```bash
python -m labeler.main finalize
```

V `checkpoints/` najdeš:

- `final_labels.jsonl` (+ `.json`) — silver dataset (NER + KEYPHRASE_* + relace + triples)
- `final_labels_synthetic.jsonl` (+ `.json`) — pokud měl extractor synthetic vzorky
- `normalization_report.json` — counters per item, distribuce labelů před/po
- `stats.json` — souhrnné statistiky

**Žádné spaCy split exporty.** Pokud potřebuješ pro spaCy NER training
jen hard NER labely (bez KEYPHRASE_*), filtruj si downstream:

```bash
jq -c 'select(.graph.entities[].category | test("^(PRODUKT|ORGANIZACE|MISTO|DATUM|HODNOTA|MATERIAL|OSOBA_ROLE|PRODUKT_TYP)$"))' \
    checkpoints/final_labels.jsonl > core_ner.jsonl
```

Smoke test (`--phase finalize`) navíc vyrobí **OWL ontologii** (.owl +
.ttl + volitelně .html přes pylode) jako ukázku formátu na malém vzorku.

Pokud z nějakého důvodu cleaning nechceš, vypni přes `--no-normalize`
(raw LLM extractor output). Není doporučeno pro produkční trénink.

### Výstupní soubory

| Soubor | Co obsahuje | Použití |
|---|---|---|
| `final_labels.jsonl` (+ `.json`) | Plný silver dataset (entity + relace + triples po cleaningu) | Knowledge Graph, NER/RE training (filtruj `jq` downstream), audit, debug |
| `final_labels_synthetic.jsonl` (+ `.json`) | Synthetic vzorky (pokud extractor vrátil `is_synthetic=true`) | Augmentace tréningových dat |
| `stats.json` | Souhrnné statistiky finalize fáze | Audit |
| `normalization_report.json` | Detailní counters cleaning vrstvy (offsets_fixed, relations_reversed_fixed, structural_identifiers_dropped, distribuce labelů před/po) | Audit kvality |

Pro spaCy NER training filtruj `final_labels.jsonl` downstream přes `jq` —
hard NER labely jsou: `PRODUKT, PRODUKT_TYP, ORGANIZACE, OSOBA_ROLE, MISTO,
DATUM, HODNOTA, MATERIAL`. KEYPHRASE_* labely jsou určené pro keyphrase
extraction, ne pro core NER training.

**Smoke test (`scripts/run_smoke_test.{sh,ps1}`) navíc vyrobí**
`final_labels.owl` + `.ttl` (+ `.html` pokud pylode) jako ukázku ontologie
na vzorku 100 itemů — pro validaci formátu před produkčním během.

### Bezpečnostní invarianty (deterministická validace)

Default běh `finalize` (cleaning automatický) garantuje tyto invarianty:

| Invariant | Řeší kde |
|---|---|
| `input_text[start_char:end_char] == entity.text` | `labeler/positions.py` + normalize fix |
| Žádné překrývající se entity | `_step_drop_overlaps` |
| Každá relace odkazuje jen na existující entity | `normalize_relations` |
| `subjectId != objectId` (žádné self-loops) | `normalize_relations` |
| Žádné duplicitní relace `(subject, relation, object)` | `normalize_relations` |
| `issued_by` má správný směr (PRODUKT → ORGANIZACE) | `relation_normalize` reverse fix |
| `located_in` neukazuje na URL | `relation_normalize` blacklist |
| `suitable_for` mezi nesmyslnými typy se dropne | `relation_normalize` type rules |
| `covers` s obecným objectem ("auto", "škoda") se dropne | `relation_normalize` unsafe terms |
| `triples` jsou VŽDY deterministicky regenerované z entities+relations | `SkoGraph._regenerate_triples_invariant` |
| Core NER export neobsahuje KEYPHRASE_* | `_split_record_for_spacy` |
| Data-driven label rules (`force_text_match`, `unsafe_texts` per label v taxonomy) | `_step_apply_taxonomy_text_rules` |

LLM může udělat chybu (špatný směr issued_by, halucinovaná triple, generický covers) — všechny tyto chyby zachytí deterministická vrstva ještě před tím, než se dostane do spaCy exportu.

## Struktura projektu

```
mudrc-labeler/
├── labeler/                          # Python core
│   ├── schemas.py                    # Pydantic: Entity/Relation/SkoGraph + validace
│   ├── prompts.py                    # Prompty pro extractor/critic + revisor
│   ├── llm_client.py                 # Async HTTP klient s retry + reasoning mode
│   ├── io_utils.py                   # JSONL writer, hash dedup, resume logika
│   ├── positions.py                  # Kódový výpočet start_char/end_char (anti-halucinace)
│   ├── datasets.py                   # Readery (Stage export + generic JSON)
│   ├── taxonomy.py                   # FÁZE 0 (default auto): auto/user discovery NER taxonomie
│   ├── extractor.py                  # FÁZE 1: extrakce
│   ├── critic.py                     # FÁZE 2: validace
│   ├── revise.py                     # FÁZE 2b: standalone revisor (feedback Gemma→Qwen)
│   ├── ner_re_split.py               # Split NER/RE (default ON): extract/critic/revise každé jako 2 volání
│   ├── normalize.py                  # Normalizační vrstva (label remapping, fix offsetů, auto-expand) — volitelná před finalize
│   ├── relation_normalize.py         # Relation type rules + reverse fix + drop type-mismatch
│   ├── finalize.py                   # FÁZE 3: merge + stats
│   ├── rdf_export.py                 # OWL/RDF export přes rdflib (.owl + .ttl + volitelně .html, jen smoke)
│   ├── smoke_test.py                 # Production smoke test — empirický time estimate
│   ├── config.py                     # Env config (EXTRACTOR_*, CRITIC_*, REVISOR_*)
│   └── main.py                       # CLI entry point
├── examples/
│   └── entity_taxonomy.example.json  # Příklad user-defined NER taxonomie
├── tests/
│   ├── test_normalize.py             # Unit testy normalizační vrstvy
│   ├── test_relation_normalize.py    # Unit testy relation normalize (neutral fixtures)
│   ├── test_spacy_export.py          # Unit testy spaCy split exporty
│   ├── test_taxonomy.py              # Unit testy taxonomy stage
│   ├── test_ner_re_split.py          # Unit testy split NER/RE (assemble_graph, critic merge)
│   ├── test_smoke_test.py            # Unit testy smoke test extrapolace
│   └── test_rdf_export.py            # Unit testy RDF / Turtle export
├── scripts/
│   ├── download_models.sh            # Stáhne HF modely
│   ├── run_production.sh             # Produkční běh (Linux) — docker | native
│   ├── run_production.ps1            # Produkční běh (Windows)
│   ├── run_smoke_test.sh             # Production smoke test (Linux) — empirický time estimate
│   ├── run_smoke_test.ps1            # Production smoke test (Windows)
│   ├── install_sglang.sh             # Auto-install SGLang + flashinfer (Linux, native mode)
│   ├── install_sglang.ps1            # Auto-install SGLang (Windows)
│   ├── start_native_llm.sh           # Start SGLang server v background (Linux)
│   ├── start_native_llm.ps1          # Start SGLang server (Windows)
│   ├── stop_native_llm.sh            # Stop SGLang server (Linux)
│   ├── stop_native_llm.ps1           # Stop SGLang server (Windows)
│   ├── run_test.sh                   # Lokální dev smoke test mini modelů (Linux)
│   └── run_test.ps1                  # Lokální dev smoke test mini modelů (Windows)
├── docker-compose.yml                # Produkce (SGLang + vLLM backend chain)
├── docker-compose.test.yml           # Override pro test (mini modely)
├── Dockerfile.labeler                # Python labeler kontejner
├── requirements.txt                  # httpx + pydantic (2 deps celkem)
├── .env.example                      # Šablona konfigurace
├── .gitattributes                    # LF line endings (Windows safety)
└── README.md                         # Tento soubor
```

## Formát výstupu

`final_labels.jsonl` — jeden záznam per řádek:

```json
{
  "item_id": "eshop-demo:12345",
  "source_dataset": "eshop-demo",
  "input_text": "Tento držák ACME Basic má chromový povrch.",
  "input_hash": "a1b2c3d4...",
  "is_synthetic": false,
  "graph": {
    "entities": [
      {"id": "e1", "text": "ACME Basic",
       "lemma": "ACME Basic", "category": "PRODUKT",
       "start_char": 12, "end_char": 22},
      {"id": "e2", "text": "chromový povrch",
       "lemma": "chromový povrch", "category": "KOMPONENTA",
       "start_char": 26, "end_char": 41}
    ],
    "relations": [
      {"subjectId": "e1", "objectId": "e2",
       "relation": "has_component",
       "relationCs": "má komponentu", "relationEn": "has_component"}
    ],
    "triples": [
      {"subject": "ACME Basic",
       "relation": "has_component",
       "object": "chromový povrch"}
    ]
  },
  "critic_status": "ok",
  "critic_reasoning": "Entity i relace odpovídají textu.",
  "extractor_model": "RedHatAI/Qwen3.5-122B-A10B-NVFP4",
  "critic_model": "nvidia/Gemma-4-31B-IT-NVFP4",
  "was_revised": false
}
```

`start_char` / `end_char` doplnil deterministický kód (`labeler.positions`),
ne LLM. Garantovaný invariant: `input_text[start_char:end_char] == text`.

### OWL ontology export (jen v rámci production smoke testu)

Plná **OWL 2 ontologie** se generuje **POUZE během production smoke testu**
(`smoke-test --phase finalize`), nikde jinde. Důvod: OWL výstup slouží
jako **ukázka schematu + dat na vzorku 100 itemů** — uživatel si ověří, že
ontologie vypadá jak má, než pustí celý produkční běh.

V **produkční pipeline se OWL negeneruje** — `finalize` zapíše JSONL/JSON
a stats, nic víc. Tím se vyhneme tomu, aby gigabajtové produkční datasety
měly další gigabajtové duplicate `.owl` exporty.

**Implementace**: hand-written serializery jsou pryč. Vše je postaveno
přes [rdflib](https://github.com/RDFLib/rdflib) (`requirements.txt`).
Modul [labeler/rdf_export.py](labeler/rdf_export.py) sestaví `rdflib.Graph`
s plným TBox + ABox a knihovna sama serializuje do `.owl` (RDF/XML) a
`.ttl` (Turtle). Volitelně se vygeneruje i `.html` dokumentace přes
[pyLODE](https://github.com/RDFLib/pyLODE) — `pip install pylode`
aktivuje. Bez pyLODE smoke test prostě HTML doc nevyrobí, `.owl` a `.ttl`
jsou tak jako tak.

**Co je v ontologii (vše, co máme — entity, relace, metadata):**

TBox (schema):
- `owl:Ontology` declaration s rdfs:label + rdfs:comment
- `:Document` (`owl:Class`) + `:Entity` (`owl:Class`, disjoint s Document)
- Per-label `owl:Class` jako `rdfs:subClassOf :Entity` (`:PRODUKT`,
  `:ORGANIZACE`, `:KEYPHRASE_BENEFIT`, …)
- Per-relation `owl:ObjectProperty` (`:issued_by`, `:has_component`,
  `:covers`, `:suitable_for`, `:part_of`, …) s `rdfs:domain :Entity`
  a `rdfs:range :Entity`
- `:hasEntity` (`owl:ObjectProperty`, Document → Entity)
- `owl:DatatypeProperty` pro `:sourceDataset`, `:inputText`,
  `:criticStatus`, `:wasRevised`, `:text`, `:lemma`, `:category`,
  `:startChar`, `:endChar` (správné xsd typy včetně `xsd:nonNegativeInteger`
  pro offsety)

ABox (data):
- Každý FinalLabel record → `owl:NamedIndividual` typu `:Document` s
  všemi datatype properties
- Každá entity → `owl:NamedIndividual` typu konkrétního label (subclass
  `:Entity`)
- `:hasEntity` linky Document → Entity
- Per-relation předikáty mezi entity individuals (`<ent1> :issued_by <ent2>`)

**Layout**: smoke vytvoří OBOJÍ:
- `final_labels.owl` / `.ttl` / `.html` — combined (všechny dokumenty,
  vhodné pro single-import do triple store / SPARQL endpoint)
- `final_labels_<source>.owl` / `.ttl` / `.html` — per-dataset split
  (každý source má vlastní ontologii)

Triggeruje se automaticky:

```bash
# Linux/WSL2:
./scripts/run_smoke_test.sh
# → po finalize fázi: checkpoints/smoke/final_labels.owl
#                   + checkpoints/smoke/final_labels.ttl
#                   + checkpoints/smoke/final_labels.html (pokud pylode)
#                   + per-source varianty

# Windows:
.\scripts\run_smoke_test.ps1
```

Ukázka výstupu (`checkpoints/smoke/final_labels.ttl`):

```turtle
@prefix : <http://mudrc.example/vocab#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://mudrc.example/ontology> a owl:Ontology ;
    rdfs:label "Mudrc Labeler Silver Dataset Ontology"@en .

:Document a owl:Class ;
    rdfs:label "Document"@en ;
    owl:disjointWith :Entity .

:Entity a owl:Class ;
    rdfs:label "Entity"@en .

:PRODUKT_TYP a owl:Class ;
    rdfs:subClassOf :Entity ;
    rdfs:label "PRODUKT_TYP"@cs .

:ORGANIZACE a owl:Class ;
    rdfs:subClassOf :Entity ;
    rdfs:label "ORGANIZACE"@cs .

:issued_by a owl:ObjectProperty ;
    rdfs:domain :Entity ;
    rdfs:range :Entity .

:hasEntity a owl:ObjectProperty ;
    rdfs:domain :Document ;
    rdfs:range :Entity .

# Datatype properties: :sourceDataset, :inputText, :criticStatus,
# :wasRevised, :text, :lemma, :category, :startChar, :endChar (s xsd typy)
# ...

<urn:mudrc:doc:demo:abc> a owl:NamedIndividual, :Document ;
    :sourceDataset "demo" ;
    :inputText "Životní pojištění od ACME." ;
    :wasRevised false ;
    :hasEntity <urn:mudrc:ent:demo:abc:e1>, <urn:mudrc:ent:demo:abc:e2> .

<urn:mudrc:ent:demo:abc:e1> a owl:NamedIndividual, :PRODUKT_TYP ;
    :text "Životní pojištění" ;
    :lemma "životní pojištění" ;
    :startChar "0"^^xsd:nonNegativeInteger ;
    :endChar "17"^^xsd:nonNegativeInteger ;
    :issued_by <urn:mudrc:ent:demo:abc:e2> .

<urn:mudrc:ent:demo:abc:e2> a owl:NamedIndividual, :ORGANIZACE ;
    :text "ACME" ;
    :startChar "21"^^xsd:nonNegativeInteger ;
    :endChar "25"^^xsd:nonNegativeInteger .
```

Výstup je validní OWL 2 DL — importovatelný do Protégé, Apache Jena,
GraphDB, Stardog, Virtuoso, RDFLib, Owlready2 a dalších.

**Filtrace anti-halucinace**: OWL export aplikuje stejné schema-validity
checky jako JSONL (label regex `^[A-Z][A-Z0-9_]{0,49}$`, relation
`^[a-z][a-z0-9_]{0,49}$`, self-loop drop, dangling subject/object drop)
— žádné invalid třídy/property se do ontologie nedostanou.

## Tuning paralelizace a max_tokens

Defaulty jsou laděné na produkční HW (B200) a na labelování dlouhých
odstavců s mnoha entitami. Pro slabší HW nebo specifické use-case si je lze
přepsat v `.env`.

### Paralelizace

```bash
# .env — kolik HTTP requestů labeler pošle paralelně na endpoint
EXTRACTOR_CONCURRENCY=32     # default 32; pro slabší HW sniž
CRITIC_CONCURRENCY=64        # default 64
REVISOR_CONCURRENCY=32       # default = EXTRACTOR_CONCURRENCY

# Backend (sglang/vllm) max-running-requests — kolik requestů zpracuje server zaráz
EXTRACTOR_BATCH_SIZE=32      # default 32; vyšší = lepší throughput, víc VRAM
CRITIC_BATCH_SIZE=64         # default 64
```

**Pravidlo palce**: `*_CONCURRENCY` ≈ `*_BATCH_SIZE`. Concurrency výrazně vyšší
než batch jen plní queue na serveru (žádný throughput benefit, jen latency).
Concurrency nižší než batch znamená, že server nepoužije celou kapacitu.

### Max output tokens

Pro labelování dlouhých odstavců s mnoha entitami a relacemi je třeba dost
prostoru pro výstupní JSON. Defaulty jsou velkorysé:

```bash
# .env
EXTRACTOR_MAX_TOKENS=24576     # default 24k; ~7-8k na 30+ entit
CRITIC_MAX_TOKENS=32768        # default 32k; reasoning ON potřebuje extra prostor
REVISOR_MAX_TOKENS=24576       # default = EXTRACTOR_MAX_TOKENS

# Context window (input + output dohromady) na backendu
EXTRACTOR_CONTEXT_LEN=65536    # default 64k — pohodlně pojme odstavec + velký output
CRITIC_CONTEXT_LEN=65536       # default 64k
```

**Proč tak velké?** Critic má `CRITIC_REASONING=true` defaultně, takže Gemma
dostane navíc thinking trace (klidně 8-16k tokenů) před vlastním JSON
výstupem. Extractor má reasoning OFF, ale s 30+ entitami a relacemi může mít
sám JSON přes 6 kB. Bezpečná rezerva = nezkrácený výstup = nezhoupnuté itemy.

**VRAM dopad**: KV cache škáluje s `context_length × batch_size`. Pokud ti
dochází VRAM (OOM při startu nebo během běhu), v tomto pořadí:
1. Sniž `EXTRACTOR_BATCH_SIZE` / `CRITIC_BATCH_SIZE` (z 32/64 → 16/32)
2. Sniž `*_CONTEXT_LEN` (z 65536 → 32768)
3. Sniž `mem-fraction-static` v compose (z 0.85 → 0.75)

### Chunkování dlouhých textů

```bash
CHUNK_MAX_CHARS=12000   # default 12k znaků (~3-4k tokenů); větší → méně chunků
                        # ale větší input do LLM
```

Texty delší než `CHUNK_MAX_CHARS` se ve fázi 1 dělí na chunky podle vět.
Ve fázi 3 (finalize) se chunky stejného dokumentu spojí zpět do jednoho
záznamu a entity se přemapují na merged text (s deduplikací a recompute pozic).

### Timeout + retries

```bash
LLM_TIMEOUT_S=1800       # default 1800s (30 min) — velkorysé pro reasoning + velký output
LLM_MAX_RETRIES=5        # default 5 — exponential backoff na transient errors
```

## Troubleshooting

**"CUDA error: no kernel image is available for execution on the device"**
→ Driver / CUDA toolkit / image jsou nekompatibilní s Blackwell SM120 + NVFP4.
Pipeline VYŽADUJE **CUDA 13.0+ runtime** + NVIDIA **driver ≥ 580**. Zkontroluj:
1. `nvidia-smi` v hlavičce ukazuje `CUDA Version: 13.x` — pokud ne, upgrade
   driver (https://www.nvidia.com/Download/index.aspx).
2. Image tagy v `.env` obsahují `-cu130` (default `lmsysorg/sglang:latest-cu130`,
   `vllm/vllm-openai:latest`).
3. `./scripts/run_production.sh` má preflight check který tohle ověří před
   nahozením kontejnerů. Pro override: `SKIP_CUDA_CHECK=1 ./scripts/run_production.sh`.

**WARNING: "Your GPU does not have native support for FP4"**
→ Marlin fallback v rámci samotného backendu. Na B200 Blackwell by se to
objevit nemělo — pokud ano, ověř že máš správnou stable image (viz výše)
a aktualizovaný NVIDIA driver.

**Gemma 4 nejde stáhnout (401/403)**
→ Je to gated model. Jdi na https://huggingface.co/google/gemma-4-31B-it,
odsouhlas licenci, vygeneruj token na huggingface.co/settings/tokens,
dej ho do `.env` jako `HF_TOKEN=hf_...`. NVFP4 verze (`nvidia/Gemma-4-31B-IT-NVFP4`)
vyžaduje stejný souhlas.

**Server se nahrává 10+ minut**
→ Normální pro Qwen3.5-122B, první start zkompiluje CUDA kernely.
Pokud na stejném MODEL_CACHE_DIR spustíš znovu, bude to ~2 min.

**První (studený) běh: kontejner se strhne / `rc=1` / "nestihlo to doběhnout"**
→ První stažení vah z HF (Qwen3.5-122B-NVFP4 ~80 GB) může přeleznout docker
healthcheck okno → docker prohlásí extractor/critic za `unhealthy` a `docker
compose run` ho strhne (smoke i produkce pak hlásí `rc=1` a zkouší další
backend). Default okno je ~90 min (`LLM_HEALTH_START_PERIOD=3600s` + retries).
Na pomalé lince zvyš v `.env`, např. `LLM_HEALTH_START_PERIOD=7200s` (MUSÍ mít
jednotku). Po prvním stažení je model v `./models` cache → další starty jsou
rychlé a velká hodnota je nezpomaluje (healthy se hlásí hned po prvním
úspěšném checku). Compose se čte čerstvý → změna nevyžaduje rebuild.

**OOM při startu**
→ Sniž `EXTRACTOR_BATCH_SIZE` z 32 na 16, nebo `mem-fraction-static` z 0.85 na 0.75.

**Labeler končí s validation error**
→ Pipeline má self-repair (1 retry). Neúspěšné itemy najdeš v logu
jako `[id] Extrakce selhala i po self-repairu`. Resume vyhodí pouze
tyhle problematické itemy a ostatní už jsou v JSONL.

## Limitace

- Critic může mít false positives (zamítne správné extrakce) nebo negatives
  (nechá projít špatné). Revision rate 5-15 % je typické.
- Syntetické texty jsou z vice rizikové než reálné — proto samostatný dataset.
- LLM pipeline produkuje **training data**, ne final production output.
  Mudrc má svou deterministickou Stanza/spaCy pipeline pro runtime inference.
