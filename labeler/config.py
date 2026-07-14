"""
Konfigurace pipeline. Všechny defaulty jsou přes environment proměnné,
které docker-compose nastaví z .env souboru.

Tři typy LLM rolí:
  - EXTRACTOR_*  primární extrakce (Qwen3.5-122B)
  - CRITIC_*     nezávislá validace (Gemma 4 31B)
  - REVISOR_*    re-evaluace kritiky — defaultně je shodný s extractorem
                 (Qwen, který popisek původně labeloval, se k němu sám vrátí
                 a rozhodne, zda kritiku zapracuje). Když chceš revizi pouštět
                 na jiném modelu, přepiš REVISOR_BASE_URL / REVISOR_MODEL.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

# Manual .env fallback (since python-dotenv is not in requirements.txt)
# This ensures native runs (without docker compose) still respect .env
_env_path = Path(".env")
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for _line in f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            if "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip()
                # Strip quotes if present
                if (_v.startswith('"') and _v.endswith('"')) or (_v.startswith("'") and _v.endswith("'")):
                    _v = _v[1:-1]
                # Only set if not already in environment (environment has priority)
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

def _getenv_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# Default REVISOR_* hodnoty se odvozují z EXTRACTOR_* (revizor = extractor),
# pokud uživatel REVISOR_* neexplicitně neoverridne.
_DEFAULT_EXTRACTOR_BASE_URL = os.environ.get(
    "EXTRACTOR_BASE_URL", "http://extractor:8000/v1"
)
_DEFAULT_EXTRACTOR_MODEL = os.environ.get(
    "EXTRACTOR_MODEL", "RedHatAI/Qwen3.5-122B-A10B-NVFP4"
)


@dataclass(frozen=True)
class Config:
    # Endpointy LLM serverů
    extractor_base_url: str = _DEFAULT_EXTRACTOR_BASE_URL
    extractor_model: str = _DEFAULT_EXTRACTOR_MODEL
    critic_base_url: str = os.environ.get("CRITIC_BASE_URL", "http://critic:8001/v1")
    critic_model: str = os.environ.get("CRITIC_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")
    # Revisor = LLM, který rozhodne, zda kritiku Gemmy zapracuje. Default
    # = stejný endpoint a model jako extractor (Qwen3.5).
    revisor_base_url: str = os.environ.get(
        "REVISOR_BASE_URL", _DEFAULT_EXTRACTOR_BASE_URL
    )
    revisor_model: str = os.environ.get("REVISOR_MODEL", _DEFAULT_EXTRACTOR_MODEL)

    extractor_reasoning: bool = _getenv_bool("EXTRACTOR_REASONING", False)
    critic_reasoning: bool = _getenv_bool("CRITIC_REASONING", True)
    revisor_reasoning: bool = _getenv_bool(
        "REVISOR_REASONING", _getenv_bool("EXTRACTOR_REASONING", False)
    )

    # Concurrency = počet HTTP requestů, které labeler pošle paralelně na endpoint.
    # Defaulty jsou laděné na produkční backend (sglang / vLLM s velkou batch
    # capacity); pro slabší HW sniž v .env. Concurrency by měla zhruba odpovídat
    # backend `max-running-requests` (EXTRACTOR_BATCH_SIZE / CRITIC_BATCH_SIZE),
    # výrazně vyšší klient číslo jen plní queue na serveru.
    extractor_concurrency: int = int(os.environ.get("EXTRACTOR_CONCURRENCY", "32"))
    critic_concurrency: int = int(os.environ.get("CRITIC_CONCURRENCY", "64"))
    revisor_concurrency: int = int(
        os.environ.get(
            "REVISOR_CONCURRENCY", os.environ.get("EXTRACTOR_CONCURRENCY", "32")
        )
    )

    max_retries: int = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    # Per-request read timeout pro HTTP klient. NON-streaming response =
    # server drží connection otevřenou až do dokončení generace.
    #
    # Matematika: při 32 paralelních requestech sdílí throughput backendu.
    # Pro Qwen3.5-122B-A10B na B200 je aggregate ~1300 tok/s → per-request
    # ~40 tok/s. EXTRACTOR_MAX_TOKENS=24576 → ~10 min per request v plné
    # paralelizaci → s timeout=600 race condition (ReadTimeout retries).
    #
    # Default 1800s (30 min) dává 3× headroom i v nejhorším případě
    # (všech 32 requestů hits max_tokens). Pokud máš nižší CONCURRENCY
    # nebo MAX_TOKENS, můžeš si snížit přes LLM_TIMEOUT_S env var.
    request_timeout_s: float = float(os.environ.get("LLM_TIMEOUT_S", "1800"))

    # Velkorysé max_tokens defaulty — pro labelování celých dlouhých odstavců
    # s mnoha entitami a relacemi. Output JSON s 30+ entitami a triplety
    # se může natáhnout na ~8 kB textu (~3-4 k tokenů); s reasoning ON
    # (critic) ještě + thinking trace. Defaulty 24 kB / 32 kB dávají bezpečnou
    # rezervu, aby nedošlo k uťatí výstupu.
    extractor_max_tokens: int = int(os.environ.get("EXTRACTOR_MAX_TOKENS", "24576"))
    critic_max_tokens: int = int(os.environ.get("CRITIC_MAX_TOKENS", "49152"))
    revisor_max_tokens: int = int(
        os.environ.get(
            "REVISOR_MAX_TOKENS", os.environ.get("EXTRACTOR_MAX_TOKENS", "24576")
        )
    )

    data_dir: Path = Path(os.environ.get("DATA_DIR", "/app/data"))
    checkpoint_dir: Path = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))

    # Fáze 0 (taxonomy auto-discovery): kolik ukázek per dataset/doména se
    # pošle LLM na návrh jednotné taxonomie. JEDNORÁZOVÁ konstanta nezávislá
    # na velikosti datasetu (auto-discovery sampluje fixní počet, neškáluje se).
    # Default 10 — vyšší = širší pokrytí domén, ale větší prompt (pozor na
    # context window). Řídí produkci i smoke test stejně.
    taxonomy_samples_per_dataset: int = int(
        os.environ.get("TAXONOMY_SAMPLES_PER_DATASET", "10")
    )

    # --- Fáze 0 / Pilíř A: disjointness audit taxonomie --------------------
    # Po návrhu taxonomie (discover_taxonomy_auto) proběhne automatický audit,
    # který na held-out vzorcích měří, jak často model neumí rozhodnout mezi
    # dvěma labely (konfuze), a zaměňované páry sám zostří (přepíše description
    # + decision_rule + counter_examples). Plně automatické, žádný člověk, žádná
    # hard-coded slova. Best-effort: když audit selže, ponechá se taxonomie
    # z návrhu (běh se nezastaví).
    taxonomy_audit_enabled: bool = _getenv_bool("TAXONOMY_AUDIT_ENABLED", True)
    # Kolik kol audit maximálně proběhne (každé kolo = 1 měření + zostření párů).
    taxonomy_audit_max_rounds: int = int(
        os.environ.get("TAXONOMY_AUDIT_MAX_ROUNDS", "3")
    )
    # Cílová konfuze (průměrná nejistota top-1 labelu, 0-1). Pod tímto prahem
    # se taxonomie považuje za dostatečně disjunktní a audit skončí.
    taxonomy_audit_target_score: float = float(
        os.environ.get("TAXONOMY_AUDIT_TARGET_SCORE", "0.15")
    )
    # Kolik nejzaměňovanějších párů se zostří v jednom kole.
    taxonomy_audit_max_pairs: int = int(
        os.environ.get("TAXONOMY_AUDIT_MAX_PAIRS", "2")
    )
    # Minimální počet zaměněných spanů, aby se pár vůbec považoval za problém
    # (chrání před zostřováním na základě jednoho náhodného výskytu).
    taxonomy_audit_min_support: int = int(
        os.environ.get("TAXONOMY_AUDIT_MIN_SUPPORT", "2")
    )
    # Strukturální úpravy (rozděl / sluč) — mění POČET labelů. Default OFF:
    # audit napřed jen zostřuje hranice; rozdělování/slučování se zapíná až
    # když z prvních běhů uvidíš čísla. Zapni přes TAXONOMY_AUDIT_STRUCTURAL_EDITS.
    taxonomy_audit_structural_edits: bool = _getenv_bool(
        "TAXONOMY_AUDIT_STRUCTURAL_EDITS", False
    )

    # --- Fáze 3 (finalize): korpusové hlasování o kategoriích ---------------
    # Finalize jako jediná fáze vidí celý korpus → statistika rozdělení
    # kategorií per lemma opraví menšinové labely (stejná entita jednou
    # PRODUKT, jinde PRODUKT_TYP → sjednotí se na dominantní). Plně
    # data-driven, žádná hard-coded slova; nejednoznačné klíče se NEmění.
    # Stejné statistiky řídí i konflikt/auto-expand rozhodování v normalize.
    category_vote_enabled: bool = _getenv_bool("CATEGORY_VOTE_ENABLED", True)
    # Minimální podíl dominantní kategorie (0-1), aby se menšina přelabelovala.
    category_vote_dominance: float = float(
        os.environ.get("CATEGORY_VOTE_DOMINANCE", "0.8")
    )
    # Minimální počet výskytů lemmatu v korpusu, aby hlas platil.
    category_vote_min_support: int = int(
        os.environ.get("CATEGORY_VOTE_MIN_SUPPORT", "3")
    )
    # Auto-expand: výraz obsažený ve VÍCE než tomto podílu dokumentů je
    # statistické generikum / funkční slovo → nikdy se neexpanduje.
    # (Nahrazuje dřívější hard-coded stopword seznam v normalize.)
    auto_expand_max_df_ratio: float = float(
        os.environ.get("AUTO_EXPAND_MAX_DF_RATIO", "0.5")
    )

    # --- Fáze 3 (finalize): deterministická lemmatizace přes Stanza ---------
    # LLM vrací jen entity + relace (porozumění); lemma je mechanická úloha
    # → počítá ji Stanza deterministicky (stejně jako pozice počítá
    # positions.py). Stanza se používá VÝHRADNĚ na lemma — žádné POS filtry.
    # Best-effort: bez nainstalované stanzy / modelu se krok přeskočí
    # (ponechají se LLM lemmata) a běh pokračuje.
    stanza_lemma_enabled: bool = _getenv_bool("STANZA_LEMMA_ENABLED", True)
    # Jazyk Stanza modelu (konfigurace, ne hardcode — ~70 podporovaných jazyků).
    stanza_lang: str = os.environ.get("STANZA_LANG", "cs")
    stanza_use_gpu: bool = _getenv_bool("STANZA_USE_GPU", False)

    # Split NER/RE: každou fázi (extract / critic / revise) rozdělit na DVĚ
    # samostatná LLM volání — jedno na entity (NER), druhé na relace (RE) —
    # místo jednoho volání nad celým grafem. Model se pak soustředí na jednu
    # úlohu (lepší kvalita, méně malformed relací). Cena: 2× LLM volání na
    # fázi → delší běh. Default ON; vypni přes SPLIT_NER_RE=false.
    split_ner_re: bool = _getenv_bool("SPLIT_NER_RE", True)

    # --- Anti-runaway pro split NER/RE -------------------------------------
    # Regrese po zavedení splitu: NER volání (combined prompt + přílepek "POUZE
    # ENTITY") se pod plnou dávkou občas zacyklí a generuje až do max_tokens →
    # useknutý JSON → reject. Combined volání (SPLIT_NER_RE=false) tím netrpí.
    # Dvě cílené brzdy JEN pro split volání (combined cesta beze změny):
    #
    # 1) Menší token strop pro NER — entity jsou malý JSON (typicky < 3k tokenů
    #    i pro hustý chunk), takže 8192 je 3× rezerva pro legit výstup a zároveň
    #    bounduje případný runaway na zlomek 24576 (rychlejší fail, míň compute).
    extractor_ner_max_tokens: int = int(
        os.environ.get("EXTRACTOR_NER_MAX_TOKENS", "8192")
    )
    # 2) Repetition penalty na split volání (NER i RE) — rozbije repetiční
    #    smyčku, kvůli které model nedojede k zavírací závorce. Mírná hodnota
    #    1.05 má minimální dopad na kvalitu JSON; 1.0 = vypnuto. Combined cesta
    #    ji NEdostává (je prověřená beze změny).
    extractor_split_repetition_penalty: float = float(
        os.environ.get("EXTRACTOR_SPLIT_REPETITION_PENALTY", "1.05")
    )
    # Critic split: stejná anti-runaway brzda. Critic jede reasoning=ON (thinking
    # trace) a na SGLangu (bez gemma4 reasoning-parseru) se thinking občas zacyklí
    # a dojede do max_tokens → useknutý JSON → reject. Rep penalty smyčku rozbije.
    # 1.0 = vypnuto. (Tvrdší alternativa: úplně vypnout thinking přes
    # CRITIC_REASONING=false — critic pak vrací JSON přímo, bez thinkingu.)
    critic_split_repetition_penalty: float = float(
        os.environ.get("CRITIC_SPLIT_REPETITION_PENALTY", "1.05")
    )

    extraction_jsonl_name: str = "extractions.jsonl"
    critic_jsonl_name: str = "critic_results.jsonl"
    final_jsonl_name: str = "final_labels.jsonl"

    extraction_synthetic_jsonl_name: str = "extractions_synthetic.jsonl"
    critic_synthetic_jsonl_name: str = "critic_results_synthetic.jsonl"
    final_synthetic_jsonl_name: str = "final_labels_synthetic.jsonl"
    stats_json_name: str = "stats.json"


CONFIG = Config()


def extraction_path(synthetic: bool = False) -> Path:
    name = (
        CONFIG.extraction_synthetic_jsonl_name
        if synthetic
        else CONFIG.extraction_jsonl_name
    )
    return CONFIG.checkpoint_dir / name


def critic_path(synthetic: bool = False) -> Path:
    name = CONFIG.critic_synthetic_jsonl_name if synthetic else CONFIG.critic_jsonl_name
    return CONFIG.checkpoint_dir / name


def final_path(synthetic: bool = False) -> Path:
    name = CONFIG.final_synthetic_jsonl_name if synthetic else CONFIG.final_jsonl_name
    return CONFIG.checkpoint_dir / name


def stats_path() -> Path:
    return CONFIG.checkpoint_dir / CONFIG.stats_json_name
