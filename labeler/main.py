"""
CLI entry point pro Mudrc Labeler.

Data se načítají AUTOMATICKY ze složky DATA_DIR (výchozí: /app/data).
Stačí zkopírovat JSON soubory do složky — pipeline je zpracuje bez jakýchkoliv
dalších argumentů.

Výběr readeru (automatický):
  Stage export (root má 'version' + 'schemas' + 'dataset') → iter_stage_export
  Ostatní *.json soubory → iter_generic_json (univerzální, auto-detekce polí)

Volitelný konfigurační soubor labeler_config.json ve složce DATA_DIR:
  {
    "produkty.json": {
      "domain": "e-shop s elektronikou",
      "text_field": "popis",
      "id_field": "catalogId",
      "context_fields": ["nazev", "kategorie"]
    }
  }
  Bez konfigurace se doména odvodí z názvu souboru a textové pole se auto-detekuje.

Použití:
  python -m labeler.main extract
  python -m labeler.main extract --limit 10        # dry run
  python -m labeler.main critic                    # fáze 2: jen Gemma kritika
  python -m labeler.main revise                    # fáze 2b: revisor (Qwen)
                                                   # projde non-ok kritiky a doplní
                                                   # extractor_revised
  python -m labeler.main finalize
  python -m labeler.main pipeline                  # 4 fáze sekvenčně:
                                                   # extract → critic → revise → finalize

Přepsat složku dat nebo doménu:
  python -m labeler.main extract --data-dir /cesta/k/datum
  python -m labeler.main extract --domain "realitní inzeráty"

Design:
  Pipeline běží v 4 oddělených fázích — v každé fázi je na GPU jen JEDEN model:

    Fáze 1 (extract):  extractor (Qwen3.5)
    Fáze 2 (critic):   critic (Gemma 4)
    Fáze 2b (revise):  revisor (defaultně Qwen3.5 — viz REVISOR_*)
                       projde non-ok kritiky a uloží extractor_revised
    Fáze 3 (finalize): bez LLM, merge

  Tím nikdy neběží dva modely souběžně, ani v produkci, ani ve smoke testu.
  Důvod: ušetříme VRAM = větší batch / KV cache = vyšší throughput, a stejná
  architektura pro malé GPU (12 GB smoke test) i velké B200.

  Konfigurace revisoru přes env (defaulty = EXTRACTOR_*):
    REVISOR_BASE_URL, REVISOR_MODEL, REVISOR_REASONING, REVISOR_CONCURRENCY
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

from .config import CONFIG
from .critic import run_critic
from .datasets import LabelItem, scan_data_dir
from .extractor import run_extraction
from .finalize import run_finalize
from .revise import run_revise
from .taxonomy import (
    TaxonomyMode,
    load_taxonomy_if_exists,
    load_user_taxonomy,
    run_taxonomy_discovery,
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _collect_items(
    data_dir: Path,
    default_domain: Optional[str],
    limit_per_dataset: Optional[int],
) -> Iterator[LabelItem]:
    """Načte všechny JSON soubory ze složky data_dir."""
    return scan_data_dir(
        data_dir,
        default_domain=default_domain,
        limit=limit_per_dataset,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labeler",
        description="Mudrc NER/RE labeling pipeline — data se načítají automaticky z DATA_DIR",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_synthetic_flags(parser: argparse.ArgumentParser) -> None:
        grp = parser.add_mutually_exclusive_group()
        grp.add_argument("--skip-synthetic", action="store_true",
                         help="Vynechat synteticky poskládané texty")
        grp.add_argument("--only-synthetic", action="store_true",
                         help="Zpracovat POUZE synteticky poskládané texty")

    def add_data_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data-dir", type=Path, default=None, metavar="DIR",
            help=f"Složka se vstupními JSON soubory (výchozí: DATA_DIR env, tj. {CONFIG.data_dir})",
        )
        parser.add_argument(
            "--domain", default=None, metavar="DOMAIN",
            help="Přepíše doménu pro všechny generické JSON soubory "
                 "(výchozí: název souboru nebo labeler_config.json)",
        )
        parser.add_argument(
            "--limit", type=int, default=None, metavar="N",
            help="Dry run: zpracuj jen prvních N itemů per soubor",
        )

    def add_taxonomy_flags(parser: argparse.ArgumentParser) -> None:
        """Společné flagy pro taxonomy stage (extract + pipeline)."""
        parser.add_argument(
            "--taxonomy-mode", choices=["off", "auto", "user"], default="auto",
            help="Fáze 0 — taxonomy discovery: auto (DEFAULT — LLM analyzuje "
                 "vzorky a navrhne jednotnou taxonomii pro všechny datasety; "
                 "běží v produkci i ve smoke testu), off (16 originálních "
                 "kategorií, žádné LLM volání), user (načti JSON z "
                 "--user-taxonomy). Pokud auto-discovery selže (extractor "
                 "nedostupný, prázdná data, …), spadne se na 16 kategorií a "
                 "běh POKRAČUJE. Pokud existuje entity_taxonomy.json v "
                 "checkpointech, extract ho použije automaticky.",
        )
        parser.add_argument(
            "--user-taxonomy", type=Path, default=None, metavar="FILE",
            help="Cesta k user-defined taxonomy JSON (pro --taxonomy-mode user). "
                 "Viz examples/entity_taxonomy.example.json.",
        )
        parser.add_argument(
            "--samples-per-dataset", type=int,
            default=CONFIG.taxonomy_samples_per_dataset, metavar="N",
            help="Počet ukázek per dataset pro auto taxonomy discovery "
                 "(default z TAXONOMY_SAMPLES_PER_DATASET, jinak 10).",
        )

    # taxonomy — samostatná fáze 0
    ptx = sub.add_parser(
        "taxonomy",
        help="Fáze 0 (volitelná) — návrh jednotné NER taxonomie pro všechny "
             "datasety. Výstup: entity_taxonomy.json v checkpointech.",
    )
    ptx.add_argument(
        "--mode", choices=["off", "auto", "user"], default="auto",
        help="off (default 16 kategorií), auto (LLM návrh), user (z --user-taxonomy).",
    )
    ptx.add_argument(
        "--user-taxonomy", type=Path, default=None, metavar="FILE",
        help="Cesta k user taxonomy JSON (pro --mode user).",
    )
    ptx.add_argument(
        "--samples-per-dataset", type=int,
        default=CONFIG.taxonomy_samples_per_dataset,
        help="Počet ukázek per dataset pro auto mode "
             "(default z TAXONOMY_SAMPLES_PER_DATASET, jinak 10).",
    )

    # extract
    pe = sub.add_parser("extract", help="Fáze 1 — extrakce Qwen3.5-122B")
    add_data_flags(pe)
    add_synthetic_flags(pe)
    add_taxonomy_flags(pe)

    # critic
    pc = sub.add_parser(
        "critic",
        help="Fáze 2 — validace Gemma 4 31B (samotná kritika, žádný revisor)",
    )
    pc.add_argument("--limit", type=int, default=None)
    add_synthetic_flags(pc)

    # revise — standalone revisor pro non-ok kritiky
    pr = sub.add_parser(
        "revise",
        help="Fáze 2b — revisor (defaultně Qwen) projde non-ok kritiky a "
             "doplní extractor_revised. Běží sám na GPU, žádné 2 modely "
             "souběžně.",
    )
    pr.add_argument("--limit", type=int, default=None)
    add_synthetic_flags(pr)

    def add_normalize_flags(parser: argparse.ArgumentParser) -> None:
        """
        Společné flagy pro normalizační vrstvu (finalize + pipeline).

        FILOZOFIE: cleaning běží AUTOMATICKY v každém produkčním běhu i
        smoke testu — uživatel nemusí nic explicitně zapínat. Default ON:

          * normalizace entit (label remap, fix offsetů, drop překryvů)
          * strict offsets (drop entit s nedohledatelným offsetem)
          * auto-expand mentions (doplnění chybějících výskytů)
          * drop structural identifiers (URL/EMAIL/TELEFON regex)
          * write report (normalization_report.json)

        Výstup je VŽDY `final_labels.{jsonl,json}` (+ normalization_report.json).
        spaCy split exporty (core_ner/keyphrases/re) NEvznikají — uživatel
        si je může dělat downstream sám, pokud potřebuje. Smoke test
        navíc vyrobí OWL ontology (.owl + .ttl + volitelně .html).

        Pro výjimečné případy (debugging, raw extractor output) lze
        jakoukoliv část vypnout pomocí `--no-*` flagů.
        """
        # === OPT-OUT FLAGS (default chování je ZAPNUTO; flag vypne) =====
        parser.add_argument(
            "--no-normalize", action="store_true",
            help="VYPNE celou normalizační vrstvu (label remap, fix offsetů, "
                 "drop překryvů). DEFAULT: normalizace běží. Použij jen pro "
                 "debugging — výstup bude raw LLM extractor output.",
        )
        parser.add_argument(
            "--no-strict-offsets", action="store_true",
            help="VYPNE strict offset filtraci. DEFAULT: entity s "
                 "nedohledatelným offsetem se dropnou. Tento flag je nechá "
                 "v datasetu s needs_review.",
        )
        parser.add_argument(
            "--no-auto-expand-mentions", action="store_true",
            help="VYPNE auto-expand mentions. DEFAULT: pipeline automaticky "
                 "doplní chybějící výskyty stejné entity (jen safe — vlastní "
                 "jména, kódy; nikdy obecné termíny).",
        )
        parser.add_argument(
            "--no-write-report", action="store_true",
            help="VYPNE zápis normalization_report.json. DEFAULT: report "
                 "se zapíše do checkpoints/normalization_report.json.",
        )
        parser.add_argument(
            "--keep-structural-identifiers", action="store_true",
            help="Default: entity, jejichž surface form je URL / email / "
                 "telefon, se DROPnou (regex-detekovatelné struktury, ne NER "
                 "entity). Tento flag drop vypne — ponechá je v JSONL "
                 "výstupu (zřídka užitečné, jen pro debugging).",
        )

        # === BACKWARD-COMPAT NO-OPS (default je teď ON, flagy jen potvrzují) ===
        # Tyto flagy zachováváme, aby starší skripty / dokumentace fungovaly.
        # Bez nich je chování stejné jako s nimi (default ON).
        parser.add_argument(
            "--normalize-entities", action="store_true",
            help="(Deprecated, default ON.) Stará form opt-in flagu — chování "
                 "je teď default. Pro vypnutí použij --no-normalize.",
        )
        parser.add_argument(
            "--auto-expand-mentions", action="store_true",
            help="(Deprecated, default ON.) Pro vypnutí --no-auto-expand-mentions.",
        )
        parser.add_argument(
            "--strict-offsets", action="store_true",
            help="(Deprecated, default ON.) Pro vypnutí --no-strict-offsets.",
        )
        parser.add_argument(
            "--write-report", action="store_true",
            help="(Deprecated, default ON.) Pro vypnutí --no-write-report.",
        )
        parser.add_argument(
            "--quality-preset", choices=["spacy"], default=None,
            help="(Deprecated, no-op.) Cleaning běží automaticky. spaCy "
                 "split exporty byly odstraněny — výstup je vždy "
                 "final_labels.{jsonl,json} + normalization_report.json.",
        )

        # === STILL OPT-IN (advanced / aggressive options) =================
        parser.add_argument(
            "--drop-uncertain-labels", action="store_true",
            help="(Opt-in, agresivní.) Zahodit entity s labelem mimo "
                 "--allowed-ner-labels (typicky KEYPHRASE_* kategorie). "
                 "POZN.: tohle zmenší dataset — KEYPHRASE entity skončí jen "
                 "v keyphrases.jsonl, ne v core_ner.jsonl. Pro produkční "
                 "spaCy NER training obvykle zbytečné.",
        )
        parser.add_argument(
            "--allowed-ner-labels", default=None, metavar="LABELS",
            help="Čárkou oddělené labely povolené v NER datasetu "
                 "(default: PRODUKT,PRODUKT_TYP,ORGANIZACE,OSOBA_ROLE,MISTO,"
                 "DATUM,HODNOTA,MATERIAL). URL/EMAIL/TELEFON tu zÁměrnE NEJSOU "
                 "— jsou to regex-detekovatelné struktury, ne NER kategorie. "
                 "Aktivní jen s --drop-uncertain-labels.",
        )

    # finalize
    pf = sub.add_parser("finalize", help="Fáze 3 — merge extractor + critic")
    pf.add_argument(
        "--keep-intermediate", action="store_true",
        help="Po merge NEmaže extractions*.jsonl a critic_results*.jsonl",
    )
    add_normalize_flags(pf)

    # pipeline (vše najednou — fáze 0(volitelně) → 4 fáze sekvenčně)
    pp = sub.add_parser(
        "pipeline",
        help="Spustí všechny fáze sekvenčně: [taxonomy] → extract → critic → "
             "revise → finalize",
    )
    add_data_flags(pp)
    pp.add_argument(
        "--keep-intermediate", action="store_true",
        help="Viz finalize --keep-intermediate",
    )
    add_synthetic_flags(pp)
    add_taxonomy_flags(pp)
    add_normalize_flags(pp)

    # smoke-test (production smoke test + empirický time estimate)
    ps = sub.add_parser(
        "smoke-test",
        help="Production smoke test: 100 items per dataset, měří wall-clock "
             "každé fáze, lineárně extrapoluje na full produkční rozsah "
             "(pure empirical — žádné teoretické scaling factory).",
    )
    ps.add_argument(
        "--phase",
        choices=["taxonomy", "extract", "critic", "revise", "finalize", "report"],
        required=True,
        help="Která fáze se má teď změřit. Spouští se sekvenčně mezi tím, jak "
             "uživatel přepíná docker-compose profily. 'taxonomy' (fáze 0) "
             "změří auto-discovery taxonomie. 'report' vygeneruje "
             "finální odhad z dosud naměřených timingů.",
    )
    ps.add_argument(
        "--hw-label", default="",
        help="Volný text popisující hardware, na kterém smoke běží (např. "
             "'1x B200', '2x B200', '4x B200', '8x B200'). Použije "
             "se POUZE pro dokumentaci v reportu — pro výpočet se používá "
             "naměřený wall-clock přímo, žádné scaling factory. Pro jiný "
             "HW spusť smoke znovu na tom HW.",
    )
    ps.add_argument(
        "--gpu-config", default=None,
        help="(Deprecated alias pro --hw-label.)",
    )
    ps.add_argument(
        "--docker-startup-s", type=float, default=0.0,
        help="Volitelný odhad per-phase docker startup overhead (model load, "
             "container init). Pokud zadáš, přičte se k reportu jako "
             "informace (× 4 fáze). Default 0 (nepřičítá se).",
    )
    ps.add_argument(
        "--samples-per-dataset", type=int, default=100,
        help="Počet itemů per dataset pro smoke vzorek (default 100). "
             "Vyšší = přesnější odhad, ale delší smoke run.",
    )
    add_data_flags(ps)

    return p


def _apply_quality_preset(args: argparse.Namespace) -> None:
    """
    Backward-compat no-op. Cleaning preset běží AUTOMATICKY v každém běhu
    (viz `_build_normalize_opts` níže s default-ON semantikou). Tento
    helper zachováván, aby starší volání nepuklo.

    Historie: dříve uživatel musel `--quality-preset spacy` zadat ručně;
    teď je to default chování. `--quality-preset spacy` je no-op.
    """
    pass


def _opt_enabled(args: argparse.Namespace, no_flag: str) -> bool:
    """
    Pomocná funkce pro default-ON semantiku: vrátí True (chování zapnuto)
    POKUD `--no-<no_flag>` NEBYLO předáno. Tj. uživatel ho musí explicit
    vypnout přes opt-out flag.

    Příklad: `_opt_enabled(args, "no_normalize")` vrátí True když uživatel
    nepředal `--no-normalize`, jinak False (vypnuto).
    """
    return not bool(getattr(args, no_flag, False))


def _build_normalize_opts(args: argparse.Namespace):
    """Sestaví NormalizeOpts z argparse args.

    DEFAULT chování (žádné flagy potřeba):
      - normalize entities: ON
      - strict offsets: ON
      - auto-expand mentions: ON
      - drop structural identifiers (URL/EMAIL/TELEFON regex): ON

    Pro vypnutí kterékoli části použij `--no-*` flag (viz add_normalize_flags).

    Taxonomy: pokud existuje checkpoints/entity_taxonomy.json (z fáze 0),
    je přibalen do opts.taxonomy — normalize step pak aplikuje
    force_text_match a unsafe_texts per label (data-driven). Bez taxonomy
    se text rules step přeskočí (žádný hard-coding heuristik).
    """
    from .normalize import NormalizeOpts, DEFAULT_NER_LABELS
    _apply_quality_preset(args)  # no-op pro backward compat
    allowed: Optional[frozenset[str]] = None
    if getattr(args, "allowed_ner_labels", None):
        allowed = frozenset(
            lb.strip().upper() for lb in args.allowed_ner_labels.split(",")
            if lb.strip()
        )
    else:
        allowed = DEFAULT_NER_LABELS
    return NormalizeOpts(
        enabled=_opt_enabled(args, "no_normalize"),
        strict_offsets=_opt_enabled(args, "no_strict_offsets"),
        auto_expand_mentions=_opt_enabled(args, "no_auto_expand_mentions"),
        # drop_uncertain_labels zůstává opt-in (agresivní, mění rozsah dat)
        drop_uncertain_labels=bool(getattr(args, "drop_uncertain_labels", False)),
        drop_structural_identifiers=(
            not bool(getattr(args, "keep_structural_identifiers", False))
        ),
        allowed_ner_labels=allowed,
        taxonomy=load_taxonomy_if_exists(),
    )


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    """Vrátí data_dir z argumentu nebo z CONFIG."""
    data_dir = getattr(args, "data_dir", None)
    return data_dir if data_dir is not None else CONFIG.data_dir


async def _run_smoke_test(args: argparse.Namespace) -> int:
    """
    Production smoke test dispatcher. Měří wall-clock pro vybranou fázi,
    lineárně extrapoluje na full dataset. Viz labeler/smoke_test.py pro
    detaily — pure empirický odhad, žádné teoretické scaling factory.
    """
    import time as _time
    from . import smoke_test as st
    from .config import CONFIG

    phase: str = args.phase
    samples_n: int = max(1, int(args.samples_per_dataset or st.DEFAULT_SAMPLES_PER_DATASET))
    # HW label — primární --hw-label, fallback na deprecated --gpu-config
    hw_label: str = (
        getattr(args, "hw_label", None)
        or getattr(args, "gpu_config", None)
        or ""
    )
    docker_startup_s: float = float(getattr(args, "docker_startup_s", 0.0) or 0.0)

    # `report` jen čte timingy, nic nepouští
    if phase == "report":
        report = st.build_report(hw_label, samples_n, docker_startup_s=docker_startup_s)
        md_path, json_path = st.write_report_files(report)
        print()
        print(st.format_report_markdown(report))
        print()
        print(f"Report uložen:")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")
        return 0

    # Pro běhy fáze přepneme checkpointy do smoke složky (nezasahuje produkci).
    # CONFIG je frozen dataclass — používáme object.__setattr__ jako bypass.
    smoke_dir = st.smoke_checkpoint_dir()
    object.__setattr__(CONFIG, "checkpoint_dir", smoke_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"[smoke-test] checkpoint_dir přepnut na {smoke_dir}")
    logger.info(f"[smoke-test] hardware label: {hw_label!r}, samples/dataset: {samples_n}")

    # Ulož HW label pro pozdější report (pokud uživatel zadal)
    if hw_label:
        st.record_hw_label(hw_label)

    data_dir = _resolve_data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Nejdřív (jen jednou) změř plný rozsah datasetů — uloží se do
    # smoke_timings.json. Příště se přečte z cache.
    timings = st.load_timings()
    if not timings.get("datasets"):
        full_stats = st.measure_dataset_sizes(data_dir)
        st.annotate_sample_stats(full_stats, samples_n, data_dir)
        st.record_dataset_stats(full_stats, samples_n)
        timings = st.load_timings()
    elif timings.get("datasets_samples_per_dataset") != samples_n:
        # Drahý full count je nacachovaný, ALE sample_* sloupce byly změřeny s
        # jiným --samples-per-dataset (stale cache → report ukazoval např.
        # sample_items=1, i když fáze běžely na 100). Re-annotuj LEVNĚ vzorek
        # pro aktuální samples_n, ať dataset tabulka i "Total samples processed"
        # sedí s tím, co reálně poběží ve fázích.
        logger.info(
            f"[smoke-test] samples/dataset se změnil "
            f"({timings.get('datasets_samples_per_dataset')} → {samples_n}); "
            f"re-annotuji dataset vzorek (full count zůstává z cache)."
        )
        full_stats = st.dataset_stats_from_cache(timings)
        st.annotate_sample_stats(full_stats, samples_n, data_dir)
        st.record_dataset_stats(full_stats, samples_n)
        timings = st.load_timings()

    # --------------------------------------------------------------- fáze 0
    # Taxonomy auto-discovery: JEDNORÁZOVÁ konstanta. Sampluje fixní počet
    # vzorků nezávisle na velikosti datasetu, takže se NEškáluje na full
    # rozsah — měříme reálný wall-clock a v reportu ho přičteme jako konstantu.
    # Auto je teď default i v PRODUKCI (ne jen smoke), takže tahle konstanta
    # odpovídá reálnému produkčnímu běhu — měříme přesně to, co poběží ostře.
    # Pozn.: smoke `--samples-per-dataset` (default 100) se týká vzorku pro
    # extract/critic; auto-discovery používá vlastní count (CONFIG, default 10).
    if phase == "taxonomy":
        object.__setattr__(CONFIG, "data_dir", data_dir)
        tax_samples = CONFIG.taxonomy_samples_per_dataset
        t0 = _time.perf_counter()
        tax_path = await run_taxonomy_discovery(
            mode=TaxonomyMode.AUTO,
            samples_per_dataset=tax_samples,
        )
        wall = _time.perf_counter() - t0
        st.record_phase_timing(
            "taxonomy",
            items_processed=0,
            input_chars=0,
            output_chars=0,
            wall_clock_s=wall,
        )
        status = "OK" if tax_path else "fallback na default 16 kategorií"
        print(
            f"\n[smoke-test] taxonomy (fáze 0) dokončena: {wall:.1f}s ({status}). "
            f"Zaznamenáno jako jednorázová konstanta (neškáluje se s rozsahem)."
        )
        return 0

    # Spočítej vstupní chars pro tento smoke vzorek (jen text + context)
    sample_items = list(_collect_items(data_dir, getattr(args, "domain", None), samples_n))
    sample_input_chars = sum(
        len(it.text or "") + len(it.context or "") for it in sample_items
    )
    logger.info(
        f"[smoke-test] smoke vzorek: {len(sample_items)} items, "
        f"{sample_input_chars:,} input chars"
    )

    # --------------------------------------------------------------- per-fáze
    if phase == "extract":
        from .taxonomy import load_taxonomy_if_exists
        t0 = _time.perf_counter()
        await run_extraction(
            sample_items,
            dry_run_limit=None,
            skip_synthetic=False,
            only_synthetic=False,
            taxonomy=load_taxonomy_if_exists(),
        )
        wall = _time.perf_counter() - t0
        st.record_phase_timing(
            "extract",
            items_processed=len(sample_items),
            input_chars=sample_input_chars,
            output_chars=0,  # neznáme bez parsování JSONL
            wall_clock_s=wall,
        )
        print(f"\n[smoke-test] extract dokončen: {len(sample_items)} items, "
              f"{wall:.1f}s. Timing zaznamenán.")
        return 0

    if phase == "critic":
        # critic input = původní text + extractor JSON (aprox. 2× input chars)
        approx_input_chars = sample_input_chars * 2
        t0 = _time.perf_counter()
        await run_critic(
            dry_run_limit=None, skip_synthetic=False, only_synthetic=False,
        )
        wall = _time.perf_counter() - t0
        nonok = _count_critic_nonok()
        st.record_phase_timing(
            "critic",
            items_processed=len(sample_items),
            input_chars=approx_input_chars,
            output_chars=0,
            wall_clock_s=wall,
            critic_nonok_count=nonok,
        )
        print(f"\n[smoke-test] critic dokončen: {len(sample_items)} items, "
              f"{wall:.1f}s, non-ok = {nonok}. Timing zaznamenán.")
        return 0

    if phase == "revise":
        # revise input = jen non-ok subset; čteme z critic_results.jsonl
        nonok = _count_critic_nonok()
        # Approx per-item chars na non-ok subsetu (text + critic feedback + JSON)
        avg_per_item = sample_input_chars / max(1, len(sample_items))
        approx_input_chars = int(nonok * avg_per_item * 2.5)
        t0 = _time.perf_counter()
        await run_revise(
            dry_run_limit=None, skip_synthetic=False, only_synthetic=False,
        )
        wall = _time.perf_counter() - t0
        st.record_phase_timing(
            "revise",
            items_processed=nonok,
            input_chars=approx_input_chars,
            output_chars=0,
            wall_clock_s=wall,
        )
        print(f"\n[smoke-test] revise dokončen: {nonok} items, {wall:.1f}s. "
              f"Timing zaznamenán.")
        return 0

    if phase == "finalize":
        # finalize je CPU only — input chars = vše co finalize čte.
        # CLEANING běží automaticky (default ON) PŘED OWL exportem:
        #   normalize entities + strict offsets + auto-expand + structural
        #   identifier drop + write normalization_report.json.
        # OWL ontology export NAVÍC (jen ve smoke testu, ne v produkci).
        # spaCy split exporty NEvznikají — výstup je vždy jen
        # final_labels.{jsonl,json}. Smoke test má navíc:
        #   - final_labels.owl + .ttl (+ .html pokud pylode)
        #   - smoke_timings.json + smoke_test_report.{md,json}
        opts = _build_normalize_opts(args)
        t0 = _time.perf_counter()
        run_finalize(
            keep_intermediate=True,  # ve smoke testu chceme zachovat intermediate
            normalize_opts=opts,
            write_report=_opt_enabled(args, "no_write_report"),
            export_spacy=False,        # žádné spaCy splits ani ve smoke
            export_owl=True,           # OWL jen ve smoke testu
            owl_per_source=True,
            owl_combined=True,
            owl_max_text_len=5000,
            owl_write_html=True,       # pyLODE HTML doc, pokud je nainstalováno
        )
        wall = _time.perf_counter() - t0
        st.record_phase_timing(
            "finalize",
            items_processed=len(sample_items),
            input_chars=sample_input_chars,
            output_chars=0,
            wall_clock_s=wall,
        )
        print(f"\n[smoke-test] finalize dokončen: {len(sample_items)} items, "
              f"{wall:.1f}s. Cleaning + OWL ontology export do checkpoints/smoke/.")
        return 0

    raise ValueError(f"Neznámá smoke-test fáze: {phase}")


def _count_critic_nonok() -> int:
    """Spočítá počet záznamů v critic_results*.jsonl, kde status != 'ok'."""
    import json as _json
    from .config import critic_path
    count = 0
    for synthetic in (False, True):
        p = critic_path(synthetic=synthetic)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    # CriticResult se serializuje PLOŠE — status je top-level
                    # klíč, ne vnořený pod "critic". (Dřív se četlo rec["critic"]
                    # → vždy {} → count=0, čímž smoke-test odhad fáze revise
                    # podhodnocoval vstup na nulu.)
                    if isinstance(rec, dict):
                        status = (rec.get("status") or "").lower()
                        if status and status != "ok":
                            count += 1
        except OSError:
            continue
    return count


async def _maybe_run_taxonomy(args: argparse.Namespace, data_dir: Path) -> None:
    """
    Helper pro extract/pipeline: spustí fázi 0 (taxonomy discovery) PŘED extract.

    Default --taxonomy-mode je 'auto' → fáze 0 běží i v PRODUKCI (ne jen ve
    smoke testu). 'off' ji explicitně vypne (16 originálních kategorií, žádné
    LLM volání). 'user' načte JSON z --user-taxonomy.

    Discovery je BEST-EFFORT: pokud auto selže (extractor nedostupný, prázdná
    data, přetečení kontextu…), run_taxonomy_discovery spadne na 16 kategorií
    a vrátí None — běh NEzastaví (extract pokračuje s defaultem).

    `data_dir` (už vyřešený z --data-dir / CONFIG) nastavíme do CONFIG, aby
    auto-discovery samplovala ze STEJNÉ složky, ze které pak extract čte.
    """
    mode_str = getattr(args, "taxonomy_mode", None) or "auto"
    if mode_str == "off":
        return
    object.__setattr__(CONFIG, "data_dir", data_dir)
    mode = TaxonomyMode(mode_str)
    await run_taxonomy_discovery(
        mode=mode,
        user_taxonomy_path=getattr(args, "user_taxonomy", None),
        samples_per_dataset=getattr(
            args, "samples_per_dataset", CONFIG.taxonomy_samples_per_dataset
        ),
    )


async def _run(args: argparse.Namespace) -> int:
    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "taxonomy":
        mode = TaxonomyMode(args.mode)
        path = await run_taxonomy_discovery(
            mode=mode,
            user_taxonomy_path=getattr(args, "user_taxonomy", None),
            samples_per_dataset=args.samples_per_dataset,
        )
        if path:
            print(f"Taxonomy saved: {path}")
        else:
            print("Taxonomy mode = off (žádný výstup).")
    elif args.cmd == "extract":
        data_dir = _resolve_data_dir(args)
        data_dir.mkdir(parents=True, exist_ok=True)
        # Fáze 0 (taxonomy discovery) — default auto, běží i v produkci.
        await _maybe_run_taxonomy(args, data_dir)
        items = _collect_items(data_dir, getattr(args, "domain", None), args.limit)
        await run_extraction(
            items,
            dry_run_limit=None,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
            taxonomy=load_taxonomy_if_exists(),
        )
    elif args.cmd == "critic":
        await run_critic(
            dry_run_limit=args.limit,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
        )
    elif args.cmd == "revise":
        await run_revise(
            dry_run_limit=args.limit,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
        )
    elif args.cmd == "finalize":
        opts = _build_normalize_opts(args)  # default-ON cleaning
        # Produkční výstup: final_labels.{jsonl,json} + normalization_report.json.
        # OWL ontology + spaCy splits se v produkci NEGENERUJÍ.
        run_finalize(
            keep_intermediate=args.keep_intermediate,
            normalize_opts=opts,
            write_report=_opt_enabled(args, "no_write_report"),
            export_spacy=False,   # spaCy splits jen jako optional Python API
            export_owl=False,     # OWL jen ve smoke testu
        )
    elif args.cmd == "smoke-test":
        return await _run_smoke_test(args)
    elif args.cmd == "pipeline":
        data_dir = _resolve_data_dir(args)
        data_dir.mkdir(parents=True, exist_ok=True)
        # Fáze 0 (taxonomy discovery) — default auto, běží i v produkci.
        await _maybe_run_taxonomy(args, data_dir)
        items = _collect_items(data_dir, getattr(args, "domain", None), args.limit)
        await run_extraction(
            items,
            dry_run_limit=None,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
            taxonomy=load_taxonomy_if_exists(),
        )
        await run_critic(
            dry_run_limit=args.limit,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
        )
        await run_revise(
            dry_run_limit=args.limit,
            skip_synthetic=args.skip_synthetic,
            only_synthetic=args.only_synthetic,
        )
        opts = _build_normalize_opts(args)  # default-ON cleaning
        # Pipeline (produkční): final_labels.{jsonl,json} + normalization_report.json.
        # OWL + spaCy splits NEGENEROVAT — to je jen smoke-test artefakt.
        run_finalize(
            keep_intermediate=args.keep_intermediate,
            normalize_opts=opts,
            write_report=_opt_enabled(args, "no_write_report"),
            export_spacy=False,   # spaCy splits jen jako optional Python API
            export_owl=False,     # OWL jen ve smoke testu
        )
    else:
        raise ValueError(f"Neznámý command: {args.cmd}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)
    try:
        code = asyncio.run(_run(args))
        sys.exit(code)
    except KeyboardInterrupt:
        print(
            "\n⚠️  Přerušeno uživatelem. State je uložen v checkpoints/, "
            "další spuštění naváže.",
            file=sys.stderr,
        )
        sys.exit(130)


if __name__ == "__main__":
    main()
