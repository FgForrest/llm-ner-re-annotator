"""
Fáze 2b — Standalone revisor (post-processing critic_results).

Tahle fáze existuje pro scénář, kdy nemůžeme držet critic i revisor v paměti
souběžně (např. smoke test na 1× 12 GB GPU). V produkci s dostatečnou VRAM
běží revisor automaticky uvnitř critic fáze (critic.py).

Postup:
  1. Načte critic_results*.jsonl (output fáze 2 — critic sám, bez revisoru).
  2. Pro každý záznam, kde status != "ok" a extractor_revised není vyplněný:
     - načte odpovídající extractor result (z extractions*.jsonl)
     - sestaví revision prompt (text + původní výstup + kritika)
     - zavolá revisora (defaultně tentýž model jako extractor — Qwen)
     - doplní extractor_revised + extractor_revised_reasoning
  3. Atomicky přepíše critic_results*.jsonl s doplněnými poli.

Resume-safe: záznamy s už vyplněným extractor_revised se přeskakují.
Idempotent: opětovné spuštění nezpůsobí duplikaci ani přepsání hotových revizí.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import ValidationError

from .config import CONFIG, critic_path, extraction_path
from .io_utils import ProgressTracker, atomic_write_jsonl, read_jsonl
from .llm_client import LLMClient, LLMFatalError, LLMTransientError
from .positions import compute_entity_positions
from .prompts import build_extractor_revision_messages
from .schemas import CriticResult, ExtractorResult, SkoGraph

logger = logging.getLogger(__name__)


async def _revise_one(
    revisor_client: LLMClient,
    extraction: ExtractorResult,
    critic_result: CriticResult,
    *,
    max_tokens: int,
    reasoning: bool,
    taxonomy=None,
) -> Optional[CriticResult]:
    """
    Pro jeden CriticResult zavolá revisora a doplní extractor_revised.

    `taxonomy` (fáze 0) = povolený slovník kategorií; bez něj revisor neví,
    co je platný label, a potvrdil by i chybnou "opravu" od critica.

    Vrací nový CriticResult (s vyplněnými extractor_revised polí), nebo None
    pokud revisor selhal nebo vrátil nevalidní graf — v takovém případě se
    původní CriticResult ponechá (volání to vrátí jako "no-op").
    """
    # Split NER/RE (default ON): re-eval entit a relací zvlášť.
    if CONFIG.split_ner_re:
        from .ner_re_split import revise_split

        return await revise_split(
            revisor_client, extraction, critic_result,
            max_tokens=max_tokens, reasoning=reasoning, taxonomy=taxonomy,
        )

    original_graph_dict = extraction.graph.model_dump(
        mode="json", exclude={"triples"}
    )
    corrected_dict = (
        critic_result.corrected.model_dump(mode="json", exclude={"triples"})
        if critic_result.corrected
        else None
    )

    messages = build_extractor_revision_messages(
        text=extraction.input_text,
        original_graph=original_graph_dict,
        critic_status=critic_result.status,
        critic_reasoning=critic_result.reasoning,
        corrected_graph=corrected_dict,
        taxonomy=taxonomy,
    )

    try:
        resp = await revisor_client.chat(
            messages, max_tokens=max_tokens, reasoning=reasoning
        )
    except (LLMFatalError, LLMTransientError) as e:
        logger.warning(f"[{extraction.item_id}] Revisor chyba: {e}")
        return None

    if resp.parsed is None:
        logger.warning(
            f"[{extraction.item_id}] Revisor nevrátil validní JSON"
        )
        return None

    try:
        # Bare list = entity list; zabal jako entities (combined revisor žádá graf).
        parsed = resp.parsed
        if isinstance(parsed, list):
            parsed = {"entities": parsed}
        revised = SkoGraph(**parsed)
        revised.build_triples()
        compute_entity_positions(extraction.input_text, revised)
    except (ValidationError, Exception) as e:
        logger.warning(
            f"[{extraction.item_id}] Revisor nevalidní graf: {e}"
        )
        return None

    # Vytvoříme nový CriticResult s doplněnými poli (zachová původní status,
    # reasoning, corrected — jen přidá extractor_revised).
    updated = critic_result.model_copy(
        update={
            "extractor_revised": revised,
            "extractor_revised_reasoning": (
                f"Revisor re-eval po kritice (status={critic_result.status})"
            ),
        }
    )
    logger.info(
        f"[{extraction.item_id}] Revisor OK: "
        f"{len(revised.entities)} entit, {len(revised.relations)} relací"
    )
    return updated


async def _revise_branch(
    *,
    label: str,
    is_synth: bool,
    revisor_client: LLMClient,
    dry_run_limit: Optional[int],
    taxonomy=None,
) -> None:
    """Zpracuje jednu větev (REAL nebo SYNTH)."""
    crit_p = critic_path(synthetic=is_synth)
    ext_p = extraction_path(synthetic=is_synth)

    if not crit_p.exists():
        logger.info(f"[{label}] {crit_p} neexistuje, přeskakuji.")
        return
    if not ext_p.exists():
        logger.warning(
            f"[{label}] {ext_p} chybí — bez extrakcí nelze revisora pustit."
        )
        return

    # Načti všechny critic results
    critic_records: list[CriticResult] = []
    for rec in read_jsonl(crit_p):
        try:
            critic_records.append(CriticResult(**rec))
        except ValidationError as e:
            logger.warning(
                f"[{label}] Skip nevalidní critic record: {e.errors()[:1]}"
            )

    # Načti odpovídající extractor results do mapy podle item_id
    ext_map: dict[str, ExtractorResult] = {}
    for rec in read_jsonl(ext_p):
        try:
            ex = ExtractorResult(**rec)
            ext_map[ex.item_id] = ex
        except ValidationError:
            continue

    # Vyber kandidáty na revizi: status != "ok" AND extractor_revised není vyplněný
    candidates: list[tuple[CriticResult, ExtractorResult]] = []
    skipped_already_revised = 0
    skipped_ok = 0
    skipped_missing_ext = 0

    for cr in critic_records:
        if cr.status == "ok":
            skipped_ok += 1
            continue
        if cr.extractor_revised is not None:
            skipped_already_revised += 1
            continue
        ex = ext_map.get(cr.item_id)
        if ex is None:
            skipped_missing_ext += 1
            logger.warning(
                f"[{label}] {cr.item_id}: critic record bez odpovídající extrakce"
            )
            continue
        candidates.append((cr, ex))

    logger.info(
        f"[{label}] Critic záznamů celkem: {len(critic_records)} "
        f"(ok={skipped_ok}, už zrevidováno={skipped_already_revised}, "
        f"bez extrakce={skipped_missing_ext}, k revizi={len(candidates)})"
    )

    if dry_run_limit:
        candidates = candidates[:dry_run_limit]
        logger.warning(
            f"[{label}] DRY RUN: revize jen prvních {dry_run_limit} kandidátů"
        )

    if not candidates:
        logger.info(f"[{label}] Nic k revizi, přeskakuji.")
        return

    tracker = ProgressTracker(
        total=len(candidates), label=f"revise-{label.lower()}"
    )
    updated_by_id: dict[str, CriticResult] = {}

    item_sem = asyncio.Semaphore(CONFIG.revisor_concurrency)

    async def process(cr: CriticResult, ex: ExtractorResult) -> None:
        async with item_sem:
            try:
                updated = await _revise_one(
                    revisor_client,
                    ex,
                    cr,
                    max_tokens=CONFIG.revisor_max_tokens,
                    reasoning=CONFIG.revisor_reasoning,
                    taxonomy=taxonomy,
                )
                if updated is None:
                    tracker.tick(error=True)
                    return
                updated_by_id[updated.item_id] = updated
                tracker.tick()
            except Exception as e:
                logger.exception(f"[{ex.item_id}] Revise unexpected: {e}")
                tracker.tick(error=True)

    tasks = [process(cr, ex) for cr, ex in candidates]
    await asyncio.gather(*tasks, return_exceptions=False)

    # Atomicky přepiš critic_results.jsonl s doplněnými poli.
    if updated_by_id:
        merged = [updated_by_id.get(cr.item_id, cr) for cr in critic_records]
        atomic_write_jsonl(crit_p, [r.model_dump(mode="json") for r in merged])
        logger.info(tracker.summary())
        logger.info(
            f"[{label}] Aktualizováno {len(updated_by_id)} záznamů v {crit_p.name}"
        )
    else:
        logger.info(tracker.summary())
        logger.info(f"[{label}] Žádné záznamy se nepodařilo zrevidovat.")


async def run_revise(
    *,
    dry_run_limit: Optional[int] = None,
    skip_synthetic: bool = False,
    only_synthetic: bool = False,
    taxonomy=None,
) -> None:
    """
    Hlavní loop fáze 2b — standalone revisor.

    Použije CONFIG.revisor_* (defaultně shodné s extractorem). Načte
    critic_results*.jsonl, pro non-ok záznamy bez extractor_revised zavolá
    revisora a atomicky přepíše soubor.
    """
    if skip_synthetic and only_synthetic:
        raise ValueError("skip_synthetic a only_synthetic jsou vzájemně vyloučené")

    branches: list[tuple[str, bool]] = []
    if not only_synthetic:
        branches.append(("REAL", False))
    if not skip_synthetic:
        branches.append(("SYNTH", True))

    logger.info(
        f"Revisor: model={CONFIG.revisor_model} @ {CONFIG.revisor_base_url}"
    )

    async with LLMClient(
        base_url=CONFIG.revisor_base_url,
        model=CONFIG.revisor_model,
        max_concurrency=CONFIG.revisor_concurrency,
        request_timeout=CONFIG.request_timeout_s,
        max_retries=CONFIG.max_retries,
        context_len=CONFIG.extractor_context_len,
    ) as revisor_client:
        await revisor_client.wait_ready(max_wait_s=1200)

        for label, is_synth in branches:
            await _revise_branch(
                label=label,
                is_synth=is_synth,
                revisor_client=revisor_client,
                dry_run_limit=dry_run_limit,
                taxonomy=taxonomy,
            )
