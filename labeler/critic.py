"""
Fáze 2 — Validace (critic).

Čte extractions.jsonl (výstup fáze 1), pro každou extrakci pustí Gemma 4 31B
s novou session a nechá ji zhodnotit, jestli je JSON správný. Critic vrací:
  - status: "ok"       → ponechat beze změny
  - status: "revised"  → critic navrhne 'corrected' graf
  - status: "reject"   → critic doporučuje zahodit

CriticResult se uloží BEZ extractor_revised (to doplní fáze 2b — revise).
Tím v této fázi běží na GPU pouze critic (Gemma), žádný extractor / revisor.

Návazná fáze 2b ([revise.py](revise.py)) pak nahodí Qwen jako revisora a projde
non-ok záznamy: revisor sám rozhodne, jestli kritiku zapracuje, nebo zůstane
u původního výstupu.

Resume-safe: načte už validované item_ids a přeskakuje je.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional

from pydantic import ValidationError

from .config import CONFIG, critic_path, extraction_path
from .io_utils import JsonlWriter, ProgressTracker, load_completed_ids, read_jsonl
from .llm_client import LLMClient, LLMFatalError, LLMTransientError
from .positions import compute_entity_positions
from .prompts import build_critic_messages
from .schemas import CriticResult, ExtractorResult, SkoGraph

logger = logging.getLogger(__name__)


async def _critique_one(
    client: LLMClient,
    extraction: ExtractorResult,
    *,
    max_tokens: int,
) -> Optional[CriticResult]:
    """
    Zavolá critic s novou "session" (žádný shared context s extractorem).
    Vrací CriticResult (BEZ extractor_revised — to doplní revise fáze).
    """
    # Split NER/RE (default ON): validace entit a relací zvlášť, merge do
    # jednoho CriticResult (formát beze změny).
    if CONFIG.split_ner_re:
        from .ner_re_split import critique_split

        return await critique_split(client, extraction, max_tokens=max_tokens)

    messages = build_critic_messages(
        text=extraction.input_text,
        proposed=extraction.graph.model_dump(mode="json", exclude={"triples"}),
        context=extraction.context,
    )

    try:
        resp = await client.chat(
            messages,
            max_tokens=max_tokens,
            reasoning=CONFIG.critic_reasoning,
        )
    except LLMFatalError as e:
        logger.error(f"[{extraction.item_id}] Critic fatal: {e}")
        return None
    except LLMTransientError as e:
        logger.error(f"[{extraction.item_id}] Critic transient exhausted: {e}")
        return None

    if resp.parsed is None:
        # CRITIC SELHAL — nevrátil validní JSON. NESMÍME dát status="ok",
        # to by tichu prošlo do final_labels jako "validováno" a uživatel
        # by neměl tušení, že critic nedoběhl. Místo toho status="reject"
        # s explicit důvodem v reasoning — finalize si to zachová jako
        # was_revised=False, critic_status="reject", uživatel vidí v
        # stats.json kolik itemů critic nezvládl validovat.
        logger.warning(
            f"[{extraction.item_id}] Critic nevrátil validní JSON: {resp.content[:200]}"
        )
        return CriticResult(
            item_id=extraction.item_id,
            status="reject",
            reasoning=(
                "Critic nevrátil validní JSON — extractor output zůstal "
                "BEZ kritické validace. NENÍ to potvrzení správnosti, "
                "jen 'critic selhal, surový extractor output'."
            ),
            corrected=None,
            extractor_revised=None,
            extractor_revised_reasoning=None,
            model=client.model,
            latency_ms=resp.latency_ms,
        )

    # Model může vrátit bare list místo objektu se statusem; pro .get přístup
    # ho degradujeme na {} (→ status 'ok', žádná korekce = ponech extractor output).
    raw = resp.parsed if isinstance(resp.parsed, dict) else {}
    status = raw.get("status", "ok")
    reasoning = raw.get("reasoning", "(bez zdůvodnění)")
    corrected_data = raw.get("corrected")

    if status not in ("ok", "revised", "reject"):
        logger.warning(
            f"[{extraction.item_id}] Neznámý status '{status}', přetvářím na ok"
        )
        status = "ok"
        reasoning = f"[auto-fix] původní status '{raw.get('status')}': {reasoning}"

    corrected_graph: Optional[SkoGraph] = None
    if status == "revised":
        if not corrected_data:
            logger.warning(
                f"[{extraction.item_id}] status=revised ale chybí 'corrected', "
                f"degrading na ok"
            )
            status = "ok"
            reasoning = f"[auto-fix] revised bez corrected: {reasoning}"
        else:
            try:
                corrected_graph = SkoGraph(**corrected_data)
                corrected_graph.build_triples()
                compute_entity_positions(extraction.input_text, corrected_graph)
            except ValidationError as e:
                logger.warning(
                    f"[{extraction.item_id}] corrected graph nevaliadní, "
                    f"degrading na ok: {e.errors(include_url=False)[:2]}"
                )
                status = "ok"
                reasoning = f"[auto-fix] corrected validation failed: {reasoning}"
                corrected_graph = None

    try:
        return CriticResult(
            item_id=extraction.item_id,
            status=status,  # type: ignore[arg-type]
            reasoning=reasoning[:2000],
            corrected=corrected_graph,
            extractor_revised=None,
            extractor_revised_reasoning=None,
            model=client.model,
            latency_ms=resp.latency_ms,
        )
    except ValidationError as e:
        logger.error(f"[{extraction.item_id}] Nelze sestavit CriticResult: {e}")
        return None


async def run_critic(
    *,
    dry_run_limit: Optional[int] = None,
    skip_synthetic: bool = False,
    only_synthetic: bool = False,
) -> None:
    """
    Hlavní loop fáze 2 — pouze critic (Gemma).

    Zpracovává extractions.jsonl + extractions_synthetic.jsonl do dvou výstupů:
      critic_results.jsonl            — validace reálných popisků
      critic_results_synthetic.jsonl  — validace syntetických textů

    Revisor (Qwen re-evaluation kritiky) se nevolá v této fázi — to dělá
    samostatná fáze 2b (revise.py). Tím se zaručí, že v této fázi běží na
    GPU pouze critic, žádný druhý LLM model souběžně.

    Flags:
      skip_synthetic:  ignoruj synthetic soubor
      only_synthetic:  zpracuj POUZE synthetic
    """
    if skip_synthetic and only_synthetic:
        raise ValueError("skip_synthetic a only_synthetic jsou vzájemně vyloučené")

    branches: list[tuple[str, bool]] = []
    if not only_synthetic:
        branches.append(("REAL", False))
    if not skip_synthetic:
        branches.append(("SYNTH", True))

    logger.info(
        f"Critic: model={CONFIG.critic_model} @ {CONFIG.critic_base_url} "
        f"(reasoning={CONFIG.critic_reasoning})"
    )

    async with LLMClient(
        base_url=CONFIG.critic_base_url,
        model=CONFIG.critic_model,
        max_concurrency=CONFIG.critic_concurrency,
        request_timeout=CONFIG.request_timeout_s,
        max_retries=CONFIG.max_retries,
    ) as critic_client:
        await critic_client.wait_ready(max_wait_s=1200)
        await _run_critic_branches(
            branches=branches,
            critic_client=critic_client,
            dry_run_limit=dry_run_limit,
        )


async def _run_critic_branches(
    *,
    branches: list[tuple[str, bool]],
    critic_client: LLMClient,
    dry_run_limit: Optional[int],
) -> None:
    for label, is_synth in branches:
        input_path = extraction_path(synthetic=is_synth)
        output_path = critic_path(synthetic=is_synth)

        if not input_path.exists():
            logger.warning(
                f"[{label}] Chybí {input_path}, přeskakuji tuto větev."
            )
            continue

        completed_ids = load_completed_ids(output_path, "item_id")
        logger.info(f"[{label}] {len(completed_ids)} už zvalidovaných")

        extractions: list[ExtractorResult] = []
        for rec in read_jsonl(input_path):
            try:
                extractions.append(ExtractorResult(**rec))
            except ValidationError as e:
                logger.warning(
                    f"[{label}] Skip nevalidní extraction "
                    f"{rec.get('item_id')}: {e.errors()[:1]}"
                )

        if dry_run_limit:
            extractions = extractions[:dry_run_limit]
            logger.warning(f"[{label}] DRY RUN: omezeno na prvních {dry_run_limit}")

        total = len(extractions)
        pending = [e for e in extractions if e.item_id not in completed_ids]
        logger.info(f"[{label}] Ke zpracování: {len(pending)} / {total}")

        if not pending:
            logger.info(f"[{label}] Nic k práci, přeskakuji.")
            continue

        tracker = ProgressTracker(total=total, label=f"critic-{label.lower()}")

        item_sem = asyncio.Semaphore(CONFIG.critic_concurrency)

        with JsonlWriter(output_path) as writer:

            async def process(ex: ExtractorResult) -> None:
                async with item_sem:
                    try:
                        result = await _critique_one(
                            critic_client,
                            ex,
                            max_tokens=CONFIG.critic_max_tokens,
                        )
                        if result is None:
                            tracker.tick(error=True)
                            return
                        writer.write(result)
                        tracker.tick()
                    except Exception as e:
                        logger.exception(f"[{ex.item_id}] Critic unexpected: {e}")
                        tracker.tick(error=True)

            tasks = [process(ex) for ex in pending]
            await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(tracker.summary())
        logger.info(f"[{label}] Výstup: {output_path.name}")
