"""
Fáze 1 — Extrakce.

Iteruje dataset, pro každý item sestaví prompt, zavolá Qwen3.5-122B,
validuje odpověď přes Pydantic, zapíše do JSONL. Resume-safe.

Self-repair: pokud extractor vrátí nevalidní JSON nebo validace selže, pustíme
retry s self-repair promptem (dá mu jeho předchozí odpověď + validační chybu a
žádá opravu). Max 1x self-repair aby to nebyl nekonečný loop.

Pozice entit (start_char/end_char) se NEPŘEBÍRAJÍ od LLM — vypočítá je
deterministický kód v labeler.positions z `entity.text` + původního textu.
Tím je zaručen invariant text[start:end] == entity.text potřebný pro
spaCy/Stanza training.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

from pydantic import ValidationError

from .config import CONFIG, extraction_path
from .datasets import LabelItem
from .io_utils import (
    JsonlWriter,
    ProgressTracker,
    load_completed_ids,
    load_hash_cache,
    text_hash,
)
from .llm_client import LLMClient, LLMFatalError, LLMTransientError
from .positions import compute_entity_positions
from .prompts import build_extractor_messages
from .schemas import ExtractorResult, SkoGraph
from .taxonomy import load_taxonomy_if_exists

logger = logging.getLogger(__name__)


SELF_REPAIR_MSG = """Tvá předchozí odpověď nebyla validní. Chyba:
{error}

Oprav chybu a vrať VALIDNÍ JSON s klíči "entities" a "relations". Žádný text navíc."""


async def _extract_one(
    client: LLMClient,
    item: LabelItem,
    *,
    max_tokens: int,
    taxonomy=None,
) -> Optional[ExtractorResult]:
    """
    Zavolá extractor pro jeden item. Vrátí ExtractorResult nebo None při fatal error.
    Implementuje jedno kolo self-repairu při validační chybě.

    Pokud `taxonomy` je předáno (z labeler.taxonomy), použije se místo defaultu
    16 kategorií — prompt bude přizpůsoben labelům z taxonomie.
    """
    # Split NER/RE (default ON): entity a relace zvlášť, dvě LLM volání.
    if CONFIG.split_ner_re:
        from .ner_re_split import extract_graph_split

        res = await extract_graph_split(
            client, item, max_tokens=max_tokens, taxonomy=taxonomy
        )
        if res is None:
            return None
        graph, latency = res
        return ExtractorResult(
            item_id=item.item_id,
            source_dataset=item.source,
            input_hash=text_hash(item.text),
            input_text=item.text,
            is_synthetic=item.is_synthetic,
            graph=graph,
            model=client.model,
            latency_ms=latency,
            retries=0,
            context=item.context,
        )

    messages = build_extractor_messages(
        text=item.text,
        domain=item.domain,
        context=item.context,
        taxonomy=taxonomy,
    )

    total_retries = 0
    resp = None
    validation_error: Optional[str] = None

    for attempt in range(2):  # 0 = initial, 1 = self-repair
        try:
            resp = await client.chat(
                messages,
                max_tokens=max_tokens,
                reasoning=CONFIG.extractor_reasoning,
            )
        except LLMFatalError as e:
            logger.error(f"[{item.item_id}] LLMFatalError: {e}")
            return None
        except LLMTransientError as e:
            logger.error(f"[{item.item_id}] Transient error vyčerpal retries: {e}")
            return None

        total_retries += resp.retries
        if resp.parsed is None:
            if getattr(resp, "finish_reason", None) == "length":
                validation_error = (
                    f"RUNAWAY — model dojel do max_tokens "
                    f"(finish_reason=length, out_tokens={resp.output_tokens}) "
                    f"→ useknutý JSON"
                )
            else:
                validation_error = "Odpověď nebyla validní JSON"
        else:
            try:
                # Bare list (bez {"entities":...,"relations":...}) = entity list;
                # combined prompt žádá graf, takže list zabalíme jako entities.
                parsed = resp.parsed
                if isinstance(parsed, list):
                    parsed = {"entities": parsed}
                graph = SkoGraph(**parsed)
                graph.build_triples()
                compute_entity_positions(item.text, graph)
                return ExtractorResult(
                    item_id=item.item_id,
                    source_dataset=item.source,
                    input_hash=text_hash(item.text),
                    input_text=item.text,
                    is_synthetic=item.is_synthetic,
                    graph=graph,
                    model=client.model,
                    latency_ms=resp.latency_ms,
                    retries=total_retries,
                    context=item.context,
                )
            except ValidationError as e:
                validation_error = (
                    f"Pydantic validation: {e.errors(include_url=False)[:3]}"
                )
            except Exception as e:
                validation_error = f"{type(e).__name__}: {e}"

        if attempt == 0:
            messages = messages + [
                {"role": "assistant", "content": resp.content if resp else "(empty)"},
                {
                    "role": "user",
                    "content": SELF_REPAIR_MSG.format(error=validation_error),
                },
            ]

    logger.warning(
        f"[{item.item_id}] Extrakce selhala i po self-repairu: {validation_error}"
    )
    return None


async def run_extraction(
    items: Iterable[LabelItem],
    *,
    dry_run_limit: Optional[int] = None,
    skip_synthetic: bool = False,
    only_synthetic: bool = False,
    taxonomy=None,
) -> None:
    """
    Hlavní loop fáze 1.

    items:          iterator ze scan_data_dir()
    dry_run_limit:  pokud zadáno, zpracuj jen prvních N itemů (pro test)
    skip_synthetic: vynech itemy z syntetického textu (source_kind='synthetic')
    only_synthetic: zpracuj POUZE syntetické itemy (pro separátní run)

    Výstup:
      extractions.jsonl            — reálné popisky (productDescription)
      extractions_synthetic.jsonl  — synteticky poskládané texty
    """
    if skip_synthetic and only_synthetic:
        raise ValueError("skip_synthetic a only_synthetic jsou vzájemně vyloučené")

    real_path = extraction_path(synthetic=False)
    synth_path = extraction_path(synthetic=True)

    real_completed = load_completed_ids(real_path, "item_id")
    synth_completed = load_completed_ids(synth_path, "item_id")
    real_hash_cache = load_hash_cache(real_path, "input_hash")
    synth_hash_cache = load_hash_cache(synth_path, "input_hash")

    # Pokud uživatel neposlal taxonomy, zkusíme načíst z checkpoints/entity_taxonomy.json
    # (volitelná fáze 0). Pokud neexistuje, použije se default 16 kategorií z EntityCategory.
    if taxonomy is None:
        taxonomy = load_taxonomy_if_exists()
    if taxonomy is not None:
        logger.info(
            f"Extractor používá vlastní taxonomii ({len(taxonomy.labels)} labelů, "
            f"mode={taxonomy.mode}): {taxonomy.codes}"
        )
    else:
        logger.info(
            "Extractor používá DEFAULT taxonomii (16 kategorií). "
            "Pro vlastní spusť `python -m labeler.main taxonomy --mode auto|user`."
        )

    logger.info(
        f"Extraktor start: REAL {len(real_completed)} hotovo / {len(real_hash_cache)} hash, "
        f"SYNTH {len(synth_completed)} hotovo / {len(synth_hash_cache)} hash"
    )

    all_items = list(items)
    if dry_run_limit:
        all_items = all_items[:dry_run_limit]
        logger.warning(f"DRY RUN: omezeno na prvních {dry_run_limit} itemů")

    real_pending = []
    synth_pending = []
    for it in all_items:
        if it.is_synthetic:
            if skip_synthetic:
                continue
            if it.item_id not in synth_completed:
                synth_pending.append(it)
        else:
            if only_synthetic:
                continue
            if it.item_id not in real_completed:
                real_pending.append(it)

    total_real = sum(
        1 for it in all_items if not it.is_synthetic and not only_synthetic
    )
    total_synth = sum(1 for it in all_items if it.is_synthetic and not skip_synthetic)
    logger.info(
        f"Ke zpracování: REAL {len(real_pending)}/{total_real}, "
        f"SYNTH {len(synth_pending)}/{total_synth}"
    )

    tracker = ProgressTracker(
        total=len(real_pending) + len(synth_pending), label="extract"
    )

    async with LLMClient(
        base_url=CONFIG.extractor_base_url,
        model=CONFIG.extractor_model,
        max_concurrency=CONFIG.extractor_concurrency,
        request_timeout=CONFIG.request_timeout_s,
        max_retries=CONFIG.max_retries,
    ) as client:
        await client.wait_ready(max_wait_s=1200)

        with JsonlWriter(real_path) as real_writer, JsonlWriter(
            synth_path
        ) as synth_writer:

            # Omezí počet současně zpracovávaných itemů. Bez tohoto by se u
            # split NER/RE fáze spustilo všech 250k itemů najednou, první část (NER)
            # by zaplnila frontu HTTP klienta, a druhá část (RE) by se zařadila
            # až na konec 250k dlouhé fronty. Žádný item by tak nebyl kompletně
            # zpracován a zapsán po celé hodiny.
            item_sem = asyncio.Semaphore(CONFIG.extractor_concurrency)

            def _pick_writer_and_cache(item: LabelItem):
                if item.is_synthetic:
                    return synth_writer, synth_hash_cache
                return real_writer, real_hash_cache

            async def process(item: LabelItem) -> None:
                async with item_sem:
                    try:
                        writer, hash_cache = _pick_writer_and_cache(item)
                        h = text_hash(item.text)
                        if h in hash_cache:
                            cached = dict(hash_cache[h])
                            cached["item_id"] = item.item_id
                            cached["source_dataset"] = item.source
                            cached["is_synthetic"] = item.is_synthetic
                            writer.write(cached)
                            tracker.tick(skipped=True)
                            return

                        result = await _extract_one(
                            client, item,
                            max_tokens=CONFIG.extractor_max_tokens,
                            taxonomy=taxonomy,
                        )
                        if result is None:
                            tracker.tick(error=True)
                            return
                        writer.write(result)
                        hash_cache[h] = result.model_dump(mode="json")
                        tracker.tick()
                    except Exception as e:
                        logger.exception(f"[{item.item_id}] Neočekávaná chyba: {e}")
                        tracker.tick(error=True)

            tasks = [process(it) for it in real_pending + synth_pending]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)

    logger.info(tracker.summary())
    logger.info(f"Výstupy: {real_path.name} + {synth_path.name}")
