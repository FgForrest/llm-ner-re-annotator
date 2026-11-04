"""
Split NER/RE (CONFIG.split_ner_re) — každou fázi rozdělí na DVĚ samostatná
LLM volání: jedno na entity (NER), druhé na relace (RE).

Motivace: model se soustředí na jednu úlohu (lepší kvalita, méně malformed
relací). RE volání vždy dostane už hotové entity jako kontext — relace na ně
odkazují přes subjectId/objectId.

Tady je centralizovaná orchestrace pro všechny tři fáze (extract / critic /
revise). Fázové soubory (extractor.py, critic.py, revise.py) jen vETVÍ na
CONFIG.split_ner_re a zavolají odpovídající funkci níže.

Tolerantní sestavení (`assemble_graph`): z částí (entity + relace) udělá VŽDY
validní SkoGraph — dropne nevalidní entity/relace, dangling relace (na
neexistující entitu), self-loops i duplicitní ID. Nikdy nevyhodí výjimku.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import CONFIG
from .datasets import LabelItem
from .llm_client import LLMClient, LLMFatalError, LLMTransientError
from .positions import compute_entity_positions
from .prompts import (
    build_critic_ner_messages,
    build_critic_re_messages,
    build_extractor_ner_messages,
    build_extractor_re_messages,
    build_revision_ner_messages,
    build_revision_re_messages,
)
from .schemas import CriticResult, Entity, ExtractorResult, Relation, SkoGraph

logger = logging.getLogger(__name__)

_VALID_STATUS = ("ok", "revised", "reject")


# ---------------------------------------------------------------------------
# Pure helpers (deterministické, snadno testovatelné)
# ---------------------------------------------------------------------------


def extract_list(parsed, key: str) -> list:
    """Vytáhne list pod `key` ('entities'/'relations') z parsed odpovědi.
    Toleruje: bare list, {key:[...]}, i zabalení do 'corrected'/'graph'."""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    v = parsed.get(key)
    if isinstance(v, list):
        return v
    for wrap in ("corrected", "graph"):
        w = parsed.get(wrap)
        if isinstance(w, dict) and isinstance(w.get(key), list):
            return w[key]
    return []


def _as_dict(parsed) -> dict:
    """Bezpečný dict pohled na parsed odpověď pro .get přístup (status/reasoning).
    Model občas vrátí bare list (entity/relace) místo objektu se statusem —
    `(list).get(...)` by hodil AttributeError, proto list degradujeme na {}
    (= default 'ok', žádné poznámky). Pro extrakci entit/relací se používá
    extract_list(), který bare list zpracuje korektně."""
    return parsed if isinstance(parsed, dict) else {}


def assemble_graph(entity_dicts: list, relation_dicts: list) -> SkoGraph:
    """Sestaví VALIDNÍ SkoGraph z částí. Tolerantně dropne nevalidní entity/
    relace, dangling relace (na neexistující entitu), self-loops a duplicitní
    ID — takže SkoGraph konstruktor (který by na tyhle případy vyhodil
    výjimku) projde. Nikdy neraisuje."""
    entities: list[Entity] = []
    seen: set[str] = set()
    for ed in entity_dicts or []:
        if not isinstance(ed, dict):
            continue
        try:
            e = Entity(**ed)
        except Exception:  # noqa: BLE001 — nevalidní entitu prostě zahodíme
            continue
        if e.id in seen:
            continue
        seen.add(e.id)
        entities.append(e)

    relations: list[Relation] = []
    for rd in relation_dicts or []:
        if not isinstance(rd, dict):
            continue
        try:
            # self-loop (subjectId == objectId) → Relation raisuje → skip
            r = Relation(**rd)
        except Exception:  # noqa: BLE001
            continue
        if r.subjectId not in seen or r.objectId not in seen:
            continue  # dangling reference
        relations.append(r)

    graph = SkoGraph(entities=entities, relations=relations)
    graph.build_triples()
    return graph


# ---------------------------------------------------------------------------
# Jedno LLM volání → parsed dict
# ---------------------------------------------------------------------------


async def _call_json(
    client: LLMClient,
    messages: list[dict],
    max_tokens: int,
    reasoning: bool,
    *,
    item_id: str = "",
    what: str = "",
    repetition_penalty: Optional[float] = None,
):
    """Jedno chat volání → (parsed_dict_or_None, resp_or_None).

    BEZ self-repair retry — když model nevrátí validní JSON, NEopakujeme (retry
    by jen 2× zopakoval drahé generování, a když model jede do max_tokens, je to
    plýtvání). Místo toho jen ZALOGUJEME, co je špatně: fáze, item, DÉLKU
    odpovědi (odhalí runaway do max_tokens) a snippet."""
    tag = (f"[{item_id}] split {what}").strip()
    try:
        resp = await client.chat(
            messages,
            max_tokens=max_tokens,
            reasoning=reasoning,
            repetition_penalty=repetition_penalty,
        )
    except (LLMFatalError, LLMTransientError) as e:
        logger.warning(f"{tag}: LLM chyba: {e}")
        return None, None
    if resp.parsed is None:
        content = resp.content or ""
        runaway = getattr(resp, "finish_reason", None) == "length"
        cause = (
            "RUNAWAY — model dojel do max_tokens (finish_reason=length), "
            "JSON je useknutý"
            if runaway
            else "nevrátil validní JSON"
        )
        logger.warning(
            f"{tag}: {cause} (délka odpovědi {len(content)} znaků, "
            f"out_tokens={getattr(resp, 'output_tokens', '?')}, "
            f"retries={getattr(resp, 'retries', '?')}): {content[:300]!r}"
        )
        return None, resp
    return resp.parsed, resp


def _lat(*resps) -> int:
    return int(sum(getattr(r, "latency_ms", 0) or 0 for r in resps if r is not None))


# ---------------------------------------------------------------------------
# FÁZE 1 — extrakce split
# ---------------------------------------------------------------------------


async def extract_graph_split(
    client: LLMClient, item: LabelItem, *, max_tokens: int, taxonomy=None
):
    """NER volání (entity) → RE volání (relace nad entitami) → sestaví graf.

    Vrací (SkoGraph, latency_ms) nebo None, když NER vůbec nevrátí použitelný
    JSON (extrakce pro item selhala — stejně jako combined cesta vrací None)."""
    # NER volání: menší token strop (entity = malý JSON) + repetition penalty —
    # cílené brzdy proti runaway, který se objevil po zavedení splitu. Combined
    # cesta (extractor._extract_one bez splitu) je beze změny.
    ner_parsed, ner_resp = await _call_json(
        client,
        build_extractor_ner_messages(item.text, item.domain, item.context, taxonomy),
        CONFIG.extractor_ner_max_tokens,
        CONFIG.extractor_reasoning,
        item_id=item.item_id,
        what="extract-NER",
        repetition_penalty=CONFIG.extractor_split_repetition_penalty,
    )
    if ner_parsed is None:
        return None
    entity_dicts = extract_list(ner_parsed, "entities")

    re_parsed, re_resp = await _call_json(
        client,
        build_extractor_re_messages(
            item.text, item.domain, entity_dicts, item.context, taxonomy
        ),
        max_tokens,
        CONFIG.extractor_reasoning,
        item_id=item.item_id,
        what="extract-RE",
        repetition_penalty=CONFIG.extractor_split_repetition_penalty,
    )
    relation_dicts = extract_list(re_parsed, "relations")

    graph = assemble_graph(entity_dicts, relation_dicts)
    compute_entity_positions(item.text, graph)
    return graph, _lat(ner_resp, re_resp)


# ---------------------------------------------------------------------------
# FÁZE 2 — critic split
# ---------------------------------------------------------------------------


async def critique_split(
    client: LLMClient, extraction: ExtractorResult, *, max_tokens: int,
    taxonomy=None,
) -> Optional[CriticResult]:
    """Critic split — NER NEJDŘÍV, pak RE (s entitami, jak je vyhodnotil NER
    krok, jako kontextem). Merge do jednoho CriticResult (formát beze změny;
    revise/finalize nic nepozná).

    Striktní pořadí: RE critic dostane entity PO NER kroku (po opravě/dropnutí),
    ne původní extractor výstup → relace se validují vůči skutečně potvrzeným
    entitám.
    """
    text = extraction.input_text
    g = extraction.graph.model_dump(mode="json", exclude={"triples"})
    orig_entities = g.get("entities", [])
    orig_relations = g.get("relations", [])
    ctx = extraction.context

    # Anti-runaway brzda i pro critic: thinking (reasoning=ON) se na SGLangu
    # občas zacyklí a dojede do max_tokens. Rep penalty smyčku rozbije.
    rep = CONFIG.critic_split_repetition_penalty

    # 1) NER critic NEJDŘÍV — validuje/opraví entity.
    ner_parsed, ner_resp = await _call_json(
        client,
        build_critic_ner_messages(text, orig_entities, ctx, taxonomy),
        max_tokens,
        CONFIG.critic_reasoning,
        item_id=extraction.item_id,
        what="critic-NER",
        repetition_penalty=rep,
    )
    ner_status = _as_dict(ner_parsed).get("status", "ok")
    if ner_status not in _VALID_STATUS:
        ner_status = "ok"
    ner_reason = _as_dict(ner_parsed).get("reasoning")

    # NER reject → entity jsou nesmysl → reject celý item (RE už nevoláme).
    if ner_status == "reject":
        return CriticResult(
            item_id=extraction.item_id,
            status="reject",
            reasoning=f"[NER reject] {(ner_reason or '')[:1200]}"[:2000],
            corrected=None,
            extractor_revised=None,
            extractor_revised_reasoning=None,
            model=client.model,
            latency_ms=_lat(ner_resp),
        )

    # Entity PO NER kroku → jdou do RE jako kontext (striktní NER → RE handoff).
    if ner_status == "revised":
        ner_corr = extract_list(ner_parsed, "entities")
        entities_after_ner = ner_corr if ner_corr else orig_entities
    else:
        entities_after_ner = orig_entities

    # 2) RE critic — dostane entity, jak je vyhodnotil NER krok (ne původní).
    re_parsed, re_resp = await _call_json(
        client,
        build_critic_re_messages(text, orig_relations, entities_after_ner, ctx),
        max_tokens,
        CONFIG.critic_reasoning,
        item_id=extraction.item_id,
        what="critic-RE",
        repetition_penalty=rep,
    )
    latency = _lat(ner_resp, re_resp)

    # Obě půlky bez JSON → critic nedoběhl → reject (neoznačit tiše jako ok).
    if ner_parsed is None and re_parsed is None:
        return CriticResult(
            item_id=extraction.item_id,
            status="reject",
            reasoning=(
                "Critic (NER i RE) nevrátil validní JSON — extractor output "
                "zůstal BEZ kritické validace (není to potvrzení správnosti)."
            ),
            corrected=None,
            extractor_revised=None,
            extractor_revised_reasoning=None,
            model=client.model,
            latency_ms=latency,
        )

    re_status = _as_dict(re_parsed).get("status", "ok")
    if re_status not in _VALID_STATUS:
        re_status = "ok"
    re_reason = _as_dict(re_parsed).get("reasoning")

    reasoning = (
        f"[NER {ner_status}] {(ner_reason or 'critic nevrátil validní JSON')[:600]} | "
        f"[RE {re_status}] {(re_reason or 'critic nevrátil validní JSON')[:600]}"
    )[:2000]

    # Výsledné relace.
    if re_status == "reject":
        relations_final: list = []  # zahoď všechny relace
    elif re_status == "revised":
        re_corr = extract_list(re_parsed, "relations")
        relations_final = re_corr if re_corr else orig_relations
    else:
        relations_final = orig_relations

    changed = (ner_status == "revised") or (re_status in ("revised", "reject"))
    if changed:
        graph = assemble_graph(entities_after_ner, relations_final)
        compute_entity_positions(text, graph)
        return CriticResult(
            item_id=extraction.item_id,
            status="revised",
            reasoning=reasoning,
            corrected=graph,
            extractor_revised=None,
            extractor_revised_reasoning=None,
            model=client.model,
            latency_ms=latency,
        )

    return CriticResult(
        item_id=extraction.item_id,
        status="ok",
        reasoning=reasoning,
        corrected=None,
        extractor_revised=None,
        extractor_revised_reasoning=None,
        model=client.model,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# FÁZE 2b — revise split
# ---------------------------------------------------------------------------


async def revise_split(
    client: LLMClient,
    extraction: ExtractorResult,
    critic_result: CriticResult,
    *,
    max_tokens: int,
    reasoning: bool,
    taxonomy=None,
) -> Optional[CriticResult]:
    """Revise split — NER NEJDŘÍV, pak RE (s NER-revidovanými entitami jako
    kontextem). Sestaví extractor_revised. Vrací updated CriticResult, nebo
    None (no-op) když oba revisor calls selhaly."""
    text = extraction.input_text
    g = extraction.graph.model_dump(mode="json", exclude={"triples"})
    orig_entities = g.get("entities", [])
    orig_relations = g.get("relations", [])
    ctx = extraction.context

    corr = (
        critic_result.corrected.model_dump(mode="json", exclude={"triples"})
        if critic_result.corrected
        else None
    )
    corr_entities = (corr or {}).get("entities") if corr else None
    corr_relations = (corr or {}).get("relations") if corr else None

    # Stejné anti-runaway brzdy jako u extract splitu (revisor = defaultně tentýž
    # Qwen, stejný režim i riziko zacyklení): NER má menší token strop a obě
    # volání dostanou repetition_penalty. Sdílí konfig s extractem.
    rep = CONFIG.extractor_split_repetition_penalty

    # 1) NER revise NEJDŘÍV.
    ner_parsed, _ = await _call_json(
        client,
        build_revision_ner_messages(
            text, orig_entities, critic_result.status, critic_result.reasoning,
            corr_entities, ctx, taxonomy,
        ),
        CONFIG.extractor_ner_max_tokens,
        reasoning,
        item_id=extraction.item_id,
        what="revise-NER",
        repetition_penalty=rep,
    )
    # Entity PO NER revisi → jdou do RE jako kontext (striktní NER → RE handoff).
    entities_after_ner = extract_list(ner_parsed, "entities") or orig_entities

    # 2) RE revise — dostane NER-revidované entity (ne původní).
    re_parsed, _ = await _call_json(
        client,
        build_revision_re_messages(
            text, orig_relations, entities_after_ner, critic_result.status,
            critic_result.reasoning, corr_relations, ctx,
        ),
        max_tokens,
        reasoning,
        item_id=extraction.item_id,
        what="revise-RE",
        repetition_penalty=rep,
    )
    if ner_parsed is None and re_parsed is None:
        return None  # revisor selhal → no-op (ponechá se původní CriticResult)

    relations_final = extract_list(re_parsed, "relations") or orig_relations

    graph = assemble_graph(entities_after_ner, relations_final)
    compute_entity_positions(text, graph)

    updated = critic_result.model_copy(
        update={
            "extractor_revised": graph,
            "extractor_revised_reasoning": (
                f"Revisor split NER/RE re-eval (status={critic_result.status})"
            ),
        }
    )
    logger.info(
        f"[{extraction.item_id}] Revisor split OK: "
        f"{len(graph.entities)} entit, {len(graph.relations)} relací"
    )
    return updated
