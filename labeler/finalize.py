"""
Fáze 3 — Finalizace (merge + stats).

Produkuje finální datasety:

  - final_labels.jsonl
  - final_labels_synthetic.jsonl

a zároveň klasické JSON soubory:

  - final_labels.json
  - final_labels_synthetic.json

Důležité:
- JSONL se stále zachová kvůli práci s většími datasety.
- JSON se vytvoří jako klasické pole objektů, tedy formálně validní JSON soubor.
- Finální soubory se vždy přepisují atomicky, takže při opakovaném spuštění
  finalize nevzniknou duplicity.

Logika merge je pro obě větve stejná:

  - status='ok'       → extractor graph
  - status='revised'  → critic.corrected graph
  - status='reject'   → zachovat extractor graph, flag was_rejected

Navíc:
- Pokud jsou dokumenty rozdělené na chunky pomocí suffixu #chunkN,
  finalize je sloučí zpět do jednoho finálního záznamu podle původního item_id.

Příklad:

  demo:documents:abc#chunk0
  demo:documents:abc#chunk1
  demo:documents:abc#chunk2

se sloučí do:

  demo:documents:abc
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from .category_vote import CorpusCategoryStats, apply_corpus_votes
from .config import CONFIG, critic_path, extraction_path, final_path, stats_path
from .io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from .normalize import (
    DEFAULT_NER_LABELS,
    NormalizationResult,
    NormalizeOpts,
    normalize_graph,
)
from .positions import compute_entity_positions
from .schemas import (
    CriticResult,
    Entity,
    ExtractorResult,
    FinalLabel,
    Relation,
    SkoGraph,
)

logger = logging.getLogger(__name__)


def _to_jsonable(record: Any) -> dict:
    """
    Převede Pydantic model nebo dict na JSON serializovatelný dict.
    """
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")

    if isinstance(record, dict):
        return record

    raise TypeError(f"Nelze serializovat typ {type(record)}")


def _json_array_path(jsonl_path: Path) -> Path:
    """
    Z cesty k JSONL souboru vytvoří odpovídající cestu ke klasickému JSON souboru.

    Např.:
      final_labels.jsonl -> final_labels.json
      final_labels_synthetic.jsonl -> final_labels_synthetic.json
    """
    if jsonl_path.suffix == ".jsonl":
        return jsonl_path.with_suffix(".json")

    return jsonl_path.with_name(jsonl_path.name + ".json")


def _write_final_outputs(jsonl_path: Path, records: list[dict]) -> tuple[Path, Path]:
    """
    Zapíše finální výstup ve dvou formátech:

      1. JSONL — vhodné pro datasety a streamování (atomic přepis, ne append)
      2. JSON  — klasický JSON soubor jako pole objektů

    Vrací:
      (jsonl_path, json_path)
    """
    json_path = _json_array_path(jsonl_path)

    atomic_write_jsonl(jsonl_path, records)
    atomic_write_json(json_path, records)

    return jsonl_path, json_path


# ---------------------------------------------------------------------------
# Oddělení keyphrase entit od hlavního datasetu (FINAL_EXCLUDE_KEYPHRASES)
# ---------------------------------------------------------------------------


def _is_keyphrase_category(category: Any) -> bool:
    """Keyphrase = kategorie po normalizaci s prefixem KEYPHRASE_ (kind=keyphrase
    v taxonomii, remap_to KEYPHRASE_*, downgrade unsafe_texts na KEYPHRASE_SUBJECT)."""
    return str(category or "").startswith("KEYPHRASE_")


# Obecná slova z unsafe_texts (normalize je záměrně vyřadil z entitních tříd) —
# relace na ně se NEZACHOVÁVAJÍ, byly by nesmyslné ("PRODUKT has_benefit pojištění").
_GENERIC_KEYPHRASE = "KEYPHRASE_SUBJECT"


def _split_keyphrases(record: dict) -> tuple[dict, dict]:
    """
    Oddělí keyphrase entity od NER grafu, ale relace NEZAHODÍ.

    Výsledný záznam:
      graph       NER entity (bez keyphrase) + relace jen mezi nimi + triples —
                  čistá data pro NER trénink (záznam bez entit = platný
                  negativní příklad, text zůstává)
      keyphrases  {"entities", "relations", "triples"} — keyphrase entity
                  s platnou pozicí a relace, které se jich týkají. Relace
                  odkazují na id z graph.entities i keyphrases.entities (id
                  jsou v rámci záznamu unikátní) → pro RE trénink spoj obojí.

    Relace s keyphrase koncem se zachová JEN když je bezpečná:
      - oba konce existují, nejde o smyčku ani duplicitu,
      - aspoň jeden konec je NER entita (keyphrase ↔ keyphrase = šum),
      - keyphrase konec není KEYPHRASE_SUBJECT (obecné slovo) a má platný
        span v textu.
    Vrací (záznam, statistika) — záznam bez keyphrase se nemění.
    """
    stats: Counter[str] = Counter()
    graph = record.get("graph") or {}
    entities = graph.get("entities") or []
    relations = graph.get("relations") or []
    key_ids = {e["id"] for e in entities if _is_keyphrase_category(e.get("category"))}
    if not key_ids:
        return record, stats

    text = record.get("input_text", "")
    by_id = {e["id"]: e for e in entities}
    main_ents = [e for e in entities if e["id"] not in key_ids]
    main_rels, kp_rels, seen = [], [], set()
    for r in relations:
        s, o = r.get("subjectId"), r.get("objectId")
        if s not in key_ids and o not in key_ids:
            main_rels.append(r)
            continue
        kp_ends = [by_id[x] for x in (s, o) if x in key_ids]
        if s not in by_id or o not in by_id:
            reason = "dangling"
        elif s == o:
            reason = "self_loop"
        elif len(kp_ends) == 2:
            reason = "keyphrase_to_keyphrase"
        elif any(e.get("category") == _GENERIC_KEYPHRASE for e in kp_ends):
            reason = "generic_keyphrase"
        elif not all(_entity_has_valid_span(e, text) for e in kp_ends):
            reason = "invalid_span"
        elif (s, r.get("relation"), o) in seen:
            reason = "duplicate"
        else:
            seen.add((s, r.get("relation"), o))
            kp_rels.append(r)
            stats["relations_kept"] += 1
            continue
        stats[f"relations_dropped_{reason}"] += 1

    kp_ents = [
        e for e in entities
        if e["id"] in key_ids and _entity_has_valid_span(e, text)
    ]
    stats["entities_moved"] += len(kp_ents)
    stats["entities_dropped_invalid_span"] += len(key_ids) - len(kp_ents)

    g = SkoGraph(**{**graph, "entities": main_ents, "relations": main_rels})
    g.build_triples()
    lemma_by_id = {e["id"]: (e.get("lemma") or e.get("text", "")) for e in entities}
    kp_triples = [
        {"subject": lemma_by_id[r["subjectId"]], "relation": r["relation"],
         "object": lemma_by_id[r["objectId"]]}
        for r in kp_rels
    ]
    out = {
        **record,
        "graph": g.model_dump(mode="json"),
        "keyphrases": {"entities": kp_ents, "relations": kp_rels, "triples": kp_triples},
    }
    return out, stats


# ---------------------------------------------------------------------------
# spaCy split exports
# ---------------------------------------------------------------------------


def _entity_has_valid_span(entity: dict, text: str) -> bool:
    """True pokud entita má validní offset a substring sedí."""
    s = entity.get("start_char")
    e = entity.get("end_char")
    if s is None or e is None:
        return False
    if not (0 <= s < e <= len(text)):
        return False
    return text[s:e] == entity.get("text", "")


def _split_record_for_spacy(
    record: dict, allowed_ner_labels: frozenset[str],
) -> tuple[dict, dict, dict]:
    """
    Z jednoho final_label záznamu vytvoří 3 verze:
      - core_ner: jen entity s labely v allowed_ner_labels (PRODUKT, ORGANIZACE, …)
      - keyphrases: jen entity s labely KEYPHRASE_*
      - re: jen relace, kde subject i object mají validní span (může používat
             i keyphrase entity, ale invariant `text[start:end] == entity.text`
             musí platit)

    Každá verze má STEJNÉ top-level meta (item_id, source_dataset, input_text),
    ale `graph` jen s relevantními entitami / relacemi.
    """
    text = record.get("input_text", "")
    graph = record.get("graph", {})
    # Po FINAL_EXCLUDE_KEYPHRASES jsou keyphrase entity + jejich relace
    # v bloku "keyphrases" — pro splity je spojíme zpět.
    kp_block = record.get("keyphrases") or {}
    entities = list(graph.get("entities", [])) + list(kp_block.get("entities", []))
    relations = list(graph.get("relations", [])) + list(kp_block.get("relations", []))

    core_ents = [
        e for e in entities
        if e.get("category") in allowed_ner_labels
        and _entity_has_valid_span(e, text)
    ]
    key_ents = [
        e for e in entities
        if str(e.get("category", "")).startswith("KEYPHRASE_")
        and _entity_has_valid_span(e, text)
    ]

    # Pro RE: relace musí mít subject i object s validním spanem
    valid_ents_by_id = {
        e["id"]: e for e in entities if _entity_has_valid_span(e, text)
    }
    re_relations = [
        r for r in relations
        if r["subjectId"] in valid_ents_by_id
        and r["objectId"] in valid_ents_by_id
    ]
    # Pro RE export vrátíme jen entity zapojené v relacích
    re_ent_ids = set()
    for r in re_relations:
        re_ent_ids.add(r["subjectId"])
        re_ent_ids.add(r["objectId"])
    re_ents = [valid_ents_by_id[i] for i in re_ent_ids if i in valid_ents_by_id]

    def _wrap(ents: list[dict], rels: list[dict]) -> dict:
        return {
            **{k: v for k, v in record.items() if k not in ("graph", "keyphrases")},
            "graph": {
                "entities": ents,
                "relations": rels,
                # triples se v exportech přepočítají z entities+relations
                "triples": _rebuild_triples(ents, rels),
            },
        }

    return (
        _wrap(core_ents, []),       # core_ner: jen entity, žádné relace
        _wrap(key_ents, []),        # keyphrases: jen keyphrase entity
        _wrap(re_ents, re_relations),  # re: entity zapojené v relacích
    )


def _rebuild_triples(entities: list[dict], relations: list[dict]) -> list[dict]:
    """Deterministicky rebuild triples z entities + relations."""
    emap = {e["id"]: e for e in entities}
    out = []
    for r in relations:
        s = emap.get(r["subjectId"])
        o = emap.get(r["objectId"])
        if not s or not o:
            continue
        out.append({
            "subject": s.get("lemma") or s.get("text", ""),
            "relation": r["relation"],
            "object": o.get("lemma") or o.get("text", ""),
        })
    return out


def _write_spacy_split_exports(
    jsonl_path: Path, records: list[dict],
    *, allowed_ner_labels: frozenset[str],
) -> dict:
    """
    Zapíše 3 odvozené exporty vedle final_labels:
      - final_labels_core_ner.{jsonl,json}    pro spaCy NER trénink
      - final_labels_keyphrases.{jsonl,json}  keyphrase entity
      - final_labels_re.{jsonl,json}          relations s validními spany

    Vrací slovník s počty per export.
    """
    core_path = jsonl_path.with_name(jsonl_path.stem + "_core_ner" + jsonl_path.suffix)
    key_path = jsonl_path.with_name(jsonl_path.stem + "_keyphrases" + jsonl_path.suffix)
    re_path = jsonl_path.with_name(jsonl_path.stem + "_re" + jsonl_path.suffix)

    core_rows: list[dict] = []
    key_rows: list[dict] = []
    re_rows: list[dict] = []
    for rec in records:
        c, k, r = _split_record_for_spacy(rec, allowed_ner_labels)
        # Skip zápis prázdných záznamů (žádné entity → není co v dataset)
        if c["graph"]["entities"]:
            core_rows.append(c)
        if k["graph"]["entities"]:
            key_rows.append(k)
        if r["graph"]["relations"]:
            re_rows.append(r)

    atomic_write_jsonl(core_path, core_rows)
    atomic_write_json(_json_array_path(core_path), core_rows)
    atomic_write_jsonl(key_path, key_rows)
    atomic_write_json(_json_array_path(key_path), key_rows)
    atomic_write_jsonl(re_path, re_rows)
    atomic_write_json(_json_array_path(re_path), re_rows)

    return {
        "core_ner": {
            "path_jsonl": str(core_path),
            "records": len(core_rows),
            "entities": sum(len(r["graph"]["entities"]) for r in core_rows),
        },
        "keyphrases": {
            "path_jsonl": str(key_path),
            "records": len(key_rows),
            "entities": sum(len(r["graph"]["entities"]) for r in key_rows),
        },
        "re": {
            "path_jsonl": str(re_path),
            "records": len(re_rows),
            "relations": sum(len(r["graph"]["relations"]) for r in re_rows),
        },
    }


def _base_item_id(item_id: str) -> str:
    """
    Vrátí původní ID dokumentu bez chunk suffixu.

    Např.:
      demo:documents:abc#chunk0 -> demo:documents:abc
      demo:documents:abc#chunk1 -> demo:documents:abc
      eshop:123                 -> eshop:123
    """
    if "#chunk" in item_id:
        return item_id.rsplit("#chunk", 1)[0]

    return item_id


def _chunk_index(item_id: str) -> int:
    """
    Vrátí pořadí chunku podle suffixu #chunkN.
    Záznam bez chunku má index 0.
    """
    if "#chunk" not in item_id:
        return 0

    try:
        return int(item_id.rsplit("#chunk", 1)[1])
    except ValueError:
        return 0


def _norm_entity_text(value: str) -> str:
    """
    Normalizace pro deduplikaci entit mezi chunky.

    Cíl:
      "ACME Pojišťovna, a.s."
      "acme pojišťovna a.s."
    se mají chovat jako podobná / stejná entita.
    """
    return " ".join(
        value.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .replace(":", " ")
        .split()
    )


def _entity_merge_key(entity: Entity) -> tuple[str, str]:
    """
    Klíč pro spojování stejných entit z různých chunků.

    Používáme:
      - kategorii entity
      - lemma, pokud existuje
      - jinak text entity
    """
    text = entity.lemma or entity.text
    return entity.category, _norm_entity_text(text)


def _merge_graphs(graphs: list[SkoGraph]) -> SkoGraph:
    """
    Sloučí grafy z více chunků jednoho dokumentu do jednoho grafu.

    Co řeší:
      - přečísluje entity na e1, e2, ...
      - stejné entity napříč chunky deduplikuje
      - relace přemapuje na nová ID entit
      - odstraní duplicitní relace
      - odstraní relace, které po deduplikaci vedou z entity na sebe sama
      - znovu sestaví triples
    """
    merged_entities: list[Entity] = []
    merged_relations: list[Relation] = []

    entity_key_to_new_id: dict[tuple[str, str], str] = {}
    old_to_new_id: dict[tuple[int, str], str] = {}

    next_entity_id = 1

    for graph_index, graph in enumerate(graphs):
        for entity in graph.entities:
            key = _entity_merge_key(entity)

            if key in entity_key_to_new_id:
                new_id = entity_key_to_new_id[key]
            else:
                new_id = f"e{next_entity_id}"
                next_entity_id += 1

                entity_data = entity.model_dump(mode="json")
                entity_data["id"] = new_id

                merged_entities.append(Entity(**entity_data))
                entity_key_to_new_id[key] = new_id

            old_to_new_id[(graph_index, entity.id)] = new_id

    seen_relations: set[tuple[str, str, str]] = set()

    for graph_index, graph in enumerate(graphs):
        for relation in graph.relations:
            new_subject = old_to_new_id.get((graph_index, relation.subjectId))
            new_object = old_to_new_id.get((graph_index, relation.objectId))

            if not new_subject or not new_object:
                continue

            if new_subject == new_object:
                continue

            relation_key = (new_subject, relation.relation, new_object)

            if relation_key in seen_relations:
                continue

            seen_relations.add(relation_key)

            relation_data = relation.model_dump(mode="json")
            relation_data["subjectId"] = new_subject
            relation_data["objectId"] = new_object

            merged_relations.append(Relation(**relation_data))

    merged = SkoGraph(
        entities=merged_entities,
        relations=merged_relations,
    )
    merged.build_triples()

    return merged


def _merge_chunk_final_records(records: list[dict]) -> list[dict]:
    """
    Sloučí finální záznamy podle base item_id.

    Bez této funkce zůstávají chunky jako samostatné finální položky:

      demo:x:y#chunk0
      demo:x:y#chunk1

    Po sloučení vznikne:

      demo:x:y
    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        grouped[_base_item_id(record["item_id"])].append(record)

    merged_records: list[dict] = []

    for base_id, group in grouped.items():
        group = sorted(group, key=lambda r: _chunk_index(r["item_id"]))

        # Pokud dokument nebyl chunkovaný, necháme ho beze změny.
        if len(group) == 1 and group[0]["item_id"] == base_id:
            merged_records.append(group[0])
            continue

        graphs = [SkoGraph(**record["graph"]) for record in group]
        merged_graph = _merge_graphs(graphs)

        first = group[0]
        merged = dict(first)

        merged["item_id"] = base_id

        merged["input_text"] = "\n\n".join(
            record.get("input_text", "") for record in group
        )

        # Po merge chunků jsou ofsety z původních chunků neplatné vůči nové
        # konkatenaci. Dohledáme je znovu nad merged textem (FG162 — entity
        # span musí odpovídat input_text[start:end]).
        compute_entity_positions(merged["input_text"], merged_graph)

        merged["input_hash"] = first["input_hash"]

        merged["graph"] = merged_graph.model_dump(mode="json")

        statuses = [record.get("critic_status", "ok") for record in group]

        if "revised" in statuses:
            merged["critic_status"] = "revised"
        elif "reject" in statuses:
            merged["critic_status"] = "reject"
        else:
            merged["critic_status"] = "ok"

        merged["critic_reasoning"] = "\n\n".join(
            f"[{record['item_id']}] {record.get('critic_reasoning', '')}"
            for record in group
            if record.get("critic_reasoning")
        )

        merged["revisor_reasoning"] = "\n\n".join(
            f"[{record['item_id']}] {record.get('revisor_reasoning', '')}"
            for record in group
            if record.get("revisor_reasoning")
        ) or None

        sources = {
            record.get("final_graph_source", "extractor_original") for record in group
        }
        merged["final_graph_source"] = sources.pop() if len(sources) == 1 else "mixed"

        # changes_applied: sjednotit deterministické diffy z jednotlivých chunků
        merged_changes = {
            "entities_added": [], "entities_removed": [],
            "relations_added": [], "relations_removed": [],
        }
        for record in group:
            rc = record.get("changes_applied") or {}
            for k in merged_changes:
                merged_changes[k].extend(rc.get(k, []))
        merged["changes_applied"] = {k: sorted(set(v)) for k, v in merged_changes.items()}

        merged["was_revised"] = any(record.get("was_revised", False) for record in group)

        merged_records.append(merged)

    return merged_records


def _entity_keys(graph: SkoGraph) -> set:
    """Content key entity: (lemma|text lowercase, category) — stabilní napříč
    re-číslováním ID, takže diff měří OBSAH, ne náhodné ID."""
    keys = set()
    for e in graph.entities:
        label = (e.lemma or e.text or "").strip().lower()
        if label:
            keys.add((label, e.category))
    return keys


def _relation_keys(graph: SkoGraph) -> set:
    """Content key relace: (subj_label, relation, obj_label) — ID → lemma."""
    emap = {e.id: (e.lemma or e.text or "").strip().lower() for e in graph.entities}
    keys = set()
    for r in graph.relations:
        s, o = emap.get(r.subjectId), emap.get(r.objectId)
        if s and o:
            keys.add((s, r.relation, o))
    return keys


def _graph_diff(orig_e: set, orig_r: set, final_graph: SkoGraph) -> dict:
    """DETERMINISTICKÝ diff extractor-original → finální graf. Ground truth o
    tom, co se reálně změnilo (na rozdíl od LLM critic_reasoning, který může
    grafu odporovat). `orig_e`/`orig_r` jsou nasnapshotované PŘED mutací."""
    fe, fr = _entity_keys(final_graph), _relation_keys(final_graph)
    return {
        "entities_added": sorted(f"{t} ({c})" for t, c in (fe - orig_e)),
        "entities_removed": sorted(f"{t} ({c})" for t, c in (orig_e - fe)),
        "relations_added": sorted(f"{s} -{rel}-> {o}" for s, rel, o in (fr - orig_r)),
        "relations_removed": sorted(f"{s} -{rel}-> {o}" for s, rel, o in (orig_r - fr)),
    }


def _process_branch(
    *,
    label: str,
    is_synth: bool,
    normalize_opts: Optional[NormalizeOpts] = None,
    report_accumulator: Optional[NormalizationResult] = None,
) -> Optional[dict]:
    """
    Zpracuje jednu větev datasetu:

      - REAL
      - SYNTHETIC

    Vrátí statistiky větve, nebo None, pokud větev neexistuje.
    """
    ext_p = extraction_path(synthetic=is_synth)
    crit_p = critic_path(synthetic=is_synth)
    out_jsonl_p = final_path(synthetic=is_synth)

    if not ext_p.exists():
        logger.info(f"[{label}] {ext_p} neexistuje, přeskakuji")
        return None

    crit_map: dict[str, CriticResult] = {}

    if crit_p.exists():
        for rec in read_jsonl(crit_p):
            try:
                cr = CriticResult(**rec)
                crit_map[cr.item_id] = cr
            except ValidationError as e:
                logger.warning(
                    f"[{label}] Nevalidní critic {rec.get('item_id')}: "
                    f"{e.errors()[:1]}"
                )

        logger.info(f"[{label}] Načteno {len(crit_map)} critic výsledků")

    else:
        logger.warning(
            f"[{label}] {crit_p} neexistuje — finalizuji bez validace. "
            f"Všechny recordy dostanou status='ok'."
        )

    missing_critic = 0
    revised_examples: list[dict] = []
    final_records: list[dict] = []

    for rec in read_jsonl(ext_p):
        try:
            ex = ExtractorResult(**rec)
        except ValidationError as e:
            logger.warning(f"[{label}] Skip nevalidní extraction: {e.errors()[:1]}")
            continue

        # Snapshot PŮVODNÍHO (extractor) grafu PŘED jakoukoli mutací — pro
        # deterministický diff (co kritik/revisor reálně změnil).
        orig_e_keys = _entity_keys(ex.graph)
        orig_r_keys = _relation_keys(ex.graph)

        cr = crit_map.get(ex.item_id)
        revisor_reasoning: Optional[str] = None

        if cr is None:
            missing_critic += 1
            final_graph = ex.graph
            status = "ok"
            critic_reasoning = "Critic chybí nebo tento záznam nebyl zpracován."
            final_graph_source = "extractor_original"

        elif cr.status == "revised" and cr.extractor_revised is not None:
            # Feedback smyčka: revisor re-evaluoval kritiku a vrátil výsledek
            final_graph = cr.extractor_revised
            final_graph.build_triples()
            status = "revised"
            critic_reasoning = cr.reasoning
            revisor_reasoning = cr.extractor_revised_reasoning
            final_graph_source = "extractor_revised"

        elif cr.status == "revised" and cr.corrected is not None:
            # Standardní revised: použijeme graf navrhnutý criticem
            final_graph = cr.corrected
            final_graph.build_triples()
            status = "revised"
            critic_reasoning = cr.reasoning
            final_graph_source = "critic_corrected"

        elif cr.status == "reject" and cr.extractor_revised is not None:
            # Feedback smyčka pro reject: revisor re-evaluoval a rozhodl se
            final_graph = cr.extractor_revised
            final_graph.build_triples()
            status = "revised"
            critic_reasoning = cr.reasoning
            revisor_reasoning = cr.extractor_revised_reasoning
            final_graph_source = "extractor_revised"

        elif cr.status == "reject":
            final_graph = ex.graph
            status = "reject"
            critic_reasoning = cr.reasoning
            final_graph_source = "extractor_original"

        else:
            final_graph = ex.graph
            status = "ok"
            critic_reasoning = cr.reasoning
            final_graph_source = "extractor_original"

        # DEFENSIVE: recompute entity offsets proti aktuálnímu input_text.
        # Pozice už byly spočítané v extract/critic/revise fázích, ALE merge
        # boundary tu může selhat tiše: pokud LLM vrátil entitu s text="X"
        # a offsety byly spočítané proti chunkovanému textu, po sloučení
        # chunků (_merge_chunk_final_records) by mohly být offsety mimo.
        # `compute_entity_positions` udělá idempotentní recheck.
        # Garantuje invariant text[start:end] == entity.text po finalize.
        try:
            compute_entity_positions(ex.input_text, final_graph)
        except Exception as e:
            logger.warning(
                f"[{label}] compute_entity_positions selhalo pro "
                f"{ex.item_id}: {e}; pozice nechány tak, jak jsou."
            )

        # Pro jistotu vždy znovu sestavit triples podle finálních entit a relací.
        final_graph.build_triples()

        # GROUND-TRUTH diff: co se reálně změnilo extractor-original → finále.
        # (critic_reasoning může LLM-tvrdit jinak; tohle je počítané z grafů.)
        changes_applied = _graph_diff(orig_e_keys, orig_r_keys, final_graph)

        fl = FinalLabel(
            item_id=ex.item_id,
            source_dataset=ex.source_dataset,
            input_text=ex.input_text,
            input_hash=ex.input_hash,
            is_synthetic=ex.is_synthetic,
            graph=final_graph,
            critic_status=status,  # type: ignore[arg-type]
            critic_reasoning=critic_reasoning,
            revisor_reasoning=revisor_reasoning,
            final_graph_source=final_graph_source,
            changes_applied=changes_applied,
            extractor_model=ex.model,
            critic_model=cr.model if cr else "(missing)",
            was_revised=(status == "revised"),
        )

        fl_dict = _to_jsonable(fl)
        final_records.append(fl_dict)

        if status == "revised" and len(revised_examples) < 20:
            revised_examples.append(
                {
                    "item_id": ex.item_id,
                    "reasoning": critic_reasoning[:300],
                    "revisor_reasoning": (revisor_reasoning or "")[:300],
                    "final_graph_source": final_graph_source,
                    "changes_applied": changes_applied,
                    "orig_entities": len(ex.graph.entities),
                    "new_entities": len(final_graph.entities),
                    "orig_relations": len(ex.graph.relations),
                    "new_relations": len(final_graph.relations),
                }
            )

    # === Deterministická lemmatizace (Stanza, viz labeler.lemma) ============
    # PŘED merge chunků a PŘED hlasováním: konzistentní lemmata zlepšují
    # chunk-merge dedup klíče (_entity_merge_key) i hlasovací klíče
    # (category_vote.entity_key). Stanza dělá JEN lemma; best-effort — bez
    # knihovny/modelu se přeskočí a zůstanou LLM lemmata.
    lemma_summary: Optional[dict] = None
    if CONFIG.stanza_lemma_enabled and final_records:
        from .lemma import apply_lemmas_to_records, get_lemmatizer
        lemmatizer = get_lemmatizer()
        if lemmatizer.available:
            lemma_report = apply_lemmas_to_records(final_records, lemmatizer)
            lemma_summary = lemma_report.to_dict()
            logger.info(
                f"[{label}] stanza lemma: {lemma_report.lemmas_changed} lemmat "
                f"přepsáno ({lemma_report.docs_processed} dokumentů)"
            )
        else:
            logger.info(
                f"[{label}] stanza nedostupná — lemmata zůstávají z LLM "
                f"(pip install stanza pro deterministickou lemmatizaci)"
            )

    final_records = _merge_chunk_final_records(final_records)

    # === Korpusové hlasování o kategoriích (viz labeler.category_vote) ======
    # Finalize jako jediná fáze vidí celý korpus → statistika rozdělení
    # kategorií per lemma umí opravit menšinové/nekonzistentní labely, které
    # per-dokument fáze (extractor/critic) nemohly vidět. Plně data-driven,
    # žádná hard-coded slova, entropy guard (nejednoznačné klíče se nemění).
    # Stejné statistiky pak dostane normalize (konflikty, auto-expand safety).
    vote_stats: Optional[CorpusCategoryStats] = None
    vote_summary: Optional[dict] = None
    if CONFIG.category_vote_enabled and final_records:
        vote_stats = CorpusCategoryStats.build(
            final_records,
            dominance=CONFIG.category_vote_dominance,
            min_support=CONFIG.category_vote_min_support,
            max_df_ratio=CONFIG.auto_expand_max_df_ratio,
        )
        vote_report = apply_corpus_votes(final_records, vote_stats)
        vote_summary = vote_report.to_dict()
        logger.info(
            f"[{label}] category vote: {vote_report.relabeled} přelabelováno, "
            f"{vote_report.decisions} klíčů s jasnou dominancí, "
            f"{vote_report.ambiguous} nejednoznačných (nezměněno)"
        )

    # === Normalizační vrstva (volitelná) ====================================
    # Pokud opts.enabled=True, projde každý záznam přes normalize.normalize_graph
    # (fix offsets, auto-expand mentions, label remapping, conflict resolve,
    # overlap drop, drop uncertain, rebuild relations). Akumuluje counter do
    # `report_accumulator`, který run_finalize zapíše jako normalization_report.
    if normalize_opts and normalize_opts.enabled:
        logger.info(
            f"[{label}] Normalizace zapnuta: strict_offsets={normalize_opts.strict_offsets}, "
            f"auto_expand_mentions={normalize_opts.auto_expand_mentions}, "
            f"drop_uncertain_labels={normalize_opts.drop_uncertain_labels}"
        )
        for record in final_records:
            try:
                graph = SkoGraph(**record["graph"])
                norm_res = normalize_graph(
                    graph, record["input_text"], opts=normalize_opts,
                    corpus_stats=vote_stats,
                )
                record["graph"] = norm_res.graph.model_dump(mode="json")
                if report_accumulator is not None:
                    report_accumulator.merge(norm_res)
            except ValidationError as e:
                logger.warning(
                    f"[{label}] Normalizace selhala pro {record.get('item_id')}: "
                    f"{e.errors()[:1]}"
                )

    n_total = 0
    by_source: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    cat_counter: Counter[str] = Counter()
    rel_counter: Counter[str] = Counter()

    entity_counts: list[int] = []
    relation_counts: list[int] = []

    for record in final_records:
        graph = SkoGraph(**record["graph"])
        graph.build_triples()
        record["graph"] = graph.model_dump(mode="json")

        n_total += 1
        by_source[record["source_dataset"]] += 1
        by_status[record["critic_status"]] += 1

        entity_counts.append(len(graph.entities))
        relation_counts.append(len(graph.relations))

        for entity in graph.entities:
            cat_counter[entity.category] += 1

        for relation in graph.relations:
            rel_counter[relation.relation] += 1

    # Reject záznamy (critic označil 50 %+ entit za nesmysl) NEpatří do
    # hlavního trénovacího datasetu — naivní konzument by jinak trénoval na
    # vadných labelech. Necháme je v odděleném souboru pro audit/transparenci.
    # Statistiky (by_status, reject_rate) jsou spočítané OUTŠ nad VŠEMI záznamy,
    # takže reject_rate zůstává přesný.
    accepted_records = [
        r for r in final_records if r.get("critic_status") != "reject"
    ]
    rejected_records = [
        r for r in final_records if r.get("critic_status") == "reject"
    ]

    # Keyphrase entity (KEYPHRASE_*) nepatří do NER grafu — přesun do bloku
    # "keyphrases" v témže záznamu; bezpečné relace se zachovají (viz
    # _split_keyphrases). Zapnutí/vypnutí: FINAL_EXCLUDE_KEYPHRASES v .env.
    kp_stats: Counter[str] = Counter()
    kp_items = 0
    if CONFIG.final_exclude_keyphrases:
        split_records: list[dict] = []
        for r in accepted_records:
            new_r, st = _split_keyphrases(r)
            split_records.append(new_r)
            kp_stats.update(st)
            kp_items += int("keyphrases" in new_r)
        accepted_records = split_records

    written_jsonl_path, written_json_path = _write_final_outputs(
        out_jsonl_p,
        accepted_records,
    )

    if CONFIG.final_exclude_keyphrases:
        dropped = {
            k.replace("relations_dropped_", ""): v
            for k, v in kp_stats.items() if k.startswith("relations_dropped_")
        }
        logger.info(
            f"[{label}] keyphrase mimo NER graf: {kp_stats['entities_moved']} entit "
            f"v {kp_items} záznamech → blok 'keyphrases'; relace zachováno "
            f"{kp_stats['relations_kept']}, zahozeno {sum(dropped.values())} {dropped}"
        )

    # Rejected soubor zapisujeme vždy (i prázdný) — atomický přepis tak nikdy
    # nenechá stát starý reject výstup z předchozího běhu.
    rejected_jsonl_p = out_jsonl_p.with_name(
        out_jsonl_p.stem + "_rejected" + out_jsonl_p.suffix
    )
    rejected_jsonl_path, rejected_json_path = _write_final_outputs(
        rejected_jsonl_p,
        rejected_records,
    )
    if rejected_records:
        logger.info(
            f"[{label}] {len(rejected_records)} reject záznamů odděleno do "
            f"{rejected_jsonl_path.name} (mimo hlavní {written_jsonl_path.name})"
        )

    avg_entities = (
        round(sum(entity_counts) / len(entity_counts), 2) if entity_counts else 0
    )
    avg_relations = (
        round(sum(relation_counts) / len(relation_counts), 2) if relation_counts else 0
    )

    return {
        "label": label,
        "dataset_kind": "synthetic" if is_synth else "real",
        "total_items": n_total,
        "category_vote": vote_summary,
        "stanza_lemma": lemma_summary,
        "missing_critic": missing_critic,
        "by_source_dataset": dict(by_source),
        "by_critic_status": dict(by_status),
        "revision_rate_pct": round(
            100 * by_status.get("revised", 0) / max(n_total, 1),
            2,
        ),
        "reject_rate_pct": round(
            100 * by_status.get("reject", 0) / max(n_total, 1),
            2,
        ),
        "avg_entities_per_item": avg_entities,
        "avg_relations_per_item": avg_relations,
        "entity_categories_top": dict(cat_counter.most_common(20)),
        "relation_types_top": dict(rel_counter.most_common(20)),
        "revised_examples_first_20": revised_examples,
        "output_file_jsonl": str(written_jsonl_path),
        "output_file_json": str(written_json_path),
        "output_file": str(written_json_path),
        "accepted_items": len(accepted_records),
        "rejected_items": len(rejected_records),
        "keyphrases_excluded": CONFIG.final_exclude_keyphrases,
        "keyphrase_items": kp_items,
        "keyphrase_stats": dict(kp_stats),
        "output_file_rejected_jsonl": str(rejected_jsonl_path),
        "output_file_rejected_json": str(rejected_json_path),
        "extraction_file": str(ext_p),
        "critic_file": str(crit_p) if crit_p.exists() else None,
    }


def run_finalize(
    *,
    keep_intermediate: bool = False,
    normalize_opts: Optional[NormalizeOpts] = None,
    write_report: bool = False,
    export_spacy: bool = False,
    export_owl: bool = False,
    owl_per_source: bool = True,
    owl_combined: bool = True,
    owl_max_text_len: int = 5000,
    owl_write_html: bool = True,
) -> None:
    """
    Fáze 3 — sloučí extractor + critic do finálních datasetů.

    Výstupy:

      - final_labels.jsonl + final_labels.json
      - final_labels_synthetic.jsonl + final_labels_synthetic.json
      - stats.json
      - normalization_report.json (jen pokud write_report=True a normalize_opts.enabled)

    Parametry:
      keep_intermediate: nesmaže extractions*.jsonl a critic_results*.jsonl
                         po finalize (default False — výstup je čistý).
      normalize_opts:    NormalizeOpts. Pokud None nebo enabled=False, žádná
                         normalizace neproběhne (zachování existujícího chování).
      write_report:      zapsat normalization_report.json s detailními counters.
    """
    # Akumulátor pro normalization_report (sčítá per-branch counters)
    accumulator = NormalizationResult(graph=SkoGraph()) if write_report else None
    norm_active = normalize_opts is not None and normalize_opts.enabled

    real_stats = _process_branch(
        label="REAL", is_synth=False,
        normalize_opts=normalize_opts,
        report_accumulator=accumulator,
    )
    synth_stats = _process_branch(
        label="SYNTH", is_synth=True,
        normalize_opts=normalize_opts,
        report_accumulator=accumulator,
    )

    if real_stats is None and synth_stats is None:
        raise FileNotFoundError(
            "Nenalezeny žádné extraction soubory. " "Nejdřív spusť extract fázi."
        )

    combined_total = (real_stats["total_items"] if real_stats else 0) + (
        synth_stats["total_items"] if synth_stats else 0
    )

    stats = {
        "combined_total_items": combined_total,
        "real": real_stats,
        "synthetic": synth_stats,
        "intermediate_kept": keep_intermediate,
    }

    # spaCy split exports (pokud --export-spacy)
    if export_spacy:
        allowed = (
            normalize_opts.effective_allowed()
            if normalize_opts is not None else DEFAULT_NER_LABELS
        )
        spacy_stats: dict[str, dict] = {}
        for is_synth, branch in [(False, real_stats), (True, synth_stats)]:
            if branch is None:
                continue
            base_jsonl = final_path(synthetic=is_synth)
            records = list(read_jsonl(base_jsonl))
            sub = _write_spacy_split_exports(
                base_jsonl, records, allowed_ner_labels=allowed,
            )
            key = "real" if not is_synth else "synthetic"
            spacy_stats[key] = sub
            logger.info(
                f"[{key.upper()}] spaCy splits: "
                f"core_ner {sub['core_ner']['records']} rec / {sub['core_ner']['entities']} ents, "
                f"keyphrases {sub['keyphrases']['records']} rec / {sub['keyphrases']['entities']} ents, "
                f"re {sub['re']['records']} rec / {sub['re']['relations']} rels"
            )
        stats["spacy_exports"] = spacy_stats

    # OWL ontology export (volitelný — jen ve smoke testu)
    if export_owl:
        from .rdf_export import export_owl as _export_owl
        owl_stats: dict[str, dict] = {}
        for is_synth, branch in [(False, real_stats), (True, synth_stats)]:
            if branch is None:
                continue
            base_jsonl = final_path(synthetic=is_synth)
            try:
                records = list(read_jsonl(base_jsonl))
            except FileNotFoundError:
                continue
            sub = _export_owl(
                records, base_jsonl,
                per_source=owl_per_source,
                combined=owl_combined,
                max_text_len=owl_max_text_len,
                write_html=owl_write_html,
            )
            key = "real" if not is_synth else "synthetic"
            owl_stats[key] = sub
            combined_info = sub.get("combined") or {}
            logger.info(
                f"[{key.upper()}] OWL ontology: "
                f"combined={combined_info.get('owl')!r} "
                f"(triples={combined_info.get('triples')}), "
                f"per_source_files={len(sub.get('per_source', []))}"
            )
        stats["owl_exports"] = owl_stats

    atomic_write_json(stats_path(), stats)

    # Zapsat category_vote_report.json (pokud hlasování běželo) — auditní
    # stopa korpusového hlasování: co se přelabelovalo a které klíče zůstaly
    # nejednoznačné (= vstup pro disjointness audit v příštím běhu fáze 0).
    vote_reports = {
        key: branch.get("category_vote")
        for key, branch in (("real", real_stats), ("synthetic", synth_stats))
        if branch is not None and branch.get("category_vote") is not None
    }
    if vote_reports:
        vote_report_path = stats_path().parent / "category_vote_report.json"
        atomic_write_json(vote_report_path, vote_reports)
        logger.info(f"Category vote report: {vote_report_path}")

    # Zapsat normalization_report.json (pokud zapnuto)
    if write_report and accumulator is not None and norm_active:
        report_path = stats_path().parent / "normalization_report.json"
        atomic_write_json(report_path, accumulator.to_dict())
        logger.info(f"Normalization report: {report_path}")

    logger.info("=" * 70)
    logger.info("FINALIZE HOTOVO")
    logger.info("=" * 70)

    for name, branch_stats in [
        ("REAL", real_stats),
        ("SYNTHETIC", synth_stats),
    ]:
        if branch_stats is None:
            logger.info(f"[{name}] branch se neběžel")
            continue

        logger.info(
            f"[{name}] items={branch_stats['total_items']} "
            f"revision_rate={branch_stats['revision_rate_pct']}% "
            f"reject_rate={branch_stats['reject_rate_pct']}% "
            f"avg_ents={branch_stats['avg_entities_per_item']} "
            f"avg_rels={branch_stats['avg_relations_per_item']}"
        )

        logger.info(f"         JSONL → {branch_stats['output_file_jsonl']}")
        logger.info(f"         JSON  → {branch_stats['output_file_json']}")

    logger.info(f"Combined total items: {combined_total}")
    logger.info(f"Stats JSON: {stats_path()}")

    if keep_intermediate:
        logger.info(
            "keep_intermediate=True → mezivýstupy zachovány "
            "(extractions*.jsonl, critic_results*.jsonl)."
        )
    else:
        _cleanup_intermediate(real_stats, synth_stats)


def _cleanup_intermediate(
    real_stats: Optional[dict],
    synth_stats: Optional[dict],
) -> None:
    """
    Smaže mezivýstupy po úspěšném merge.

    Maže pouze soubory větví, které opravdu proběhly.
    Finální výstupy nikdy nemaže.
    """
    to_delete: list[Path] = []

    if real_stats is not None:
        to_delete.extend(
            [
                extraction_path(synthetic=False),
                critic_path(synthetic=False),
            ]
        )

    if synth_stats is not None:
        to_delete.extend(
            [
                extraction_path(synthetic=True),
                critic_path(synthetic=True),
            ]
        )

    deleted = 0

    for path in to_delete:
        if not path.exists():
            continue

        try:
            path.unlink()
            deleted += 1
            logger.info(f"Smazán mezivýstup: {path.name}")
        except OSError as e:
            logger.warning(f"Nepodařilo se smazat {path.name}: {e}")

    if deleted:
        logger.info(
            f"Cleanup hotov — smazáno {deleted} mezivýstupů. "
            f"Výsledné soubory: "
            f"final_labels.json, final_labels_synthetic.json, "
            f"final_labels.jsonl, final_labels_synthetic.jsonl, stats.json."
        )