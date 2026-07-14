"""
Relation normalization / type validation.

Deterministická vrstva nad LLM relacemi (žádné hard-coding domén):
  - drop relací s neexistujícím subject/object
  - drop relací se subject==object
  - deduplikace (subjectId, relation, objectId)
  - typová validace podle (subject_label, relation, object_label)
  - reverse fix jednoznačných obrácených směrů (ORG issued_by PRODUKT → PRODUKT issued_by ORG)
  - drop typově nesmyslných relací (KEYPHRASE_SUBJECT suitable_for OSOBA_ROLE)
  - drop located_in mířících na URL (Ministerstvo located_in https://...)

Generic-term filtrace pro objekty (covers "auto" / "pojištění" / "škoda") je
DATA-DRIVEN — definuje se v `entity_taxonomy.json` přes `unsafe_texts` per label.
Generic termíny se downgradují na KEYPHRASE_SUBJECT v normalize.py kroku 2 a tady
neprojdou typovou validací (KEYPHRASE_SUBJECT není v RISK_LIKE ani TARGET_LIKE).

Volá se z labeler.normalize.normalize_graph jako vrstva G+. Statistiky se zapisují
do `NormalizationResult` (countery relations_reversed_fixed, relations_dropped_*).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .schemas import Entity, Relation, SkoGraph

if TYPE_CHECKING:
    from .normalize import NormalizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label sets (kanonické po remappingu v normalize.py)
# ---------------------------------------------------------------------------

PRODUCT_LIKE = frozenset({"PRODUKT", "PRODUKT_TYP"})
ORG_LIKE = frozenset({"ORGANIZACE"})
KOMPONENTA_LIKE = frozenset({"KEYPHRASE_KOMPONENTA"})
PARAM_LIKE = frozenset({"KEYPHRASE_PARAMETR"})
VALUE_LIKE = frozenset({"HODNOTA"})
MATERIAL_LIKE = frozenset({"MATERIAL"})
BENEFIT_LIKE = frozenset({"KEYPHRASE_BENEFIT"})
TECH_LIKE = frozenset({"KEYPHRASE_TECH"})
SPEC_LIKE = frozenset({"KEYPHRASE_SPEC"})
PROPERTY_LIKE = frozenset({"KEYPHRASE_PROPERTY"})
# Pro covers — pojistná rizika a pojistné události.
# POZNÁMKA: KEYPHRASE_SUBJECT zde ZÁMĚRNĚ není — slouží jako "generic fallback"
# label, kam taxonomy.unsafe_texts shazuje příliš obecné termíny ("pojištění",
# "auto"). Pokud něco skončí v KEYPHRASE_SUBJECT, nemá se objevit jako platný
# object covers/excludes — tím se chrání kvalita RE datasetu.
RISK_LIKE = frozenset({
    "KEYPHRASE_USECASE", "KEYPHRASE_BENEFIT", "KEYPHRASE_PROPERTY",
})
# Pro suitable_for — cílová skupina, role, použití
TARGET_LIKE = frozenset({
    "OSOBA_ROLE", "KEYPHRASE_USECASE",
})
PLACE_LIKE = frozenset({"MISTO"})
URL_LIKE = frozenset({"URL"})

# Speciální token pro "libovolný label"
ANY = "*"


# ---------------------------------------------------------------------------
# Pravidla per relation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationTypeRule:
    """
    Typové pravidlo pro jednu relaci.

    Atributy:
      subject_labels: povolené labely subjektu (frozenset, nebo ANY)
      object_labels: povolené labely objektu (frozenset, nebo ANY)
      reverse_if_swapped: pokud True a subject/object jsou prohozené (subject
                          je v object_labels a object v subject_labels), relace
                          se otočí. Jinak se zachová jen pokud match strict.
      object_label_blacklist: pokud object má label v této množině, relace se
                              dropne (např. located_in s URL).

    POZN.: Generic-term filtrace pro objects (covers "auto" / "škoda") je
    DATA-DRIVEN přes taxonomy.unsafe_texts — pokud taxonomy obsahuje "auto"
    v unsafe_texts pro PRODUKT, entity "auto" se downgraduje na KEYPHRASE_SUBJECT
    a covers tu pak nepustí jako platný RISK_LIKE objekt. Žádné hard-coding zde.
    """
    subject_labels: object  # frozenset[str] | str (ANY)
    object_labels: object   # frozenset[str] | str (ANY)
    reverse_if_swapped: bool = False
    object_label_blacklist: frozenset[str] = field(default_factory=frozenset)


RELATION_TYPE_RULES: dict[str, RelationTypeRule] = {
    # === Vlastnictví / vydavatel ===
    "issued_by": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=ORG_LIKE,
        reverse_if_swapped=True,
    ),
    "manufactured_by": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=ORG_LIKE,
        reverse_if_swapped=True,
    ),
    "is_brand": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=ORG_LIKE,
        reverse_if_swapped=True,
    ),

    # === Struktura / komponenty ===
    "has_component": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=KOMPONENTA_LIKE,
        reverse_if_swapped=True,
    ),
    # part_of je sémanticky obecné — povolíme jakýkoli směr (LLM ho nemůže
    # spolehlivě otočit z labelů); typování jen blacklist URL jako celek.
    "part_of": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
        object_label_blacklist=URL_LIKE,
    ),
    "belongs_to": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
        object_label_blacklist=URL_LIKE,
    ),

    # === Parametry / hodnoty ===
    "has_parameter": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=PARAM_LIKE,
        reverse_if_swapped=True,
    ),
    "has_value": RelationTypeRule(
        subject_labels=PARAM_LIKE | PRODUCT_LIKE,
        object_labels=VALUE_LIKE,
        reverse_if_swapped=True,
    ),
    "has_dimension": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=VALUE_LIKE,
        reverse_if_swapped=True,
    ),
    "has_color": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=VALUE_LIKE,
        reverse_if_swapped=True,
    ),
    "has_battery_life": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=VALUE_LIKE,
        reverse_if_swapped=True,
    ),
    "has_abbreviation": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
    ),

    # === Materiál ===
    "has_material": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=MATERIAL_LIKE,
        reverse_if_swapped=True,
    ),
    "made_of": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=MATERIAL_LIKE,
        reverse_if_swapped=True,
    ),
    "covered_with": RelationTypeRule(
        subject_labels=PRODUCT_LIKE | KOMPONENTA_LIKE,
        object_labels=MATERIAL_LIKE,
        reverse_if_swapped=True,
    ),

    # === Technologie / specifikace ===
    "has_technology": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=TECH_LIKE | SPEC_LIKE,
        reverse_if_swapped=True,
    ),
    "has_feature": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=TECH_LIKE | SPEC_LIKE | PROPERTY_LIKE,
        reverse_if_swapped=True,
    ),
    "has_style": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=PROPERTY_LIKE,
        reverse_if_swapped=True,
    ),
    "has_design": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=PROPERTY_LIKE,
        reverse_if_swapped=True,
    ),
    "has_connectivity": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=SPEC_LIKE | TECH_LIKE,
        reverse_if_swapped=True,
    ),
    "has_property": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=PROPERTY_LIKE,
        reverse_if_swapped=True,
    ),

    # === Benefit / použití ===
    "provides_benefit": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=BENEFIT_LIKE,
        reverse_if_swapped=True,
    ),
    "provides_audio_quality": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=BENEFIT_LIKE,
        reverse_if_swapped=True,
    ),
    "covers": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=RISK_LIKE,
        reverse_if_swapped=True,
    ),
    "protects_against": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=RISK_LIKE,
        reverse_if_swapped=True,
    ),
    "suitable_for": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=TARGET_LIKE,
        reverse_if_swapped=True,
    ),
    "allows": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=BENEFIT_LIKE | RISK_LIKE,
        reverse_if_swapped=True,
    ),
    "supports_charging": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=SPEC_LIKE | TECH_LIKE,
        reverse_if_swapped=True,
    ),
    "supports_function": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=BENEFIT_LIKE | TECH_LIKE,
        reverse_if_swapped=True,
    ),

    # === Excluze / podmínky ===
    "excludes": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=RISK_LIKE,
        reverse_if_swapped=True,
    ),
    "requires": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
    ),
    "valid_for": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=frozenset({"DATUM"}) | TARGET_LIKE,
    ),
    "applies_to": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=TARGET_LIKE,
    ),

    # === Geografické ===
    "located_in": RelationTypeRule(
        subject_labels=ANY,
        object_labels=PLACE_LIKE,
        object_label_blacklist=URL_LIKE,
    ),
    "available_in": RelationTypeRule(
        subject_labels=PRODUCT_LIKE,
        object_labels=PLACE_LIKE | frozenset({"DATUM"}),
        object_label_blacklist=URL_LIKE,
    ),

    # === Hierarchie / klasifikace ===
    "is_a": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
    ),

    # === Obecná (related_to) — povolíme cokoliv, ale fallback ===
    "related_to": RelationTypeRule(
        subject_labels=ANY,
        object_labels=ANY,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches(label: str, allowed: object) -> bool:
    """True pokud label match pravidlu (frozenset nebo ANY)."""
    if allowed == ANY:
        return True
    return label in allowed  # type: ignore[operator]


def _swapped(
    subj_label: str, obj_label: str, rule: RelationTypeRule
) -> bool:
    """True pokud subject a object jsou prohozené (rule reversed match)."""
    if rule.subject_labels == ANY or rule.object_labels == ANY:
        return False
    return _matches(obj_label, rule.subject_labels) and _matches(
        subj_label, rule.object_labels
    )


_KNOWN_CONSOLIDATED: Optional[frozenset[str]] = None


def _known_consolidated() -> frozenset[str]:
    """
    Množina KONSOLIDOVANÝCH label kódů, pro které jsou typová pravidla
    (PRODUCT_LIKE, RISK_LIKE, ...) autoritativní. Odvozeno z
    normalize.DEFAULT_LABEL_MAPPING (žádná duplicitní definice) + URL.

    Pokud má entita label MIMO tuto množinu, jde o vlastní kód z Phase-0
    taxonomie — natvrdo zadaná pravidla pro něj NEPLATÍ a typovou neshodu
    pak NEDROPUJEME (jinak by custom taxonomie ztratila všechny typované
    relace). Lazy import zabraňuje cyklu (normalize importuje relation_normalize
    až ve funkci normalize_graph).
    """
    global _KNOWN_CONSOLIDATED
    if _KNOWN_CONSOLIDATED is None:
        from .normalize import DEFAULT_LABEL_MAPPING
        _KNOWN_CONSOLIDATED = frozenset(DEFAULT_LABEL_MAPPING.values()) | {"URL"}
    return _KNOWN_CONSOLIDATED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_relations(
    graph: SkoGraph,
    *,
    result: Optional["NormalizationResult"] = None,
) -> SkoGraph:
    """
    Aplikuje pravidla na všechny relace v grafu. Modifikuje graph.relations
    in-place. Counter se přičítají do `result` (pokud zadáno).

    Pořadí kroků:
      1. Drop self-loop a missing subject/object
      2. Type validation + reverse fix
      3. Object label blacklist (located_in → URL)
      4. Object text unsafe (covers "pojištění")
      5. Deduplikace (subjectId, relation, objectId)
    """
    entities_by_id: dict[str, Entity] = {e.id: e for e in graph.entities}
    initial = list(graph.relations)
    out: list[Relation] = []

    for r in initial:
        # --- 1. Existence + self-loop -------------------------------------
        if r.subjectId not in entities_by_id or r.objectId not in entities_by_id:
            if result is not None:
                result.relations_dropped_missing_entity += 1
            logger.debug(
                f"relation_normalize: drop missing-entity "
                f"{r.subjectId}-{r.relation}-{r.objectId}"
            )
            continue
        if r.subjectId == r.objectId:
            if result is not None:
                result.relations_dropped_self_loop += 1
            logger.debug(
                f"relation_normalize: drop self-loop {r.subjectId} {r.relation}"
            )
            continue

        # --- 2. Type validation ------------------------------------------
        subj = entities_by_id[r.subjectId]
        obj = entities_by_id[r.objectId]
        subj_lb = subj.category
        obj_lb = obj.category
        rule = RELATION_TYPE_RULES.get(r.relation)

        if rule is not None:
            forward = _matches(subj_lb, rule.subject_labels) and _matches(
                obj_lb, rule.object_labels
            )
            if not forward:
                # Zkus reverse
                if rule.reverse_if_swapped and _swapped(subj_lb, obj_lb, rule):
                    # Otoč subject/object
                    r.subjectId, r.objectId = r.objectId, r.subjectId
                    if result is not None:
                        result.relations_reversed_fixed += 1
                    logger.info(
                        f"relation_normalize: REVERSED "
                        f"{obj_lb} {r.relation} {subj_lb} → "
                        f"{subj_lb} {r.relation} {obj_lb}"
                    )
                    # Re-fetch po reverse
                    subj = entities_by_id[r.subjectId]
                    obj = entities_by_id[r.objectId]
                    subj_lb = subj.category
                    obj_lb = obj.category
                else:
                    # #5: pokud je některý endpoint MIMO známý konsolidovaný
                    # set (custom Phase-0 taxonomy kód), typová pravidla pro
                    # něj neplatí — RELATION_TYPE_RULES zná jen default labely.
                    # Dropnout by znamenalo, že custom taxonomie ztratí VŠECHNY
                    # relace, kterých se její kódy účastní. Proto relaci
                    # ponecháme (bez reverse, ten taky nejde ověřit).
                    if (
                        subj_lb not in _known_consolidated()
                        or obj_lb not in _known_consolidated()
                    ):
                        out.append(r)
                        continue
                    if result is not None:
                        result.relations_dropped_type_mismatch += 1
                    logger.info(
                        f"relation_normalize: drop type-mismatch "
                        f"{subj_lb}[{subj.text!r}] {r.relation} "
                        f"{obj_lb}[{obj.text!r}]"
                    )
                    continue

            # --- 3. Object label blacklist (např. located_in s URL) -----
            if rule.object_label_blacklist and obj_lb in rule.object_label_blacklist:
                if result is not None:
                    result.relations_dropped_type_mismatch += 1
                logger.info(
                    f"relation_normalize: drop blacklisted-object-label "
                    f"{r.relation} → {obj_lb}[{obj.text!r}]"
                )
                continue

            # --- 4. Generic object filtrace ------------------------
            # POZN.: V dřívější verzi byl zde object_text_unsafe seznam, ale
            # byl to hard-coding. Místo toho filtrace generic termů probíhá
            # už v _step_apply_taxonomy_text_rules (data-driven přes taxonomy
            # unsafe_texts) — generic objekt se downgraduje na KEYPHRASE_SUBJECT
            # a tady pak neprojde typovou validací (KEYPHRASE_SUBJECT není
            # v RISK_LIKE ani TARGET_LIKE). Žádná akce zde není potřeba.

        out.append(r)

    # --- 5. Deduplikace (subjectId, relation, objectId) -------------------
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Relation] = []
    for r in out:
        key = (r.subjectId, r.relation, r.objectId)
        if key in seen:
            if result is not None:
                result.relations_deduplicated += 1
            continue
        seen.add(key)
        deduped.append(r)

    graph.relations = deduped
    # Triples se přepočítají v normalize_graph na konci
    return graph
