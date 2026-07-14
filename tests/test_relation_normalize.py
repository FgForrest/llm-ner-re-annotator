"""
Unit testy pro labeler.relation_normalize + integrované scénáře v normalize_graph.

DŮLEŽITÉ: Tento soubor je doménově NEUTRÁLNÍ. Žádné konkrétní pojmy z reálných
zdrojových dat tu být nesmí — testujeme pouze
obecnou pipeline logiku přes taxonomy fixturu se zástupnými termíny
(Alpha/Beta/Gamma, "thing"/"item"/"stuff", "category one"/"category two").

Testovaná chování (univerzální, nezávislá na doméně):
  - issued_by obrácený směr → reverse fix
  - issued_by mezi neproduktovými entitami → drop (type mismatch)
  - suitable_for mezi nesmyslnými typy → drop
  - located_in s URL jako object → drop
  - halucinované triples bez podporujících entit → ignored
  - data-driven force_text_match / unsafe_texts z taxonomy (žádný hard-coding)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.normalize import NormalizeOpts, normalize_graph
from labeler.relation_normalize import (
    RELATION_TYPE_RULES,
    normalize_relations,
)
from labeler.schemas import Entity, Relation, SkoGraph, Triple
from labeler.taxonomy import Taxonomy, TaxonomyLabel


def _neutral_taxonomy():
    """
    Helper: minimální taxonomy se zástupnými NEUTRÁLNÍMI termíny.

    Záměrně používá fiktivní zástupné názvy ("Alpha", "category one") místo
    reálných doménových výrazů — pipeline kód musí být doménově univerzální.
    Reálné doménové taxonomie (pojišťovnictví, e-shop, ...) jsou v
    examples/entity_taxonomy.example.json a v testech zde nemají co dělat.
    """
    return Taxonomy(labels=[
        TaxonomyLabel(
            code="PRODUKT",
            description="Konkrétní pojmenovaný produkt / služba.",
            examples=["Alpha", "Beta"],
            force_text_match=["Alpha", "Beta", "Gamma"],
            case_sensitive=True,
            unsafe_texts=["thing", "item", "stuff"],
        ),
        TaxonomyLabel(
            code="PRODUKT_TYP",
            description="Typ produktu (kategorie) obecně.",
            examples=["category one"],
            force_text_match=[
                "category one", "category two", "category three",
            ],
            case_sensitive=False,
            unsafe_texts=["thing", "item"],
        ),
        TaxonomyLabel(
            code="ORGANIZACE",
            description="Firma, instituce, značka.",
            examples=["Alpha Corp"],
        ),
    ])


def _make_graph(entities_spec, relations_spec=None, triples_spec=None):
    """entities_spec = list[(id, text, label, start, end)] (lemma=text)"""
    entities = [
        Entity(id=eid, text=txt, lemma=txt, category=lb, start_char=s, end_char=e)
        for eid, txt, lb, s, e in entities_spec
    ]
    relations = [
        Relation(subjectId=sub, objectId=obj, relation=rel,
                 relationCs=rel, relationEn=rel)
        for sub, obj, rel in (relations_spec or [])
    ]
    triples = [
        Triple(subject=s, relation=r, object=o)
        for s, r, o in (triples_spec or [])
    ]
    g = SkoGraph(entities=entities, relations=relations, triples=triples)
    return g


# === Test 1: issued_by REVERSE FIX =======================================

def test_issued_by_reverse_fix():
    """ORG issued_by TYP → TYP issued_by ORG (reverse fix)."""
    text = "category one od Alpha Corp."
    g = _make_graph(
        entities_spec=[
            ("e1", "Alpha Corp", "ORGANIZACE", 16, 26),
            ("e2", "category one", "PRODUKT_TYP", 0, 12),
        ],
        relations_spec=[("e1", "e2", "issued_by")],  # špatný směr
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 1
    r = g.relations[0]
    # Po reverse: e2 issued_by e1
    assert r.subjectId == "e2"
    assert r.objectId == "e1"
    assert res.relations_reversed_fixed == 1


# === Test 2: issued_by TYPE MISMATCH DROP ================================

def test_issued_by_type_mismatch_drop():
    """SUBJECT issued_by ROLE → drop (žádná z stran není PRODUKT/ORG)."""
    text = "personal data and role."
    g = _make_graph(
        entities_spec=[
            ("e1", "personal data", "KEYPHRASE_SUBJECT", 0, 13),
            ("e2", "role", "OSOBA_ROLE", 18, 22),
        ],
        relations_spec=[("e1", "e2", "issued_by")],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 0
    assert res.relations_dropped_type_mismatch == 1


# === Test 3: suitable_for NONSENSE DROP ===================================

def test_suitable_for_nonsense_drop():
    """SUBJECT suitable_for ROLE → drop (subject není PRODUKT)."""
    text = "personal data and role."
    g = _make_graph(
        entities_spec=[
            ("e1", "personal data", "KEYPHRASE_SUBJECT", 0, 13),
            ("e2", "role", "OSOBA_ROLE", 18, 22),
        ],
        relations_spec=[("e1", "e2", "suitable_for")],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 0
    assert res.relations_dropped_type_mismatch == 1


# === Test 4: located_in WITH URL DROP =====================================

def test_located_in_with_url_drop():
    """ORG located_in URL → drop (URL ne MISTO)."""
    text = "Alpha Ministry at https://example.gov/."
    g = _make_graph(
        entities_spec=[
            ("e1", "Alpha Ministry", "ORGANIZACE", 0, 14),
            ("e2", "https://example.gov/", "URL", 18, 38),
        ],
        relations_spec=[("e1", "e2", "located_in")],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 0
    assert res.relations_dropped_type_mismatch == 1


# === Test 5: HALLUCINATED TRIPLES IGNORED =================================

def test_hallucinated_triples_ignored():
    """LLM-provided triples se subject/object mimo entities → ignored po init."""
    # Vstup grafu obsahuje 'triples' pole, ale žádné entity ani relace
    g = SkoGraph(
        entities=[],
        relations=[],
        triples=[
            Triple(subject="ghost subject", relation="ghost_rel",
                   object="ghost object"),
        ],
    )
    # SkoGraph model_validator přepsal triples na [] (žádné odpovídající
    # entities/relations)
    assert g.triples == [], (
        f"BEZPEČNOSTNÍ INVARIANT: triples musí být [] pokud žádné entities, "
        f"ale dostal {g.triples}"
    )


def test_hallucinated_triples_with_real_entities_replaced():
    """Vstup má hallucinated triples + reálné entities/relations → triples se přepíší."""
    g = SkoGraph(
        entities=[
            Entity(id="e1", text="iPhone", lemma="iPhone", category="PRODUKT",
                   start_char=0, end_char=6),
            Entity(id="e2", text="Apple", lemma="Apple", category="ORGANIZACE",
                   start_char=10, end_char=15),
        ],
        relations=[
            Relation(subjectId="e1", objectId="e2", relation="issued_by",
                     relationCs="vydává", relationEn="issued_by"),
        ],
        triples=[
            # Halucinace — neexistuje v entities
            Triple(subject="vesmír", relation="ovlivňuje", object="entropie"),
            # Reálný triple ale špatným pořadím
            Triple(subject="WRONG", relation="WRONG", object="WRONG"),
        ],
    )
    # Po init musí triples odpovídat entities+relations, halucinace pryč
    assert len(g.triples) == 1
    assert g.triples[0].subject == "iPhone"
    assert g.triples[0].relation == "issued_by"
    assert g.triples[0].object == "Apple"


# === Test 6: SELF-LOOP DROP ===============================================

def test_self_loop_drop():
    """
    Self-loop: subject==object. Pydantic Relation validator už brání tvorbě
    self-loop přes Relation(...) — tento test ověří, že normalize_relations
    se postaví i k tomu, kdyby self-loop nějak prošel (např. po reindex IDs
    nebo po reverse fix v rohových případech).
    """
    # Sestavíme graf, který má 2 entity, ale relace se subjectId==objectId.
    # Použijeme model_construct (Pydantic v2 bypass validator) pro simulaci.
    from labeler.schemas import Relation
    rel = Relation.model_construct(
        subjectId="e1", objectId="e1", relation="is_a",
        relationCs="je", relationEn="is_a",
    )
    g = _make_graph(entities_spec=[("e1", "X", "PRODUKT", 0, 1)])
    g.relations = [rel]
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 0
    assert res.relations_dropped_self_loop == 1


# === Test 7: DEDUPLICATION ================================================

def test_deduplication():
    text = "iPhone od Apple."
    g = _make_graph(
        entities_spec=[
            ("e1", "iPhone", "PRODUKT", 0, 6),
            ("e2", "Apple", "ORGANIZACE", 10, 15),
        ],
        relations_spec=[
            ("e1", "e2", "issued_by"),
            ("e1", "e2", "issued_by"),  # duplicit
            ("e1", "e2", "issued_by"),  # duplicit
        ],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 1
    assert res.relations_deduplicated == 2


# === Test 8: COVERS s generic-fallback objektem → DROP (type-mismatch) ====

def test_covers_with_generic_subject_object_drop():
    """PRODUKT covers KEYPHRASE_SUBJECT → drop (SUBJECT ne v RISK_LIKE).

    Filtrace generic termů je data-driven přes taxonomy: text v unsafe_texts
    PRODUKT/PRODUKT_TYP → downgrade na KEYPHRASE_SUBJECT (generic fallback
    label, který není v žádné relation target množině). Po downgrade
    relation_normalize odhalí type-mismatch a relaci zahodí.
    """
    text = "Alpha covers thing."
    g = _make_graph(
        entities_spec=[
            ("e1", "Alpha", "PRODUKT", 0, 5),
            ("e2", "thing", "KEYPHRASE_SUBJECT", 13, 18),
        ],
        relations_spec=[("e1", "e2", "covers")],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 0
    # SUBJECT object → type-mismatch drop (RISK_LIKE záměrně bez KEYPHRASE_SUBJECT)
    assert res.relations_dropped_type_mismatch == 1


# === Test 9: COVERS se specifickým KEYPHRASE_USECASE → KEEP ==============

def test_covers_with_specific_usecase_keep():
    """PRODUKT covers KEYPHRASE_USECASE → keep (USECASE je v RISK_LIKE)."""
    text = "Alpha covers specific risk scenario X."
    g = _make_graph(
        entities_spec=[
            ("e1", "Alpha", "PRODUKT", 0, 5),
            ("e2", "specific risk scenario X",
             "KEYPHRASE_USECASE", 13, 37),
        ],
        relations_spec=[("e1", "e2", "covers")],
    )
    from labeler.normalize import NormalizationResult
    res = NormalizationResult(graph=g)
    normalize_relations(g, result=res)
    assert len(g.relations) == 1  # zachováno
    assert res.relations_dropped_weak == 0


# === Test 10: TAXONOMY DATA-DRIVEN HEURISTICS =============================

def test_taxonomy_rules_force_match_to_produkt():
    """Text v PRODUKT.force_text_match (case-sensitive) → label se přepíše na PRODUKT."""
    text = "Alpha je produkt."
    g = _make_graph([("e1", "Alpha", "KATEGORIE", 0, 5)])
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, taxonomy=_neutral_taxonomy(),
    ))
    assert res.graph.entities[0].category == "PRODUKT"
    assert res.taxonomy_rules_applied >= 1


def test_taxonomy_rules_force_match_to_produkt_typ():
    """Text v PRODUKT_TYP.force_text_match (case-insensitive) → PRODUKT_TYP."""
    text = "Category One je typ produktu."
    g = _make_graph([("e1", "Category One", "PRODUKT", 0, 12)])
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, taxonomy=_neutral_taxonomy(),
    ))
    assert res.graph.entities[0].category == "PRODUKT_TYP"


def test_taxonomy_rules_unsafe_text_downgrade():
    """Text v PRODUKT.unsafe_texts a entity má label PRODUKT → KEYPHRASE_SUBJECT."""
    text = "thing obecně."
    g = _make_graph([("e1", "thing", "PRODUKT", 0, 5)])
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, taxonomy=_neutral_taxonomy(),
    ))
    assert res.graph.entities[0].category == "KEYPHRASE_SUBJECT"


def test_no_taxonomy_no_hard_coded_heuristics():
    """
    Bez taxonomy: žádné force_text_match ani unsafe_texts pravidla se
    NEAPLIKUJÍ — entita zůstane jak ji dal LLM (po general remap KATEGORIE
    → PRODUKT_TYP). Žádné hard-coding doménové heuristiky.
    """
    text = "Alpha je produkt."
    g = _make_graph([("e1", "Alpha", "KATEGORIE", 0, 5)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True, taxonomy=None))
    # Bez taxonomy "Alpha" projde jen general remap KATEGORIE → PRODUKT_TYP
    assert res.graph.entities[0].category == "PRODUKT_TYP"
    assert res.taxonomy_rules_applied == 0  # žádné text rules


# === Test 11: AUTO-EXPAND UNSAFE TERMS ====================================

def test_auto_expand_skips_unsafe_terms_from_taxonomy():
    """'thing' je v PRODUKT_TYP.unsafe_texts → nedoplní se další výskyty."""
    text = "thing is one. another thing. third thing here."
    g = _make_graph([("e1", "thing", "PRODUKT_TYP", 0, 5)])
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, auto_expand_mentions=True, taxonomy=_neutral_taxonomy(),
    ))
    # Taxonomy unsafe → expand se zastaví → zůstane 1 entita
    # (Navíc po unsafe_texts downgrade entity přejde na KEYPHRASE_SUBJECT)
    assert len(res.graph.entities) == 1
    assert res.mentions_expanded == 0


# === Test 12: COMPLETE PIPELINE — issued_by reverse + taxonomy ============

def test_pipeline_full_issued_by_with_taxonomy():
    """End-to-end: LLM dal špatný směr + špatný label; taxonomy + normalize opraví."""
    text = "Category One od Alpha Corp."
    g = _make_graph(
        entities_spec=[
            ("e1", "Alpha Corp", "ORGANIZACE", 16, 26),
            ("e2", "Category One", "PRODUKT", 0, 12),  # LLM dal PRODUKT
        ],
        relations_spec=[("e1", "e2", "issued_by")],  # špatný směr
    )
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, taxonomy=_neutral_taxonomy(),
    ))
    # Po taxonomy rules: Category One → PRODUKT_TYP (force_text_match)
    e_typ = [e for e in res.graph.entities
             if e.text.lower() == "category one"][0]
    assert e_typ.category == "PRODUKT_TYP"
    # Po relation normalize: reverse fix
    assert len(res.graph.relations) == 1
    r = res.graph.relations[0]
    subj = next(e for e in res.graph.entities if e.id == r.subjectId)
    obj = next(e for e in res.graph.entities if e.id == r.objectId)
    assert subj.category == "PRODUKT_TYP"
    assert obj.category == "ORGANIZACE"
    assert res.relations_reversed_fixed == 1


# === Runner ===============================================================

if __name__ == "__main__":
    import inspect
    tests = [
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR  {name}: {type(e).__name__}: {e}")
    print(f"\nTotal: {len(tests)} tests, {failed} failed.")
    sys.exit(1 if failed else 0)
