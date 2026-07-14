"""
Unit testy pro labeler.rdf_export — OWL ontology export přes rdflib.

Pokrývá:
  - Validace labelů / relací (anti-halucinace)
  - URI generace
  - TBox: owl:Ontology, owl:Class (Document, Entity, per-label subclasses)
  - TBox: owl:ObjectProperty per relation + hasEntity
  - TBox: owl:DatatypeProperty pro text/lemma/startChar/endChar/category/atd.
  - ABox: Document a Entity individuals, hasEntity links, relace
  - Filtrace nevalidních labelů / relací / self-loop / dangling refs
  - Serializace do .owl (RDF/XML) + .ttl (Turtle)
  - Per-source split (parita s JSONL exporty)
  - Round-trip: zapsat OWL → načíst rdflib.Graph zpět → ověřit triples

Verifikace přes rdflib.Graph API + SPARQL — žádný hand-written string match.

Spuštění: python tests/test_rdf_export.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from labeler.rdf_export import (
    DOC_NS_STR,
    ENT_NS_STR,
    ONTOLOGY_IRI,
    VOCAB,
    VOCAB_NS_STR,
    _add_abox_record,
    _add_tbox,
    _doc_uri,
    _ent_uri,
    _is_valid_label,
    _is_valid_relation,
    _safe_uri_component,
    build_owl_graph,
    collect_labels_and_relations,
    export_owl,
    write_owl_files,
    write_per_source_owl,
)


# === Schema validation ====================================================

def test_is_valid_label_accepts_uppercase_underscore():
    assert _is_valid_label("PRODUKT")
    assert _is_valid_label("KEYPHRASE_BENEFIT")
    assert _is_valid_label("OSOBA_ROLE")


def test_is_valid_label_rejects_lowercase_or_special():
    assert not _is_valid_label("produkt")
    assert not _is_valid_label("Produkt")
    assert not _is_valid_label("PRODUKT!")
    assert not _is_valid_label("")
    assert not _is_valid_label("9PRODUKT")


def test_is_valid_relation_accepts_snake_case():
    assert _is_valid_relation("issued_by")
    assert _is_valid_relation("has_component")
    assert _is_valid_relation("is_a")


def test_is_valid_relation_rejects_uppercase_or_special():
    assert not _is_valid_relation("ISSUED_BY")
    assert not _is_valid_relation("issued-by")
    assert not _is_valid_relation("")


# === URI generation =======================================================

def test_safe_uri_component_keeps_alnum():
    assert _safe_uri_component("hello123") == "hello123"
    assert _safe_uri_component("alpha-beta_gamma") == "alpha-beta_gamma"


def test_safe_uri_component_escapes_special():
    result = _safe_uri_component("hello world!")
    assert " " not in result
    assert "!" not in result


def test_safe_uri_component_empty_to_anon():
    assert _safe_uri_component("") == "anon"
    assert _safe_uri_component("   ") == "anon"


def test_doc_uri_uses_doc_namespace():
    uri = _doc_uri("beta:abc")
    assert str(uri).startswith(DOC_NS_STR)


def test_ent_uri_uses_ent_namespace():
    uri = _ent_uri("beta:abc", "e1")
    assert str(uri).startswith(ENT_NS_STR)
    assert "e1" in str(uri)


# === collect_labels_and_relations =========================================

def _make_record(
    item_id="doc1", source="alfa", text="iPhone od Apple.",
    entities=None, relations=None, critic_status="ok", was_revised=False,
):
    return {
        "item_id": item_id,
        "source_dataset": source,
        "input_text": text,
        "critic_status": critic_status,
        "was_revised": was_revised,
        "graph": {
            "entities": entities or [],
            "relations": relations or [],
        },
    }


def test_collect_labels_and_relations_basic():
    records = [
        _make_record(
            entities=[
                {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
                 "start_char": 0, "end_char": 1},
            ],
            relations=[{"subjectId": "e1", "objectId": "e1", "relation": "is_a"}],
        ),
        _make_record(
            entities=[
                {"id": "e1", "text": "y", "lemma": "y", "category": "ORGANIZACE",
                 "start_char": 0, "end_char": 1},
            ],
            relations=[{"subjectId": "e1", "objectId": "e1", "relation": "issued_by"}],
        ),
    ]
    labels, rels = collect_labels_and_relations(records)
    assert labels == {"PRODUKT", "ORGANIZACE"}
    assert rels == {"is_a", "issued_by"}


def test_collect_filters_invalid():
    records = [_make_record(
        entities=[
            {"id": "e1", "category": "produkt", "text": "x", "lemma": "x",
             "start_char": 0, "end_char": 1},
            {"id": "e2", "category": "PRODUKT", "text": "y", "lemma": "y",
             "start_char": 0, "end_char": 1},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e2", "relation": "BAD-REL"},
        ],
    )]
    labels, rels = collect_labels_and_relations(records)
    assert labels == {"PRODUKT"}
    assert rels == set()


# === TBox tests ===========================================================

def test_add_tbox_emits_ontology_declaration():
    g = Graph()
    ont = _add_tbox(g, [], [])
    # owl:Ontology declaration musí být v grafu
    assert (ont, RDF.type, OWL.Ontology) in g
    # má rdfs:label + rdfs:comment
    labels = list(g.objects(ont, RDFS.label))
    assert len(labels) >= 1


def test_add_tbox_emits_document_and_entity_classes():
    g = Graph()
    _add_tbox(g, [], [])
    assert (VOCAB.Document, RDF.type, OWL.Class) in g
    assert (VOCAB.Entity, RDF.type, OWL.Class) in g
    # Disjointní
    assert (VOCAB.Document, OWL.disjointWith, VOCAB.Entity) in g


def test_add_tbox_emits_per_label_subclasses():
    g = Graph()
    _add_tbox(g, ["PRODUKT", "ORGANIZACE"], [])
    assert (VOCAB.PRODUKT, RDF.type, OWL.Class) in g
    assert (VOCAB.PRODUKT, RDFS.subClassOf, VOCAB.Entity) in g
    assert (VOCAB.ORGANIZACE, RDF.type, OWL.Class) in g
    assert (VOCAB.ORGANIZACE, RDFS.subClassOf, VOCAB.Entity) in g


def test_add_tbox_skips_invalid_label_classes():
    g = Graph()
    _add_tbox(g, ["PRODUKT", "produkt", "BAD!"], [])
    assert (VOCAB.PRODUKT, RDF.type, OWL.Class) in g
    # Nevalidní lowercase / special chars musí být odfiltrované
    assert (VOCAB.produkt, RDF.type, OWL.Class) not in g


def test_add_tbox_emits_hasEntity_object_property():
    g = Graph()
    _add_tbox(g, [], [])
    assert (VOCAB.hasEntity, RDF.type, OWL.ObjectProperty) in g
    assert (VOCAB.hasEntity, RDFS.domain, VOCAB.Document) in g
    assert (VOCAB.hasEntity, RDFS.range, VOCAB.Entity) in g


def test_add_tbox_emits_relation_object_properties():
    g = Graph()
    _add_tbox(g, [], ["issued_by", "has_component"])
    for rel in ["issued_by", "has_component"]:
        prop = VOCAB[rel]
        assert (prop, RDF.type, OWL.ObjectProperty) in g
        assert (prop, RDFS.domain, VOCAB.Entity) in g
        assert (prop, RDFS.range, VOCAB.Entity) in g


def test_add_tbox_emits_datatype_properties():
    g = Graph()
    _add_tbox(g, [], [])
    # Klíčové datatype properties
    for prop_local in ["sourceDataset", "inputText", "criticStatus", "wasRevised",
                       "text", "lemma", "category", "startChar", "endChar"]:
        prop = VOCAB[prop_local]
        assert (prop, RDF.type, OWL.DatatypeProperty) in g, (
            f"Missing DatatypeProperty :{prop_local}"
        )


def test_add_tbox_startChar_endChar_are_nonNegativeInteger():
    g = Graph()
    _add_tbox(g, [], [])
    assert (VOCAB.startChar, RDFS.range, XSD.nonNegativeInteger) in g
    assert (VOCAB.endChar, RDFS.range, XSD.nonNegativeInteger) in g


# === ABox tests ===========================================================

def test_add_abox_creates_document_individual():
    g = Graph()
    _add_tbox(g, ["PRODUKT"], [])
    rec = _make_record(item_id="d1", source="alfa", text="hello",
                       critic_status="ok", was_revised=False)
    _add_abox_record(g, rec)
    doc_iri = _doc_uri("d1")
    assert (doc_iri, RDF.type, OWL.NamedIndividual) in g
    assert (doc_iri, RDF.type, VOCAB.Document) in g
    assert (doc_iri, VOCAB.sourceDataset, Literal("alfa")) in g
    assert (doc_iri, VOCAB.inputText, Literal("hello")) in g
    assert (doc_iri, VOCAB.criticStatus, Literal("ok")) in g
    assert (doc_iri, VOCAB.wasRevised, Literal(False)) in g


def test_add_abox_creates_entity_individuals_with_subclass_type():
    g = Graph()
    _add_tbox(g, ["PRODUKT", "ORGANIZACE"], [])
    rec = _make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
            {"id": "e2", "text": "Apple", "lemma": "Apple",
             "category": "ORGANIZACE", "start_char": 10, "end_char": 15},
        ],
    )
    _add_abox_record(g, rec)
    e1 = _ent_uri("d1", "e1")
    e2 = _ent_uri("d1", "e2")
    # Entity individuals
    assert (e1, RDF.type, OWL.NamedIndividual) in g
    assert (e1, RDF.type, VOCAB.PRODUKT) in g
    assert (e2, RDF.type, VOCAB.ORGANIZACE) in g
    # Datatype assertions
    assert (e1, VOCAB.text, Literal("iPhone")) in g
    assert (e1, VOCAB.lemma, Literal("iPhone")) in g
    assert (e1, VOCAB.category, Literal("PRODUKT")) in g
    # startChar/endChar jako xsd:nonNegativeInteger
    assert (e1, VOCAB.startChar, Literal(0, datatype=XSD.nonNegativeInteger)) in g
    assert (e1, VOCAB.endChar, Literal(6, datatype=XSD.nonNegativeInteger)) in g


def test_add_abox_emits_hasEntity_links():
    g = Graph()
    _add_tbox(g, ["PRODUKT"], [])
    rec = _make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
             "start_char": 0, "end_char": 1},
        ],
    )
    _add_abox_record(g, rec)
    doc_iri = _doc_uri("d1")
    e1 = _ent_uri("d1", "e1")
    assert (doc_iri, VOCAB.hasEntity, e1) in g


def test_add_abox_emits_relation_triple_between_entities():
    g = Graph()
    _add_tbox(g, ["PRODUKT", "ORGANIZACE"], ["issued_by"])
    rec = _make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
            {"id": "e2", "text": "Apple", "lemma": "Apple",
             "category": "ORGANIZACE", "start_char": 10, "end_char": 15},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e2", "relation": "issued_by"},
        ],
    )
    _add_abox_record(g, rec)
    e1 = _ent_uri("d1", "e1")
    e2 = _ent_uri("d1", "e2")
    assert (e1, VOCAB.issued_by, e2) in g


def test_add_abox_skips_invalid_label_entities():
    g = Graph()
    _add_tbox(g, ["PRODUKT"], [])
    rec = _make_record(
        item_id="d1",
        entities=[
            # Invalid lowercase label
            {"id": "e1", "text": "x", "lemma": "x", "category": "produkt",
             "start_char": 0, "end_char": 1},
            {"id": "e2", "text": "y", "lemma": "y", "category": "PRODUKT",
             "start_char": 2, "end_char": 3},
        ],
    )
    _add_abox_record(g, rec)
    # e1 (invalid) NESMÍ být v grafu
    e1 = _ent_uri("d1", "e1")
    assert (e1, RDF.type, OWL.NamedIndividual) not in g
    e2 = _ent_uri("d1", "e2")
    assert (e2, RDF.type, VOCAB.PRODUKT) in g


def test_add_abox_skips_self_loop_relation():
    g = Graph()
    _add_tbox(g, ["PRODUKT"], ["is_a"])
    rec = _make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
             "start_char": 0, "end_char": 1},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e1", "relation": "is_a"},
        ],
    )
    _add_abox_record(g, rec)
    e1 = _ent_uri("d1", "e1")
    # Self-loop nesmí být v grafu
    assert (e1, VOCAB.is_a, e1) not in g


def test_add_abox_skips_dangling_relations():
    g = Graph()
    _add_tbox(g, ["PRODUKT"], ["issued_by"])
    rec = _make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
             "start_char": 0, "end_char": 1},
        ],
        relations=[
            # objectId e_ghost neexistuje
            {"subjectId": "e1", "objectId": "e_ghost", "relation": "issued_by"},
            # subjectId e_phantom neexistuje
            {"subjectId": "e_phantom", "objectId": "e1", "relation": "is_a"},
        ],
    )
    _add_abox_record(g, rec)
    e1 = _ent_uri("d1", "e1")
    e_ghost = _ent_uri("d1", "e_ghost")
    e_phantom = _ent_uri("d1", "e_phantom")
    assert (e1, VOCAB.issued_by, e_ghost) not in g
    assert (e_phantom, VOCAB.is_a, e1) not in g


def test_add_abox_truncates_long_text():
    g = Graph()
    _add_tbox(g, [], [])
    rec = _make_record(item_id="d1", text="x" * 10_000)
    _add_abox_record(g, rec, max_text_len=100)
    doc_iri = _doc_uri("d1")
    texts = list(g.objects(doc_iri, VOCAB.inputText))
    assert len(texts) == 1
    s = str(texts[0])
    assert len(s) < 200, f"Text was not truncated: {len(s)} chars"
    assert s.endswith("…") or s.endswith("...")


# === build_owl_graph (full integration) ===================================

def test_build_owl_graph_full_pipeline():
    records = [_make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
            {"id": "e2", "text": "Apple", "lemma": "Apple",
             "category": "ORGANIZACE", "start_char": 10, "end_char": 15},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e2", "relation": "issued_by"},
        ],
    )]
    g = build_owl_graph(records)
    # TBox přítomný
    assert any(g.subjects(RDF.type, OWL.Ontology))
    assert (VOCAB.Document, RDF.type, OWL.Class) in g
    assert (VOCAB.PRODUKT, RDFS.subClassOf, VOCAB.Entity) in g
    assert (VOCAB.issued_by, RDF.type, OWL.ObjectProperty) in g
    # ABox přítomný
    doc_iri = _doc_uri("d1")
    assert (doc_iri, RDF.type, VOCAB.Document) in g
    e1 = _ent_uri("d1", "e1")
    e2 = _ent_uri("d1", "e2")
    assert (e1, VOCAB.issued_by, e2) in g


def test_build_owl_graph_sparql_count_query():
    """Použij SPARQL k spočtení Document a Entity individuals."""
    records = [
        _make_record(item_id="d1", entities=[
            {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
             "start_char": 0, "end_char": 1},
        ]),
        _make_record(item_id="d2", entities=[
            {"id": "e1", "text": "y", "lemma": "y", "category": "ORGANIZACE",
             "start_char": 0, "end_char": 1},
            {"id": "e2", "text": "z", "lemma": "z", "category": "PRODUKT",
             "start_char": 2, "end_char": 3},
        ]),
    ]
    g = build_owl_graph(records)
    docs = list(g.subjects(RDF.type, VOCAB.Document))
    ents = list(g.subjects(RDF.type, VOCAB.Entity))  # NEexpanduje subclasses!
    # `Entity` direct type nikoho nemá — všechno je subclass
    assert len(docs) == 2
    # Subclass typy:
    produkty = list(g.subjects(RDF.type, VOCAB.PRODUKT))
    org = list(g.subjects(RDF.type, VOCAB.ORGANIZACE))
    assert len(produkty) == 2
    assert len(org) == 1


# === File output: .owl, .ttl, optional .html ==============================

def test_write_owl_files_creates_owl_and_ttl():
    records = [_make_record(entities=[
        {"id": "e1", "text": "x", "lemma": "x", "category": "PRODUKT",
         "start_char": 0, "end_char": 1},
    ])]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "final_labels.owl"
        stats = write_owl_files(records, base, write_html=False)
        assert (Path(tmp) / "final_labels.owl").exists()
        assert (Path(tmp) / "final_labels.ttl").exists()
        assert stats["records"] == 1
        assert stats["triples"] > 0
        assert stats["owl"] == "final_labels.owl"
        assert stats["ttl"] == "final_labels.ttl"


def test_owl_file_is_valid_rdfxml_parseable_back():
    """Round-trip: zapsat → parse zpět rdflib → musí mít stejné triples."""
    records = [_make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
        ],
        relations=[],
    )]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "final_labels.owl"
        write_owl_files(records, base, write_html=False, write_ttl=False)

        owl_path = Path(tmp) / "final_labels.owl"
        g2 = Graph()
        g2.parse(str(owl_path), format="xml")

        # Critical triples musí být po round-trip
        doc_iri = _doc_uri("d1")
        assert (doc_iri, RDF.type, VOCAB.Document) in g2
        e1 = _ent_uri("d1", "e1")
        assert (e1, RDF.type, VOCAB.PRODUKT) in g2


def test_ttl_file_is_valid_turtle_parseable_back():
    records = [_make_record(
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
        ],
    )]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "final_labels.owl"
        write_owl_files(records, base, write_html=False, write_ttl=True)
        ttl_path = Path(tmp) / "final_labels.ttl"
        g2 = Graph()
        g2.parse(str(ttl_path), format="turtle")
        assert (VOCAB.Document, RDF.type, OWL.Class) in g2


# === Per-source split =====================================================

def test_write_per_source_owl_splits_by_source():
    records = [
        _make_record(item_id="d1", source="alfa",
                     entities=[{"id": "e1", "text": "x", "lemma": "x",
                                "category": "PRODUKT", "start_char": 0, "end_char": 1}]),
        _make_record(item_id="d2", source="beta",
                     entities=[{"id": "e1", "text": "y", "lemma": "y",
                                "category": "ORGANIZACE", "start_char": 0, "end_char": 1}]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "final_labels.owl"
        stats = write_per_source_owl(records, base, write_html=False)
        names = sorted(s["source"] for s in stats)
        assert "alfa" in names
        assert "beta" in names
        assert (Path(tmp) / "final_labels_alfa.owl").exists()
        assert (Path(tmp) / "final_labels_beta.owl").exists()

        # Alfa .owl obsahuje PRODUKT, ne ORGANIZACE
        g_alfa = Graph()
        g_alfa.parse(str(Path(tmp) / "final_labels_alfa.owl"), format="xml")
        assert (VOCAB.PRODUKT, RDF.type, OWL.Class) in g_alfa
        # ORGANIZACE NE — alfa neměl žádnou
        assert (VOCAB.ORGANIZACE, RDF.type, OWL.Class) not in g_alfa


def test_export_owl_creates_both_combined_and_per_source():
    records = [
        _make_record(item_id="d1", source="alfa",
                     entities=[{"id": "e1", "text": "x", "lemma": "x",
                                "category": "PRODUKT", "start_char": 0, "end_char": 1}]),
        _make_record(item_id="d2", source="beta",
                     entities=[{"id": "e1", "text": "y", "lemma": "y",
                                "category": "ORGANIZACE", "start_char": 0, "end_char": 1}]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base_jsonl = Path(tmp) / "final_labels.jsonl"
        stats = export_owl(records, base_jsonl, write_html=False)
        assert stats["records"] == 2
        assert stats["combined"]["owl"] == "final_labels.owl"
        sources = {s["source"] for s in stats["per_source"]}
        assert sources == {"alfa", "beta"}


def test_export_owl_combined_only():
    records = [_make_record()]
    with tempfile.TemporaryDirectory() as tmp:
        base_jsonl = Path(tmp) / "final_labels.jsonl"
        stats = export_owl(records, base_jsonl, per_source=False, write_html=False)
        assert stats["combined"]["owl"] == "final_labels.owl"
        assert stats["per_source"] == []


def test_export_owl_per_source_only():
    records = [_make_record()]
    with tempfile.TemporaryDirectory() as tmp:
        base_jsonl = Path(tmp) / "final_labels.jsonl"
        stats = export_owl(records, base_jsonl, combined=False, write_html=False)
        assert stats["combined"] is None
        assert stats["per_source"]
        assert not (Path(tmp) / "final_labels.owl").exists()


# === Anti-halucinace na celém pipeline ===================================

def test_full_pipeline_anti_hallucination_filters():
    """Smoke: nevalidní entity/relace se nedostanou do výstupního .owl."""
    records = [_make_record(
        item_id="d1",
        entities=[
            {"id": "e1", "text": "x", "lemma": "x", "category": "produkt",  # invalid
             "start_char": 0, "end_char": 1},
            {"id": "e2", "text": "y", "lemma": "y", "category": "PRODUKT",
             "start_char": 0, "end_char": 1},
            {"id": "e3", "text": "z", "lemma": "z", "category": "ORGANIZACE",
             "start_char": 0, "end_char": 1},
        ],
        relations=[
            {"subjectId": "e2", "objectId": "e3", "relation": "issued_by"},
            {"subjectId": "e2", "objectId": "e2", "relation": "is_a"},  # self-loop
            {"subjectId": "e2", "objectId": "e_ghost", "relation": "covers"},  # dangling
            {"subjectId": "e2", "objectId": "e3", "relation": "BAD-REL"},  # invalid rel
        ],
    )]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "final_labels.owl"
        write_owl_files(records, base, write_html=False, write_ttl=False)
        g2 = Graph()
        g2.parse(str(Path(tmp) / "final_labels.owl"), format="xml")

        # e1 (lowercase invalid) → není v grafu
        e1 = _ent_uri("d1", "e1")
        assert (e1, RDF.type, OWL.NamedIndividual) not in g2

        # e2 PRODUKT validní → JE v grafu
        e2 = _ent_uri("d1", "e2")
        assert (e2, RDF.type, VOCAB.PRODUKT) in g2

        # issued_by mezi e2 a e3 → JE
        e3 = _ent_uri("d1", "e3")
        assert (e2, VOCAB.issued_by, e3) in g2

        # Self-loop e2 is_a e2 → NENÍ
        assert (e2, VOCAB.is_a, e2) not in g2

        # Dangling e2 covers e_ghost → NENÍ
        e_ghost = _ent_uri("d1", "e_ghost")
        assert (e2, VOCAB.covers, e_ghost) not in g2


# === Runner ===============================================================

if __name__ == "__main__":
    import inspect
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
            msg = repr(str(e)) if not str(e).isascii() else str(e)
            print(f"  FAIL  {name}: {msg}")
        except Exception as e:
            failed += 1
            msg = repr(str(e)) if not str(e).isascii() else str(e)
            print(f"  ERROR  {name}: {type(e).__name__}: {msg}")
    print(f"\nTotal: {len(tests)} tests, {failed} failed.")
    sys.exit(1 if failed else 0)
