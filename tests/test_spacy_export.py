"""
Unit testy pro labeler.finalize split exports (--export-spacy / --quality-preset spacy).

Pokrývá:
  - core_ner export obsahuje JEN hard NER labely (PRODUKT, ORGANIZACE, DATUM, …)
    a NEMÁ KEYPHRASE_*
  - keyphrases export obsahuje JEN KEYPHRASE_* entity
  - re export má relace mezi entitami s VALIDNÍMI spany (text[s:e] == entity.text)
  - entity s nedohledatelným offsetem se NEdostanou ani do core_ner, ani do RE
  - triples v exportech jsou deterministicky regenerované z entities+relations
    (žádné LLM halucinace)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.finalize import (
    _entity_has_valid_span,
    _rebuild_triples,
    _split_record_for_spacy,
)
from labeler.normalize import DEFAULT_NER_LABELS


def _make_record(input_text, entities, relations=None):
    """Sestaví final_label-like dict pro testování."""
    return {
        "item_id": "test:1",
        "source_dataset": "test",
        "input_text": input_text,
        "is_synthetic": False,
        "input_hash": "h",
        "extractor_model": "X", "critic_model": "Y", "was_revised": False,
        "critic_status": "ok", "critic_reasoning": "ok",
        "graph": {
            "entities": entities,
            "relations": relations or [],
            "triples": [],
        },
    }


# === Test 1: core_ner BEZ KEYPHRASE_* labelů ==============================

def test_core_ner_excludes_keyphrase():
    text = "OMIKRON je produkt od Vzorová pojišťovna ABC a má limit plnění."
    rec = _make_record(text, [
        {"id": "e1", "text": "OMIKRON", "lemma": "OMIKRON",
         "category": "PRODUKT", "start_char": 0, "end_char": 7},
        {"id": "e2", "text": "Vzorová pojišťovna ABC", "lemma": "Vzorová pojišťovna ABC",
         "category": "ORGANIZACE", "start_char": 22, "end_char": 44},
        {"id": "e3", "text": "limit plnění", "lemma": "limit plnění",
         "category": "KEYPHRASE_PARAMETR", "start_char": 53, "end_char": 65},
        {"id": "e4", "text": "asistence", "lemma": "asistence",
         "category": "KEYPHRASE_BENEFIT", "start_char": 0, "end_char": 9},  # úmyslně špatně, ale to test 3 řeší
    ])
    core, keyphrases, re_ = _split_record_for_spacy(rec, DEFAULT_NER_LABELS)

    core_labels = {e["category"] for e in core["graph"]["entities"]}
    assert "PRODUKT" in core_labels
    assert "ORGANIZACE" in core_labels
    assert "KEYPHRASE_PARAMETR" not in core_labels
    assert "KEYPHRASE_BENEFIT" not in core_labels


def test_keyphrases_only_keyphrase():
    text = "OMIKRON a limit plnění."
    rec = _make_record(text, [
        {"id": "e1", "text": "OMIKRON", "lemma": "OMIKRON",
         "category": "PRODUKT", "start_char": 0, "end_char": 7},
        {"id": "e2", "text": "limit plnění", "lemma": "limit plnění",
         "category": "KEYPHRASE_PARAMETR", "start_char": 10, "end_char": 22},
    ])
    _, keyphrases, _ = _split_record_for_spacy(rec, DEFAULT_NER_LABELS)
    labels = {e["category"] for e in keyphrases["graph"]["entities"]}
    assert labels == {"KEYPHRASE_PARAMETR"}


# === Test 2: invalid offset entity NOT in core_ner =========================

def test_invalid_offset_dropped_from_export():
    text = "Apple iPhone."
    rec = _make_record(text, [
        {"id": "e1", "text": "Apple iPhone", "lemma": "Apple iPhone",
         "category": "PRODUKT", "start_char": 0, "end_char": 12},
        # Tato entita má špatný offset (nesedí substring)
        {"id": "e2", "text": "Samsung", "lemma": "Samsung",
         "category": "PRODUKT", "start_char": 0, "end_char": 7},
    ])
    core, _, _ = _split_record_for_spacy(rec, DEFAULT_NER_LABELS)
    # Jen e1 by mělo projít
    assert len(core["graph"]["entities"]) == 1
    assert core["graph"]["entities"][0]["id"] == "e1"


# === Test 3: RE export má jen relace s validními spany ====================

def test_re_export_only_valid_relations():
    text = "iPhone od Apple."
    rec = _make_record(
        text,
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
            {"id": "e2", "text": "Apple", "lemma": "Apple",
             "category": "ORGANIZACE", "start_char": 10, "end_char": 15},
            # Entita s nevaliděním offsetem — relace na ni musí být dropnuta
            {"id": "e3", "text": "neexistuje", "lemma": "neexistuje",
             "category": "PRODUKT", "start_char": 100, "end_char": 110},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e2", "relation": "issued_by",
             "relationCs": "vydává", "relationEn": "issued_by"},
            # Tato relace má entitu s nevalidním offsetem
            {"subjectId": "e1", "objectId": "e3", "relation": "is_a",
             "relationCs": "je", "relationEn": "is_a"},
        ],
    )
    _, _, re_ = _split_record_for_spacy(rec, DEFAULT_NER_LABELS)
    # Jen jedna relace prošla (e1→e2)
    assert len(re_["graph"]["relations"]) == 1
    assert re_["graph"]["relations"][0]["relation"] == "issued_by"


# === Test 4: triples v exportech jsou regenerované, ne LLM-provided =======

def test_triples_in_export_are_regenerated():
    text = "iPhone od Apple."
    rec = _make_record(
        text,
        entities=[
            {"id": "e1", "text": "iPhone", "lemma": "iPhone",
             "category": "PRODUKT", "start_char": 0, "end_char": 6},
            {"id": "e2", "text": "Apple", "lemma": "Apple",
             "category": "ORGANIZACE", "start_char": 10, "end_char": 15},
        ],
        relations=[
            {"subjectId": "e1", "objectId": "e2", "relation": "issued_by",
             "relationCs": "vydává", "relationEn": "issued_by"},
        ],
    )
    # Vložíme hallucinated triples — exporty je musí ignorovat
    rec["graph"]["triples"] = [
        {"subject": "vesmír", "relation": "ovlivňuje", "object": "entropie"},
    ]
    _, _, re_ = _split_record_for_spacy(rec, DEFAULT_NER_LABELS)
    triples = re_["graph"]["triples"]
    assert len(triples) == 1
    assert triples[0]["subject"] == "iPhone"
    assert triples[0]["relation"] == "issued_by"
    assert triples[0]["object"] == "Apple"


# === Test 5: _entity_has_valid_span ========================================

def test_entity_has_valid_span():
    text = "Apple iPhone 15."
    # Validní
    assert _entity_has_valid_span(
        {"text": "Apple iPhone", "start_char": 0, "end_char": 12}, text,
    )
    # Špatný substring
    assert not _entity_has_valid_span(
        {"text": "Samsung", "start_char": 0, "end_char": 7}, text,
    )
    # None pozice
    assert not _entity_has_valid_span(
        {"text": "Apple", "start_char": None, "end_char": 5}, text,
    )
    # Out of range
    assert not _entity_has_valid_span(
        {"text": "Apple", "start_char": 0, "end_char": 100}, text,
    )


# === Test 6: _rebuild_triples =============================================

def test_rebuild_triples_deterministic():
    entities = [
        {"id": "e1", "text": "iPhone", "lemma": "iPhone"},
        {"id": "e2", "text": "Apple", "lemma": "Apple"},
    ]
    relations = [
        {"subjectId": "e1", "objectId": "e2", "relation": "issued_by"},
    ]
    triples = _rebuild_triples(entities, relations)
    assert len(triples) == 1
    assert triples[0]["subject"] == "iPhone"
    assert triples[0]["relation"] == "issued_by"
    assert triples[0]["object"] == "Apple"


def test_rebuild_triples_skips_missing_entity():
    """Relace s entitou mimo entities → triple se neobjeví."""
    entities = [
        {"id": "e1", "text": "iPhone", "lemma": "iPhone"},
    ]
    relations = [
        {"subjectId": "e1", "objectId": "e999", "relation": "issued_by"},
    ]
    triples = _rebuild_triples(entities, relations)
    assert triples == []


# === Test 7: custom allowed_ner_labels =====================================

def test_custom_allowed_labels():
    text = "iPhone a Praha a 100 Kč."
    rec = _make_record(text, [
        {"id": "e1", "text": "iPhone", "lemma": "iPhone",
         "category": "PRODUKT", "start_char": 0, "end_char": 6},
        {"id": "e2", "text": "Praha", "lemma": "Praha",
         "category": "MISTO", "start_char": 9, "end_char": 14},
        {"id": "e3", "text": "100 Kč", "lemma": "100 Kč",
         "category": "HODNOTA", "start_char": 17, "end_char": 23},
    ])
    # Custom set — jen PRODUKT a MISTO
    core, _, _ = _split_record_for_spacy(rec, frozenset({"PRODUKT", "MISTO"}))
    labels = {e["category"] for e in core["graph"]["entities"]}
    assert labels == {"PRODUKT", "MISTO"}
    assert "HODNOTA" not in labels


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
