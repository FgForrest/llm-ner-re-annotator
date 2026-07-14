"""
Unit testy pro deterministický graph diff ve finalize (changes_applied).

Ověřuje, že `changes_applied` je GROUND TRUTH (počítaný z grafů), content-based
(stabilní přes re-číslování ID) — nikdy neodporuje finálnímu grafu, na rozdíl
od LLM critic_reasoning.

Spuštění: python tests/test_finalize_diff.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.finalize import _entity_keys, _graph_diff, _relation_keys
from labeler.schemas import Entity, Relation, SkoGraph


def _g(entities, relations):
    return SkoGraph(
        entities=[Entity(**e) for e in entities],
        relations=[Relation(**r) for r in relations],
    )


def _diff(orig, final):
    return _graph_diff(_entity_keys(orig), _relation_keys(orig), final)


def test_diff_removed_entity_and_relation():
    orig = _g(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
         {"id": "e2", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e2", "relation": "located_in"}],
    )
    final = _g([{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}], [])
    d = _diff(orig, final)
    assert d["entities_removed"] == ["brno (MISTO)"], d
    assert d["relations_removed"] == ["praha -located_in-> brno"], d
    assert d["entities_added"] == [] and d["relations_added"] == []


def test_diff_added_entity_and_relation():
    orig = _g([{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}], [])
    final = _g(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
         {"id": "e2", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e2", "relation": "located_in"}],
    )
    d = _diff(orig, final)
    assert d["entities_added"] == ["brno (MISTO)"], d
    assert d["relations_added"] == ["praha -located_in-> brno"], d


def test_diff_no_change_is_empty():
    g = _g([{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}], [])
    d = _diff(g, g)
    assert d == {
        "entities_added": [], "entities_removed": [],
        "relations_added": [], "relations_removed": [],
    }


def test_diff_stable_across_reid():
    """Stejný obsah, jiná ID → žádný diff (klíč je obsahový, ne ID)."""
    orig = _g(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
         {"id": "e2", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e2", "relation": "located_in"}],
    )
    final = _g(
        [{"id": "e9", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
         {"id": "e7", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
        [{"subjectId": "e9", "objectId": "e7", "relation": "located_in"}],
    )
    d = _diff(orig, final)
    assert all(v == [] for v in d.values()), d


def test_diff_category_change_shows_add_and_remove():
    """Změna kategorie = remove (stará) + add (nová) — ground truth."""
    orig = _g([{"id": "e1", "text": "šperky", "lemma": "šperk", "category": "PRODUKT_TYP"}], [])
    final = _g([{"id": "e1", "text": "šperky", "lemma": "šperk", "category": "PRODUKT"}], [])
    d = _diff(orig, final)
    assert d["entities_removed"] == ["šperk (PRODUKT_TYP)"], d
    assert d["entities_added"] == ["šperk (PRODUKT)"], d


def test_finalize_revised_item_endtoend():
    """E2E: run_finalize na 'revised' itemu — ověří, že revised cesta NEpadá
    (regrese na NameError v revised_examples) a že changes_applied / oddělený
    reasoning / final_graph_source jsou správně. Tohle unit testy diffu
    nezachytí, protože nejedou přes _process_branch."""
    import json
    import tempfile
    from labeler.config import CONFIG, critic_path, extraction_path, final_path
    from labeler.finalize import run_finalize
    from labeler.schemas import CriticResult, ExtractorResult

    old = CONFIG.checkpoint_dir
    tmp = Path(tempfile.mkdtemp())
    object.__setattr__(CONFIG, "checkpoint_dir", tmp)
    try:
        text = "Praha je krásné město a Brno je taky město."
        ex = ExtractorResult(
            item_id="x", source_dataset="s", input_hash="h", input_text=text,
            graph=_g(
                [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
                 {"id": "e2", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
                [{"subjectId": "e1", "objectId": "e2", "relation": "located_in"}],
            ),
            model="ex", latency_ms=1,
        )
        cr = CriticResult(
            item_id="x", status="revised", reasoning="Brno je halucinace, mažu ji.",
            corrected=_g([{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}], []),
            extractor_revised=None, extractor_revised_reasoning=None,
            model="cr", latency_ms=1,
        )
        extraction_path(synthetic=False).write_text(
            json.dumps(ex.model_dump(mode="json"), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        critic_path(synthetic=False).write_text(
            json.dumps(cr.model_dump(mode="json"), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        run_finalize(
            keep_intermediate=True, normalize_opts=None, write_report=False,
            export_spacy=False, export_owl=False,
        )
        recs = [
            json.loads(l)
            for l in final_path(synthetic=False).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert len(recs) == 1, recs
        r = recs[0]
        assert r["was_revised"] is True
        assert r["final_graph_source"] == "critic_corrected", r["final_graph_source"]
        assert r["changes_applied"]["entities_removed"] == ["brno (MISTO)"], r["changes_applied"]
        assert r["changes_applied"]["relations_removed"] == ["praha -located_in-> brno"]
        assert "Brno je halucinace" in r["critic_reasoning"]
    finally:
        object.__setattr__(CONFIG, "checkpoint_dir", old)


# ---------- Runner ---------------------------------------------------------

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
