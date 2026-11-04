"""
Unit testy pro oddělení keyphrase entit z NER grafu (finalize,
FINAL_EXCLUDE_KEYPHRASES).

Pokrývá:
  - graph: bez KEYPHRASE_* entit, jen relace mezi NER entitami, triples
  - blok "keyphrases": keyphrase entity + BEZPEČNÉ relace, které se jich týkají
  - pojistky: obecné slovo (KEYPHRASE_SUBJECT), keyphrase↔keyphrase,
    neplatný span, duplicita → relace se nezachová (a spočítá se důvod)
  - záznam bez keyphrase se nemění
  - spaCy split vidí keyphrase z bloku
  - přepínač FINAL_EXCLUDE_KEYPHRASES z env i z .env souboru

Spuštění: python tests/test_keyphrase_exclusion.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from labeler.finalize import (
    _is_keyphrase_category,
    _split_keyphrases,
    _split_record_for_spacy,
)

TEXT = "Pojištění ALFA od ACME Pojišťovny chrání dům a přináší klid i jistotu."


def _ent(eid, text, category, start=None):
    start = TEXT.index(text) if start is None else start
    return {"id": eid, "text": text, "lemma": text, "category": category,
            "start_char": start, "end_char": start + len(text)}


def _rel(s, rel, o):
    return {"subjectId": s, "objectId": o, "relation": rel}


def _record():
    return {
        "item_id": "demo:x:1",
        "source_dataset": "demo",
        "input_text": TEXT,
        "critic_status": "ok",
        "graph": {
            "entities": [
                _ent("e1", "ALFA", "PRODUKT"),
                _ent("e2", "ACME Pojišťovny", "ORGANIZACE"),
                _ent("e3", "chrání dům", "KEYPHRASE_BENEFIT"),
                _ent("e4", "klid", "KEYPHRASE_SUBJECT"),
                _ent("e5", "jistotu", "KEYPHRASE_BENEFIT"),
                _ent("e6", "přináší", "KEYPHRASE_BENEFIT", start=0),  # špatný span
            ],
            "relations": [
                _rel("e2", "offers", "e1"),          # NER ↔ NER → graph
                _rel("e1", "has_benefit", "e3"),     # bezpečná → zachovat
                _rel("e1", "has_benefit", "e3"),     # duplicita → zahodit
                _rel("e1", "has_benefit", "e4"),     # obecné slovo → zahodit
                _rel("e3", "related_to", "e5"),      # keyphrase ↔ keyphrase → zahodit
                _rel("e1", "has_benefit", "e6"),     # neplatný span → zahodit
            ],
            "triples": [],
        },
    }


def test_is_keyphrase_category():
    assert _is_keyphrase_category("KEYPHRASE_BENEFIT")
    assert not _is_keyphrase_category("PRODUKT")
    assert not _is_keyphrase_category(None)


def test_graph_is_clean_ner():
    out, _ = _split_keyphrases(_record())
    g = out["graph"]
    assert [e["category"] for e in g["entities"]] == ["PRODUKT", "ORGANIZACE"]
    assert [r["relation"] for r in g["relations"]] == ["offers"]
    assert len(g["triples"]) == 1
    assert out["input_text"] == TEXT and out["item_id"] == "demo:x:1"


def test_safe_relations_kept_in_keyphrase_block():
    out, stats = _split_keyphrases(_record())
    kp = out["keyphrases"]
    assert [(r["subjectId"], r["relation"], r["objectId"]) for r in kp["relations"]] == [
        ("e1", "has_benefit", "e3")
    ]
    assert kp["triples"] == [{"subject": "ALFA", "relation": "has_benefit", "object": "chrání dům"}]
    # keyphrase entity s platným spanem jsou v bloku, s neplatným ne
    assert sorted(e["id"] for e in kp["entities"]) == ["e3", "e4", "e5"]
    # každý konec zachované relace je dohledatelný v graph nebo v bloku
    ids = {e["id"] for e in out["graph"]["entities"]} | {e["id"] for e in kp["entities"]}
    assert all(r["subjectId"] in ids and r["objectId"] in ids for r in kp["relations"])
    assert stats["relations_kept"] == 1
    assert stats["relations_dropped_duplicate"] == 1
    assert stats["relations_dropped_generic_keyphrase"] == 1
    assert stats["relations_dropped_keyphrase_to_keyphrase"] == 1
    assert stats["relations_dropped_invalid_span"] == 1
    assert stats["entities_dropped_invalid_span"] == 1


def test_record_without_keyphrases_unchanged():
    rec = _record()
    rec["graph"]["entities"] = rec["graph"]["entities"][:2]
    rec["graph"]["relations"] = rec["graph"]["relations"][:1]
    out, stats = _split_keyphrases(rec)
    assert out is rec and "keyphrases" not in out and not stats


def test_spacy_split_sees_keyphrases_from_block():
    out, _ = _split_keyphrases(_record())
    core, key, re_ = _split_record_for_spacy(out, frozenset({"PRODUKT", "ORGANIZACE"}))
    assert {e["category"] for e in core["graph"]["entities"]} == {"PRODUKT", "ORGANIZACE"}
    assert {e["id"] for e in key["graph"]["entities"]} == {"e3", "e4", "e5"}
    assert {r["relation"] for r in re_["graph"]["relations"]} == {"offers", "has_benefit"}
    assert "keyphrases" not in core


def _config_flag(env_extra: dict, dotenv: str = "") -> str:
    """Hodnota CONFIG.final_exclude_keyphrases v čistém procesu (config čte
    env + .env z CWD při importu)."""
    with tempfile.TemporaryDirectory() as tmp:
        if dotenv:
            Path(tmp, ".env").write_text(dotenv, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "FINAL_EXCLUDE_KEYPHRASES"}
        env.update(env_extra)
        env["PYTHONPATH"] = str(REPO)
        r = subprocess.run(
            [sys.executable, "-c",
             "from labeler.config import CONFIG; print(CONFIG.final_exclude_keyphrases)"],
            cwd=tmp, env=env, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()


def test_toggle_via_env_and_dotenv():
    assert _config_flag({}) == "True"                                    # default ON
    assert _config_flag({"FINAL_EXCLUDE_KEYPHRASES": "false"}) == "False"
    assert _config_flag({}, dotenv="FINAL_EXCLUDE_KEYPHRASES=false\n") == "False"
    assert _config_flag({}, dotenv="FINAL_EXCLUDE_KEYPHRASES=true\n") == "True"


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
