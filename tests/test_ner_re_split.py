"""
Unit testy pro labeler.ner_re_split (split NER/RE).

Pokrývá:
  - extract_list: vytažení entities/relations z různých tvarů odpovědi
  - assemble_graph: tolerantní sestavení (drop dangling / self-loop / dup id /
    nevalidní; alias source/target; lemma fill)
  - critique_split: merge logika (ok / revised / reject kombinace) přes fake klienta

Spuštění: python tests/test_ner_re_split.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.llm_client import LLMClient, LLMResponse
from labeler.ner_re_split import assemble_graph, critique_split, extract_list
from labeler.schemas import Entity, ExtractorResult, Relation, SkoGraph


# ---------- extract_list ---------------------------------------------------

def test_extract_list_direct_key():
    assert extract_list({"entities": [{"id": "e1"}]}, "entities") == [{"id": "e1"}]


def test_extract_list_wrapped_in_corrected():
    assert extract_list({"corrected": {"relations": [{"x": 1}]}}, "relations") == [{"x": 1}]


def test_extract_list_bare_list_and_none():
    assert extract_list([{"a": 1}], "entities") == [{"a": 1}]
    assert extract_list(None, "entities") == []
    assert extract_list({"foo": "bar"}, "entities") == []


# ---------- assemble_graph -------------------------------------------------

def test_assemble_basic_valid():
    g = assemble_graph(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"},
         {"id": "e2", "text": "Brno", "lemma": "Brno", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e2", "relation": "located_in"}],
    )
    assert isinstance(g, SkoGraph)
    assert {e.id for e in g.entities} == {"e1", "e2"}
    assert len(g.relations) == 1


def test_assemble_drops_dangling_relation():
    g = assemble_graph(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e9", "relation": "located_in"}],
    )
    assert g.relations == []  # e9 neexistuje


def test_assemble_drops_self_loop():
    g = assemble_graph(
        [{"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}],
        [{"subjectId": "e1", "objectId": "e1", "relation": "located_in"}],
    )
    assert g.relations == []


def test_assemble_drops_duplicate_and_missing_id():
    g = assemble_graph(
        [{"id": "e1", "text": "A", "lemma": "A", "category": "MISTO"},
         {"id": "e1", "text": "dup", "lemma": "dup", "category": "MISTO"},
         {"text": "no id", "category": "MISTO"}],
        [],
    )
    assert [e.id for e in g.entities] == ["e1"]
    assert g.entities[0].text == "A"  # první vyhrává


def test_assemble_relation_field_aliases():
    """source/target a subject/predicate/object se zmapují (A2 tolerantní příjem)."""
    g = assemble_graph(
        [{"id": "e1", "text": "A", "lemma": "A", "category": "PRODUKT"},
         {"id": "e2", "text": "B", "lemma": "B", "category": "MATERIAL"}],
        [{"source": "e1", "target": "e2", "relation": "made_of"},
         {"subject": "e2", "predicate": "has_component", "object": "e1"}],
    )
    assert len(g.relations) == 2


def test_assemble_fills_missing_lemma_from_text():
    g = assemble_graph(
        [{"id": "e1", "text": "Foto ACME", "category": "ORGANIZACE"}], []
    )
    assert g.entities[0].lemma == "Foto ACME"


def test_assemble_never_raises_on_garbage():
    g = assemble_graph(["not a dict", 42, {"id": "bad-id"}], ["junk", {"relation": "x"}])
    assert isinstance(g, SkoGraph)
    assert g.entities == [] and g.relations == []


# ---------- critique_split: merge logika -----------------------------------

class _Resp:
    def __init__(self, parsed):
        self.parsed = parsed
        self.content = json.dumps(parsed) if parsed is not None else ""
        self.latency_ms = 5
        self.retries = 0


class _FakeClient:
    """Vrací NER vs RE odpověď podle obsahu posledního user message."""
    model = "fake-critic"

    def __init__(self, ner_parsed, re_parsed):
        self._ner = ner_parsed
        self._re = re_parsed

    async def chat(self, messages, **kw):
        content = messages[-1]["content"]
        return _Resp(self._ner if "POUZE ENTITY" in content else self._re)


def _extraction():
    text = "Praha je město a Brno je město."
    graph = SkoGraph(
        entities=[
            Entity(id="e1", text="Praha", lemma="Praha", category="MISTO"),
            Entity(id="e2", text="Brno", lemma="Brno", category="MISTO"),
        ],
        relations=[Relation(subjectId="e1", objectId="e2", relation="located_in")],
    )
    return ExtractorResult(
        item_id="x", source_dataset="s", input_hash="h", input_text=text,
        graph=graph, model="m", latency_ms=1,
    )


def _run(ner, re_):
    return asyncio.run(
        critique_split(_FakeClient(ner, re_), _extraction(), max_tokens=256)
    )


def test_critique_both_ok():
    r = _run({"status": "ok", "reasoning": "fine"}, {"status": "ok", "reasoning": "fine"})
    assert r.status == "ok"
    assert r.corrected is None
    assert r.latency_ms == 10  # 5 + 5


def test_critique_ner_revised_makes_revised():
    ner = {"status": "revised", "reasoning": "fix",
           "corrected": {"entities": [
               {"id": "e1", "text": "Praha", "lemma": "Praha", "category": "MISTO"}]}}
    r = _run(ner, {"status": "ok", "reasoning": "ok"})
    assert r.status == "revised"
    assert r.corrected is not None


def test_critique_re_reject_drops_relations():
    r = _run({"status": "ok", "reasoning": "ok"},
             {"status": "reject", "reasoning": "relace nesmysl"})
    assert r.status == "revised"
    assert r.corrected is not None
    assert r.corrected.relations == []  # všechny relace zahozeny


def test_critique_ner_reject_rejects_item():
    r = _run({"status": "reject", "reasoning": "entity halucinace"},
             {"status": "ok", "reasoning": "ok"})
    assert r.status == "reject"
    assert r.corrected is None


def test_critique_both_fail_is_reject():
    r = _run(None, None)
    assert r.status == "reject"
    assert "nevrátil validní JSON" in r.reasoning


# ---------- NER → RE handoff: RE dostane entity z NER kroku ----------------

class _CapturingClient:
    """Zachytí obsah RE volání — pro ověření, že RE dostane entity z NER kroku.
    Routuje podle markeru: 'POUZE ENTITY' → NER odpověď, jinak RE (+ uloží prompt)."""
    model = "fake"

    def __init__(self, ner_parsed, re_parsed):
        self._ner = ner_parsed
        self._re = re_parsed
        self.re_prompt = None

    async def chat(self, messages, **kw):
        content = messages[-1]["content"]
        if "POUZE ENTITY" in content:
            return _Resp(self._ner)
        self.re_prompt = content
        return _Resp(self._re)


def test_critic_re_receives_ner_corrected_entities():
    """RE critic dostane entity PO NER kroku (ne původní). NER přepíše
    KATEGORII na marker → marker se MUSÍ objevit v RE promptu (orig entity
    ho nemají). Pozn.: lemma jako marker použít nejde — z LLM payloadů se
    stripuje (_strip_lemma_entities), od LLM se nepřebírá."""
    ner = {"status": "revised", "reasoning": "fix",
           "corrected": {"entities": [
               {"id": "e1", "text": "Praha", "category": "MISTO_NER_MARKER"}]}}
    client = _CapturingClient(ner, {"status": "ok", "reasoning": "ok"})
    asyncio.run(critique_split(client, _extraction(), max_tokens=256))
    assert client.re_prompt is not None, "RE critic se vůbec nezavolal"
    assert "MISTO_NER_MARKER" in client.re_prompt  # NER-corrected entita předaná do RE
    # Lemma se do LLM promptů neposílá vůbec (dopočítá ho Stanza ve finalize)
    assert '"lemma"' not in client.re_prompt


def test_extract_re_receives_ner_entities():
    """Extract: RE volání dostane entity detekované NER krokem (marker
    v kategorii — lemma se z LLM payloadů stripuje)."""
    from labeler.datasets import LabelItem
    from labeler.ner_re_split import extract_graph_split
    item = LabelItem(item_id="x", source="s", domain="e-shop", text="Praha je město.")
    ner = {"entities": [
        {"id": "e1", "text": "Praha", "category": "MISTO_NER_MARKER"}],
        "relations": []}
    client = _CapturingClient(ner, {"relations": []})
    asyncio.run(extract_graph_split(client, item, max_tokens=256))
    assert client.re_prompt is not None, "RE extract se vůbec nezavolal"
    assert "MISTO_NER_MARKER" in client.re_prompt
    assert '"lemma"' not in client.re_prompt


def test_critic_ner_reject_skips_re_call():
    """NER reject → RE critic se vůbec nezavolá (short-circuit, NER před RE)."""
    client = _CapturingClient(
        {"status": "reject", "reasoning": "halucinace"},
        {"status": "ok", "reasoning": "ok"},
    )
    r = asyncio.run(critique_split(client, _extraction(), max_tokens=256))
    assert r.status == "reject"
    assert client.re_prompt is None  # RE se po NER reject nevolá


# ---------- regrese: bare JSON list z modelu (Qwen3.5 NER/RE) --------------
# PRODUKČNÍ BUG: Qwen3.5 vracel entity jako bare list [..] místo {"entities":[..]}.
# _parse_json bare list zahodil (vracel jen dict → None) → extract_graph_split
# dostalo ner_parsed=None → KAŽDÝ item zahozen → prázdný extractions.jsonl
# (0 uloženo z 250k itemů, 5h běh, žádná chyba, žádný runaway). Combined cesta
# (SPLIT_NER_RE=false) tím netrpěla, protože smoke jel s jiným modelem.

def test_parse_json_accepts_bare_list():
    pj = LLMClient._parse_json
    assert pj('[{"id":"e1"}]') == [{"id": "e1"}]
    # thinking trace + bare list = přesný produkční tvar (~1940 tok, finish=stop)
    assert pj('<think>\nanalyza\n</think>\n[{"id":"e1"}]') == [{"id": "e1"}]
    # ```json fence``` s bare listem
    assert pj('```json\n[{"id":"e1"}]\n```') == [{"id": "e1"}]
    # objekt pořád funguje (žádná regrese)
    assert pj('{"entities":[{"id":"e1"}]}') == {"entities": [{"id": "e1"}]}
    # vyloženě ne-JSON → pořád None
    assert pj("jen text bez jsonu") is None


class _RawClient:
    """Vrací SYROVÝ obsah modelu a parsuje ho SKUTEČNÝM _parse_json (na rozdíl
    od _FakeClient, který parsed obchází) — testuje celou parse cestu vč. bare list."""
    model = "fake"

    def __init__(self, ner_raw, re_raw):
        self._ner = ner_raw
        self._re = re_raw

    async def chat(self, messages, **kw):
        raw = self._ner if "POUZE ENTITY" in messages[-1]["content"] else self._re
        return LLMResponse(
            content=raw, parsed=LLMClient._parse_json(raw),
            input_tokens=0, output_tokens=0, latency_ms=5, retries=0,
            finish_reason="stop",
        )


def test_extract_graph_split_bare_list_not_dropped():
    """Regrese: model vrátí entity jako bare list (+ thinking) → item se NESMÍ
    zahodit. Dřív extract_graph_split vracel None (0 uloženo)."""
    from labeler.datasets import LabelItem
    from labeler.ner_re_split import extract_graph_split
    item = LabelItem(item_id="x", source="s", domain="e-shop", text="Praha je město.")
    ner_raw = ('<think>\nNajdu entity\n</think>\n'
               '[{"id":"e1","text":"Praha","lemma":"Praha","category":"MISTO"}]')
    re_raw = '<think>\nVztahy\n</think>\n[]'
    res = asyncio.run(
        extract_graph_split(_RawClient(ner_raw, re_raw), item, max_tokens=256)
    )
    assert res is not None, "item zahozen — bare list bug je zpět!"
    graph, _ = res
    assert [e.text for e in graph.entities] == ["Praha"]


def test_critique_split_bare_list_does_not_crash():
    """Critic dostane bare list (ne objekt se statusem) → _as_dict ho degraduje
    na {} (status ok), nesmí spadnout na AttributeError."""
    r = _run([{"id": "e1"}], [{"subjectId": "e1"}])
    assert r is not None
    assert r.status in ("ok", "revised", "reject")


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
