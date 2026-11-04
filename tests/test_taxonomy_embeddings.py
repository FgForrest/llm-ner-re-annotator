"""
Unit testy pro labeler.taxonomy_embeddings (fáze 0b) a embedding ruler auditu.

Bez stahování modelu: fake backend deterministicky zakóduje text jako
bag-of-words vektor (hashované tokeny) → třídy s podobným popisem mají
podobné vektory, odlišné třídy odlišné.

Pokrývá:
  - run_taxonomy_embedding_analysis: zápis 3 výstupů, všechny dvojice,
    heuristický disclaimer (formal_disjointness_proof = false)
  - measure_taxonomy_confusion: nejpodobnější dvojice je první
  - audit s embedding rulerem: zostří nejpodobnější pár bez dat z korpusu
  - ruler 'auto' bez dostupných embeddingů → spadne na LLM ruler
  - zostření řízené embeddingy mění taxonomii, pravidla z více párů se sčítají
  - anotace (confusable_with / disjoint_candidates / embedding_cluster /
    embedding_analysis) se zapíše do entity_taxonomy.json a do promptu

Spuštění: python tests/test_taxonomy_embeddings.py
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import labeler.taxonomy as tax_mod
from labeler.datasets import LabelItem
from labeler.taxonomy import Taxonomy, TaxonomyLabel, audit_and_refine_taxonomy
from labeler.taxonomy_embeddings import (
    EMBEDDINGS_FILE,
    REPORT_JSON_FILE,
    REPORT_MD_FILE,
    TaxonomyEmbeddingConfig,
    _EmbeddingRulerState,
    measure_taxonomy_confusion,
    run_taxonomy_embedding_analysis,
)


class _FakeBackend:
    """Bag-of-words hashovaný do 256 dimenzí — deterministický, bez modelu."""
    name = "fake"
    model_name = "fake-bow"

    def encode(self, texts, batch_size):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().replace(",", " ").replace(".", " ").split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % 256
                out[i, h] += 1.0
        return out


def _tax():
    # DATUM a TERMIN mají skoro stejný popis → nejpodobnější dvojice
    return Taxonomy(labels=[
        TaxonomyLabel(code="DATUM", description="Kalendářní datum nebo časový okamžik události."),
        TaxonomyLabel(code="TERMIN", description="Kalendářní datum nebo časový okamžik lhůty."),
        TaxonomyLabel(code="PRODUKT", description="Konkrétní prodávaný výrobek se značkou a modelem."),
        TaxonomyLabel(code="OSOBA", description="Jméno fyzické osoby, zákazníka nebo zaměstnance."),
    ])


def _cfg(**kw):
    return TaxonomyEmbeddingConfig(model="fake-bow", device="cpu", **kw)


def test_analysis_writes_reports_for_all_pairs():
    tax = _tax()
    with tempfile.TemporaryDirectory() as tmp:
        tax_path = Path(tmp) / "entity_taxonomy.json"
        tax_path.write_text(json.dumps(tax.to_dict(), ensure_ascii=False), encoding="utf-8")
        report = run_taxonomy_embedding_analysis(tax_path, Path(tmp), _cfg(), backend=_FakeBackend())
        for name in (REPORT_JSON_FILE, REPORT_MD_FILE, EMBEDDINGS_FILE):
            assert (Path(tmp) / name).exists(), name
        data = json.loads((Path(tmp) / REPORT_JSON_FILE).read_text(encoding="utf-8"))
    n = len(tax.labels)
    assert len(report.pairs) == n * (n - 1) // 2
    assert json.dumps(data).count('"formal_disjointness_proof": false') >= 1


def test_measure_confusion_ranks_most_similar_pair_first():
    m = measure_taxonomy_confusion(_tax(), _cfg(), _FakeBackend())
    a, b, sim, _z = m.pairs[0]
    assert {a, b} == {"DATUM", "TERMIN"}
    assert sim == max(p[2] for p in m.pairs)
    assert m.score >= 0.0


class _SharpenOnlyClient:
    """Fake LLM: embedding ruler neměří přes LLM → volá se jen sharpen."""
    model = "fake"

    def __init__(self):
        self.sharpen_pairs = []

    async def chat(self, messages, **kw):
        user = messages[-1]["content"]
        codes = [c for c in ("DATUM", "TERMIN", "PRODUKT", "OSOBA") if c in user]
        self.sharpen_pairs.append(tuple(sorted(codes)))

        class _Resp:
            content = ""
            parsed = {"labels": [
                {"code": c, "description": f"Ostřejší popis třídy {c} bez překryvu.",
                 "decision_rule": f"Když váháš, rozhodni pro {c} podle kontextu.",
                 "counter_examples": ["x"]}
                for c in codes[:2]
            ]}
        return _Resp()


def test_audit_with_embedding_ruler_sharpens_top_pair_without_corpus():
    orig_create, orig_scan = tax_mod.create_embedding_ruler, tax_mod.scan_data_dir
    tax_mod.create_embedding_ruler = lambda: _EmbeddingRulerState(config=_cfg(), backend=_FakeBackend())

    def _no_scan(_dir):
        raise AssertionError("embedding ruler nemá číst korpus")
    tax_mod.scan_data_dir = _no_scan
    client = _SharpenOnlyClient()
    try:
        out = asyncio.run(audit_and_refine_taxonomy(
            client=client, tax=_tax(), ruler="embedding", max_rounds=2, max_pairs=1,
        ))
    finally:
        tax_mod.create_embedding_ruler, tax_mod.scan_data_dir = orig_create, orig_scan
    assert client.sharpen_pairs and client.sharpen_pairs[0] == ("DATUM", "TERMIN")
    assert "embedding" in out.notes
    assert out.validate() == []


def test_auto_ruler_falls_back_to_llm_when_embeddings_unavailable():
    orig = tax_mod.create_embedding_ruler
    tax_mod.create_embedding_ruler = lambda: None
    try:
        # prázdný korpus → LLM ruler hned skončí; důležité je, že neselže
        # a poznamená LLM ruler
        orig_scan = tax_mod.scan_data_dir
        tax_mod.scan_data_dir = lambda _dir: []
        try:
            out = asyncio.run(audit_and_refine_taxonomy(
                client=_SharpenOnlyClient(), tax=_tax(), ruler="auto",
            ))
        finally:
            tax_mod.scan_data_dir = orig_scan
    finally:
        tax_mod.create_embedding_ruler = orig
    assert [lb.code for lb in out.labels] == [lb.code for lb in _tax().labels]


def _pair(a, b, score, support=1, sources="llm"):
    return tax_mod.ConfusablePair(a=a, b=b, score=score, support=support, sources=sources)


def _report(score, pairs, *, ruler, n_spans=10, evidence=None):
    return tax_mod.ConfusionReport(
        score=score, n_spans=n_spans, pairs=pairs, evidence=evidence or {}, ruler=ruler,
    )


def test_combine_merges_scores_pairs_and_evidence():
    emb = _report(0.4, [_pair("DATUM", "TERMIN", 0.8, 1, "emb"),
                        _pair("OSOBA", "PRODUKT", 0.4, 1, "emb")], ruler="embedding")
    llm = _report(0.2, [_pair("DATUM", "TERMIN", 2.0, 3),
                        _pair("MISTO", "OSOBA", 1.0, 1)], ruler="llm",
                 evidence={("DATUM", "TERMIN"): ["2 roky"]})
    out = tax_mod._combine_confusion_reports(emb, llm, 0.5)
    assert out.ruler == "combined"
    assert out.score == 0.5 * 0.4 + 0.5 * 0.2
    by = {(p.a, p.b): p for p in out.pairs}
    # pár nalezený OBĚMA rulery je první a má nejvyšší skóre
    assert out.pairs[0].a, out.pairs[0].b == ("DATUM", "TERMIN")
    assert by[("DATUM", "TERMIN")].sources == "both"
    assert by[("DATUM", "TERMIN")].score == 0.5 * 1.0 + 0.5 * 1.0
    assert by[("OSOBA", "PRODUKT")].sources == "emb"
    assert by[("MISTO", "OSOBA")].sources == "llm"
    # evidence (reálně zaměněné výrazy pro zostřovací prompt) z LLM probe
    assert out.evidence[("DATUM", "TERMIN")] == ["2 roky"]


def test_combine_falls_back_when_one_side_missing_or_empty():
    llm = _report(0.3, [_pair("A", "B", 1.0)], ruler="llm")
    emb = _report(0.5, [_pair("A", "B", 0.9, 1, "emb")], ruler="embedding")
    assert tax_mod._combine_confusion_reports(None, llm, 0.5) is llm
    assert tax_mod._combine_confusion_reports(emb, None, 0.5) is emb
    assert tax_mod._combine_confusion_reports(_report(0.0, [], ruler="embedding", n_spans=0), llm, 0.5) is llm
    assert tax_mod._combine_confusion_reports(emb, _report(0.0, [], ruler="llm", n_spans=0), 0.5) is emb


def test_combined_support_filter_applies_only_to_llm_only_pairs():
    rep = _report(0.3, [
        _pair("A", "B", 0.9, 1, "llm"),    # jen LLM, support < min → vyfiltrovat
        _pair("C", "D", 0.8, 1, "emb"),    # embedding → projde bez supportu
        _pair("E", "F", 0.7, 3, "both"),   # oba → projde
    ], ruler="combined")
    got = [(p.a, p.b) for p in tax_mod._select_confusable_pairs(rep, max_pairs=5, min_support=2)]
    assert got == [("C", "D"), ("E", "F")]


class _ProbeAndSharpenClient:
    """Fake LLM: na probe vrací zaměněné spany, na zostření nové popisy."""
    model = "fake"

    def __init__(self):
        self.probe_calls = 0
        self.sharpen_calls = 0

    async def chat(self, messages, **kw):
        system = messages[0]["content"]
        if "auditor NER taxonomie" in system:
            self.probe_calls += 1
            parsed = {"spans": [
                {"text": "15. 3. 2026", "top1": "DATUM", "top2": "TERMIN", "confidence": 0.5},
                {"text": "do konce roku", "top1": "TERMIN", "top2": "DATUM", "confidence": 0.5},
            ]}
        else:
            self.sharpen_calls += 1
            parsed = {"labels": [
                {"code": "DATUM", "description": "Datum události, ne lhůty.",
                 "decision_rule": "Lhůta → TERMIN.", "counter_examples": ["do konce roku"]},
                {"code": "TERMIN", "description": "Datum lhůty, ne události.",
                 "decision_rule": "Událost → DATUM.", "counter_examples": ["15. 3. 2026"]},
            ]}

        class _Resp:
            content = ""
        _Resp.parsed = parsed
        return _Resp()


def _run_audit(ruler, client, *, items=None, embedding=True):
    orig_create, orig_scan = tax_mod.create_embedding_ruler, tax_mod.scan_data_dir
    tax_mod.create_embedding_ruler = (
        (lambda: _EmbeddingRulerState(config=_cfg(), backend=_FakeBackend()))
        if embedding else (lambda: None)
    )
    tax_mod.scan_data_dir = lambda _dir: (items if items is not None else [
        LabelItem(item_id=f"i{i}", source="s", domain="d", text=f"Text {i} o produktu.")
        for i in range(20)
    ])
    try:
        return asyncio.run(audit_and_refine_taxonomy(
            client=client, tax=_tax(), ruler=ruler, max_rounds=2, max_pairs=1,
            min_support=1,
        ))
    finally:
        tax_mod.create_embedding_ruler, tax_mod.scan_data_dir = orig_create, orig_scan


def test_combined_ruler_uses_both_embeddings_and_llm():
    client = _ProbeAndSharpenClient()
    out = _run_audit("combined", client)
    assert client.probe_calls >= 1          # LLM probe proběhl
    assert "embedding+llm" in out.notes     # a zároveň embeddingy
    assert client.sharpen_calls >= 1
    assert out.validate() == []


def test_combined_without_corpus_measures_embeddings_only():
    client = _ProbeAndSharpenClient()
    out = _run_audit("combined", client, items=[])
    assert client.probe_calls == 0
    assert "disjointness audit (embedding)" in out.notes


def test_auto_keeps_original_either_or_behaviour():
    """auto = jen JEDNO měření: embeddingy, když jsou; jinak LLM probe."""
    client = _ProbeAndSharpenClient()
    out = _run_audit("auto", client)
    assert client.probe_calls == 0                       # LLM probe se nevolá
    assert "disjointness audit (embedding)" in out.notes

    client = _ProbeAndSharpenClient()
    out = _run_audit("auto", client, embedding=False)    # bez embeddingů → LLM
    assert client.probe_calls >= 1
    assert "disjointness audit (llm)" in out.notes


def test_explicit_single_rulers():
    client = _ProbeAndSharpenClient()
    out = _run_audit("embedding", client)
    assert client.probe_calls == 0 and "(embedding)" in out.notes

    client = _ProbeAndSharpenClient()
    out = _run_audit("llm", client)
    assert client.probe_calls >= 1 and "(llm)" in out.notes


def _tax6():
    # Dvě skupiny zaměnitelných tříd (DATUM/TERMIN/LHUTA a PRODUKT/VYROBEK)
    # + OSOBA daleko od všech.
    return Taxonomy(labels=[
        TaxonomyLabel(code="DATUM", description="Kalendářní datum nebo časový okamžik události."),
        TaxonomyLabel(code="TERMIN", description="Kalendářní datum nebo časový okamžik lhůty."),
        TaxonomyLabel(code="LHUTA", description="Kalendářní datum nebo časový okamžik splatnosti."),
        TaxonomyLabel(code="PRODUKT", description="Konkrétní prodávaný výrobek se značkou a modelem."),
        TaxonomyLabel(code="VYROBEK", description="Konkrétní prodávaný výrobek se značkou a typem."),
        TaxonomyLabel(code="OSOBA", description="Jméno fyzické osoby, zákazníka nebo zaměstnance."),
    ])


class _PairSharpenClient:
    """Fake LLM pro zostření: pravidlo nese kód partnera → lze ověřit, že se
    pravidla z více párů u jedné třídy SČÍTAJÍ, ne přepisují."""
    model = "fake"

    def __init__(self):
        self.pairs = []

    async def chat(self, messages, **kw):
        user = messages[-1]["content"]
        head = user.split("REÁLNĚ")[0]
        codes = [lb.code for lb in _tax6().labels if f"  {lb.code}:" in head]
        a, b = codes
        self.pairs.append((a, b))

        class _Resp:
            content = ""
            parsed = {"labels": [
                {"code": a, "description": f"Ostřejší popis {a}, odlišný od {b}.",
                 "decision_rule": f"{a} ne {b}.", "counter_examples": [f"ex-{b}"]},
                {"code": b, "description": f"Ostřejší popis {b}, odlišný od {a}.",
                 "decision_rule": f"{b} ne {a}.", "counter_examples": [f"ex-{a}"]},
            ]}
        return _Resp()


def _with_fake_backend(fn):
    import labeler.taxonomy_embeddings as emb_mod
    orig = emb_mod.create_backend
    emb_mod.create_backend = lambda _cfg: _FakeBackend()
    try:
        return fn()
    finally:
        emb_mod.create_backend = orig


def test_embedding_refinement_changes_taxonomy_and_accumulates_rules():
    client = _PairSharpenClient()
    out = _with_fake_backend(lambda: asyncio.run(
        tax_mod.refine_taxonomy_with_embeddings(client, _tax6(), max_pairs=10)
    ))
    assert client.pairs, "embeddingy měly označit zaměnitelné páry"
    flagged_codes = {c for pair in client.pairs for c in pair}
    assert "OSOBA" not in flagged_codes
    by = {lb.code: lb for lb in out.labels}
    # každá zostřená třída má pravidla pro VŠECHNY své partnery (sčítají se)
    for code in flagged_codes:
        partners = {b if a == code else a for a, b in client.pairs if code in (a, b)}
        for other in partners:
            assert f"{code} ne {other}." in by[code].decision_rule
            assert f"ex-{other}" in by[code].counter_examples
    assert by["OSOBA"].decision_rule == ""
    assert "embedding zostření" in out.notes
    assert out.validate() == []


def test_merge_refinement_ignores_codes_outside_pair():
    tax = _tax6()
    rules, described = {}, set()
    changed = tax_mod._merge_refinement(
        tax, [{"code": "OSOBA", "description": "Něco úplně jiného dlouhého.",
               "decision_rule": "x"}],
        ("DATUM", "TERMIN"), rules, described,
    )
    assert changed == 0 and rules == {}


def test_annotation_writes_relations_into_taxonomy():
    import labeler.taxonomy_embeddings as emb_mod
    tax = _tax6()
    with tempfile.TemporaryDirectory() as tmp:
        tax_path = Path(tmp) / "entity_taxonomy.json"
        tax_path.write_text(json.dumps(tax.to_dict(), ensure_ascii=False), encoding="utf-8")
        report = emb_mod.run_taxonomy_embedding_analysis(
            tax_path, Path(tmp), _cfg(), backend=_FakeBackend(),
        )
    tax_mod.annotate_taxonomy_with_embeddings(tax, report)
    by = {lb.code: lb for lb in tax.labels}
    assert by["DATUM"].confusable_with and by["DATUM"].confusable_with[0] in ("TERMIN", "LHUTA")
    assert "OSOBA" not in by["DATUM"].confusable_with
    assert all(lb.embedding_cluster >= 0 for lb in tax.labels)
    assert tax.embedding_analysis["model"] == "fake-bow"
    assert tax.embedding_analysis["formal_disjointness_proof"] is False
    # anotace přežije round-trip a nemění content hash (není vstupem analýzy)
    tax2 = Taxonomy.from_dict(tax.to_dict())
    assert tax2.labels[0].confusable_with == tax.labels[0].confusable_with
    assert tax2.embedding_analysis == tax.embedding_analysis
    assert emb_mod.taxonomy_content_hash(tax2) == emb_mod.taxonomy_content_hash(_tax6())


def test_prompt_block_renders_confusable_with():
    from labeler.prompts import _taxonomy_block
    tax = _tax6()
    tax.labels[0].confusable_with = ["TERMIN", "LHUTA"]
    block = _taxonomy_block(tax)
    assert "Pozor na záměnu s: TERMIN, LHUTA" in block


def test_taxonomy_discovery_runs_embedding_analysis_automatically():
    """Fáze 0b běží sama po uložení taxonomie (cesta smoke testu i produkce)."""
    import labeler.taxonomy_embeddings as emb_mod
    from labeler.config import CONFIG
    from labeler.taxonomy import TaxonomyMode, run_taxonomy_discovery

    orig_backend, orig_ckpt = emb_mod.create_backend, CONFIG.checkpoint_dir
    emb_mod.create_backend = lambda _cfg: _FakeBackend()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            object.__setattr__(CONFIG, "checkpoint_dir", Path(tmp) / "ckpt")
            user_tax = Path(tmp) / "user_taxonomy.json"
            user_tax.write_text(json.dumps(_tax().to_dict(), ensure_ascii=False), encoding="utf-8")
            path = asyncio.run(run_taxonomy_discovery(
                mode=TaxonomyMode.USER, user_taxonomy_path=user_tax,
            ))
            assert CONFIG.taxonomy_embedding_enabled  # default ON
            for name in (REPORT_JSON_FILE, REPORT_MD_FILE, EMBEDDINGS_FILE):
                assert (path.parent / name).exists(), name
            # výsledky embeddingů jsou zapsané PŘÍMO v entity_taxonomy.json
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["embedding_analysis"]["model"] == CONFIG.taxonomy_embedding_model
            by = {lb["code"]: lb for lb in saved["labels"]}
            assert set(by["DATUM"]["confusable_with"]) == {"TERMIN"}
            assert all("embedding_cluster" in lb for lb in saved["labels"])
    finally:
        emb_mod.create_backend = orig_backend
        object.__setattr__(CONFIG, "checkpoint_dir", orig_ckpt)


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
