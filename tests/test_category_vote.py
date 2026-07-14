"""
Unit testy pro labeler.category_vote (korpusové hlasování o kategoriích).

DŮLEŽITÉ: Doménově NEUTRÁLNÍ — zástupné termíny (Alpha, Beta, widget),
žádná reálná doménová slova. Testuje se:

  - norm_key / entity_key normalizace
  - CorpusCategoryStats.build (rejecty se nepočítají)
  - dominant / support / preferred_category (determinismus, remíza = None)
  - text_doc_ratio (word boundary — "auto" se nenajde v "automobil")
  - is_generic_surface (statistický stopword — vysoká DF)
  - is_safe_expand (korpusová konzistence rescue pro krátké texty)
  - apply_corpus_votes (přelabelování menšiny, entropy guard, report)
  - integrace s normalize (konflikty přes korpus, auto-expand DF blok)

Spuštění: python tests/test_category_vote.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.category_vote import (
    CorpusCategoryStats,
    VoteReport,
    _contains_word,
    apply_corpus_votes,
    entity_key,
    norm_key,
)
from labeler.normalize import NormalizeOpts, normalize_graph
from labeler.schemas import Entity, SkoGraph


# ---------- Helpers ----------------------------------------------------------

def _rec(item_id, text, entities, critic_status="ok"):
    """Záznam ve tvaru final_records (dict tvar po merge chunků).
    entities = list[(text, lemma, category)]."""
    return {
        "item_id": item_id,
        "critic_status": critic_status,
        "input_text": text,
        "graph": {
            "entities": [
                {"id": f"e{i}", "text": t, "lemma": lm, "category": c}
                for i, (t, lm, c) in enumerate(entities, 1)
            ],
            "relations": [],
            "triples": [],
        },
    }


def _corpus_alpha_dominant():
    """Korpus: 'Alpha' 4× WIDGET + 1× GADGET (dominance 0.8), 'Beta' 1:1
    remíza (ale support 2 < 3), 'Gamma' 2:2 remíza se supportem 4 (ambiguous)."""
    return [
        _rec("d1", "Alpha je tady. Beta taky. Gamma s tim.",
             [("Alpha", "Alpha", "WIDGET"), ("Beta", "Beta", "WIDGET"),
              ("Gamma", "Gamma", "WIDGET")]),
        _rec("d2", "Alpha znovu. Gamma podruhe.",
             [("Alpha", "Alpha", "WIDGET"), ("Gamma", "Gamma", "WIDGET")]),
        _rec("d3", "Alpha potřetí. Beta zas. Gamma potreti.",
             [("Alpha", "Alpha", "WIDGET"), ("Beta", "Beta", "GADGET"),
              ("Gamma", "Gamma", "GADGET")]),
        _rec("d4", "Alpha počtvrté. Gamma poctvrte.",
             [("Alpha", "Alpha", "WIDGET"), ("Gamma", "Gamma", "GADGET")]),
        _rec("d5", "Alpha jinak.", [("Alpha", "Alpha", "GADGET")]),
    ]


# ---------- norm_key / entity_key -------------------------------------------

def test_norm_key_strips_punct_and_case():
    assert norm_key("ACME Pojišťovna, a.s.") == "acme pojišťovna a s"
    assert norm_key("  Alpha   Beta ") == "alpha beta"
    assert norm_key("") == ""


def test_entity_key_prefers_lemma():
    assert entity_key({"lemma": "widget", "text": "widgety"}) == "widget"
    assert entity_key({"text": "Widgety"}) == "widgety"  # fallback na text
    assert entity_key({}) == ""


# ---------- _contains_word (word boundary) ----------------------------------

def test_contains_word_boundaries():
    assert _contains_word("auto je vůz", "auto") is True
    assert _contains_word("automobil je vůz", "auto") is False  # uvnitř slova
    assert _contains_word("další auto.", "auto") is True        # interpunkce OK
    assert _contains_word("", "auto") is False
    assert _contains_word("cokoliv", "") is False


# ---------- build / dominant / support / preferred_category ------------------

def test_build_counts_and_dominant():
    stats = CorpusCategoryStats.build(_corpus_alpha_dominant(),
                                      dominance=0.8, min_support=3)
    assert stats.n_docs == 5
    assert stats.support("alpha") == 5
    cat, share = stats.dominant("alpha")
    assert cat == "WIDGET"
    assert abs(share - 0.8) < 1e-9
    # Beta: 1:1 remíza → entropy guard → None
    assert stats.dominant("beta") is None


def test_build_excludes_rejects():
    records = _corpus_alpha_dominant()
    records.append(_rec("d6", "Alpha reject.", [("Alpha", "Alpha", "GADGET")],
                        critic_status="reject"))
    stats = CorpusCategoryStats.build(records)
    # Reject se nepočítá — support zůstává 5, dominance 0.8
    assert stats.n_docs == 5
    assert stats.support("alpha") == 5


def test_preferred_category_restricted_to_candidates():
    stats = CorpusCategoryStats.build(_corpus_alpha_dominant())
    # Mezi kandidáty WIDGET/GADGET vyhraje WIDGET (4 > 1)
    assert stats.preferred_category("alpha", ["GADGET", "WIDGET"]) == "WIDGET"
    # Neznámý klíč → None
    assert stats.preferred_category("gamma", ["WIDGET", "GADGET"]) is None
    # Kandidáti bez korpusového počtu → None
    assert stats.preferred_category("alpha", ["NEEXISTUJE"]) is None
    # Remíza (beta 1:1) → None
    assert stats.preferred_category("beta", ["WIDGET", "GADGET"]) is None


def test_stats_deterministic():
    s1 = CorpusCategoryStats.build(_corpus_alpha_dominant())
    s2 = CorpusCategoryStats.build(_corpus_alpha_dominant())
    assert s1.counts == s2.counts
    assert s1.dominant("alpha") == s2.dominant("alpha")


# ---------- text_doc_ratio / is_generic_surface ------------------------------

def test_text_doc_ratio_and_generic():
    records = [
        _rec("d1", "widget se používá tady", [("widget", "widget", "WIDGET")]),
        _rec("d2", "widget se hodí", [("widget", "widget", "WIDGET")]),
        _rec("d3", "něco jiného se děje", []),
        _rec("d4", "úplně jiný text bez", []),
    ]
    stats = CorpusCategoryStats.build(records, max_df_ratio=0.5)
    # "se" je ve 3/4 dokumentů → generikum (funkční slovo, statisticky)
    assert stats.text_doc_ratio("se") == 0.75
    assert stats.is_generic_surface("se") is True
    # "widget" je ve 2/4 → není generikum (na hraně 0.5, ne víc)
    assert stats.text_doc_ratio("widget") == 0.5
    assert stats.is_generic_surface("widget") is False


# ---------- is_safe_expand ---------------------------------------------------

def test_is_safe_expand_corpus_rescue_short_name():
    """Krátký název ('Axo') s konzistentní kategorií a nízkou DF je safe —
    korpus nahrazuje dřívější hard-coded safe-label seznam."""
    records = [
        _rec("d1", "Axo je produkt", [("Axo", "Axo", "WIDGET")]),
        _rec("d2", "Axo znovu", [("Axo", "Axo", "WIDGET")]),
        _rec("d3", "Axo potřetí", [("Axo", "Axo", "WIDGET")]),
        _rec("d4", "text bez zmínky", []),
        _rec("d5", "další text jinde", []),
        _rec("d6", "a ještě jeden", []),
        _rec("d7", "sedmý dokument", []),
    ]
    stats = CorpusCategoryStats.build(records, dominance=0.8, min_support=3,
                                      max_df_ratio=0.5)
    assert stats.is_safe_expand("axo", "WIDGET", "Axo") is True
    # Jiná kategorie než dominantní → unsafe
    assert stats.is_safe_expand("axo", "GADGET", "Axo") is False
    # Nedostatečný support → unsafe
    assert stats.is_safe_expand("neznamy", "WIDGET", "neznamy") is False


# ---------- apply_corpus_votes -----------------------------------------------

def test_apply_votes_relabels_minority():
    records = _corpus_alpha_dominant()
    stats = CorpusCategoryStats.build(records, dominance=0.8, min_support=3)
    rep = apply_corpus_votes(records, stats)
    # Menšinový GADGET (d5) → WIDGET
    d5_ents = records[4]["graph"]["entities"]
    assert d5_ents[0]["category"] == "WIDGET"
    assert rep.relabeled == 1
    assert rep.relabel_pairs["GADGET->WIDGET"] == 1
    # Beta (support 2 < 3) se NEZMĚNILA — pod prahem podpory
    d3_ents = records[2]["graph"]["entities"]
    assert d3_ents[1]["category"] == "GADGET"
    # Gamma (2:2 remíza, support 4) se NEZMĚNILA — entropy guard + report
    assert d3_ents[2]["category"] == "GADGET"
    assert records[0]["graph"]["entities"][2]["category"] == "WIDGET"
    assert rep.ambiguous >= 1


def test_apply_votes_respects_min_support():
    records = [
        _rec("d1", "Gamma tu", [("Gamma", "Gamma", "WIDGET")]),
        _rec("d2", "Gamma tam", [("Gamma", "Gamma", "GADGET")]),
    ]
    stats = CorpusCategoryStats.build(records, dominance=0.5, min_support=3)
    rep = apply_corpus_votes(records, stats)
    # Support 2 < 3 → žádné rozhodnutí, nic se nemění
    assert rep.relabeled == 0
    assert records[0]["graph"]["entities"][0]["category"] == "WIDGET"
    assert records[1]["graph"]["entities"][0]["category"] == "GADGET"


def test_apply_votes_skips_rejects():
    records = _corpus_alpha_dominant()
    records.append(_rec("dr", "Alpha reject", [("Alpha", "Alpha", "GADGET")],
                        critic_status="reject"))
    stats = CorpusCategoryStats.build(records)
    apply_corpus_votes(records, stats)
    # Reject záznam zůstává nedotčený (není v hlavním datasetu)
    assert records[-1]["graph"]["entities"][0]["category"] == "GADGET"


def test_vote_report_to_dict():
    records = _corpus_alpha_dominant()
    stats = CorpusCategoryStats.build(records)
    rep = apply_corpus_votes(records, stats)
    d = rep.to_dict()
    assert d["relabeled"] == rep.relabeled
    assert "relabel_pairs" in d
    assert "ambiguous_sample" in d


# ---------- Integrace s normalize -------------------------------------------

def test_normalize_conflict_uses_corpus_stats():
    """Within-doc konflikt WIDGET vs GADGET rozhodne korpus (ne dokument)."""
    # Korpus: "Delta" je jasně GADGET (4:0), min_support=3
    corpus = [
        _rec(f"d{i}", "Delta je vec", [("Delta", "Delta", "GADGET")])
        for i in range(4)
    ]
    stats = CorpusCategoryStats.build(corpus, dominance=0.8, min_support=3)

    # Dokument s konfliktem: 2× WIDGET, 1× GADGET → dokumentová většina by
    # řekla WIDGET, ale korpus říká GADGET → vyhrává korpus.
    text = "Delta tady. Delta tam. Delta vsude."
    g = SkoGraph(entities=[
        Entity(id="e1", text="Delta", lemma="Delta", category="WIDGET",
               start_char=0, end_char=5),
        Entity(id="e2", text="Delta", lemma="Delta", category="WIDGET",
               start_char=12, end_char=17),
        Entity(id="e3", text="Delta", lemma="Delta", category="GADGET",
               start_char=24, end_char=29),
    ])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True),
                          corpus_stats=stats)
    labels = {e.category for e in res.graph.entities}
    assert labels == {"GADGET"}
    assert res.label_conflicts_resolved == 1


def test_normalize_auto_expand_blocked_by_generic_df():
    """Výraz s vysokou DF (v každém dokumentu) se NEexpanduje — statistický
    stopword filtr místo hard-coded seznamu."""
    corpus = [
        _rec(f"d{i}", "Vsude je slovo widget a widget.", [])
        for i in range(4)
    ]
    stats = CorpusCategoryStats.build(corpus, max_df_ratio=0.5)
    assert stats.is_generic_surface("widget") is True

    text = "Prvni widget je tu. Druhy widget je tam. Treti widget."
    g = SkoGraph(entities=[
        Entity(id="e1", text="widget", lemma="widget", category="WIDGET",
               start_char=7, end_char=13),
    ])
    res = normalize_graph(
        g, text, opts=NormalizeOpts(enabled=True, auto_expand_mentions=True),
        corpus_stats=stats,
    )
    # Bez expanze (DF blok) — jen 1 entita
    assert len(res.graph.entities) == 1
    assert res.mentions_expanded == 0


def test_normalize_auto_expand_corpus_rescue():
    """Krátký název s korpusovou konzistencí SE expanduje (dřív by ho
    zachránil jen hard-coded safe-label seznam)."""
    corpus = [
        _rec("d1", "Axo je tu", [("Axo", "Axo", "WIDGET")]),
        _rec("d2", "Axo je tam", [("Axo", "Axo", "WIDGET")]),
        _rec("d3", "Axo je vsude", [("Axo", "Axo", "WIDGET")]),
        _rec("d4", "jiny text uplne", []),
        _rec("d5", "dalsi text jinde", []),
        _rec("d6", "a jeste jeden text", []),
        _rec("d7", "sedmy dokument tady", []),
    ]
    stats = CorpusCategoryStats.build(corpus, dominance=0.8, min_support=3,
                                      max_df_ratio=0.5)
    text = "Axo je produkt. Axo se hodi. Dalsi Axo."
    g = SkoGraph(entities=[
        Entity(id="e1", text="Axo", lemma="Axo", category="WIDGET",
               start_char=0, end_char=3),
    ])
    res = normalize_graph(
        g, text, opts=NormalizeOpts(enabled=True, auto_expand_mentions=True),
        corpus_stats=stats,
    )
    # 3 výskyty "Axo" → expandováno na 3 entity (korpus rescue krátkého textu)
    assert len(res.graph.entities) == 3
    assert res.mentions_expanded == 2


# ---------- Runner -----------------------------------------------------------

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
