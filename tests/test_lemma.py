"""
Unit testy pro labeler.lemma (deterministická lemmatizace přes Stanza).

Testuje se BEZ závislosti na stanza (fake lemmatizer) — pure logika:
  - lemma_for_span (mapování spanů na tokeny, překryvy, hranice)
  - apply_lemmas_to_records (přepis lemmat, standalone fallback, report)
  - graceful no-op když lemmatizer není available

Volitelný integrační test s reálnou Stanzou se SKIPne, pokud stanza nebo
český model nejsou k dispozici (CI/offline stroje).

Spuštění: python tests/test_lemma.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.lemma import (
    LemmaReport,
    StanzaLemmatizer,
    apply_lemmas_to_records,
    lemma_for_span,
)


# ---------- Helpers ----------------------------------------------------------

class FakeLemmatizer:
    """Duck-typed lemmatizer: pevná mapa text → tokens (žádná stanza)."""

    def __init__(self, tokens_by_text, standalone=None, available=True):
        self.available = available
        self._by_text = tokens_by_text
        self._standalone = standalone or {}
        self.calls = 0

    def tokens_for(self, text):
        self.calls += 1
        return self._by_text.get(text)

    def lemma_of_text(self, text):
        return self._standalone.get(text)


def _rec(text, entities):
    """entities = list[(text, lemma, category, start, end)]"""
    return {
        "item_id": "x",
        "critic_status": "ok",
        "input_text": text,
        "graph": {
            "entities": [
                {"id": f"e{i}", "text": t, "lemma": lm, "category": c,
                 "start_char": s, "end_char": e}
                for i, (t, lm, c, s, e) in enumerate(entities, 1)
            ],
            "relations": [],
            "triples": [],
        },
    }


# ---------- lemma_for_span (pure) --------------------------------------------

# "Widgety jsou skvele" → tokeny (0,7,widget) (8,12,byt) (13,19,skvely)
_TOKENS = [(0, 7, "widget"), (8, 12, "byt"), (13, 19, "skvely")]


def test_lemma_for_span_single_token():
    assert lemma_for_span(_TOKENS, 0, 7) == "widget"


def test_lemma_for_span_multiword_join():
    assert lemma_for_span(_TOKENS, 0, 12) == "widget byt"
    assert lemma_for_span(_TOKENS, 0, 19) == "widget byt skvely"


def test_lemma_for_span_partial_overlap_counts():
    # Span končící uprostřed tokenu → token se počítá (překryv)
    assert lemma_for_span(_TOKENS, 0, 3) == "widget"
    # Span přes hranici dvou tokenů
    assert lemma_for_span(_TOKENS, 5, 10) == "widget byt"


def test_lemma_for_span_no_tokens_or_invalid():
    assert lemma_for_span(_TOKENS, 100, 110) is None   # mimo text
    assert lemma_for_span(_TOKENS, None, 7) is None    # chybějící offset
    assert lemma_for_span(_TOKENS, 7, 7) is None       # prázdný span
    assert lemma_for_span([], 0, 7) is None            # žádné tokeny


# ---------- apply_lemmas_to_records ------------------------------------------

def test_apply_lemmas_overwrites_llm_lemma():
    text = "Widgety jsou skvele"
    rec = _rec(text, [("Widgety", "Widgety", "WIDGET", 0, 7)])
    fake = FakeLemmatizer({text: _TOKENS})
    rep = apply_lemmas_to_records([rec], fake)
    ent = rec["graph"]["entities"][0]
    assert ent["lemma"] == "widget"          # LLM "Widgety" → Stanza "widget"
    assert rep.lemmas_changed == 1
    assert rep.docs_processed == 1
    assert rep.entities_seen == 1
    assert any("widget" in s for s in rep.changed_sample)


def test_apply_lemmas_unchanged_when_same():
    text = "Widgety jsou skvele"
    rec = _rec(text, [("Widgety", "widget", "WIDGET", 0, 7)])  # už správné
    fake = FakeLemmatizer({text: _TOKENS})
    rep = apply_lemmas_to_records([rec], fake)
    assert rep.lemmas_changed == 0
    assert rec["graph"]["entities"][0]["lemma"] == "widget"


def test_apply_lemmas_standalone_fallback_without_offsets():
    text = "Widgety jsou skvele"
    rec = _rec(text, [("Widgety", "Widgety", "WIDGET", None, None)])
    fake = FakeLemmatizer(
        {text: _TOKENS}, standalone={"Widgety": "widget"},
    )
    rep = apply_lemmas_to_records([rec], fake)
    assert rec["graph"]["entities"][0]["lemma"] == "widget"
    assert rep.entities_standalone_fallback == 1


def test_apply_lemmas_keeps_llm_lemma_when_stanza_gives_nothing():
    text = "Widgety jsou skvele"
    rec = _rec(text, [("Neznamy", "puvodni-lemma", "WIDGET", None, None)])
    fake = FakeLemmatizer({text: _TOKENS})  # standalone nic nevrátí
    rep = apply_lemmas_to_records([rec], fake)
    assert rec["graph"]["entities"][0]["lemma"] == "puvodni-lemma"
    assert rep.lemmas_changed == 0


def test_apply_lemmas_doc_failure_counts_and_continues():
    """Dokument, na kterém tokens_for vrátí None, se přeskočí (docs_failed),
    ostatní se zpracují."""
    t1, t2 = "selze", "Widgety jsou skvele"
    r1 = _rec(t1, [("selze", "selze", "WIDGET", 0, 5)])
    r2 = _rec(t2, [("Widgety", "Widgety", "WIDGET", 0, 7)])
    fake = FakeLemmatizer({t2: _TOKENS})   # t1 chybí → None
    rep = apply_lemmas_to_records([r1, r2], fake)
    assert rep.docs_failed == 1
    assert rep.docs_processed == 1
    assert r2["graph"]["entities"][0]["lemma"] == "widget"
    assert r1["graph"]["entities"][0]["lemma"] == "selze"  # nedotčeno


def test_apply_lemmas_noop_when_unavailable():
    rec = _rec("text", [("text", "puvodni", "WIDGET", 0, 4)])
    fake = FakeLemmatizer({}, available=False)
    rep = apply_lemmas_to_records([rec], fake)
    assert rep.docs_processed == 0
    assert rec["graph"]["entities"][0]["lemma"] == "puvodni"
    assert fake.calls == 0   # tokens_for se vůbec nevolá


def test_apply_lemmas_skips_entityless_docs():
    rec = _rec("text bez entit", [])
    fake = FakeLemmatizer({})
    rep = apply_lemmas_to_records([rec], fake)
    assert rep.docs_processed == 0
    assert fake.calls == 0   # žádné entity → dokument se nelemmatizuje


# ---------- StanzaLemmatizer: graceful bez modelu ----------------------------

def test_lemmatizer_unavailable_is_graceful():
    """Neexistující jazyk → init selže → available=False, žádná výjimka."""
    lem = StanzaLemmatizer(lang="neexistujici-jazyk-xx")
    assert lem.available is False
    assert lem.tokens_for("cokoliv") is None
    assert lem.lemma_of_text("cokoliv") is None


# ---------- Volitelná integrace s reálnou Stanzou (skip bez modelu) ----------

def test_real_stanza_czech_if_available():
    """Reálná Stanza s českým modelem — SKIP pokud není k dispozici
    (offline stroj / CI bez modelu). Ověřuje determinismus a smysluplnost."""
    try:
        import stanza  # noqa: F401
    except ImportError:
        print("  (skip: stanza není nainstalovaná)")
        return
    lem = StanzaLemmatizer(lang="cs")
    # Init může selhat na chybějícím modelu bez internetu → graceful skip
    if not lem.available:
        print("  (skip: český model není k dispozici)")
        return
    text = "Widgety jsou skvělé."
    t1 = lem.tokens_for(text)
    t2 = lem.tokens_for(text)
    assert t1 == t2                       # determinismus
    assert t1 is not None and len(t1) >= 3
    lemma = lemma_for_span(t1, 0, 7)      # "Widgety"
    assert lemma                          # něco vrátí
    assert lemma.lower() != "widgety"     # a je to lemmatizované (ne surface)


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
