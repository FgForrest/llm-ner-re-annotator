"""
Unit testy pro labeler.normalize.

DŮLEŽITÉ: Doménově NEUTRÁLNÍ — žádné reálné pojmy ze zdrojových dat.
Test fixtury jsou se zástupnými termíny (Alpha, Beta,
"category one", item / thing / stuff). Pipeline kód i tyto testy musí být
univerzální nad libovolnou doménou.

Pokrývá:
  - opravu offsetu (case-sensitive + case-insensitive fallback)
  - doplnění všech výskytů entity (auto-expand) jen pro safe entity
  - řešení překryvů (longest > priority > earlier)
  - přemapování labelů (16 LLM → konsolidovaný set)
  - URL přesměrování (SUBJEKT obsahující URL → URL)
  - zachování/zahození relací po normalizaci
  - report statistik (counters)
  - konflikt labelů (stejný text, různé labely)
  - safety pro krátké generické termíny (jazyková heuristika + taxonomy)
  - sentence-like entity filter (univerzální anti-halucinace)

Spuštění: python -m pytest tests/test_normalize.py
       nebo: python tests/test_normalize.py
"""

import sys
from pathlib import Path

# Aby šlo spustit i bez pip install (cd worktree && python tests/test_normalize.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.normalize import (
    DEFAULT_LABEL_MAPPING,
    DEFAULT_NER_LABELS,
    NormalizeOpts,
    NormalizationResult,
    effective_ner_labels,
    normalize_graph,
    _find_all_occurrences,
    _is_sentence_like,
    _is_unsafe_for_auto_expand,
)
from labeler.schemas import Entity, EntityCategory, Relation, SkoGraph


# Pomocná funkce pro stručné sestavení grafu v testech
def _make_graph(entities_spec, relations_spec=None):
    """entities_spec = list[(id, text, label, start, end)] (lemma = text)"""
    entities = [
        Entity(id=eid, text=txt, lemma=txt, category=EntityCategory(lb),
               start_char=s, end_char=e)
        for eid, txt, lb, s, e in entities_spec
    ]
    relations = []
    for r in (relations_spec or []):
        sub, obj, rel = r
        relations.append(Relation(
            subjectId=sub, objectId=obj, relation=rel,
            relationCs=rel, relationEn=rel,
        ))
    return SkoGraph(entities=entities, relations=relations)


# ---------- Test 1: oprava offsetu (case-sensitive find) -------------------

def test_fix_offsets_case_sensitive():
    text = "Apple iPhone 15 je nejnovější model. Apple iPhone je značka."
    # Entita s ŠPATNÝM offsetem
    g = _make_graph([("e1", "Apple iPhone", "PRODUKT", 100, 112)])
    opts = NormalizeOpts(enabled=True, strict_offsets=True,
                        auto_expand_mentions=False)
    res = normalize_graph(g, text, opts=opts)
    # Měl by najít první výskyt v textu (pozice 0..12) — ale je 2 výskytů,
    # takže vybere nejbližší k původnímu offsetu (100) = druhý výskyt (37..49)
    kept = res.graph.entities
    assert len(kept) == 1
    assert text[kept[0].start_char:kept[0].end_char] == "Apple iPhone"
    assert res.offsets_fixed == 1


def test_fix_offsets_case_insensitive_fallback():
    text = "Apple iPhone 15 je super telefon."
    g = _make_graph([("e1", "apple iphone", "PRODUKT", 0, 12)])
    opts = NormalizeOpts(enabled=True, strict_offsets=True)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 1
    e = res.graph.entities[0]
    # entity.text se přepíše na přesný substring originálu
    assert e.text == "Apple iPhone"
    assert text[e.start_char:e.end_char] == "Apple iPhone"


def test_fix_offsets_dropped_when_unrecoverable():
    text = "Apple iPhone."
    g = _make_graph([("e1", "Samsung Galaxy", "PRODUKT", 0, 14)])
    opts = NormalizeOpts(enabled=True, strict_offsets=True)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 0
    assert res.offsets_unrecoverable == 1


# ---------- Test 2: doplnění všech výskytů (auto-expand) -------------------

def test_auto_expand_safe_product():
    text = "Alpha je produkt. Alpha obsahuje to a to. Další Alpha detail."
    g = _make_graph([("e1", "Alpha", "PRODUKT", 0, 5)])
    opts = NormalizeOpts(enabled=True, auto_expand_mentions=True,
                        strict_offsets=True)
    res = normalize_graph(g, text, opts=opts)
    # 3 výskyty "Alpha" v textu → 3 entity
    assert len(res.graph.entities) == 3
    assert res.mentions_expanded == 2  # 1 byla původní, 2 doplněné
    # všechny musí mít text "Alpha"
    assert all(e.text == "Alpha" for e in res.graph.entities)


def test_auto_expand_skip_unsafe_short_generic():
    # Krátký lowercase termín → jazyková heuristika unsafe (délka < 5 / single word lowercase)
    text = "auto je vůz. Naše auto. Další auto. auto auto auto."
    g = _make_graph([("e1", "auto", "PRODUKT_TYP", 0, 4)])
    opts = NormalizeOpts(enabled=True, auto_expand_mentions=True)
    res = normalize_graph(g, text, opts=opts)
    # PRODUKT_TYP "auto" je krátké lowercase → unsafe → žádný expand
    assert len(res.graph.entities) == 1
    assert res.mentions_expanded == 0


def test_auto_expand_off_by_default():
    text = "Alpha a další Alpha a další Alpha."
    g = _make_graph([("e1", "Alpha", "PRODUKT", 0, 5)])
    opts = NormalizeOpts(enabled=True, auto_expand_mentions=False)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 1  # bez expandu


# ---------- Test 3: řešení překryvů ----------------------------------------

def test_overlap_resolution_longer_wins():
    text = "Apple iPhone 15 Pro Max je telefon."
    g = _make_graph([
        ("e1", "Apple iPhone", "PRODUKT", 0, 12),
        ("e2", "Apple iPhone 15 Pro Max", "PRODUKT", 0, 23),
    ])
    opts = NormalizeOpts(enabled=True, auto_expand_mentions=False)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 1
    assert res.graph.entities[0].text == "Apple iPhone 15 Pro Max"
    assert res.overlaps_dropped == 1


# ---------- Test 4: přemapování labelů -------------------------------------

def test_label_remapping():
    """Obecný label remapping (16 → 11): KATEGORIE→PRODUKT_TYP, KOMPONENTA→KEYPHRASE_*."""
    text = "Category One a procesor"
    g = _make_graph([
        ("e1", "Category One", "KATEGORIE", 0, 12),
        ("e2", "procesor", "KOMPONENTA", 15, 23),
    ])
    # Bez taxonomy — testujeme jen obecný remap (data-driven heuristiky off)
    opts = NormalizeOpts(enabled=True, taxonomy=None)
    res = normalize_graph(g, text, opts=opts)
    labels = {e.category for e in res.graph.entities}
    assert "PRODUKT_TYP" in labels      # KATEGORIE → PRODUKT_TYP
    assert "KEYPHRASE_KOMPONENTA" in labels  # KOMPONENTA → KEYPHRASE_*
    assert res.labels_remapped == 2
    assert res.taxonomy_rules_applied == 0  # bez taxonomy se heuristiky nepoužijí


def test_label_osoba_renames_to_osoba_role():
    text = "Klient je odpovědný."
    g = _make_graph([("e1", "Klient", "OSOBA", 0, 6)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert res.graph.entities[0].category == "OSOBA_ROLE"


def test_custom_taxonomy_mapping_does_not_leak_default_renames():
    """Regression: s custom taxonomií z fáze 0 se mapping odvozuje z NÍ —
    custom label PARAMETR (kind=ner) se NESMÍ potichu přejmenovat default
    tabulkou na KEYPHRASE_PARAMETR."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PARAMETR",
                     description="V této doméně plnohodnotná NER kategorie."),
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt zde."),
    ])
    # Entity kategorie musí být validní pattern — použijeme přímo SkoGraph
    from labeler.schemas import Entity as _E, SkoGraph as _G
    text = "Limit plneni je parametr."
    g = _G(entities=[_E(id="e1", text="Limit plneni", lemma="limit plnění",
                        category="PARAMETR", start_char=0, end_char=12)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True, taxonomy=tax))
    # Mapping z taxonomie = identita → label zůstává PARAMETR
    assert res.graph.entities[0].category == "PARAMETR"
    assert res.labels_remapped == 0


def test_custom_taxonomy_keyphrase_kind_gets_prefix():
    """Custom taxonomy label s kind='keyphrase' se přemapuje na KEYPHRASE_*
    — odvozeno z DAT taxonomie, ne z hard-coded tabulky."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="VYHODA", description="Přínos nabídky pro klienta.",
                     kind="keyphrase"),
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt zde."),
    ])
    from labeler.schemas import Entity as _E, SkoGraph as _G
    text = "Velka vyhoda pro vsechny."
    g = _G(entities=[_E(id="e1", text="Velka vyhoda", lemma="velká výhoda",
                        category="VYHODA", start_char=0, end_char=12)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True, taxonomy=tax))
    assert res.graph.entities[0].category == "KEYPHRASE_VYHODA"
    assert res.labels_remapped == 1


# ---------- Test 5: Drop strukturálních identifikátorů (URL/EMAIL/TELEFON) -

def test_url_entity_is_dropped_not_relabeled():
    """LLM extrahoval URL (často jako MISTO) → dropne se. NER se na URL netrénuje."""
    text = "Více na https://www.example.com/produkt najdete vše."
    url = "https://www.example.com/produkt"
    start = text.index(url)
    g = _make_graph([("e1", url, "MISTO", start, start + len(url))])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    # Entita s URL surface se ZAHODÍ (žádný relabeling na "URL")
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


def test_email_entity_is_dropped():
    text = "Napište na info@example.com pro více info."
    g = _make_graph([("e1", "info@example.com", "SUBJEKT", 11, 27)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


def test_phone_entity_is_dropped():
    text = "Volej +420 123 456 789."
    g = _make_graph([("e1", "+420 123 456 789", "HODNOTA", 6, 22)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


def test_keep_structural_identifiers_flag_preserves_url():
    """Pokud --keep-structural-identifiers, URL/email/telefon se NEDROPnou."""
    text = "Více na https://www.example.com najdete vše."
    url = "https://www.example.com"
    start = text.index(url)
    g = _make_graph([("e1", url, "MISTO", start, start + len(url))])
    res = normalize_graph(g, text, opts=NormalizeOpts(
        enabled=True, drop_structural_identifiers=False,
    ))
    # Entita zůstává (s původním labelem MISTO — pak by ji uživatel řešil sám)
    assert len(res.graph.entities) == 1
    assert res.structural_identifiers_dropped == 0


# ---------- Test 6: zachování/zahození relací ------------------------------

def test_relations_dropped_when_entity_removed():
    text = "iPhone je telefon."
    g = _make_graph(
        entities_spec=[
            ("e1", "iPhone", "PRODUKT", 0, 6),
            ("e2", "neexistuje", "PRODUKT", 100, 110),  # bude dropnuta
        ],
        relations_spec=[("e1", "e2", "is_a")],
    )
    opts = NormalizeOpts(enabled=True, strict_offsets=True)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 1
    assert len(res.graph.relations) == 0
    assert res.relations_dropped_missing_entity == 1


# ---------- Test 7: konflikt labelů ----------------------------------------

def test_label_conflict_resolved_first_occurrence_fallback():
    text = "iPhone je PRODUKT a iPhone je KATEGORIE."
    g = _make_graph([
        ("e1", "iPhone", "PRODUKT", 0, 6),
        ("e2", "iPhone", "KATEGORIE", 20, 26),
    ])
    opts = NormalizeOpts(enabled=True)
    res = normalize_graph(g, text, opts=opts)
    # Po remap: PRODUKT vs PRODUKT_TYP. Bez korpusových statistik a s remízou
    # v dokumentu (1:1) rozhodne první výskyt → PRODUKT (start 0 < 20).
    # ŽÁDNÁ hard-coded priorita labelů.
    labels = {e.category for e in res.graph.entities}
    assert labels == {"PRODUKT"}  # všechny entity sjednoceny
    assert res.label_conflicts_resolved == 1
    assert len(res.needs_review) >= 1


def test_label_conflict_resolved_by_document_majority():
    """Dokumentová většina přebije první výskyt (2× KATEGORIE vs 1× PRODUKT)."""
    text = "iPhone tady. iPhone tam. iPhone všude."
    g = _make_graph([
        ("e1", "iPhone", "PRODUKT", 0, 6),
        ("e2", "iPhone", "KATEGORIE", 13, 19),
        ("e3", "iPhone", "KATEGORIE", 25, 31),
    ])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    labels = {e.category for e in res.graph.entities}
    # Většina = KATEGORIE → po remapu PRODUKT_TYP
    assert labels == {"PRODUKT_TYP"}
    assert res.label_conflicts_resolved == 1


# ---------- Test 8: drop uncertain labels ----------------------------------

def test_drop_uncertain_labels():
    text = "iPhone je telefon a má parametr displej."
    g = _make_graph([
        ("e1", "iPhone", "PRODUKT", 0, 6),
        ("e2", "parametr displej", "PARAMETR", 22, 38),  # KEYPHRASE
    ])
    opts = NormalizeOpts(
        enabled=True,
        drop_uncertain_labels=True,
        allowed_ner_labels=DEFAULT_NER_LABELS,
    )
    res = normalize_graph(g, text, opts=opts)
    labels = {e.category for e in res.graph.entities}
    assert "PRODUKT" in labels
    assert "KEYPHRASE_PARAMETR" not in labels  # zahozeno


# ---------- Test 9: counter konzistence ------------------------------------

def test_report_counters():
    text = "Alpha je produkt. Alpha navíc. info@example.com"
    g = _make_graph([
        ("e1", "Alpha", "PRODUKT", 0, 5),
        ("e2", "info@example.com", "SUBJEKT", 31, 47),
        ("e3", "Category One", "KATEGORIE", 100, 112),  # offset wrong
    ])
    opts = NormalizeOpts(
        enabled=True, auto_expand_mentions=True, strict_offsets=True,
    )
    res = normalize_graph(g, text, opts=opts)
    d = res.to_dict()
    assert d["entities_before"] == 3
    # info@example.com — strukturální identifikátor → dropnut (ne přelabelován)
    assert d["structural_identifiers_dropped"] == 1
    assert d["mentions_expanded"] >= 1  # Alpha 2× v textu
    assert d["offsets_unrecoverable"] == 1
    # Distribuce labelů — žádný EMAIL (entita dropnuta)
    assert "PRODUKT" in d["label_distribution_after"]
    assert "EMAIL" not in d["label_distribution_after"]


# ---------- Test 10: disabled = no-op --------------------------------------

def test_disabled_is_noop():
    text = "iPhone je telefon."
    g = _make_graph([("e1", "iPhone", "PRODUKT", 0, 6)])
    opts = NormalizeOpts(enabled=False)
    res = normalize_graph(g, text, opts=opts)
    assert len(res.graph.entities) == 1
    assert res.graph.entities[0].category == "PRODUKT"  # bez remap
    assert res.offsets_fixed == 0


# ---------- Test 11: _find_all_occurrences ---------------------------------

def test_find_all_word_boundaries():
    text = "auto je vůz. automobil je taky vůz. další auto."
    # "auto" se najde 2× (auto + auto), ne uvnitř "automobil"
    occ = _find_all_occurrences(text, "auto", case_sensitive=True)
    assert len(occ) == 2


def test_is_unsafe_for_auto_expand():
    # Bez taxonomy — jen jazykové heuristiky (krátký text, lowercase generic)
    assert _is_unsafe_for_auto_expand("auto", "PRODUKT_TYP") is True   # délka < 5
    assert _is_unsafe_for_auto_expand("Alpha", "PRODUKT") is False     # safe label
    assert _is_unsafe_for_auto_expand("Alpha Corp Inc.", "ORGANIZACE") is False
    assert _is_unsafe_for_auto_expand("test", "PRODUKT_TYP") is True

    # S taxonomy — slovo v unsafe_texts dělá ho unsafe i pro safe label
    from labeler.taxonomy import Taxonomy, TaxonomyLabel
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní pojmenovaný produkt.",
                     unsafe_texts=["thing"], case_sensitive=False),
    ])
    assert _is_unsafe_for_auto_expand("thing", "PRODUKT", tax) is True
    # "Alpha" není v unsafe → zůstane safe pro safe label
    assert _is_unsafe_for_auto_expand("Alpha", "PRODUKT", tax) is False


# ---------- effective_ner_labels: fallback vs taxonomy override ------------

def test_effective_ner_labels_without_taxonomy_is_default():
    """Bez taxonomy → fallback na DEFAULT_NER_LABELS."""
    assert effective_ner_labels(None) == DEFAULT_NER_LABELS
    # Default obsahuje rozumné e-shop / pojišťovácké NER labely
    assert "PRODUKT" in DEFAULT_NER_LABELS
    assert "ORGANIZACE" in DEFAULT_NER_LABELS
    assert "MISTO" in DEFAULT_NER_LABELS


def test_effective_ner_labels_with_taxonomy_uses_its_codes():
    """S taxonomy → labely z taxonomy (Phase 0 overrides fallback)."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="INSURANCE_PRODUCT",
                      description="Konkrétní pojistný produkt."),
        TaxonomyLabel(code="MEDICAL_CONDITION",
                      description="Lékařský stav nebo diagnóza."),
        TaxonomyLabel(code="ORGANIZACE", description="Firma, instituce."),
    ])
    result = effective_ner_labels(tax)
    assert "INSURANCE_PRODUCT" in result
    assert "MEDICAL_CONDITION" in result
    assert "ORGANIZACE" in result
    # PRODUKT (z fallbacku) NENÍ v taxonomy → není v result
    assert "PRODUKT" not in result


def test_effective_ner_labels_excludes_keyphrase_labels_from_taxonomy():
    """KEYPHRASE_* z taxonomy se do hard NER setu nezahrnou."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT_TYP", description="Typologie produktů."),
        TaxonomyLabel(code="KEYPHRASE_BENEFIT", description="Marketing přínos."),
    ])
    result = effective_ner_labels(tax)
    assert "PRODUKT_TYP" in result
    assert "KEYPHRASE_BENEFIT" not in result  # KEYPHRASE_* zfiltrované


def test_effective_ner_labels_empty_taxonomy_falls_back():
    """Defensive: taxonomy bez labelů → fallback na DEFAULT_NER_LABELS."""
    class EmptyTax:
        labels: list = []
    assert effective_ner_labels(EmptyTax()) == DEFAULT_NER_LABELS


# ---------- Regression: URL regex preserves bare-domain brand names -------

def test_bare_domain_brand_kept_not_dropped_as_url():
    """
    Regression: 'AcmeShop.cz' / 'DemoZpravy.cz' / 'VzorObchod.cz' jsou platné
    ORGANIZACE / PRODUKT entity, NE URL. URL regex je nesmí dropnout.
    """
    text = "E-shop AcmeShop.cz nabízí produkty."
    g = _make_graph([("e1", "AcmeShop.cz", "ORGANIZACE", 7, 18)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 1
    assert res.graph.entities[0].category == "ORGANIZACE"
    assert res.graph.entities[0].text == "AcmeShop.cz"
    assert res.structural_identifiers_dropped == 0


def test_url_with_path_is_dropped():
    """URL s explicit cestou (demo.cz/blog) musí být dropnuta."""
    text = "Více informací na demo.cz/blog/akne najdete."
    g = _make_graph([("e1", "demo.cz/blog/akne", "MISTO", 18, 35)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


def test_https_url_is_dropped():
    """Plnotvarná URL (https://) je vždy strukturální identifikátor."""
    text = "Web https://example.com je tu."
    g = _make_graph([("e1", "https://example.com", "MISTO", 4, 23)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


# ---------- Regression: phone regex preserves large HODNOTA ----------------

def test_large_value_kept_not_dropped_as_phone():
    """
    Regression: '100 000 000 Kč' (cena, ne telefon). Phone regex nesmí
    dropnout entity s velkými číselnými hodnotami bez phone markerů.
    """
    text = "Celková škoda 100 000 000 Kč."
    g = _make_graph([("e1", "100 000 000", "HODNOTA", 14, 25)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 1
    assert res.graph.entities[0].category == "HODNOTA"
    assert res.structural_identifiers_dropped == 0


def test_phone_with_plus_prefix_is_dropped():
    """Telefon s explicit + country prefixem se dropne."""
    text = "Volej +420 123 456 789 v pracovní době."
    g = _make_graph([("e1", "+420 123 456 789", "HODNOTA", 6, 22)])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    assert len(res.graph.entities) == 0
    assert res.structural_identifiers_dropped == 1


# ---------- CLI default-ON cleaning (regression test) ---------------------

def test_cli_default_finalize_enables_full_cleaning():
    """
    Regression test: `python -m labeler.main finalize` (bez jakýchkoli
    flagů) MUSÍ defaultně mít zapnutý cleaning + report.

    Default chování:
      - cleaning běží automaticky (normalize, offsets, auto-expand, drop)
      - normalization_report.json se zapíše
      - VÝSTUP je jen final_labels.{jsonl,json} — žádné spaCy splits
        (uživatel je nechce, splits zůstávají jako optional Python API)
      - OWL jen ve smoke testu, ne v produkci
    """
    from labeler.main import build_parser, _build_normalize_opts, _opt_enabled

    parser = build_parser()
    args = parser.parse_args(["finalize"])  # bez jakýchkoli flagů
    opts = _build_normalize_opts(args)

    # Core cleaning kroky musí být defaultně ZAPNUTÉ
    assert opts.enabled is True, "normalize musí být default ON"
    assert opts.strict_offsets is True, "strict_offsets musí být default ON"
    assert opts.auto_expand_mentions is True, "auto-expand musí být default ON"
    assert opts.drop_structural_identifiers is True, \
        "structural identifiers drop musí být default ON"

    # write_report default ON
    assert _opt_enabled(args, "no_write_report") is True

    # spaCy split exporty: NEPATŘÍ MEZI DEFAULTNÍ CHOVÁNÍ.
    # CLI je nemá ani jako opt-in flag.
    assert not hasattr(args, "no_export_spacy"), \
        "--no-export-spacy už neexistuje (spaCy splits zrušené)"

    # Advanced agresivní volby zůstávají opt-in (default OFF)
    assert opts.drop_uncertain_labels is False, \
        "drop_uncertain_labels je agresivní, musí zůstat opt-in"


def test_cli_no_normalize_disables_cleaning():
    """--no-normalize flag vypne cleaning (debugging escape hatch)."""
    from labeler.main import build_parser, _build_normalize_opts

    parser = build_parser()
    args = parser.parse_args([
        "finalize", "--no-normalize", "--no-auto-expand-mentions",
        "--no-strict-offsets",
    ])
    opts = _build_normalize_opts(args)
    assert opts.enabled is False
    assert opts.auto_expand_mentions is False
    assert opts.strict_offsets is False


def test_cli_deprecated_quality_preset_is_noop():
    """--quality-preset spacy je teď no-op (default chování)."""
    from labeler.main import build_parser, _build_normalize_opts

    parser = build_parser()
    # Bez preset
    args1 = parser.parse_args(["finalize"])
    opts1 = _build_normalize_opts(args1)
    # S preset
    args2 = parser.parse_args(["finalize", "--quality-preset", "spacy"])
    opts2 = _build_normalize_opts(args2)
    # Stejné defaulty
    assert opts1.enabled == opts2.enabled
    assert opts1.strict_offsets == opts2.strict_offsets
    assert opts1.auto_expand_mentions == opts2.auto_expand_mentions


# ---------- Test 12: sentence-like entity filter (univerzální) -------------

def test_is_sentence_like_terminator():
    """Entity končící ? nebo ! je sentence-like (LLM halucinace celé věty)."""
    assert _is_sentence_like("Kolik vám vyplatíme?") is True
    assert _is_sentence_like("Pomůžeme vám!") is True
    # Tečka na konci sama o sobě nestačí (např. "Apple Inc.")
    assert _is_sentence_like("Apple Inc.") is False


def test_is_sentence_like_inner_boundary():
    """Tečka uprostřed + mezera + velké písmeno = sentence boundary."""
    assert _is_sentence_like("První věta. Druhá věta") is True
    # Datum "1. 1. 2026" — tečky následují číslice, ne velké písmeno → ne sentence
    assert _is_sentence_like("1. 1. 2026") is False
    # Inicály "J. K. Rowling" — tečky následují velké písmeno, ale ne mezera+upper
    # ve formě sentence boundary
    assert _is_sentence_like("J. K. Rowling") is False


def test_is_sentence_like_too_many_words():
    """Příliš mnoho slov (> 15) = pravděpodobně věta, ne entita."""
    short_keyphrase = "škoda způsobená vytopením sousedního bytu"  # 5 slov
    assert _is_sentence_like(short_keyphrase) is False
    long_sentence = " ".join(["slovo"] * 20)
    assert _is_sentence_like(long_sentence) is True


def test_is_sentence_like_normal_entities_pass():
    """Normální entity nesmí být označeny jako sentence-like."""
    for txt in [
        "iPhone",
        "Apple iPhone 15 Pro Max",
        "ACME Basic Authentic",
        "https://example.com/path",
        "info@example.com",
        "+420 123 456 789",
        "100 000 Kč",
        "Ag 925/1000",
        "1. 1. 2026",
        "Praha",
        "rodiny s dětmi",
        "category one",
    ]:
        assert _is_sentence_like(txt) is False, f"Falsely flagged: {txt!r}"


def test_drop_sentence_like_entities_in_pipeline():
    """Entity vypadající jako věty (otázky atd.) se zahodí během normalizace."""
    text = "Kolik vám vyplatíme? Plnění je 100 000 Kč."
    g = _make_graph([
        ("e1", "Kolik vám vyplatíme?", "KEYPHRASE_PARAMETR", 0, 20),
        ("e2", "100 000 Kč", "HODNOTA", 31, 41),
    ])
    res = normalize_graph(g, text, opts=NormalizeOpts(enabled=True))
    # Sentence-like entita musí zmizet
    surface_forms = {e.text for e in res.graph.entities}
    assert "Kolik vám vyplatíme?" not in surface_forms
    assert "100 000 Kč" in surface_forms
    assert res.entities_dropped_sentence_like == 1


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
