"""
Unit testy pro labeler.taxonomy (fáze 0).

Pokrývá:
  - TaxonomyLabel.validate (UPPERCASE_SNAKE pattern, min description)
  - Taxonomy.from_dict / to_dict round-trip
  - load_user_taxonomy: validní + nevalidní soubor
  - example soubor v examples/ je validní
  - _deduplicate_aliases: duplicitní kódy + aliases sloučení
  - _collect_samples: deterministické (stejný seed → stejný výběr)
  - _parse_llm_taxonomy_response: dobrý JSON + špatný JSON
  - prompty obsahují vlastní taxonomy (build_extractor_messages)

Spuštění: python tests/test_taxonomy.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.taxonomy import (
    ConfusablePair,
    ConfusionReport,
    Taxonomy,
    TaxonomyLabel,
    TaxonomyMode,
    TaxonomyQualityError,
    _aggregate_confusion,
    _apply_merge,
    _apply_sharpen_result,
    _assess_taxonomy_quality,
    _collect_samples,
    _deduplicate_aliases,
    _ensure_mandatory_labels,
    _extract_json_list,
    _heldout_snippets,
    _MANDATORY_LABELS,
    _MIN_LLM_LABELS,
    _parse_llm_taxonomy_response,
    _select_confusable_pairs,
    audit_and_refine_taxonomy,
    load_taxonomy_if_exists,
    load_user_taxonomy,
    production_taxonomy_path,
    taxonomy_path,
)
from labeler.datasets import LabelItem
from labeler.prompts import build_extractor_messages, _taxonomy_block


# ---------- TaxonomyLabel.validate -----------------------------------------

def test_label_valid():
    lb = TaxonomyLabel(
        code="PRODUKT",
        description="Konkrétní pojmenovaný produkt nebo služba.",
        examples=["iPhone", "Pojištění auta OMEGA"],
        domains=["e-shop"],
    )
    assert lb.validate() == []


def test_label_invalid_code_pattern():
    lb = TaxonomyLabel(code="produkt", description="Plný popis.")
    errs = lb.validate()
    assert any("UPPERCASE_SNAKE" in e for e in errs)


def test_label_invalid_short_description():
    lb = TaxonomyLabel(code="PRODUKT", description="krátké")
    errs = lb.validate()
    assert any("description" in e for e in errs)


# ---------- Taxonomy round-trip --------------------------------------------

def test_taxonomy_roundtrip():
    tax = Taxonomy(
        labels=[
            TaxonomyLabel(code="PRODUKT", description="Produkt nebo služba.",
                         examples=["iPhone"], domains=["e-shop"]),
            TaxonomyLabel(code="URL", description="URL adresa nebo doména.",
                         examples=["https://x.com"]),
        ],
        mode="user",
        notes="test",
    )
    d = tax.to_dict()
    tax2 = Taxonomy.from_dict(d)
    assert tax2.codes == ["PRODUKT", "URL"]
    assert tax2.mode == "user"
    assert tax2.labels[0].examples == ["iPhone"]


# ---------- Validate: duplicitní kódy --------------------------------------

def test_taxonomy_duplicate_codes():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Produkt nebo služba."),
        TaxonomyLabel(code="PRODUKT", description="Konflikt — jiný popis."),
    ])
    errs = tax.validate()
    assert any("duplicit" in e for e in errs)


def test_taxonomy_too_few_labels():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Produkt nebo služba."),
    ])
    errs = tax.validate()
    assert any("alespoň 2 labely" in e for e in errs)


# ---------- load_user_taxonomy ---------------------------------------------

def test_load_user_taxonomy_valid():
    data = {
        "version": 1,
        "labels": [
            {"code": "PRODUKT", "description": "Produkt nebo služba.",
             "examples": ["iPhone"], "domains": ["e-shop"]},
            {"code": "URL", "description": "URL adresa nebo doména.",
             "examples": ["https://x.com"]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        p = Path(f.name)
    try:
        tax = load_user_taxonomy(p)
        assert tax.mode == "user"
        assert tax.codes == ["PRODUKT", "URL"]
    finally:
        p.unlink()


def test_load_user_taxonomy_invalid_raises():
    data = {"labels": [{"code": "produkt", "description": "x"}]}  # lowercase
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        p = Path(f.name)
    try:
        try:
            load_user_taxonomy(p)
            assert False, "měla vyhodit ValueError"
        except ValueError as e:
            assert "problémy" in str(e)
    finally:
        p.unlink()


def test_example_file_is_valid():
    """examples/entity_taxonomy.example.json musí být validní user taxonomy."""
    example = Path(__file__).parent.parent / "examples" / "entity_taxonomy.example.json"
    assert example.exists(), f"Example chybí: {example}"
    tax = load_user_taxonomy(example)
    assert len(tax.labels) >= 8  # rozumný počet labelů
    assert "PRODUKT" in tax.codes
    assert "ORGANIZACE" in tax.codes
    # URL / EMAIL / TELEFON do taxonomie nepatří — regex-detekovatelné
    # struktury, ne pojmenované entity (normalize je dropne).
    assert "URL" not in tax.codes
    assert "EMAIL" not in tax.codes
    assert "TELEFON" not in tax.codes


# ---------- _deduplicate_aliases -------------------------------------------

def test_deduplicate_keeps_first_keeps_aliases():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Produkt nebo služba.",
                     aliases=["ITEM"], examples=["iPhone"]),
        TaxonomyLabel(code="ITEM", description="Položka v katalogu — duplikát.",
                     examples=["MacBook"]),
    ])
    out = _deduplicate_aliases(tax)
    # ITEM by se měl sloučit do PRODUKT (aliases match)
    assert len(out.labels) == 1
    assert out.labels[0].code == "PRODUKT"
    assert "MacBook" in out.labels[0].examples  # examples zachovány
    assert "ITEM" in out.labels[0].aliases


def test_deduplicate_duplicate_codes():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="První definice produktu.",
                     examples=["iPhone"]),
        TaxonomyLabel(code="PRODUKT", description="Druhá definice produktu.",
                     examples=["MacBook"], domains=["e-shop"]),
    ])
    out = _deduplicate_aliases(tax)
    assert len(out.labels) == 1
    assert "iPhone" in out.labels[0].examples
    assert "MacBook" in out.labels[0].examples
    assert "e-shop" in out.labels[0].domains


# ---------- _collect_samples (deterministické) ------------------------------

def test_collect_samples_deterministic():
    items = [
        LabelItem(item_id=f"x{i}", source="src", domain="d1", text=f"Text {i}")
        for i in range(20)
    ]
    items += [
        LabelItem(item_id=f"y{i}", source="src", domain="d2", text=f"Other {i}")
        for i in range(15)
    ]
    s1 = _collect_samples(items, samples_per_dataset=5, seed=42)
    s2 = _collect_samples(items, samples_per_dataset=5, seed=42)
    assert s1 == s2  # stejný seed → identický výsledek
    assert set(s1.keys()) == {"d1", "d2"}
    # Rozprostření: 3 kotvy (první/prostřední/poslední) + N náhodných = N+3.
    # d1 má 20 položek (> 5+3) → 3 + 5 = 8; d2 má 15 (> 5+3) → 3 + 5 = 8.
    assert len(s1["d1"]) == 8
    assert len(s1["d2"]) == 8


def test_collect_samples_spread_includes_anchors():
    """Výběr MUSÍ obsahovat první, prostřední a poslední položku datasetu
    (kotvy), aby byl rozprostřený a ne jen 'prvních N'."""
    items = [
        LabelItem(item_id=f"x{i}", source="src", domain="d1", text=f"Text {i}")
        for i in range(20)
    ]
    s = _collect_samples(items, samples_per_dataset=5, seed=42)
    chosen = s["d1"]
    assert "Text 0" in chosen          # první
    assert "Text 10" in chosen         # prostřední (n//2)
    assert "Text 19" in chosen         # poslední
    # Pořadí zachované podle indexu (kotvy nejsou nutně na kraji výběru, ale
    # výběr je vzestupně dle indexu): první prvek je "Text 0", poslední "Text 19".
    assert chosen[0] == "Text 0"
    assert chosen[-1] == "Text 19"


def test_collect_samples_small_dataset_takes_all():
    """Malý dataset (<= samples_per_dataset + 3) se vezme celý, beze ztráty."""
    items = [
        LabelItem(item_id=f"x{i}", source="src", domain="d1", text=f"Text {i}")
        for i in range(7)  # 7 <= 5 + 3 → vše
    ]
    s = _collect_samples(items, samples_per_dataset=5, seed=42)
    assert len(s["d1"]) == 7
    assert s["d1"] == [f"Text {i}" for i in range(7)]


# ---------- _parse_llm_taxonomy_response ------------------------------------

def test_parse_llm_response_valid():
    parsed = {
        "labels": [
            {"code": "PRODUKT", "description": "Produkt nebo služba.",
             "examples": ["iPhone"], "domains": ["e-shop"]},
            {"code": "ORGANIZACE", "description": "Firma nebo instituce.",
             "examples": ["Apple"], "domains": ["e-shop"]},
        ],
    }
    tax = _parse_llm_taxonomy_response("", parsed)
    assert tax.mode == "auto"
    assert tax.codes == ["PRODUKT", "ORGANIZACE"]


def test_parse_llm_response_missing_labels():
    parsed = {"version": 1}
    try:
        _parse_llm_taxonomy_response("", parsed)
        assert False, "měla vyhodit ValueError"
    except ValueError as e:
        assert "labels" in str(e)


def test_parse_llm_response_invalid_json_fallback():
    try:
        _parse_llm_taxonomy_response("not a json {{", None)
        assert False, "měla vyhodit ValueError"
    except ValueError:
        pass  # OK


# ---------- T2: _ensure_mandatory_labels -----------------------------------

def test_ensure_mandatory_labels_adds_missing():
    """Pokud LLM nedal povinné labely, SW je doplní sám (a všechny projdou
    validací — popisy mají >= 10 znaků, kódy jsou UPPERCASE_SNAKE)."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="OSOBA", description="Jméno fyzické osoby."),
    ])
    added = _ensure_mandatory_labels(tax)
    assert set(added) == set(_MANDATORY_LABELS.keys())
    codes = tax.codes
    for code in _MANDATORY_LABELS:
        assert code in codes
    # Doplněné labely jsou validní
    assert tax.validate() == []


def test_ensure_mandatory_labels_case_insensitive_and_aliases():
    """Přítomný povinný label (i přes alias / jiný case) se NEpřidá duplicitně."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt nebo služba."),
        # ORGANIZACE schované jako alias jiného labelu
        TaxonomyLabel(code="SUBJEKT", description="Obecný subjekt nebo entita.",
                     aliases=["organizace"]),
        TaxonomyLabel(code="DATUM", description="Datum nebo časový údaj kdykoliv."),
        TaxonomyLabel(code="MISTO", description="Lokalita, adresa nebo místo."),
        TaxonomyLabel(code="HODNOTA", description="Číselná částka nebo množství."),
    ])
    added = _ensure_mandatory_labels(tax)
    # PRODUKT už je (case match), ORGANIZACE je jako alias → nepřidávat;
    # DATUM/MISTO/HODNOTA už jsou → nic nechybí.
    assert added == []
    # Žádné duplicity
    assert tax.validate() == []


def test_ensure_mandatory_labels_none_missing():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code=c, description=d) for c, d in _MANDATORY_LABELS.items()
    ])
    assert _ensure_mandatory_labels(tax) == []


# ---------- T1: _assess_taxonomy_quality ------------------------------------

def test_assess_quality_ok_when_enough_labels():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code=f"LABEL_{i}", description=f"Popis labelu číslo {i}.")
        for i in range(_MIN_LLM_LABELS)
    ])
    assert _assess_taxonomy_quality(tax, n_llm_labels=_MIN_LLM_LABELS) == []


def test_assess_quality_complains_too_few_llm_labels():
    """Málo labelů od LLM → výtka (i když SW doplnil povinné na validní set)."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code=c, description=d) for c, d in _MANDATORY_LABELS.items()
    ])
    complaints = _assess_taxonomy_quality(tax, n_llm_labels=1)
    assert complaints  # neprázdné
    assert any("málo" in c.lower() or str(_MIN_LLM_LABELS) in c for c in complaints)


def test_assess_quality_complains_invalid_labels():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="dobry_kod_ale_lowercase", description="Dost dlouhý popis."),
    ] + [
        TaxonomyLabel(code=f"LABEL_{i}", description=f"Popis labelu číslo {i}.")
        for i in range(_MIN_LLM_LABELS)
    ])
    complaints = _assess_taxonomy_quality(tax, n_llm_labels=_MIN_LLM_LABELS + 1)
    assert any("UPPERCASE_SNAKE" in c or "validní" in c.lower() for c in complaints)


def test_taxonomy_quality_error_is_exception():
    assert issubclass(TaxonomyQualityError, Exception)


# ---------- T1 integrace: discover_taxonomy_auto retry/fallback ------------

class _FakeResp:
    def __init__(self, parsed):
        self.parsed = parsed
        self.content = json.dumps(parsed)


def _good_labels_payload():
    return {"labels": [
        {"code": f"LABEL_{i}", "description": f"Popis labelu číslo {i}."}
        for i in range(_MIN_LLM_LABELS)
    ]}


def _bad_labels_payload():
    return {"labels": [{"code": "JEDINY", "description": "Jen jeden label, to je málo."}]}


def _run_discover_with_fake(responses):
    """Spustí discover_taxonomy_auto s fake klientem, který vrací `responses`
    v pořadí. Monkeypatchuje scan_data_dir (žádný filesystem). Vrací
    (výsledek_nebo_None, počet_volání, raised_exc_nebo_None)."""
    import asyncio
    import labeler.taxonomy as tax_mod

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.model = "fake-model"

        async def chat(self, messages, **kw):
            resp = responses[min(self.calls, len(responses) - 1)]
            self.calls += 1
            return _FakeResp(resp)

    orig_scan = tax_mod.scan_data_dir
    tax_mod.scan_data_dir = lambda _dir: [
        LabelItem(item_id="a", source="s", domain="d", text="Nějaký text."),
    ]
    client = FakeClient()
    try:
        result = asyncio.run(tax_mod.discover_taxonomy_auto(extractor_client=client))
        return result, client.calls, None
    except Exception as e:  # noqa: BLE001 — test chce zachytit typ
        return None, client.calls, e
    finally:
        tax_mod.scan_data_dir = orig_scan


def test_discover_taxonomy_auto_retries_then_succeeds():
    """1. pokus málo labelů → reklamace → 2. pokus OK; povinné labely doplněny."""
    tax, calls, exc = _run_discover_with_fake([
        _bad_labels_payload(), _good_labels_payload(),
    ])
    assert exc is None
    assert calls == 2  # reklamace + retry
    assert tax is not None
    assert "LABEL_0" in tax.codes
    for code in _MANDATORY_LABELS:
        assert code in tax.codes  # T2 doplnění
    assert tax.validate() == []


def test_discover_taxonomy_auto_fails_after_two_attempts():
    """Nekvalitní i po 2. pokusu → TaxonomyQualityError (caller spadne na default)."""
    tax, calls, exc = _run_discover_with_fake([
        _bad_labels_payload(), _bad_labels_payload(),
    ])
    assert tax is None
    assert calls == 2
    assert isinstance(exc, TaxonomyQualityError)


def test_discover_taxonomy_auto_succeeds_first_try():
    """Kvalitní hned napoprvé → 1 volání, žádná reklamace."""
    tax, calls, exc = _run_discover_with_fake([_good_labels_payload()])
    assert exc is None
    assert calls == 1
    assert tax is not None
    for code in _MANDATORY_LABELS:
        assert code in tax.codes


# ---------- _taxonomy_block (prompt) ---------------------------------------

def test_taxonomy_block_format():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt nebo služba.",
                     examples=["iPhone"], domains=["e-shop"]),
    ])
    block = _taxonomy_block(tax)
    assert "Povolené labely" in block
    assert "PRODUKT" in block
    assert "Konkrétní produkt" in block
    assert "iPhone" in block
    assert "e-shop" in block


# ---------- build_extractor_messages s vlastní taxonomií -------------------

def test_build_extractor_messages_with_taxonomy():
    """Vlastní taxonomie nahradí kategorie block."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt nebo služba.",
                     examples=["iPhone"]),
    ])
    msgs = build_extractor_messages(
        text="Test text",
        domain="e-shop",
        taxonomy=tax,
    )
    user = msgs[1]["content"]
    # Nový taxonomy block je tam
    assert "Povolené labely" in user
    assert "PRODUKT" in user
    assert "Konkrétní produkt nebo služba." in user  # description z taxonomy


def test_build_extractor_messages_without_taxonomy_uses_default():
    """Bez taxonomy → default 16 kategorií (čárkou oddělené)."""
    msgs = build_extractor_messages(text="Test", domain="e-shop", taxonomy=None)
    user = msgs[1]["content"]
    # Default csv list obsahuje POUZITI, BENEFIT, KATEGORIE
    assert "PRODUKT, KATEGORIE" in user  # default block je čárkou oddělený
    assert "POUZITI" in user
    assert "Povolené labely" not in user  # to je jen v taxonomy block


def test_custom_taxonomy_drops_default_examples_from_prompt():
    """
    Regression: pokud user dodá CUSTOM taxonomii (např. INSURANCE_RISK,
    MEDICAL_CONDITION), prompt NESMÍ obsahovat hardcoded EXAMPLE_OUTPUT_TECH
    a EXAMPLE_OUTPUT_REALTY se starými 16 kategoriemi (PRODUKT/KATEGORIE/atd.).
    Konfliktní signály pro LLM → degradovaná kvalita.

    Slim varianta promptu (EXTRACTOR_USER_TEMPLATE_CUSTOM_TAXONOMY)
    obsahuje jen bad_example pro strukturu, žádné default examples.
    """
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="INSURANCE_RISK",
                     description="Pojistné riziko nebo událost."),
        TaxonomyLabel(code="MEDICAL_CONDITION",
                     description="Lékařský stav nebo diagnóza."),
    ])
    msgs = build_extractor_messages(
        text="Test text", domain="medical", taxonomy=tax,
    )
    user = msgs[1]["content"]
    # Taxonomy labely tam jsou
    assert "INSURANCE_RISK" in user
    assert "MEDICAL_CONDITION" in user
    # Default examples (TECH = ProBook 450 G10; REALTY = Byt 3+kk Žižkov)
    # NESMÍ být v promptu — konfliktovaly by s "use only these labels"
    assert "ProBook 450 G10" not in user, (
        "Default EXAMPLE_OUTPUT_TECH leaked do custom taxonomy promptu"
    )
    assert "Byt 3+kk Praha Žižkov" not in user, (
        "Default EXAMPLE_OUTPUT_REALTY leaked do custom taxonomy promptu"
    )
    # KRITICKÉ PRAVIDLO je explicitní
    assert "VÝHRADNĚ" in user or "Pouze" in user.lower() or "POUZE" in user


def test_default_taxonomy_keeps_examples():
    """Bez custom taxonomy: default 16 kategorií + EXAMPLES jsou v promptu."""
    msgs = build_extractor_messages(text="Test", domain="e-shop", taxonomy=None)
    user = msgs[1]["content"]
    # Default examples MUSÍ být přítomné — slouží jako few-shot training
    assert "ProBook 450 G10" in user
    assert "Byt 3+kk Praha Žižkov" in user


def test_extractor_prompts_do_not_ask_for_lemma():
    """Regression: lemma se od LLM NEPŘEBÍRÁ (počítá ho Stanza ve finalize) —
    examples ani JSON payloady v promptech nesmí obsahovat klíč "lemma".
    Slovo 'lemma' v instrukci ('lemma NEVRACEJ') je v pořádku."""
    from labeler.prompts import (
        build_extractor_ner_messages,
        build_extractor_re_messages,
        build_critic_ner_messages,
    )
    # JSON klíč má tvar '"lemma":' — instrukční zmínka ('Pole "lemma" NEVRACEJ')
    # dvojtečku nemá, takže ji tenhle assert nechytá.
    # Combined extractor: examples bez lemma klíče
    msgs = build_extractor_messages(text="Test", domain="e-shop", taxonomy=None)
    assert '"lemma":' not in msgs[0]["content"] + msgs[1]["content"]
    # Split NER prompt
    msgs = build_extractor_ner_messages(text="Test", domain="e-shop")
    assert '"lemma":' not in msgs[0]["content"] + msgs[1]["content"]
    # Entity payload do RE / critic promptů se stripuje
    ents = [{"id": "e1", "text": "Praha", "lemma": "praha", "category": "MISTO"}]
    msgs = build_extractor_re_messages(text="Test", domain="e-shop", entities=ents)
    assert '"lemma":' not in msgs[1]["content"]
    msgs = build_critic_ner_messages(text="Test", entities=ents)
    assert '"lemma":' not in msgs[1]["content"]


# ---------- Regression: URL/EMAIL/TELEFON NEvznikají v auto taxonomy -------

def test_parse_llm_response_drops_forbidden_url_label():
    """
    Regression: pokud LLM přesto navrhne URL/EMAIL/TELEFON jako label
    (ignoroval instrukci v promptu), parser je dropne. Tyto jsou
    regex-detekovatelné struktury, ne NER kategorie.
    """
    llm_response = {
        "labels": [
            {"code": "PRODUKT", "description": "Konkrétní produkt nebo služba.",
             "examples": ["iPhone"]},
            {"code": "URL", "description": "URL adresa nebo doména.",
             "examples": ["https://x.com"]},
            {"code": "EMAIL", "description": "E-mailová adresa.",
             "examples": ["test@example.com"]},
            {"code": "TELEFON", "description": "Telefonní číslo.",
             "examples": ["+420 123 456 789"]},
            {"code": "ORGANIZACE", "description": "Firma nebo instituce.",
             "examples": ["Apple"]},
        ]
    }
    tax = _parse_llm_taxonomy_response("", llm_response)
    codes = tax.codes
    assert "PRODUKT" in codes
    assert "ORGANIZACE" in codes
    # Forbidden labels musí být dropnuté
    assert "URL" not in codes
    assert "EMAIL" not in codes
    assert "TELEFON" not in codes


def test_parse_llm_response_drops_forbidden_aliases():
    """Aliasy regex-detekovatelných struktur (WEB, MAIL, PHONE) taky dropnuty."""
    llm_response = {
        "labels": [
            {"code": "PRODUKT", "description": "Konkrétní produkt."},
            {"code": "WEB", "description": "Webová stránka."},
            {"code": "MAIL", "description": "E-mailová zpráva."},
            {"code": "PHONE", "description": "Telefonní číslo."},
        ]
    }
    tax = _parse_llm_taxonomy_response("", llm_response)
    codes = tax.codes
    assert "PRODUKT" in codes
    assert "WEB" not in codes
    assert "MAIL" not in codes
    assert "PHONE" not in codes


# ---------- Regression: smoke test reads production taxonomy ---------------

def test_production_taxonomy_path_from_env_var():
    """
    production_taxonomy_path() vrátí path z CHECKPOINT_DIR env varu,
    i když je CONFIG.checkpoint_dir override-nutý na smoke subfolder.
    """
    import os
    old = os.environ.get("CHECKPOINT_DIR")
    try:
        os.environ["CHECKPOINT_DIR"] = "/app/checkpoints"
        prod_path = production_taxonomy_path()
        assert prod_path is not None
        assert str(prod_path) == "/app/checkpoints/entity_taxonomy.json" or \
               str(prod_path) == "\\app\\checkpoints\\entity_taxonomy.json"
    finally:
        if old is None:
            os.environ.pop("CHECKPOINT_DIR", None)
        else:
            os.environ["CHECKPOINT_DIR"] = old


def test_production_taxonomy_path_from_smoke_subfolder():
    """
    production_taxonomy_path() vrátí parent, pokud CONFIG.checkpoint_dir
    končí na 'smoke' (auto-detect smoke override).
    """
    import os
    from labeler.config import CONFIG
    old_dir = CONFIG.checkpoint_dir
    old_env = os.environ.get("CHECKPOINT_DIR")
    try:
        os.environ.pop("CHECKPOINT_DIR", None)
        object.__setattr__(CONFIG, "checkpoint_dir", Path("/tmp/checkpoints/smoke"))
        prod_path = production_taxonomy_path()
        assert prod_path is not None
        assert prod_path.name == "entity_taxonomy.json"
        assert prod_path.parent.name == "checkpoints"
    finally:
        object.__setattr__(CONFIG, "checkpoint_dir", old_dir)
        if old_env:
            os.environ["CHECKPOINT_DIR"] = old_env


def test_load_taxonomy_smoke_falls_back_to_production():
    """
    Regression: smoke test override-uje CONFIG.checkpoint_dir na
    checkpoints/smoke/. Pokud tam taxonomy není, ale ve produkčním
    checkpoints/ je → smoke MUSÍ načíst produkční taxonomy (jinak
    používá default 16 cat → biased time estimate + nesprávný OWL).
    """
    import os
    from labeler.config import CONFIG
    old_dir = CONFIG.checkpoint_dir
    old_env = os.environ.get("CHECKPOINT_DIR")

    with tempfile.TemporaryDirectory() as tmp:
        prod_dir = Path(tmp) / "checkpoints"
        smoke_dir = prod_dir / "smoke"
        prod_dir.mkdir(parents=True)
        smoke_dir.mkdir(parents=True)

        # Vytvoř produkční taxonomy (BEZ smoke verze)
        prod_taxonomy = {
            "version": 1,
            "mode": "user",
            "labels": [
                {"code": "PRODUKT", "description": "Konkrétní produkt nebo služba."},
                {"code": "ORGANIZACE", "description": "Firma, instituce, značka."},
            ],
        }
        (prod_dir / "entity_taxonomy.json").write_text(
            json.dumps(prod_taxonomy), encoding="utf-8"
        )

        try:
            # Simuluj smoke test: CONFIG.checkpoint_dir = smoke_dir
            os.environ["CHECKPOINT_DIR"] = str(prod_dir)
            object.__setattr__(CONFIG, "checkpoint_dir", smoke_dir)

            # load_taxonomy_if_exists by mělo najít produkční fallback
            tax = load_taxonomy_if_exists()
            assert tax is not None, (
                "Smoke test neměl najít produkční taxonomy přes fallback"
            )
            assert "PRODUKT" in tax.codes
            assert "ORGANIZACE" in tax.codes
        finally:
            object.__setattr__(CONFIG, "checkpoint_dir", old_dir)
            if old_env is None:
                os.environ.pop("CHECKPOINT_DIR", None)
            else:
                os.environ["CHECKPOINT_DIR"] = old_env


def test_load_taxonomy_smoke_prefers_smoke_local_if_exists():
    """
    Pokud smoke checkpoint dir má vlastní taxonomy, použije ji
    (předchází produkční fallback).
    """
    import os
    from labeler.config import CONFIG
    old_dir = CONFIG.checkpoint_dir
    old_env = os.environ.get("CHECKPOINT_DIR")

    with tempfile.TemporaryDirectory() as tmp:
        prod_dir = Path(tmp) / "checkpoints"
        smoke_dir = prod_dir / "smoke"
        prod_dir.mkdir(parents=True)
        smoke_dir.mkdir(parents=True)

        # Produkční má jeden label, smoke má jiný — chceme smoke
        (prod_dir / "entity_taxonomy.json").write_text(
            json.dumps({"version": 1, "labels": [
                {"code": "PRODUKT_PROD", "description": "Produkční label X."},
                {"code": "ORGANIZACE_PROD", "description": "Produkční label Y."},
            ]}), encoding="utf-8"
        )
        (smoke_dir / "entity_taxonomy.json").write_text(
            json.dumps({"version": 1, "labels": [
                {"code": "PRODUKT_SMOKE", "description": "Smoke testovací label."},
                {"code": "ORGANIZACE_SMOKE", "description": "Smoke testovací label 2."},
            ]}), encoding="utf-8"
        )

        try:
            os.environ["CHECKPOINT_DIR"] = str(prod_dir)
            object.__setattr__(CONFIG, "checkpoint_dir", smoke_dir)
            tax = load_taxonomy_if_exists()
            assert tax is not None
            assert "PRODUKT_SMOKE" in tax.codes
            assert "PRODUKT_PROD" not in tax.codes  # smoke má přednost
        finally:
            object.__setattr__(CONFIG, "checkpoint_dir", old_dir)
            if old_env is None:
                os.environ.pop("CHECKPOINT_DIR", None)
            else:
                os.environ["CHECKPOINT_DIR"] = old_env


# ===========================================================================
# Pilíř A / krok 2 — disjointness audit
# ===========================================================================

# ---------- Nová pole TaxonomyLabel (decision_rule / counter_examples) ------

def test_label_new_fields_roundtrip():
    """decision_rule a counter_examples přežijí to_dict/from_dict round-trip."""
    lb = TaxonomyLabel(
        code="HODNOTA",
        description="Číselná hodnota s jednotkou.",
        decision_rule="Má-li jednotku času (roky, dny) → patří do DOBA.",
        counter_examples=["2 roky", "14 dnů"],
    )
    d = lb.to_dict()
    lb2 = TaxonomyLabel.from_dict(d)
    assert lb2.decision_rule == "Má-li jednotku času (roky, dny) → patří do DOBA."
    assert lb2.counter_examples == ["2 roky", "14 dnů"]
    # Defaulty: chybějící pole → prázdné (zpětná kompatibilita se starým JSON)
    lb3 = TaxonomyLabel.from_dict({"code": "X", "description": "Nějaký popis navíc."})
    assert lb3.decision_rule == ""
    assert lb3.counter_examples == []


def test_taxonomy_block_renders_boundary_hints():
    """_taxonomy_block vyrenderuje decision_rule i counter_examples, když jsou."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(
            code="DATUM", description="Časový bod nebo interval.",
            decision_rule="Je-li to trvání (5 let), patří do HODNOTA.",
            counter_examples=["5 let", "14 dnů"],
        ),
    ])
    block = _taxonomy_block(tax)
    assert "Rozhodni na hranici" in block
    assert "trvání" in block
    assert "NE sem" in block
    assert "5 let" in block


def test_taxonomy_block_omits_boundary_hints_when_empty():
    """Bez auditu (prázdná pole) blok vypadá jako dřív — žádné hranice/NE sem."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt nebo služba."),
    ])
    block = _taxonomy_block(tax)
    assert "Rozhodni na hranici" not in block
    assert "NE sem" not in block


# ---------- _aggregate_confusion (pure) ------------------------------------

def test_aggregate_confusion_basic():
    """Nejistota = průměr (1 - confidence); pár se skóruje a sbírá evidence."""
    spans = [
        {"text": "2 roky", "top1": "DATUM", "top2": "HODNOTA", "confidence": 0.5},
        {"text": "od 2024", "top1": "DATUM", "top2": "HODNOTA", "confidence": 0.6},
        {"text": "iPhone", "top1": "PRODUKT", "top2": None, "confidence": 1.0},
    ]
    rep = _aggregate_confusion(spans, {"DATUM", "HODNOTA", "PRODUKT"})
    assert rep.n_spans == 3
    # (0.5 + 0.4 + 0.0) / 3
    assert abs(rep.score - (0.9 / 3)) < 1e-9
    assert len(rep.pairs) == 1
    p = rep.pairs[0]
    assert {p.a, p.b} == {"DATUM", "HODNOTA"}
    assert p.support == 2
    assert abs(p.score - 0.9) < 1e-9  # 0.5 + 0.4
    assert set(rep.evidence[("DATUM", "HODNOTA")]) == {"2 roky", "od 2024"}


def test_aggregate_confusion_ignores_unknown_and_selfpair():
    """Labely mimo taxonomii se ignorují; top2==top1 nedělá pár."""
    spans = [
        {"text": "x", "top1": "NEEXISTUJE", "top2": "DATUM", "confidence": 0.3},
        {"text": "y", "top1": "DATUM", "top2": "DATUM", "confidence": 0.4},
        {"text": "z", "top1": "DATUM", "top2": "MRAKY", "confidence": 0.4},  # top2 unknown
    ]
    rep = _aggregate_confusion(spans, {"DATUM", "HODNOTA"})
    # jen 'y' a 'z' mají validní top1=DATUM; 'x' se zahodí
    assert rep.n_spans == 2
    # žádný validní pár (top2 buď == top1, nebo unknown)
    assert rep.pairs == []


def test_aggregate_confusion_case_insensitive():
    spans = [{"text": "a", "top1": "datum", "top2": "hodnota", "confidence": 0.5}]
    rep = _aggregate_confusion(spans, {"DATUM", "HODNOTA"})
    assert rep.n_spans == 1
    assert rep.pairs[0].a == "DATUM"
    assert rep.pairs[0].b == "HODNOTA"


def test_aggregate_confusion_empty():
    rep = _aggregate_confusion([], {"DATUM"})
    assert rep.n_spans == 0
    assert rep.score == 0.0
    assert rep.pairs == []


# ---------- _select_confusable_pairs (pure) --------------------------------

def test_select_confusable_pairs_respects_support_and_cap():
    rep = ConfusionReport(
        score=0.4, n_spans=10,
        pairs=[
            ConfusablePair("A", "B", score=3.0, support=5),
            ConfusablePair("C", "D", score=2.0, support=1),  # pod prahem support
            ConfusablePair("E", "F", score=1.5, support=3),
        ],
    )
    sel = _select_confusable_pairs(rep, max_pairs=2, min_support=2)
    # C/D vypadne (support=1); zůstanou A/B a E/F, cap 2
    assert [(p.a, p.b) for p in sel] == [("A", "B"), ("E", "F")]


# ---------- _apply_sharpen_result (pure) -----------------------------------

def test_apply_sharpen_result_updates_only_allowed_fields():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="DATUM", description="Původní popis data tady.",
                     examples=["1.1.2026"], aliases=["DATE"]),
        TaxonomyLabel(code="HODNOTA", description="Původní popis hodnoty tady."),
    ])
    refined = [
        {"code": "DATUM", "description": "Ostřejší popis: časový bod, ne trvání.",
         "decision_rule": "Trvání (5 let) → HODNOTA.",
         "counter_examples": ["5 let", "5 let"]},  # duplikát se nesmí zdvojit
        {"code": "HODNOTA", "description": "krátké"},  # < 10 znaků → ignorovat
        {"code": "NEZNAMY", "description": "Tenhle label v taxonomii není."},
    ]
    changed = _apply_sharpen_result(tax, refined)
    assert changed == 1  # jen DATUM se reálně změnil
    datum = tax.labels[0]
    assert datum.description == "Ostřejší popis: časový bod, ne trvání."
    assert datum.decision_rule == "Trvání (5 let) → HODNOTA."
    assert datum.counter_examples == ["5 let"]  # dedup
    # Nedotčená pole
    assert datum.examples == ["1.1.2026"]
    assert datum.aliases == ["DATE"]
    # HODNOTA: krátký popis se neaplikoval
    assert tax.labels[1].description == "Původní popis hodnoty tady."


# ---------- _apply_merge (pure) --------------------------------------------

def test_apply_merge_combines_and_removes():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="DATUM", description="Datum tady.", examples=["1.1."],
                     aliases=["DATE"]),
        TaxonomyLabel(code="DOBA", description="Doba tady.", examples=["5 let"],
                     domains=["muzeum"], counter_examples=["rok 2013"]),
        TaxonomyLabel(code="PRODUKT", description="Produkt tady."),
    ])
    ok = _apply_merge(tax, "DATUM", "DOBA")
    assert ok is True
    assert tax.codes == ["DATUM", "PRODUKT"]  # DOBA zmizela
    datum = tax.labels[0]
    assert "5 let" in datum.examples          # examples přeneseny
    assert "muzeum" in datum.domains
    assert "rok 2013" in datum.counter_examples
    assert "DOBA" in datum.aliases            # kód zmizelého do aliasů


def test_apply_merge_protects_two_label_floor():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="A", description="Popis labelu A dlouhý."),
        TaxonomyLabel(code="B", description="Popis labelu B dlouhý."),
    ])
    assert _apply_merge(tax, "A", "B") is False
    assert len(tax.labels) == 2


# ---------- _extract_json_list ---------------------------------------------

def test_extract_json_list_from_parsed_dict():
    assert _extract_json_list("", {"spans": [1, 2]}, "spans") == [1, 2]
    assert _extract_json_list("", {"labels": [3]}, "labels", "Labels") == [3]


def test_extract_json_list_from_bare_list_and_content_fallback():
    assert _extract_json_list("", [1, 2], "spans") == [1, 2]
    # parsed None → parsuj z content
    assert _extract_json_list('{"spans": [9]}', None, "spans") == [9]
    # nevalidní → []
    assert _extract_json_list("junk {", None, "spans") == []


# ---------- _heldout_snippets ----------------------------------------------

def test_heldout_snippets_interleaves_and_caps():
    items = [
        LabelItem(item_id=f"a{i}", source="s", domain="d1", text=f"Alfa text {i}")
        for i in range(30)
    ] + [
        LabelItem(item_id=f"b{i}", source="s", domain="d2", text=f"Beta text {i}")
        for i in range(30)
    ]
    snips = _heldout_snippets(items, samples_per_dataset=10, seed=1, max_snippets=6)
    assert len(snips) == 6
    # interleave: měly by být zastoupené obě domény
    assert any("Alfa" in s for s in snips)
    assert any("Beta" in s for s in snips)
    # deterministické pro stejný seed
    snips2 = _heldout_snippets(items, samples_per_dataset=10, seed=1, max_snippets=6)
    assert snips == snips2


# ---------- audit_and_refine_taxonomy (integrace, fake klient) -------------

def _tax_two_confusable():
    return Taxonomy(labels=[
        TaxonomyLabel(code="DATUM", description="Časový bod nebo interval tady."),
        TaxonomyLabel(code="HODNOTA", description="Číselná hodnota s jednotkou zde."),
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt nebo služba."),
    ])


class _AuditFakeClient:
    """Fake klient: probe vrací klesající konfuzi (0.5 → 0.4 → 0.1), sharpen
    vrací zostřené labely. Rozlišuje probe/sharpen podle system promptu."""
    def __init__(self, probe_confidences):
        self.model = "fake"
        self.calls = 0
        self.probe_calls = 0
        self.sharpen_calls = 0
        self._probe_conf = probe_confidences

    async def chat(self, messages, **kw):
        self.calls += 1
        system = messages[0]["content"]
        if "auditor NER taxonomie" in system:
            self.probe_calls += 1
            conf = self._probe_conf[min(self.probe_calls - 1, len(self._probe_conf) - 1)]
            parsed = {"spans": [
                {"text": "2 roky", "top1": "DATUM", "top2": "HODNOTA",
                 "confidence": conf},
                {"text": "od 2024", "top1": "DATUM", "top2": "HODNOTA",
                 "confidence": conf},
                {"text": "iPhone", "top1": "PRODUKT", "top2": None, "confidence": 1.0},
            ]}
        else:  # sharpen
            self.sharpen_calls += 1
            parsed = {"labels": [
                {"code": "DATUM", "description": "Časový bod, NIKDY trvání.",
                 "decision_rule": "Trvání → HODNOTA.", "counter_examples": ["2 roky"]},
                {"code": "HODNOTA", "description": "Veličina s jednotkou, i doba.",
                 "decision_rule": "Trvání sem.", "counter_examples": ["1.1.2026"]},
            ]}
        return _FakeResp(parsed)


def _run_audit_with_fake(client, tax, **kw):
    import asyncio
    import labeler.taxonomy as tax_mod
    orig_scan = tax_mod.scan_data_dir
    tax_mod.scan_data_dir = lambda _dir: [
        LabelItem(item_id=f"x{i}", source="s", domain="d", text=f"Text {i}")
        for i in range(30)
    ]
    try:
        return asyncio.run(audit_and_refine_taxonomy(client=client, tax=tax, **kw))
    finally:
        tax_mod.scan_data_dir = orig_scan


def test_audit_sharpens_confusable_pair_and_converges():
    """Konfuze klesá; audit zostří DATUM/HODNOTA a přidá jim hraniční vodítka."""
    client = _AuditFakeClient(probe_confidences=[0.5, 0.9])  # kolo1 vysoká konfuze, kolo2 nízká
    tax = _tax_two_confusable()
    out = _run_audit_with_fake(
        client, tax, max_rounds=3, target_score=0.15, min_support=2,
    )
    assert client.sharpen_calls >= 1  # zostření proběhlo
    datum = next(lb for lb in out.labels if lb.code == "DATUM")
    assert datum.decision_rule  # audit doplnil hraniční pravidlo
    assert "2 roky" in datum.counter_examples
    # Výstup je validní taxonomie
    assert out.validate() == []


def test_audit_never_returns_worse_than_input():
    """Když konfuze zůstává vysoká (žádné zlepšení), vrátí NEJLEPŠÍ (= vstup)."""
    client = _AuditFakeClient(probe_confidences=[0.5])  # pořád stejná vysoká konfuze
    tax = _tax_two_confusable()
    out = _run_audit_with_fake(
        client, tax, max_rounds=3, target_score=0.05, min_support=2,
    )
    # best se drží; výstup je validní a má stejné kódy jako vstup (žádný merge)
    assert set(out.codes) == {"DATUM", "HODNOTA", "PRODUKT"}
    assert out.validate() == []


def test_audit_noop_on_empty_data(monkeypatch=None):
    """Prázdný DATA_DIR → audit vrátí taxonomii beze změny, bez volání LLM."""
    import asyncio
    import labeler.taxonomy as tax_mod
    orig_scan = tax_mod.scan_data_dir
    tax_mod.scan_data_dir = lambda _dir: []
    client = _AuditFakeClient(probe_confidences=[0.5])
    tax = _tax_two_confusable()
    try:
        out = asyncio.run(audit_and_refine_taxonomy(client=client, tax=tax))
    finally:
        tax_mod.scan_data_dir = orig_scan
    assert client.calls == 0  # žádné LLM volání
    assert out.codes == tax.codes


# ===========================================================================
# Data-driven label mapping (kind / remap_to / default_taxonomy.json)
# ===========================================================================

def test_label_kind_remap_roundtrip_and_validation():
    lb = TaxonomyLabel(code="POUZITI", description="Kontext použití produktu.",
                       kind="keyphrase", remap_to="KEYPHRASE_USECASE")
    lb2 = TaxonomyLabel.from_dict(lb.to_dict())
    assert lb2.kind == "keyphrase"
    assert lb2.remap_to == "KEYPHRASE_USECASE"
    assert lb2.validate() == []
    # Defaulty pro starý JSON bez polí
    lb3 = TaxonomyLabel.from_dict({"code": "X", "description": "Dost dlouhý popis."})
    assert lb3.kind == "ner"
    assert lb3.remap_to == ""
    # Nevalidní kind / remap_to
    bad = TaxonomyLabel(code="X", description="Dost dlouhý popis.", kind="mrkev")
    assert any("kind" in e for e in bad.validate())
    bad2 = TaxonomyLabel(code="X", description="Dost dlouhý popis.",
                         remap_to="lowercase")
    assert any("remap_to" in e for e in bad2.validate())


def test_label_mapping_derivation_rules():
    """Mapping se odvozuje z dat: remap_to > kind-prefix > identita."""
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt zde."),
        TaxonomyLabel(code="OSOBA", description="Osoba nebo role zde.",
                     remap_to="OSOBA_ROLE"),
        TaxonomyLabel(code="BENEFIT", description="Přínos pro uživatele.",
                     kind="keyphrase"),
        TaxonomyLabel(code="KEYPHRASE_SPEC", description="Norma či standard.",
                     kind="keyphrase"),
    ])
    m = tax.label_mapping()
    assert m["PRODUKT"] == "PRODUKT"                    # ner → identita
    assert m["OSOBA"] == "OSOBA_ROLE"                   # explicitní remap_to
    assert m["BENEFIT"] == "KEYPHRASE_BENEFIT"          # keyphrase → prefix
    assert m["KEYPHRASE_SPEC"] == "KEYPHRASE_SPEC"      # už prefixovaný — beze změny
    # Identita pro cíle (idempotentní reprocessing)
    assert m["OSOBA_ROLE"] == "OSOBA_ROLE"
    assert m["KEYPHRASE_BENEFIT"] == "KEYPHRASE_BENEFIT"


def test_ner_label_targets_excludes_keyphrase():
    tax = Taxonomy(labels=[
        TaxonomyLabel(code="PRODUKT", description="Konkrétní produkt zde."),
        TaxonomyLabel(code="OSOBA", description="Osoba nebo role zde.",
                     remap_to="OSOBA_ROLE"),
        TaxonomyLabel(code="BENEFIT", description="Přínos pro uživatele.",
                     kind="keyphrase"),
    ])
    targets = tax.ner_label_targets()
    assert targets == frozenset({"PRODUKT", "OSOBA_ROLE"})


def test_default_taxonomy_loads_and_matches_legacy_mapping():
    """default_taxonomy.json je validní a odvozený mapping je IDENTICKÝ
    s dřívějším hard-coded DEFAULT_LABEL_MAPPING (regresní pojistka)."""
    from labeler.taxonomy import default_taxonomy
    tax = default_taxonomy()
    assert tax.validate() == []
    assert len(tax.labels) == 16
    legacy = {
        "PRODUKT": "PRODUKT", "ORGANIZACE": "ORGANIZACE", "DATUM": "DATUM",
        "MISTO": "MISTO", "HODNOTA": "HODNOTA", "MATERIAL": "MATERIAL",
        "OSOBA": "OSOBA_ROLE", "KATEGORIE": "PRODUKT_TYP",
        "BENEFIT": "KEYPHRASE_BENEFIT", "PARAMETR": "KEYPHRASE_PARAMETR",
        "KOMPONENTA": "KEYPHRASE_KOMPONENTA", "SPECIFIKACE": "KEYPHRASE_SPEC",
        "POUZITI": "KEYPHRASE_USECASE", "TECHNOLOGIE": "KEYPHRASE_TECH",
        "VLASTNOST": "KEYPHRASE_PROPERTY", "SUBJEKT": "KEYPHRASE_SUBJECT",
        "PRODUKT_TYP": "PRODUKT_TYP", "OSOBA_ROLE": "OSOBA_ROLE",
        "KEYPHRASE_BENEFIT": "KEYPHRASE_BENEFIT",
        "KEYPHRASE_PARAMETR": "KEYPHRASE_PARAMETR",
        "KEYPHRASE_KOMPONENTA": "KEYPHRASE_KOMPONENTA",
        "KEYPHRASE_SPEC": "KEYPHRASE_SPEC",
        "KEYPHRASE_USECASE": "KEYPHRASE_USECASE",
        "KEYPHRASE_TECH": "KEYPHRASE_TECH",
        "KEYPHRASE_PROPERTY": "KEYPHRASE_PROPERTY",
        "KEYPHRASE_SUBJECT": "KEYPHRASE_SUBJECT",
    }
    assert tax.label_mapping() == legacy
    assert tax.ner_label_targets() == frozenset({
        "PRODUKT", "PRODUKT_TYP", "ORGANIZACE", "OSOBA_ROLE",
        "MISTO", "DATUM", "HODNOTA", "MATERIAL",
    })


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
