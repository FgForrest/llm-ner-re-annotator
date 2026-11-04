"""Fáze 0: horní počet kategorií jde vypnout (TAXONOMY_MAX_LABELS=0)."""
from __future__ import annotations

import contextlib

from labeler.config import CONFIG
from labeler.taxonomy import label_count_rule, taxonomy_system_prompt


@contextlib.contextmanager
def label_bounds(lo: int, hi: int):
    old = (CONFIG.taxonomy_min_labels, CONFIG.taxonomy_max_labels)
    object.__setattr__(CONFIG, "taxonomy_min_labels", lo)
    object.__setattr__(CONFIG, "taxonomy_max_labels", hi)
    try:
        yield
    finally:
        object.__setattr__(CONFIG, "taxonomy_min_labels", old[0])
        object.__setattr__(CONFIG, "taxonomy_max_labels", old[1])


def test_range_is_rendered_when_max_is_set():
    rule = label_count_rule(8, 16)
    assert "8-16" in rule
    assert "HORNÍ HRANICE" not in rule


def test_no_upper_bound_when_max_is_zero():
    rule = label_count_rule(8, 0)
    assert "HORNÍ HRANICE NENÍ STANOVENÁ" in rule
    assert "aspoň 8" in rule
    assert "8-16" not in rule


def test_minimum_still_applies_without_an_upper_bound():
    """Bez stropu nesmí zmizet spodní hranice — brána na degenerovaný návrh."""
    assert "aspoň 12" in label_count_rule(12, 0)


def test_inconsistent_range_is_normalized():
    """max < min nesmí vyrobit nesmysl typu '10-5'."""
    rule = label_count_rule(10, 5)
    assert "10-5" not in rule
    assert "10-10" in rule


def test_system_prompt_has_no_leftover_placeholder():
    for lo, hi in ((8, 16), (8, 0), (10, 5)):
        with label_bounds(lo, hi):
            prompt = taxonomy_system_prompt()
            assert "{LABEL_COUNT_RULE}" not in prompt
            assert "{MIN_LABELS}" not in prompt
            assert "{MAX_LABELS}" not in prompt


def test_system_prompt_reflects_env_config():
    with label_bounds(8, 0):
        assert "HORNÍ HRANICE NENÍ STANOVENÁ" in taxonomy_system_prompt()
    with label_bounds(8, 16):
        assert "8-16" in taxonomy_system_prompt()


def test_explicit_arguments_win_over_config():
    with label_bounds(8, 16):
        assert "HORNÍ HRANICE NENÍ STANOVENÁ" in taxonomy_system_prompt(8, 0)


def test_empty_env_value_falls_back_to_default(monkeypatch):
    """TAXONOMY_MAX_LABELS= (prázdné) nesmí shodit import configu."""
    import importlib

    monkeypatch.setenv("TAXONOMY_MAX_LABELS", "")
    import labeler.config as cfg

    reloaded = importlib.reload(cfg)
    assert reloaded.CONFIG.taxonomy_max_labels == 16
    monkeypatch.delenv("TAXONOMY_MAX_LABELS", raising=False)
    importlib.reload(cfg)


# --- vypnutá spodní hranice (TAXONOMY_MIN_LABELS=0) ------------------------

def test_no_lower_bound_when_min_is_zero():
    rule = label_count_rule(0, 16)
    assert "SPODNÍ HRANICE NENÍ STANOVENÁ" in rule
    assert "nejvýš 16" in rule
    assert "aspoň" not in rule


def test_both_bounds_off():
    rule = label_count_rule(0, 0)
    assert "NENÍ NIJAK OMEZENÝ" in rule
    assert "aspoň" not in rule and "nejvýš" not in rule


def test_unbounded_wordings_keep_the_distinguishability_warning():
    """Bez stropu hrozí překrývající se kategorie — připomínka tam musí být."""
    for lo, hi in ((8, 0), (0, 16), (0, 0)):
        assert "navzájem odlišitelné" in label_count_rule(lo, hi)


def test_quality_gate_follows_the_minimum():
    from labeler.taxonomy import min_llm_labels

    with label_bounds(8, 16):
        assert min_llm_labels() == 5       # tolerance 3 pod minimem
    with label_bounds(3, 16):
        assert min_llm_labels() == 2       # nikdy pod 2, dokud je brána zapnutá
    with label_bounds(0, 16):
        assert min_llm_labels() == 0       # brána vypnutá


def test_quality_gate_stops_complaining_about_count_when_min_is_zero():
    """Vypnuté minimum = žádná reklamace na počet (3 labely < default 5)."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel, _assess_taxonomy_quality

    tax = Taxonomy(
        labels=[
            TaxonomyLabel(code=f"KATEGORIE_{i}", description=f"Testovací kategorie {i}.")
            for i in range(3)
        ]
    )
    with label_bounds(8, 16):
        complaints = _assess_taxonomy_quality(tax, n_llm_labels=3)
        assert any("málo" in c.lower() for c in complaints), "s minimem má reklamovat"
    with label_bounds(0, 16):
        assert _assess_taxonomy_quality(tax, n_llm_labels=3) == []


def test_structural_floor_of_two_labels_stays():
    """I s vypnutým minimem drží strukturální podmínka Taxonomy.validate():
    jednoprvková taxonomie nemá co rozlišovat."""
    from labeler.taxonomy import Taxonomy, TaxonomyLabel, _assess_taxonomy_quality

    single = Taxonomy(
        labels=[TaxonomyLabel(code="JEDINY", description="Jediná kategorie v taxonomii.")]
    )
    with label_bounds(0, 0):
        complaints = _assess_taxonomy_quality(single, n_llm_labels=1)
        assert complaints, "jeden label musí projít reklamací i bez minima"
        assert any("2 labely" in c for c in complaints)


def test_empty_labels_are_rejected_even_without_a_minimum():
    """Vypnutá brána nesmí pustit návrh úplně bez labelů."""
    import pytest

    from labeler.taxonomy import _parse_llm_taxonomy_response

    with label_bounds(0, 0):
        with pytest.raises(ValueError):
            _parse_llm_taxonomy_response('{"labels": []}', {"labels": []})


def test_system_prompt_without_any_bounds():
    with label_bounds(0, 0):
        prompt = taxonomy_system_prompt()
        assert "NENÍ NIJAK OMEZENÝ" in prompt
        assert "{MIN_LABELS}" not in prompt and "{MAX_LABELS}" not in prompt
