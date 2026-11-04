"""Critic a revisor musí dostat TU SAMOU taxonomii jako extractor.

Regrese: critic měl v systémovém promptu natvrdo 16 defaultních kategorií
(a revisor neměl povolený slovník vůbec), takže u auto-discovered taxonomie
z fáze 0 označil platné labely za "neexistující kategorii" a přepsal je.
"""
from __future__ import annotations

import inspect

import pytest

from labeler import critic as critic_mod
from labeler import ner_re_split, revise as revise_mod
from labeler.prompts import (
    build_critic_messages,
    build_critic_ner_messages,
    build_extractor_revision_messages,
    build_revision_ner_messages,
)
from labeler.taxonomy import Taxonomy, TaxonomyLabel

DEFAULT_16_FRAGMENT = "SPECIFIKACE, ORGANIZACE"


@pytest.fixture
def tax() -> Taxonomy:
    return Taxonomy(
        labels=[
            TaxonomyLabel(
                code="POJISTNY_PRODUKT",
                description="Konkrétní pojistný produkt nebo služba.",
                examples=["Pojištění domácnosti"],
                decision_rule="Produkt, ne riziko: má cenu a lze ho sjednat.",
                counter_examples=["povodeň"],
            ),
            TaxonomyLabel(code="RIZIKO", description="Pojištěné riziko."),
        ]
    )


def _graph() -> dict:
    return {
        "entities": [
            {"id": "e1", "text": "Pojištění domácnosti", "category": "POJISTNY_PRODUKT"}
        ],
        "relations": [],
    }


def _systems(tax_arg):
    """Systémové prompty všech čtyř volání, která o kategoriích rozhodují."""
    g = _graph()
    return {
        "critic": build_critic_messages("text", g, None, tax_arg)[0]["content"],
        "critic_ner": build_critic_ner_messages(
            "text", g["entities"], None, tax_arg
        )[0]["content"],
        "revise": build_extractor_revision_messages(
            "text", g, "revised", "důvod", None, None, tax_arg
        )[0]["content"],
        "revise_ner": build_revision_ner_messages(
            "text", g["entities"], "revised", "důvod", None, None, tax_arg
        )[0]["content"],
    }


def test_with_taxonomy_only_its_labels_are_allowed(tax):
    for name, system in _systems(tax).items():
        assert "POJISTNY_PRODUKT" in system, f"{name}: chybí label z taxonomie"
        assert "RIZIKO" in system, f"{name}: chybí label z taxonomie"
        assert DEFAULT_16_FRAGMENT not in system, (
            f"{name}: s vlastní taxonomií se NESMÍ objevit default 16 kategorií"
        )


def test_without_taxonomy_default_16_stays():
    for name, system in _systems(None).items():
        assert DEFAULT_16_FRAGMENT in system, f"{name}: bez taxonomie čekám default 16"
        assert "POJISTNY_PRODUKT" not in system


def test_no_unformatted_placeholder_leaks(tax):
    for tax_arg in (tax, None):
        for name, system in _systems(tax_arg).items():
            assert "{categories_rule}" not in system, f"{name}: placeholder v promptu"


def test_taxonomy_decision_rules_reach_the_critic(tax):
    """Critic má vidět i hraniční vodítka z disjointness auditu."""
    system = _systems(tax)["critic"]
    assert "Produkt, ne riziko" in system
    assert "povodeň" in system


def test_critic_must_not_rewrite_known_label(tax):
    """Prompt musí explicitně zakázat 'opravu' labelu, který v taxonomii je."""
    system = _systems(tax)["critic"]
    assert "VŽDY platný" in system
    assert "NIKDY ho neměň" in system


@pytest.mark.parametrize(
    "func",
    [
        critic_mod._critique_one,
        critic_mod.run_critic,
        critic_mod._run_critic_branches,
        revise_mod._revise_one,
        revise_mod.run_revise,
        revise_mod._revise_branch,
        ner_re_split.critique_split,
        ner_re_split.revise_split,
    ],
)
def test_taxonomy_is_threaded_through(func):
    assert "taxonomy" in inspect.signature(func).parameters, (
        f"{func.__module__}.{func.__name__} taxonomii nepředává"
    )


def test_main_passes_taxonomy_to_critic_and_revise():
    """Fáze 2 i 2b musí taxonomii načíst, ne jen umět přijmout."""
    import pathlib

    src = pathlib.Path("labeler/main.py").read_text(encoding="utf-8")
    for call in ("await run_critic(", "await run_revise("):
        starts = [i for i in range(len(src)) if src.startswith(call, i)]
        assert starts, f"{call} v main.py nenalezeno"
        for i in starts:
            block = src[i:src.index(")", i) + 1]
            assert "taxonomy=" in block, f"{call} bez taxonomy= (offset {i})"
