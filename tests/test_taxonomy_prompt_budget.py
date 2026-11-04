"""Fáze 0 nesmí poslat prompt, který se nevejde do kontextu serveru."""
import contextlib
import inspect

import pytest

from labeler.config import CONFIG
from labeler import taxonomy as t


@contextlib.contextmanager
def context_len(value: int):
    """CONFIG je frozen dataclass -> prepis pres object.__setattr__."""
    old = CONFIG.extractor_context_len
    object.__setattr__(CONFIG, "extractor_context_len", value)
    try:
        yield
    finally:
        object.__setattr__(CONFIG, "extractor_context_len", old)


def test_small_prompt_untouched():
    samples = {"a": ["kratky text", "druhy"], "b": ["jeden"]}
    fit, cap, note = t._fit_samples_to_budget(samples, t._prompt_char_budget(8192))
    assert note is None
    assert fit == samples
    assert cap == CONFIG.taxonomy_sample_preview_chars


def test_big_prompt_fits_budget():
    with context_len(32768):
        budget = t._prompt_char_budget(8192)
        samples = {
            f"dom{d}": [("text s diakritikou ěščřžýáíé " * 40)[:900] for _ in range(20)]
            for d in range(12)
        }
        _, raw_block = t._format_samples_for_prompt(samples)
        assert len(raw_block) > budget, "testovací prompt má být přes budget"

        fit, cap, note = t._fit_samples_to_budget(samples, budget)
        _, block = t._format_samples_for_prompt(fit, preview_chars=cap)
        assert len(block) <= budget
        assert note is not None
        # žádná doména nesmí přijít o všechny ukázky
        assert all(len(v) >= 2 for v in fit.values())


def test_budget_scales_with_context():
    with context_len(65536):
        big = t._prompt_char_budget(8192)
    with context_len(32768):
        small = t._prompt_char_budget(8192)
    assert big > small > 0


def test_budget_survives_tiny_context():
    """Kontext menší než požadovaná odpověď nesmí dát nulový/negativní budget."""
    with context_len(4096):
        assert t._prompt_char_budget(8192) > 0


def test_reasoning_floor_can_be_lowered():
    """llm_client s reasoning=True nesmí ignorovat nižší strop od volajícího."""
    from labeler.llm_client import LLMClient
    sig = inspect.signature(LLMClient.chat)
    assert "reasoning_max_tokens" in sig.parameters
    src = inspect.getsource(LLMClient.chat)
    assert "floor = 16384 if reasoning_max_tokens is None else reasoning_max_tokens" in src
