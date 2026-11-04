"""Prompt + odpověď se musí vejít do kontextu serveru.

Regrese z provozu (critic-NER na A100, kontext 32768):

    Requested token count exceeds the model's maximum context length of 32768
    tokens. You requested a total of 39011 tokens: 14435 tokens from the input
    messages and 24576 tokens for the completion.

Tedy prompt + max_tokens > kontext → HTTP 400 a ztracený item.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

from labeler.llm_client import (
    CONTEXT_RESERVE_TOKENS,
    MIN_COMPLETION_TOKENS,
    LLMClient,
    LLMPromptTooLong,
    estimate_prompt_tokens,
    fit_completion_tokens,
)

CTX = 32768


def _prompt_of_tokens(n: int) -> list[dict]:
    """Zpráva, jejíž odhad vyjde zhruba na n tokenů."""
    return [{"role": "system", "content": "x" * int(n * 2.5)}]


def test_production_case_is_clamped_to_fit():
    messages = _prompt_of_tokens(14435)
    allowed, prompt_tokens = fit_completion_tokens(messages, 24576, CTX)
    assert 0 < allowed < 24576
    assert prompt_tokens + allowed + CONTEXT_RESERVE_TOKENS <= CTX


def test_short_prompt_is_untouched():
    allowed, _ = fit_completion_tokens([{"role": "user", "content": "krátký"}], 24576, CTX)
    assert allowed == 24576


def test_unknown_context_changes_nothing():
    """Bez znalosti kontextu se chováme jako dřív (žádné omezování)."""
    allowed, _ = fit_completion_tokens(_prompt_of_tokens(14435), 24576, None)
    assert allowed == 24576


def test_prompt_filling_the_context_returns_zero():
    allowed, _ = fit_completion_tokens(_prompt_of_tokens(32000), 24576, CTX)
    assert allowed == 0


def test_estimate_counts_all_messages():
    one = estimate_prompt_tokens([{"role": "system", "content": "a" * 1000}])
    two = estimate_prompt_tokens(
        [{"role": "system", "content": "a" * 1000}, {"role": "user", "content": "a" * 1000}]
    )
    assert two > one


class _FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {"choices": [{"message": {"content": '{"status": "ok"}'}}]}


class _FakeHttpx:
    """Zachytí payload, který by šel na server."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def post(self, url, headers=None, json=None):  # noqa: A002
        self.payloads.append(json)
        return _FakeResponse()


def _client_with_fake(context_len, fake):
    client = LLMClient(
        base_url="http://x/v1", model="m", max_concurrency=2, context_len=context_len
    )
    client._client = fake
    return client


def test_chat_clamps_max_tokens_before_sending():
    fake = _FakeHttpx()
    client = _client_with_fake(CTX, fake)
    messages = _prompt_of_tokens(14435)
    asyncio.run(client.chat(messages, max_tokens=24576))
    sent = fake.payloads[0]["max_tokens"]
    assert sent < 24576
    assert estimate_prompt_tokens(messages) + sent <= CTX


def test_chat_does_not_touch_a_request_that_fits():
    fake = _FakeHttpx()
    client = _client_with_fake(CTX, fake)
    asyncio.run(client.chat([{"role": "user", "content": "ahoj"}], max_tokens=4096))
    assert fake.payloads[0]["max_tokens"] == 4096


def test_chat_refuses_to_send_a_hopeless_prompt():
    """Request, který by stejně skončil 400, se vůbec neodešle."""
    fake = _FakeHttpx()
    client = _client_with_fake(CTX, fake)
    with pytest.raises(LLMPromptTooLong):
        asyncio.run(client.chat(_prompt_of_tokens(32000), max_tokens=24576))
    assert fake.payloads == [], "request se neměl odeslat"


def test_prompt_too_long_is_fatal_so_callers_skip_the_item():
    from labeler.llm_client import LLMFatalError

    assert issubclass(LLMPromptTooLong, LLMFatalError)


def test_reasoning_floor_is_clamped_too():
    """reasoning=True tlačí max_tokens na 16384 — i to musí projít clampem."""
    fake = _FakeHttpx()
    client = _client_with_fake(CTX, fake)
    messages = _prompt_of_tokens(20000)
    asyncio.run(client.chat(messages, max_tokens=512, reasoning=True))
    sent = fake.payloads[0]["max_tokens"]
    assert sent < 16384
    assert estimate_prompt_tokens(messages) + sent <= CTX


@pytest.mark.parametrize(
    "module, expected",
    [
        ("labeler/critic.py", "CONFIG.critic_context_len"),
        ("labeler/extractor.py", "CONFIG.extractor_context_len"),
        ("labeler/revise.py", "CONFIG.extractor_context_len"),
        ("labeler/taxonomy.py", "CONFIG.extractor_context_len"),
    ],
)
def test_every_phase_tells_the_client_its_context(module, expected):
    src = pathlib.Path(module).read_text(encoding="utf-8")
    assert f"context_len={expected}" in src, f"{module} nepředává kontext serveru"


# --- samooprava z chybové odpovědi serveru ---------------------------------

PROD_ERROR_BODY = (
    '{"object":"error","message":"Requested token count exceeds the model\'s '
    "maximum context length of 32768 tokens. You requested a total of 39011 "
    "tokens: 14435 tokens from the input messages and 24576 tokens for the "
    'completion.","type":"BadRequestError","param":null,"code":400}'
)


def test_exact_budget_is_parsed_from_the_server_error():
    from labeler.llm_client import exact_completion_from_error

    assert exact_completion_from_error(PROD_ERROR_BODY) == (32768, 14435)
    assert exact_completion_from_error("něco jiného") is None


class _Fake400ThenOk:
    """Poprvé vrátí 400 kvůli kontextu, podruhé OK."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def post(self, url, headers=None, json=None):  # noqa: A002
        self.payloads.append(dict(json))
        if len(self.payloads) == 1:
            class R:
                status_code = 400
                text = PROD_ERROR_BODY
            return R()
        return _FakeResponse()


def test_server_400_is_self_corrected_once():
    """Když se odhad netrefí, přesná čísla ze serveru to srovnají."""
    fake = _Fake400ThenOk()
    # kontext schválně nastavený špatně (vyšší, než server reálně má)
    client = _client_with_fake(65536, fake)
    asyncio.run(client.chat([{"role": "user", "content": "x" * 3000}], max_tokens=24576))
    assert len(fake.payloads) == 2, "měl proběhnout právě jeden opravný pokus"
    assert fake.payloads[1]["max_tokens"] == 32768 - 14435 - 64
    # prompt se NESMÍ měnit — opravuje se jen budget odpovědi
    assert fake.payloads[0]["messages"] == fake.payloads[1]["messages"]


def test_prompt_is_never_modified_by_the_budget_logic():
    """Entity, relace ani text se clampem nijak nemění."""
    fake = _FakeHttpx()
    client = _client_with_fake(CTX, fake)
    messages = [
        {"role": "system", "content": "pravidla " * 2000},
        {"role": "user", "content": '{"entities": [{"id": "e1", "text": "Kotva"}]}'},
    ]
    asyncio.run(client.chat(messages, max_tokens=24576))
    assert fake.payloads[0]["messages"] == messages
