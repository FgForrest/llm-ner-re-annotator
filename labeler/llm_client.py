"""
HTTP klient pro OpenAI-compatible endpoint (SGLang nebo vLLM).

Features:
- Asynchronní (httpx) pro paralelní requesty
- Exponential backoff retry (transient errors, 5xx, timeouts)
- Striktní JSON-only response parsing (s fallback repairem)
- Token accounting (log input/output tokens pro monitoring)
- Semafor pro limit paralelních requestů na endpoint
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import httpx

logger = logging.getLogger(__name__)


# Pattern pro extrakci JSON z odpovědí, které občas mají žvanění okolo
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Reasoning modely emitují thinking trace a teprve PAK finální JSON. Trace
# musíme odstranit, jinak se { } závorky uvnitř úvah pletou do extrakce JSON.
# Dva formáty:
#   1) <think>...</think>  (Qwen3.5 / DeepSeek styl)
#   2) harmony kanály <|channel|>analysis<|message|>…<|channel|>final<|message|>…
#      (gpt-oss styl; emituje i nvidia/Gemma-4). Critic běží na SGLangu, který
#      pro tenhle model NEMÁ reasoning-parser (SGLang umí jen deepseek-r1 /
#      qwen3, ne gemma4), takže thinking leakuje rovnou do contentu jako
#      `<|channel>thought …`. Tady ho ořízneme, ať se finální JSON dá najít.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|?>\s*final\b.*?<\|message\|?>", re.DOTALL | re.IGNORECASE
)
_HARMONY_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|?>")


def _strip_think_blocks(text: str) -> str:
    """Odstraní reasoning trace, ať zbyde jen finální odpověď (typicky JSON).

    Podporuje:
      - <think>...</think> (i bez otevíracího tagu → bere obsah za posledním
        </think>),
      - harmony kanály <|channel|>…<|message|>: pokud existuje 'final' kanál,
        vezme obsah za posledním <|channel|>final<|message|>; pak odstraní
        zbylé harmony control tokeny (<|channel|>, <|message|>, <|end|>, …).
    Pro běžný (ne-harmony) text se nic navíc nestane — regexy nic nematchnou.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    lower = cleaned.lower()
    idx = lower.rfind("</think>")
    if idx != -1:
        cleaned = cleaned[idx + len("</think>"):]
    # Harmony 'final' kanál = finální odpověď (ne thinking) → vezmi obsah za ním.
    finals = list(_HARMONY_FINAL_RE.finditer(cleaned))
    if finals:
        cleaned = cleaned[finals[-1].end():]
    # Odstraň zbylé harmony control tokeny, aby nerušily extrakci JSON.
    cleaned = _HARMONY_TOKEN_RE.sub("", cleaned)
    return cleaned


def _try_parse_object_at(text: str, start: int) -> tuple[Optional[dict], int]:
    """Od indexu `start` (kde je '{') najde vyvážený konec objektu (respektuje
    uvozovky a escapy) a zkusí ho naparsovat. Vrací (dict|None, index_za_objektem)."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    return (obj if isinstance(obj, dict) else None, i + 1)
                except json.JSONDecodeError:
                    return (None, i + 1)
    return (None, len(text))


def _extract_last_json_object(text: str) -> Optional[dict]:
    """Najde POSLEDNÍ top-level vyvážený {...} objekt, který se naparsuje jako
    dict. Robustnější než greedy regex `\\{.*\\}`, který by spojil závorky z
    reasoning trace s finálním JSONem a selhal."""
    last_valid: Optional[dict] = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            obj, end = _try_parse_object_at(text, i)
            if obj is not None:
                last_valid = obj
                i = end
                continue
        i += 1
    return last_valid


class LLMError(Exception):
    pass


class LLMTransientError(LLMError):
    """Chyba, kterou má smysl opakovat (5xx, timeout, connection reset)."""


class LLMFatalError(LLMError):
    """Chyba, kterou nemá smysl opakovat (400, validace)."""


@dataclass
class LLMResponse:
    content: str
    parsed: Optional[Union[dict, list]]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retries: int
    # finish_reason z OpenAI-compat odpovědi: "stop" (normální konec),
    # "length" (model dojel do max_tokens = runaway → typicky useknutý JSON).
    # Slouží k odlišení runaway od jiných parse chyb v logu.
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Async klient pro OpenAI-compatible /v1/chat/completions endpoint.

    Použití:
        async with LLMClient("http://extractor:8000/v1", "qwen3.5-122b") as client:
            resp = await client.chat(messages, temperature=0)
            data = resp.parsed  # dict nebo None
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_concurrency: int = 16,
        request_timeout: float = 300.0,
        max_retries: int = 5,
        api_key: str = "not-needed",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout = request_timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=30.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def health(self) -> bool:
        """Check jestli endpoint odpovídá."""
        assert self._client
        try:
            r = await self._client.get(f"{self.base_url}/models", headers=self._headers)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Health check selhal: {e}")
            return False

    async def wait_ready(self, max_wait_s: int = 600) -> None:
        """Čeká dokud endpoint není připraven (model loaded)."""
        start = time.monotonic()
        attempt = 0
        while time.monotonic() - start < max_wait_s:
            if await self.health():
                logger.info(f"LLM endpoint {self.base_url} je připraven ({self.model})")
                return
            attempt += 1
            wait = min(10, 2**attempt)
            logger.info(f"Čekám na LLM endpoint {self.base_url}… (pokus {attempt})")
            await asyncio.sleep(wait)
        raise LLMFatalError(f"Endpoint {self.base_url} nebyl ready do {max_wait_s}s")

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format_json: bool = True,
        seed: Optional[int] = 42,
        reasoning: bool = False,
        repetition_penalty: Optional[float] = None,
    ) -> LLMResponse:
        """
        Zavolá chat completion s retry a parse JSON.

        reasoning=True zapne thinking mode pro Qwen3.5 a Gemma 4. Implementace:
          - SGLang/vLLM:   chat_template_kwargs={"enable_thinking": true}
          - OpenAI-style:  extra_body={"enable_thinking": True}
        Používáme chat_template_kwargs, což je podporováno v obou backendech.
        """
        assert self._client
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        # Anti-runaway: repetition penalty (SGLang i vLLM ho berou jako top-level
        # sampling param). 1.0 / None = neposílat (žádná penalta).
        if repetition_penalty is not None and repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if response_format_json and not reasoning:
            # Guided JSON decoding — funguje ve vLLM i SGLang.
            # POZOR: s reasoning=True nelze, protože thinking tokeny by byly
            # v JSON formátu což rozbije thinking. Řešíme parse-em JSON z textu.
            payload["response_format"] = {"type": "json_object"}
        if reasoning:
            # Qwen3.5 / Gemma 4 template flag
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            # Reasoning vyžaduje hodně prostoru pro thinking trace + finální JSON.
            # 16k je velkorysé — kritik s tímto budgetem má pohodlí i pro
            # nejsložitější validace s detailním zdůvodněním a opraveným grafem.
            payload["max_tokens"] = max(max_tokens, 16384)
        else:
            # KRITICKÉ: Qwen3 chat template má enable_thinking DEFAULTNĚ ON.
            # Bez explicitního vypnutí model i v "ne-reasoning" režimu (extractor
            # s EXTRACTOR_REASONING=false, revisor) začne emitovat <think>…,
            # což je v PŘÍMÉM konfliktu s response_format=json_object výše:
            # gramatika pustí jen validní JSON, ne <think>. Pod plnou dávkou
            # (vysoká souběžnost) se tahle hraniční generace "rozjede" a jede
            # do max_tokens → useknutý JSON → reject (100 % errorů na velkém
            # vzorku, na malém ne). Explicitní enable_thinking=False ten konflikt
            # odstraní a sjednotí chování napříč velikostí dávky.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            async with self._sem:
                t0 = time.monotonic()
                try:
                    r = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                    )
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise LLMTransientError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if r.status_code != 200:
                        raise LLMFatalError(f"HTTP {r.status_code}: {r.text[:500]}")
                    data = r.json()
                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.ReadError,
                ) as e:
                    last_exc = LLMTransientError(f"Network: {type(e).__name__}: {e}")
                except LLMTransientError as e:
                    last_exc = e
                except LLMFatalError:
                    raise
                except Exception as e:
                    last_exc = LLMTransientError(
                        f"Neočekávaná chyba: {type(e).__name__}: {e}"
                    )
                else:
                    # Úspěch
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMFatalError(f"Odpověď bez choices: {data}")
                    content = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason")
                    usage = data.get("usage", {}) or {}
                    parsed = self._parse_json(content)
                    return LLMResponse(
                        content=content,
                        parsed=parsed,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        latency_ms=latency_ms,
                        retries=attempt,
                        finish_reason=finish_reason,
                    )

            # Backoff mezi retries
            if attempt < self.max_retries:
                wait = min(60.0, 1.5**attempt + 0.5)
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} po {wait:.1f}s: {last_exc}"
                )
                await asyncio.sleep(wait)

        raise last_exc or LLMFatalError("Neznámá chyba po max_retries")

    @staticmethod
    def _parse_json(content: str) -> Optional[Union[dict, list]]:
        """
        Pokusí se extrahovat validní JSON dict NEBO list z odpovědi.
        Pokud odpověď obsahuje žvanění navíc, zkusí:
        0. odstranit reasoning trace <think>...</think>
        1. přímý json.loads (dict i bare list)
        2. extract z ```json fence``` (dict i bare list)
        3. poslední vyvážený {...} objekt (robustní vůči závorkám v úvaze)
        4. fallback greedy regex {...}
        Vrací None pokud se nepodaří nic.

        POZOR: vrací i top-level list. Modely (zvlášť Qwen3.5 v NER/RE split
        režimu) místo `{"entities":[...]}` často vrátí rovnou bare list
        `[{...},{...}]`. extract_list() to umí zpracovat; kdybychom list zahodili
        (vraceli jen dict), KAŽDÝ takový item by spadl jako "nevrátil validní
        JSON" → zahozen → prázdný extractions.jsonl. Konzumenti, kteří čekají
        dict (.get / SkoGraph(**...)), si list ošetřují sami.
        """
        if not content:
            return None
        content = content.strip()

        # 0) Odstraň reasoning trace — s reasoning=True (critic) přichází JSON až
        #    PO <think>...</think>; bez tohoto by se závorky z úvahy pletly do parse.
        content = _strip_think_blocks(content).strip()
        if not content:
            return None

        # 1) Přímý parse — akceptuj dict i bare list (viz docstring).
        try:
            obj = json.loads(content)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            pass

        # 2) ```json ... ``` fence
        m = _JSON_FENCE_RE.search(content)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                pass

        # 3) Poslední vyvážený {...} objekt (správně počítá závorky + uvozovky)
        obj = _extract_last_json_object(content)
        if obj is not None:
            return obj

        # 4) Fallback: greedy regex (zpětná kompatibilita)
        m = _JSON_OBJECT_RE.search(content)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass

        return None
