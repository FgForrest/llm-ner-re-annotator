#!/usr/bin/env python3
"""
probe_concurrency.py — najde MAXIMÁLNÍ STABILNÍ souběžnost extractor LLM serveru
bez slepého opakování celého pipeline.

PROČ: u Qwen3.5-122B (hybrid MoE+Mamba) na SGLangu se pod vysokou souběžností
generování "rozjede" (model se nezastaví, jede do max_tokens → useknutý JSON →
error). Na malém vzorku to neuvidíš, na velkém ano. Tenhle skript hranici najde
deterministicky: pro každou úroveň souběžnosti N pošle dávku reálných extract
requestů a změří, kolik jich skončí finish_reason="length" (= runaway).

CO ŘÍKÁ VÝSLEDEK:
  - Runaway až od nějakého N (např. od 16)  → je to SOUBĚŽNOST. Nastav
    EXTRACTOR_CONCURRENCY i EXTRACTOR_BATCH_SIZE na poslední stabilní úroveň.
  - Runaway UŽ na N=1                        → není to souběžnost, ale PROMPT
    (snižování concurrency nepomůže) — pošli mi tabulku, opravím prompt.

Stdlib only (urllib + vlákna) → běží na hostu bez pip installu. Server musí být
dostupný; docker mapuje extractor na localhost:8000.

PARITA S PRODUKCÍ: probe defaultně posílá STEJNÁ pole jako produkční extractor
— response_format=json_object, chat_template_kwargs.enable_thinking=False a
seed=42. Bez nich by testoval jiný tvar requestu a hlásil falešné "OK".

POUŽITÍ (z kořene repa, server běží):
  python scripts/probe_concurrency.py
  python scripts/probe_concurrency.py --levels 1,4,8,16,24,32 --mode ner
  python scripts/probe_concurrency.py --mode combined        # pro srovnání
  python scripts/probe_concurrency.py --max-tokens 24576     # pomalé, reálný strop

  # A/B test opravy enable_thinking (čekej runaway u 'default', čisto u 'off'):
  python scripts/probe_concurrency.py --thinking default     # staré (rozbité)
  python scripts/probe_concurrency.py --thinking off         # po opravě

  # CRITIC server (jiný endpoint/model, thinking ON, bez json_object, vyšší
  # max_tokens — critic generuje thinking trace + JSON, proto je pomalejší):
  python scripts/probe_concurrency.py --role critic --levels 1,16,32,48,64

Rychlost: --max-tokens default 4096 je RYCHLÝ proxy. finish_reason="length" při
4096 = model se do 4096 tokenů nezastavil = runaway (zdravá extrakce skončí
hluboko pod tím). Reálný strop (24576) testuj jen když chceš přesné časy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Realistický český vzorek (~ délka prod chunku, doména kosmetika/e-shop) —
# ať prompt vypadá přesně jako v produkci.
DOMAIN = "e-shop s kosmetikou a péčí o pleť"
SAMPLE_TEXT = (
    "Pleťové sérum Hydra Intense od značky ACME obsahuje nízkomolekulární "
    "kyselinu hyaluronovou a vitamín B5, které intenzivně hydratují a vyhlazují "
    "pokožku. Vhodné pro všechny typy pleti, zejména pro suchou a zralou pleť. "
    "Aplikujte ráno a večer na očištěnou pleť před krémem. Balení 30 ml, "
    "vyrobeno v České republice. Doporučená cena 349 Kč. Produkt je dermatologicky "
    "testován a neobsahuje parabeny ani silikony. Kombinujte s pleťovým krémem "
    "Hydra Intense pro maximální účinek. Skladujte při teplotě do 25 °C."
)


# Fake entity/relace pro RE a critic prompty (musí na čem stavět).
FAKE_ENTITIES = [
    {"id": "e1", "text": "Hydra Intense", "lemma": "Hydra Intense", "category": "PRODUKT"},
    {"id": "e2", "text": "kyselina hyaluronová", "lemma": "kyselina hyaluronová", "category": "MATERIAL"},
    {"id": "e3", "text": "ACME", "lemma": "ACME", "category": "ORGANIZACE"},
]
FAKE_RELATIONS = [
    {"subjectId": "e1", "objectId": "e2", "relation": "made_of"},
]


def build_messages(mode: str, role: str = "extractor") -> list[dict]:
    """Postaví zprávy 1:1 jako produkce pro danou roli (extractor / critic /
    revise) a fázi (ner / re / combined). Když import labeler selže (chybí deps
    na hostu), fallback na inline prompt."""
    try:
        if role == "critic":
            from labeler.prompts import (
                build_critic_messages,
                build_critic_ner_messages,
                build_critic_re_messages,
            )
            if mode == "combined":
                return build_critic_messages(
                    SAMPLE_TEXT,
                    {"entities": FAKE_ENTITIES, "relations": FAKE_RELATIONS},
                )
            if mode == "re":
                return build_critic_re_messages(SAMPLE_TEXT, FAKE_RELATIONS, FAKE_ENTITIES)
            return build_critic_ner_messages(SAMPLE_TEXT, FAKE_ENTITIES)

        if role == "revise":
            from labeler.prompts import (
                build_extractor_revision_messages,
                build_revision_ner_messages,
                build_revision_re_messages,
            )
            fake_reason = "Zkontroluj kategorie a případné halucinace entit."
            if mode == "combined":
                return build_extractor_revision_messages(
                    SAMPLE_TEXT,
                    {"entities": FAKE_ENTITIES, "relations": FAKE_RELATIONS},
                    "revised", fake_reason,
                )
            if mode == "re":
                return build_revision_re_messages(
                    SAMPLE_TEXT, FAKE_RELATIONS, FAKE_ENTITIES, "revised", fake_reason
                )
            return build_revision_ner_messages(
                SAMPLE_TEXT, FAKE_ENTITIES, "revised", fake_reason
            )

        from labeler.prompts import (
            build_extractor_messages,
            build_extractor_ner_messages,
            build_extractor_re_messages,
        )
        if mode == "combined":
            return build_extractor_messages(SAMPLE_TEXT, DOMAIN)
        if mode == "re":
            return build_extractor_re_messages(SAMPLE_TEXT, DOMAIN, FAKE_ENTITIES)
        return build_extractor_ner_messages(SAMPLE_TEXT, DOMAIN)
    except Exception as e:  # pragma: no cover - fallback když labeler není import.
        print(f"[probe] (real prompt import selhal: {e!r} → inline fallback prompt)")
        sysp = (
            'Jsi extraktor entit. Vrať POUZE validní JSON '
            '{"entities":[...],"relations":[]}. Žádný další text.'
        )
        return [
            {"role": "system", "content": sysp},
            {"role": "user", "content": SAMPLE_TEXT + "\n\nExtrahuj entity jako JSON."},
        ]


def one_request(base_url, model, messages, max_tokens, timeout, req_extra=None) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    # req_extra = pole, kterými produkce posílá request (response_format,
    # chat_template_kwargs, seed). Bez nich probe netestuje skutečný tvar
    # requestu a může hlásit "OK" tam, kde produkce padá.
    if req_extra:
        payload.update(req_extra)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — chceme zachytit cokoliv (timeout/HTTP/…)
        return {"error": str(e)[:140], "dt": time.time() - t}
    dt = time.time() - t
    ch = (j.get("choices") or [{}])[0]
    out = (ch.get("message") or {}).get("content") or ""
    usage = j.get("usage") or {}
    txt = out.strip()
    ok_json = False
    if "{" in txt and "}" in txt:
        try:
            json.loads(txt[txt.find("{") : txt.rfind("}") + 1])
            ok_json = True
        except Exception:
            ok_json = False
    return {
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "len": len(out),
        "ok_json": ok_json,
        "dt": dt,
    }


def run_level(base_url, model, messages, max_tokens, n, per_level, timeout, req_extra=None) -> list[dict]:
    out = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [
            ex.submit(one_request, base_url, model, messages, max_tokens, timeout, req_extra)
            for _ in range(per_level)
        ]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Najde max stabilní souběžnost extractor / critic serveru.")
    ap.add_argument(
        "--role", choices=["extractor", "critic", "revise"], default="extractor",
        help="Kterou roli testovat. Řídí default endpoint/model, prompty, tvar "
             "requestu a max_tokens. extractor/revise: json_object + thinking off "
             "(revise = revizní prompty na revisor endpointu = defaultně extractor); "
             "critic: thinking on, bez json_object.",
    )
    ap.add_argument("--base-url", default=None, help="Default dle role: EXTRACTOR_ / CRITIC_ / REVISOR_BASE_URL.")
    ap.add_argument("--model", default=None, help="Default dle role: EXTRACTOR_ / CRITIC_ / REVISOR_MODEL.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Default dle role (extractor 4096 = rychlý proxy, critic 16384 — "
                         "thinking trace + JSON). finish_reason=length = runaway.")
    ap.add_argument("--levels", default="1,4,8,16,24,32", help="úrovně souběžnosti, čárkou")
    ap.add_argument("--per-level", type=int, default=None, help="requestů na úroveň (default = max(N,6))")
    ap.add_argument("--mode", choices=["ner", "re", "combined"], default="ner")
    ap.add_argument("--timeout", type=float, default=900.0)
    # --- Parita s produkčním requestem (jinak probe testuje jiný tvar) -------
    ap.add_argument(
        "--thinking", choices=["off", "on", "default", "auto"], default="auto",
        help="chat_template_kwargs.enable_thinking: auto = dle role (extractor "
             "off, critic on; DEFAULT), off/on = explicitně, default = kwarg "
             "vůbec neposílat. A/B test extract opravy: --thinking off vs default.",
    )
    ap.add_argument(
        "--no-guided-json", action="store_true",
        help="NEposílat response_format=json_object. Default: posílá se "
             "(parita s produkčním extractorem, reasoning=false).",
    )
    ap.add_argument(
        "--no-seed", action="store_true",
        help="NEposílat seed=42. Default: posílá se (parita s produkcí).",
    )
    ap.add_argument(
        "--repetition-penalty", type=float, default=None, metavar="P",
        help="Pošle repetition_penalty (anti-runaway). A/B test fixu: zkus "
             "--mode ner bez tohoto vs s --repetition-penalty 1.05.",
    )
    ap.add_argument(
        "--result-file", default=None, metavar="PATH",
        help="Zapíše strojově čitelný výsledek (KEY=value) — pro automatizaci "
             "(auto_tune_concurrency.sh ho přečte a zapíše do .env). Klíče: "
             "safe_concurrency, any_stable, can_go_higher, max_tested, mode, thinking.",
    )
    args = ap.parse_args()

    # Role-based defaulty: endpoint, model, max_tokens i tvar requestu se liší
    # podle toho, co testujeme. Extractor: json_object + thinking off, malý
    # proxy max_tokens. Critic: bez json_object, thinking on (critic_reasoning),
    # větší max_tokens (thinking trace + JSON; 4096 by bylo vždy "length").
    if args.role == "critic":
        base_url = args.base_url or os.environ.get("CRITIC_BASE_URL", "http://localhost:8001/v1")
        model = args.model or os.environ.get("CRITIC_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")
        max_tokens = args.max_tokens or 16384
        thinking = "on" if args.thinking == "auto" else args.thinking
        default_guided = False  # critic nepoužívá response_format=json_object
    elif args.role == "revise":
        # Revisor = defaultně tentýž endpoint/model jako extractor (config.py
        # odvozuje REVISOR_* z EXTRACTOR_*). Stejný režim (reasoning=False →
        # json_object + thinking off), jen revizní prompty.
        base_url = (
            args.base_url or os.environ.get("REVISOR_BASE_URL")
            or os.environ.get("EXTRACTOR_BASE_URL", "http://localhost:8000/v1")
        )
        model = (
            args.model or os.environ.get("REVISOR_MODEL")
            or os.environ.get("EXTRACTOR_MODEL", "RedHatAI/Qwen3.5-122B-A10B-NVFP4")
        )
        max_tokens = args.max_tokens or 4096
        thinking = "off" if args.thinking == "auto" else args.thinking
        default_guided = True
    else:
        base_url = args.base_url or os.environ.get("EXTRACTOR_BASE_URL", "http://localhost:8000/v1")
        model = args.model or os.environ.get("EXTRACTOR_MODEL", "RedHatAI/Qwen3.5-122B-A10B-NVFP4")
        max_tokens = args.max_tokens or 4096
        thinking = "off" if args.thinking == "auto" else args.thinking
        default_guided = True

    # Tvar requestu (parita s produkcí dané role). Bez toho probe měří jiný
    # request než produkce a podhodnocuje runaway.
    guided = default_guided and not args.no_guided_json
    req_extra: dict = {}
    if guided:
        req_extra["response_format"] = {"type": "json_object"}
    if thinking == "off":
        req_extra["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking == "on":
        req_extra["chat_template_kwargs"] = {"enable_thinking": True}
    # thinking == "default" → kwarg neposílat (nech šablonu rozhodnout)
    if not args.no_seed:
        req_extra["seed"] = 42
    if args.repetition_penalty is not None:
        req_extra["repetition_penalty"] = args.repetition_penalty

    # Windows konzole bývá cp1250 → vynuť UTF-8 výstup (best-effort), ať to
    # nepadá na ne-ASCII znacích při přesměrování do souboru apod.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    messages = build_messages(args.mode, args.role)
    in_chars = sum(len(m.get("content", "")) for m in messages)
    rp = f"  repetition_penalty={args.repetition_penalty}" if args.repetition_penalty is not None else ""
    print(f"[probe] role={args.role}  endpoint={base_url}  model={model}")
    print(f"[probe] mode={args.mode}  max_tokens={max_tokens}  prompt~{in_chars} znaků  úrovně={levels}")
    print(
        f"[probe] tvar requestu: response_format={'json_object' if guided else 'off'}  "
        f"enable_thinking={thinking}  seed={'—' if args.no_seed else 42}{rp}\n"
    )
    print(f"{'N':>4} {'req':>4} {'err':>4} {'length%':>8} {'badJSON%':>9} {'med_tok':>8} {'med_s':>7}  verdikt", flush=True)
    print("-" * 66, flush=True)

    safe = None
    for n in levels:
        per = args.per_level or max(n, 6)
        print(f"\rTestuji úroveň N={n} (odesílám {per} requestů)...", end="", flush=True)
        res = run_level(base_url, model, messages, max_tokens, n, per, args.timeout, req_extra)
        print("\r" + " " * 60 + "\r", end="", flush=True)
        errs = [r for r in res if "error" in r]
        ok = [r for r in res if "error" not in r]
        if not ok:
            sample_err = errs[0]["error"] if errs else "?"
            print(f"{n:>4} {len(res):>4} {len(errs):>4} {'—':>8} {'—':>9} {'—':>8} {'—':>7}  CHYBA: {sample_err}", flush=True)
            continue
        nlen = sum(1 for r in ok if r["finish_reason"] == "length")
        nbad = sum(1 for r in ok if not r["ok_json"])
        toks = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
        secs = [r["dt"] for r in ok]
        length_pct = 100 * nlen / len(ok)
        bad_pct = 100 * nbad / len(ok)
        med_tok = int(statistics.median(toks)) if toks else -1
        med_s = statistics.median(secs) if secs else 0.0
        verdict = "OK" if (length_pct == 0 and bad_pct < 10) else "RUNAWAY"
        if verdict == "OK":
            safe = n
        print(
            f"{n:>4} {len(res):>4} {len(errs):>4} {length_pct:>7.0f}% {bad_pct:>8.0f}% "
            f"{med_tok:>8} {med_s:>7.1f}  {verdict}", flush=True
        )

    print("-" * 66)
    if safe is not None and safe == levels[-1]:
        print(f"[probe] OK: stabilní i na nejvyšší testované úrovni ({safe}). Můžeš zkusit výš.")
    elif safe is not None:
        print(f"[probe] OK: nejvyšší hrubá STABILNÍ souběžnost v tomto běhu = {safe}")
    else:
        print("[probe] POZOR: RUNAWAY už na nejnižší úrovni => NENÍ to (jen) souběžnost, ale PROMPT.")
        print("[probe]     Snižování concurrency nepomůže — pošli mi tuhle tabulku, opravím prompt.")

    # Strojově čitelný výsledek pro automatizaci (auto_tune_concurrency.sh).
    if args.result_file:
        try:
            rf = Path(args.result_file)
            rf.parent.mkdir(parents=True, exist_ok=True)
            rf.write_text(
                "# auto-generated by scripts/probe_concurrency.py\n"
                f"safe_concurrency={safe if safe is not None else 0}\n"
                f"any_stable={1 if safe is not None else 0}\n"
                f"can_go_higher={1 if (safe is not None and safe == levels[-1]) else 0}\n"
                f"max_tested={levels[-1]}\n"
                f"role={args.role}\n"
                f"mode={args.mode}\n"
                f"thinking={thinking}\n",
                encoding="utf-8",
            )
            print(f"[probe] výsledek zapsán: {args.result_file}")
        except OSError as e:
            print(f"[probe] (nepodařilo se zapsat --result-file: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
