"""
I/O utility: JSONL append-only writer, checkpoint loader, hash dedup.

Klíčový design:
- Výstup je JSONL (jeden JSON na řádek) → atomicky rozšiřitelný, streamovatelný.
- Po každém itemu se fsync-ne → crash safety.
- Při startu se načte existující JSONL do setu zpracovaných hashů → resume.
- Identické vstupy (stejný hash) se labelují jen jednou → dedup.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    """Deterministic SHA-256 hash textu (prvních 16 znaků = 64 bitů, stačí na dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> Iterator[dict]:
    """Líný iterator přes JSONL soubor. Přeskočí prázdné řádky a nevaliadní JSON (s warningem)."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{path}:{lineno} nevalidní JSON, přeskakuji: {e}")


def load_completed_ids(path: Path, id_key: str = "item_id") -> set[str]:
    """Načte množinu item_id z již existujícího JSONL (pro resume)."""
    if not path.exists():
        return set()
    ids = set()
    for rec in read_jsonl(path):
        v = rec.get(id_key)
        if v:
            ids.add(v)
    logger.info(f"Načteno {len(ids)} již zpracovaných záznamů z {path}")
    return ids


def load_hash_cache(path: Path, hash_key: str = "input_hash") -> dict[str, dict]:
    """
    Načte mapu hash → první záznam s tímto hashem.
    Umožňuje znovu použít výsledek pro identické vstupy bez volání LLM.
    """
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for rec in read_jsonl(path):
        h = rec.get(hash_key)
        if h and h not in cache:
            cache[h] = rec
    logger.info(f"Hash cache: {len(cache)} unikátních vstupů z {path}")
    return cache


class JsonlWriter:
    """
    Thread-safe append-only JSONL writer s fsync po každém záznamu.

    Použití:
        with JsonlWriter(path) as w:
            w.write(record_dict)
            w.write(pydantic_model)
    """

    def __init__(self, path: Path, fsync: bool = True):
        self.path = path
        self._fsync = fsync
        self._lock = threading.Lock()
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode - zachovává existující obsah pro resume
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        logger.info(f"Otevřen JSONL pro append: {self.path}")
        return self

    def __exit__(self, *args) -> None:
        if self._fh:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def write(self, record: Any) -> None:
        if isinstance(record, BaseModel):
            data = record.model_dump(mode="json")
        elif isinstance(record, dict):
            data = record
        else:
            raise TypeError(f"Nelze serializovat typ {type(record)}")
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            assert self._fh
            self._fh.write(line)
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())


def _format_duration(seconds: float) -> str:
    """Naformátuje trvání jako 'Xh Ymin', 'Ymin', nebo 'Xs' (pro krátké úseky)."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}min {s}s" if s else f"{m}min"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}min" if m else f"{h}h"


class ProgressTracker:
    """
    Počítadlo pro logování postupu fáze.

    Při každém log-tiku zobrazí:
      - počet/total (% dokončeno)
      - errors / skipped
      - elapsed (jak dlouho fáze běží)
      - ETA (odhad zbývajícího času ve formátu 'Xh Ymin')

    ETA se počítá z průměrného času na _reálně zpracovaný_ item (skipped =
    cache hits zdarma, ty se do průměru nezapočítávají, jinak by byl
    nadhodnocený).
    """

    def __init__(
        self,
        total: int,
        label: str,
        log_every_n: Optional[int] = None,
    ):
        self.total = total
        self.label = label
        # Adaptivni log frekvence: cca 25 log radku za fazi
        # (3 itemy → log every 1; 60000 itemů → log every 2400, capped at 1000).
        # Override přes env PROGRESS_LOG_EVERY (např. =1 → progress po KAŽDÉM
        # itemu, pro plnou viditelnost při ladění). Pozn.: chyby jednotlivých
        # itemů (runaway / nevalidní JSON) se logují VŽDY zvlášť, nezávisle na
        # téhle souhrnné frekvenci.
        if log_every_n is None:
            env_every = os.environ.get("PROGRESS_LOG_EVERY", "").strip()
            if env_every.isdigit() and int(env_every) > 0:
                log_every_n = int(env_every)
            else:
                # Změněno z 1000 na 50, aby se progress u velkých datasetů logoval 
                # podstatně častěji (každých 50 zpracovaných položek).
                log_every_n = max(1, min(50, total // 25)) if total > 0 else 1
        self.log_every_n = log_every_n
        self.done = 0
        self.errors = 0
        self.skipped = 0
        self._start_time = time.monotonic()
        if total > 0:
            logger.info(
                f"[{label}] start: {total} itemů ke zpracování (souhrnný progress "
                f"+ ETA každých {log_every_n} itemů a na konci; chyby itemů se "
                f"logují vždy zvlášť — PROGRESS_LOG_EVERY=1 pro per-item progress)"
            )

    def tick(self, *, error: bool = False, skipped: bool = False) -> None:
        self.done += 1
        if error:
            self.errors += 1
        if skipped:
            self.skipped += 1
        if self.done % self.log_every_n == 0 or self.done == self.total:
            self._log_progress()

    def elapsed_seconds(self) -> float:
        """Celkový čas běhu fáze (sekundy)."""
        return time.monotonic() - self._start_time

    def summary(self) -> str:
        """
        Vrátí jednořádkové shrnutí fáze pro finální log.
        Obsahuje: done/total, errors, skipped, total elapsed time, throughput.
        """
        elapsed = self.elapsed_seconds()
        elapsed_str = _format_duration(elapsed)
        processed = self.done - self.skipped
        rate_str = ""
        if processed > 0 and elapsed > 0:
            rate_per_min = (processed / elapsed) * 60
            rate_str = f", {rate_per_min:.1f} it/min"
        return (
            f"[{self.label}] FINISHED: {self.done}/{self.total} "
            f"(errors={self.errors}, skipped={self.skipped}) "
            f"total time {elapsed_str}{rate_str}"
        )

    def _log_progress(self) -> None:
        pct = 100 * self.done / max(self.total, 1)
        elapsed = time.monotonic() - self._start_time
        elapsed_str = _format_duration(elapsed)
        # Reálně zpracované itemy (bez cache hitů) → poctivý průměr
        processed = self.done - self.skipped
        if processed > 0 and elapsed > 0:
            rate_per_s = processed / elapsed
            remaining = self.total - self.done
            # Pokud uz vse hotovo, ETA = 0
            eta_s = remaining / rate_per_s if rate_per_s > 0 and remaining > 0 else 0
            eta_str = _format_duration(eta_s)
            rate_per_min = rate_per_s * 60
            logger.info(
                f"[{self.label}] {self.done}/{self.total} ({pct:.1f}%) "
                f"errors={self.errors} skipped={self.skipped} "
                f"elapsed={elapsed_str} ETA={eta_str} ({rate_per_min:.1f} it/min)"
            )
        else:
            # Vsechny itemy byly skipped (cache hit) nebo neuplynul cas
            logger.info(
                f"[{self.label}] {self.done}/{self.total} ({pct:.1f}%) "
                f"errors={self.errors} skipped={self.skipped} "
                f"elapsed={elapsed_str}"
            )


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomický zápis JSON (přes tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    """
    Atomický zápis JSONL (přes tmp + rename).

    Na rozdíl od JsonlWriter (append-only) plně přepíše soubor jedním tahem —
    vhodné pro post-processing, kdy se aktualizují existující záznamy
    (např. revise.py doplní extractor_revised do critic_results.jsonl) nebo
    finálních merge výstupů.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
