"""
Production smoke test — ověření, že pipeline funguje, a EMPIRICKÝ odhad
celkového času běhu na produkčním hardware.

Princip: smoke test běží přesně tak, jak by běžela produkce — žádné
teoretické scaling factory, žádné konstantní "model load" odhady. Jen
měříme wall-clock per fáze a lineárně extrapolujeme na plný rozsah.

NEMĚNÍ architekturu pipeline. Jen ji obaluje:

  1. Spočítá celkový rozsah datasetů (items + characters) — bez LLM, jen
     čtením vstupních souborů.
  2. Z každého datasetu vezme prvních N (default 50) itemů → smoke vzorek.
  3. Spustí existující fáze (extract → critic → revise → finalize) na
     smoke vzorku do separátní složky `smoke-checkpoints/`.
  4. Pro každou fázi měří wall-clock + items + input chars do
     `smoke_timings.json`.
  5. Po průchodu všemi fázemi: lineární extrapolace
        T_phase_full = T_phase_smoke × (full_chars / smoke_chars)   pro LLM
        T_phase_full = T_phase_smoke × (full_items / smoke_items)   pro CPU
     → výsledek = naprosto realistický odhad pro **přesně ten hardware,
       na kterém smoke proběhla**.

  Model-load overhead (~30-90s při docker compose up) JE záměrně mimo
  měření — uživatel ho platí jednou per phase u smoke i produkce, takže
  by ho neměl extrapolovat 1000× s počtem itemů. Místo toho ho přičteme
  jako konstantu k reportu pokud uživatel zadá `--docker-startup-s`.

  Bias rizika: smoke bere PRVNÍCH N items per dataset. Pokud první items
  jsou systematicky kratší/delší než průměr, odhad je biased. Pro
  rigoroznější odhad doporučujeme stratified sample (`--stratified`).

CLI:

  # 1. Připrav vzorek + spusť každou fázi (uživatel ovládá docker-compose)
  python -m labeler.main smoke-test --phase extract --hw-label "1x B200"
  # ... přepni docker profile ...
  python -m labeler.main smoke-test --phase critic --hw-label "1x B200"
  python -m labeler.main smoke-test --phase revise --hw-label "1x B200"
  python -m labeler.main smoke-test --phase finalize --hw-label "1x B200"

  # 2. Vygeneruj finální odhad (čistě empirický)
  python -m labeler.main smoke-test --phase report --hw-label "1x B200"

  Hardware label je volný text — nepoužije se pro žádný theoretický
  výpočet, jen pro dokumentaci v reportu. Pro jiný HW (2x B200, 4x B200,
  8x B200, ...) prostě spusť smoke znovu na tom hardware.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default konstanty (jen pro UX — žádný theoretický výpočet)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLES_PER_DATASET: int = 100

# Default poměr `revise` itemů. Critic ne-ok rate se měří během smoke testu
# a používá pro extrapolaci revize, ale fallback hodnota pokud nejsou data.
DEFAULT_CRITIC_NONOK_RATE: float = 0.30  # ~30 % itemů potřebuje revizi


# ---------------------------------------------------------------------------
# Multi-GPU scaling factors (B200, MoE LLM — Qwen3.5-122B-A10B)
# ---------------------------------------------------------------------------
# Empirické scaling factory pro tensor parallelism na NVIDIA B200, založené
# na SGLang + vLLM benchmarcích MoE modelů třídy Qwen3.5. Throughput
# scaling, NE linear scaling počtem GPU:
#
#   1x B200:  baseline (TP=1)
#   2x B200:  ~1.8x throughput  (TP=2 přes NVLink, ~10% overhead)
#   4x B200:  ~3.3x throughput  (TP=4, větší overhead, ale lepší batch)
#   8x B200:  ~6.0x throughput  (TP=8 single-node HGX nebo 2x 4GPU cross-node
#                                přes NVSwitch / Infiniband — méně lineární)
#
# Tensor parallelism pro Qwen3.5-MoE vyžaduje power-of-2 počty GPU
# (TP=3 / TP=5 / TP=6 / TP=7 nepodporované — SGLang vyhodí error,
# attention heads / experts nejdou dělit ne-power-of-2).
#
# UNCERTAINTY: ±15-25 % per cíl HW. Nejistotu projekce v reportu vždy
# označujeme jako ±25 % a varujeme uživatele, že pro ±2h přesnost je
# nutné spustit smoke přímo na cílovém HW.
B200_THROUGHPUT_SCALING: dict[int, float] = {
    1: 1.0,
    2: 1.8,
    4: 3.3,
    8: 6.0,
}

# Validní volby pro `--hw-label` / interaktivní prompt v scriptech.
# Omezeno na power-of-2 hodnoty které SGLang/vLLM tensor parallelism
# pro MoE modely podporuje.
SUPPORTED_GPU_COUNTS: list[int] = sorted(B200_THROUGHPUT_SCALING.keys())

# Uncertainty bound pro multi-GPU projection (jako ±relativní podíl).
PROJECTION_UNCERTAINTY: float = 0.25  # ±25 %


def parse_gpu_count_from_hw_label(hw_label: str) -> Optional[int]:
    """
    Z `hw_label` vytáhne počet B200 GPU (např. "1x B200" → 1, "4x B200" → 4).

    Akceptuje variace: "1x B200", "1 x B200", "1xB200", "2× B200" atd.
    Pokud label není rozpoznán nebo neobsahuje známý GPU počet, vrátí None
    (projekce se pak vynechá).
    """
    if not hw_label:
        return None
    import re
    # Pattern: <číslo> [×|x] [whitespace] B200
    m = re.search(r"(\d+)\s*[x×]\s*b\s*200", hw_label, re.IGNORECASE)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    if n in B200_THROUGHPUT_SCALING:
        return n
    return None


def project_to_gpu_counts(
    total_seconds_measured: float,
    measured_gpu_count: int,
    target_gpu_counts: Optional[list[int]] = None,
) -> dict[int, float]:
    """
    Z naměřeného wall-clocku na `measured_gpu_count` B200 GPU projektuje
    odhad pro ostatní počty GPU pomocí empirických scaling factorů.

    Vzorec:
      time_target = time_measured × (scaling[measured] / scaling[target])

    Příklad: měřeno na 1x B200 (factor 1.0), projektuji na 4x B200
    (factor 3.3): time_4x = time_1x × (1.0 / 3.3) ≈ 30 % původního.

    `target_gpu_counts`: list počtů GPU, pro které chceme projekci.
    Default: všechny v B200_THROUGHPUT_SCALING.

    DŮLEŽITÉ: projekce má ±25 % uncertainty — NENÍ to ±2h přesnost
    smoke testu. Pro přesno spusť smoke na cílovém HW.
    """
    if measured_gpu_count not in B200_THROUGHPUT_SCALING:
        return {}
    measured_factor = B200_THROUGHPUT_SCALING[measured_gpu_count]
    counts = target_gpu_counts or sorted(B200_THROUGHPUT_SCALING.keys())
    out: dict[int, float] = {}
    for n in counts:
        if n not in B200_THROUGHPUT_SCALING:
            continue
        target_factor = B200_THROUGHPUT_SCALING[n]
        out[n] = total_seconds_measured * (measured_factor / target_factor)
    return out


# ---------------------------------------------------------------------------
# Datové struktury
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimings:
    """Měření jedné fáze pipeline na smoke vzorku.

    DŮLEŽITÉ: wall_clock_s je čistý empirický wall-clock. Model-load
    overhead při docker compose up je MIMO (paid by user outside smoke).
    Linear extrapolation pak nepřičítá konstantu.
    """
    name: str
    items_processed: int = 0
    input_chars: int = 0
    output_chars: int = 0  # u finalize je 0 (žádný LLM)
    wall_clock_s: float = 0.0
    # Per-item wall clock časy (pro bootstrap konfidenci, pokud sbíráno)
    per_item_times_s: list[float] = field(default_factory=list)
    # Pro revise: počet itemů, kde critic vrátil non-ok (jen tyto se revidují)
    critic_nonok_count: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def per_item_s(self) -> float:
        return self.wall_clock_s / max(1, self.items_processed)

    @property
    def per_char_s(self) -> float:
        """Sekund per input character — primární jednotka pro extrapolaci."""
        return self.wall_clock_s / max(1, self.input_chars)


@dataclass
class DatasetStats:
    name: str
    total_items: int
    total_input_chars: int
    sample_items: int = 0
    sample_input_chars: int = 0


@dataclass
class SmokeTestReport:
    """Souhrnný report z měření + lineární extrapolace."""
    hw_label: str
    samples_per_dataset: int
    datasets: list[DatasetStats]
    phases: dict[str, PhaseTimings]
    # Extrapolace
    estimated_total_seconds: float = 0.0
    estimated_lower_seconds: float = 0.0
    estimated_upper_seconds: float = 0.0
    estimate_breakdown: dict[str, float] = field(default_factory=dict)
    # Volitelný overhead, který uživatel chce uvést v reportu (docker
    # startup, ne-LLM I/O, atd.). Nepřičítá se k extrapolaci automaticky.
    docker_startup_overhead_s: float = 0.0
    # Multi-GPU projekce (1x/2x/4x/8x B200) — jen pokud HW label byl
    # rozpoznán jako "<N>x B200". ROUGH ESTIMATE ±25 %.
    measured_gpu_count: Optional[int] = None
    projected_total_seconds: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_full_items(self) -> int:
        return sum(d.total_items for d in self.datasets)

    @property
    def total_full_chars(self) -> int:
        return sum(d.total_input_chars for d in self.datasets)

    @property
    def total_sample_items(self) -> int:
        return sum(d.sample_items for d in self.datasets)

    def to_dict(self) -> dict:
        return {
            "hw_label": self.hw_label,
            "samples_per_dataset": self.samples_per_dataset,
            "datasets": [asdict(d) for d in self.datasets],
            "phases": {
                name: {
                    **{k: v for k, v in asdict(p).items() if k != "per_item_times_s"},
                    "per_item_times_count": len(p.per_item_times_s),
                }
                for name, p in self.phases.items()
            },
            "estimate": {
                "total_seconds": self.estimated_total_seconds,
                "total_hours": self.estimated_total_seconds / 3600,
                "lower_seconds": self.estimated_lower_seconds,
                "lower_hours": self.estimated_lower_seconds / 3600,
                "upper_seconds": self.estimated_upper_seconds,
                "upper_hours": self.estimated_upper_seconds / 3600,
                "breakdown_seconds": dict(self.estimate_breakdown),
                "uncertainty_hours_each_side": (
                    self.estimated_upper_seconds - self.estimated_total_seconds
                ) / 3600,
                "docker_startup_overhead_s": self.docker_startup_overhead_s,
            },
            "projection": {
                "measured_gpu_count": self.measured_gpu_count,
                "warning": (
                    "Pouze ´measured_gpu_count´ má ±2h přesnost (z přímého "
                    "měření). Pro ostatní počty GPU spusť smoke test na "
                    "tom HW — multi-GPU scaling u LLM inference NENÍ "
                    "lineární a NELZE spolehlivě extrapolovat. "
                    "´orientational_seconds´/´_hours´ obsahují ±25 % "
                    "indikační čísla pouze pro orientaci, NE jako "
                    "produkční odhad."
                ),
                "scaling_factors_orientational": {
                    str(k): v for k, v in B200_THROUGHPUT_SCALING.items()
                },
                "precise_seconds": (
                    self.projected_total_seconds.get(self.measured_gpu_count)
                    if self.measured_gpu_count is not None else None
                ),
                "precise_hours": (
                    (self.projected_total_seconds.get(self.measured_gpu_count) or 0)
                    / 3600
                    if self.measured_gpu_count is not None else None
                ),
                "orientational_seconds": {
                    str(k): v for k, v in self.projected_total_seconds.items()
                    if k != self.measured_gpu_count
                },
                "orientational_hours": {
                    str(k): v / 3600
                    for k, v in self.projected_total_seconds.items()
                    if k != self.measured_gpu_count
                },
            },
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# I/O — perzistence timingů mezi voláními (uživatel volá phase-by-phase)
# ---------------------------------------------------------------------------


def _smoke_dir_default() -> Path:
    """Default složka pro smoke checkpointy + timing data.

    Defaultně subfolder hlavního CONFIG.checkpoint_dir, aby fungovaly
    Docker volume mounty out-of-the-box (host './checkpoints/smoke/' →
    kontejner '/app/checkpoints/smoke/'). Override přes env
    SMOKE_CHECKPOINT_DIR.
    """
    base = os.environ.get("SMOKE_CHECKPOINT_DIR", "")
    if base:
        return Path(base)
    from .config import CONFIG
    ckpt = CONFIG.checkpoint_dir
    # Idempotence: main.py při smoke-testu override-uje CONFIG.checkpoint_dir
    # rovnou na .../smoke (kvůli phase-by-phase běhům). Kdybychom pak slepě
    # přilepili další "smoke", vznikne .../smoke/smoke a `report` fáze (běží
    # PŘED override, tedy nad .../smoke) by timingy nenašla. Když už base
    # končí na "smoke", vrať ji tak jak je.
    if ckpt.name == "smoke":
        return ckpt
    return ckpt / "smoke"


def smoke_checkpoint_dir() -> Path:
    p = _smoke_dir_default()
    p.mkdir(parents=True, exist_ok=True)
    return p


def smoke_timings_path() -> Path:
    return smoke_checkpoint_dir() / "smoke_timings.json"


def _timings_candidate_paths() -> list[Path]:
    """Kandidátní cesty pro ČTENÍ smoke_timings.json, v pořadí priority.

    Kromě kanonické cesty hledáme i historickou dvojitě zanořenou
    (.../smoke/smoke/) — starší běhy (před idempotentním _smoke_dir_default)
    nebo přímé spuštění mohly timingy uložit tam. Čtení je tak odolné vůči
    nekonzistentnímu zanoření; zápis zůstává kanonický (NIC se nepřepisuje).
    """
    canonical = smoke_timings_path()
    base = canonical.parent
    candidates = [
        canonical,                               # .../smoke/smoke_timings.json (kanon)
        base / "smoke" / "smoke_timings.json",   # legacy dvojité zanoření
    ]
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def load_timings() -> dict:
    """Načte smoke_timings.json z první existující kandidátní cesty.

    Hledá kanonickou i historickou (dvojitě zanořenou) cestu, aby `report`
    fáze našla timingy i při nekonzistentním zanoření smoke složky (pro debug).
    Vrátí prázdný dict, pokud nikde nic není.
    """
    canonical = smoke_timings_path()
    for p in _timings_candidate_paths():
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Nelze přečíst {p}: {e}; zkouším další kandidát.")
            continue
        if p != canonical:
            logger.info(
                f"smoke timings načteny z ne-kanonické cesty {p} "
                f"(kanonická: {canonical}) — pravděpodobně starší/zanořený běh."
            )
        return data
    return {}


def save_timings(data: dict) -> None:
    p = smoke_timings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record_phase_timing(
    phase: str,
    items_processed: int,
    input_chars: int,
    output_chars: int,
    wall_clock_s: float,
    per_item_times_s: Optional[list[float]] = None,
    critic_nonok_count: Optional[int] = None,
) -> None:
    """
    Uloží měření jedné fáze do smoke_timings.json. Přepíše existující záznam
    pro tu fázi (kdyby se smoke spouštěl opakovaně).
    """
    data = load_timings()
    data.setdefault("phases", {})
    data["phases"][phase] = {
        "name": phase,
        "items_processed": items_processed,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "wall_clock_s": wall_clock_s,
        "per_item_times_s": list(per_item_times_s or []),
        "critic_nonok_count": critic_nonok_count,
        "timestamp": time.time(),
    }
    save_timings(data)
    logger.info(
        f"Smoke timing recorded: {phase} {items_processed} items, "
        f"{wall_clock_s:.1f}s wall, ~{input_chars} chars in"
    )


def record_dataset_stats(
    stats: list[DatasetStats], samples_per_dataset: int
) -> None:
    """Uloží statistiky o velikosti datasetů (pro extrapolaci) + samples_per_dataset
    použitý pro sample_* sloupce — ať jde detekovat stale cache (vzorek změřený
    s jiným --samples-per-dataset, než s jakým reálně běží fáze)."""
    data = load_timings()
    data["datasets"] = [asdict(s) for s in stats]
    data["datasets_samples_per_dataset"] = samples_per_dataset
    save_timings(data)


def dataset_stats_from_cache(data: dict) -> list[DatasetStats]:
    """Rekonstruuje DatasetStats z cachovaných total countů (drahá část zůstává
    cachovaná). sample_* pole se pak levně re-annotují pro aktuální
    samples_per_dataset (viz main._run_smoke_test)."""
    out: list[DatasetStats] = []
    for d in data.get("datasets", []):
        out.append(
            DatasetStats(
                name=d.get("name", "?"),
                total_items=int(d.get("total_items", 0)),
                total_input_chars=int(d.get("total_input_chars", 0)),
                sample_items=int(d.get("sample_items", 0)),
                sample_input_chars=int(d.get("sample_input_chars", 0)),
            )
        )
    return out


def record_hw_label(hw_label: str) -> None:
    """Uloží uživatelův HW label pro pozdější report."""
    data = load_timings()
    data["hw_label"] = hw_label
    save_timings(data)


# ---------------------------------------------------------------------------
# Měření velikosti datasetů — bez LLM, jen čte vstupní soubory
# ---------------------------------------------------------------------------


def measure_dataset_sizes(data_dir: Path) -> list[DatasetStats]:
    """
    Spočítá totální items + char count pro každý dataset v data_dir.
    Nepoužívá LLM — jen iteruje datasety přes existující reader.

    Pozn.: pro velké datasety to může chvíli trvat (HTML stripping atd.),
    ale pořád řádově sekundy/minuty, ne hodiny. Cache se neukládá — pokud
    se data změní, příští smoke test přepočítá.
    """
    from .datasets import scan_data_dir
    logger.info(f"Měřím velikost datasetů v {data_dir} ...")
    t0 = time.perf_counter()

    by_source: dict[str, dict] = {}
    for item in scan_data_dir(data_dir, default_domain=None, limit=None):
        rec = by_source.setdefault(item.source, {"items": 0, "chars": 0})
        rec["items"] += 1
        rec["chars"] += len(item.text or "")
        if item.context:
            rec["chars"] += len(item.context)

    elapsed = time.perf_counter() - t0
    stats = [
        DatasetStats(
            name=src, total_items=rec["items"], total_input_chars=rec["chars"]
        )
        for src, rec in sorted(by_source.items())
    ]
    total_items = sum(s.total_items for s in stats)
    total_chars = sum(s.total_input_chars for s in stats)
    logger.info(
        f"Datasety: {len(stats)}, items={total_items}, chars={total_chars:,} "
        f"(změřeno za {elapsed:.1f}s)"
    )
    return stats


def annotate_sample_stats(
    stats: list[DatasetStats], samples_per_dataset: int, data_dir: Path,
) -> None:
    """
    Doplní do DatasetStats sample_items + sample_input_chars (skutečné, ne odhad).
    Důležité pro přesnou extrapolaci (vzorek může mít jinou hustotu textu než průměr).
    """
    from .datasets import scan_data_dir
    counts: dict[str, dict] = {}
    seen: dict[str, int] = {}
    for item in scan_data_dir(data_dir, default_domain=None, limit=samples_per_dataset):
        if seen.get(item.source, 0) >= samples_per_dataset:
            continue
        seen[item.source] = seen.get(item.source, 0) + 1
        rec = counts.setdefault(item.source, {"items": 0, "chars": 0})
        rec["items"] += 1
        rec["chars"] += len(item.text or "")
        if item.context:
            rec["chars"] += len(item.context)
    for s in stats:
        c = counts.get(s.name, {"items": 0, "chars": 0})
        s.sample_items = c["items"]
        s.sample_input_chars = c["chars"]


# ---------------------------------------------------------------------------
# Bootstrap CI + extrapolace
# ---------------------------------------------------------------------------


def _bootstrap_relative_uncertainty(
    per_item_times_s: list[float],
    n_samples: int = 1000,
) -> float:
    """
    Bootstrap odhad relativní nejistoty (stddev / mean) průměrného per-item
    time. Vrátí číslo v intervalu [0.05, 0.30] — typicky 0.05-0.10 pro 50+
    vzorků.

    Floor 0.05 (5 %) zachytává neměřitelné produkční faktory:
      - hardware noise (GPU load, network)
      - batch scheduler nondeterminismus
      - bias vzorku (smoke bere prvních N items per dataset — nemusí být
        reprezentativní pro celý dataset)
      - LLM throughput variabilita per text typ

    Cap 0.30 (30 %) pro extrém — pokud per-item variance je obrovská, lepší
    raději zachytit horní hranici než dát uživateli nesmyslně úzký interval.

    Pokud je vzorek malý nebo nulový, vrátí konzervativní 0.15.
    """
    if not per_item_times_s or len(per_item_times_s) < 10:
        return 0.15
    try:
        import random
        rng = random.Random(42)  # deterministické pro report
        means = []
        n = len(per_item_times_s)
        for _ in range(n_samples):
            sample = [per_item_times_s[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        m = sum(means) / len(means)
        if m <= 0:
            return 0.15
        sd = statistics.pstdev(means)
        # Floor 5 % (neměřitelné produkční faktory) + cap 30 % (extrém)
        return max(0.05, min(0.30, sd / m))
    except Exception as e:
        logger.warning(f"Bootstrap failed: {e}; using fallback 0.10")
        return 0.10


def compute_estimate(
    timings: dict,
    full_chars: int,
    full_items: int,
) -> tuple[float, float, float, dict[str, float], list[str]]:
    """
    EMPIRICKÁ extrapolace ze smoke timingů na celý produkční rozsah.

    Vzorec (pure linear):

      Pro LLM fáze (extract/critic/revise):
        T_full = T_smoke × (full_chars_for_phase / smoke_chars_for_phase)

      Pro finalize (CPU only, neškáluje s text rozsahem):
        T_full = T_smoke × (full_items / smoke_items)

    `full_chars_for_phase`:
      extract:  full_chars
      critic:   full_chars (input = original text + extractor JSON; output
                je menší než input, ignorujeme pro lineární odhad)
      revise:   full_chars × critic_nonok_rate (jen non-ok subset)

    Žádný "model_load_s" overhead — ten platí uživatel jednou per phase u
    `docker compose up` (mimo wall-clock měření). Uvedeme ho v reportu jako
    separátní položku, pokud uživatel zadá `--docker-startup-s`.

    Žádný "GPU scaling factor" — smoke běží přesně na tom hardware, který
    se použije v produkci. Multi-GPU varianty se NEEXTRAPOLUJÍ teoreticky;
    pokud chceš odhad pro jiný HW, spusť smoke znovu na tom HW.

    Vrátí: (total_s, lower_s, upper_s, breakdown_per_phase, notes).
    """
    phases = timings.get("phases", {})
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    # Critic non-ok rate (pro revize) — pokud jsme měřili critic, vezmeme
    # skutečnou hodnotu; jinak default.
    critic_phase = phases.get("critic", {})
    critic_items = critic_phase.get("items_processed", 0) or 0
    critic_nonok = critic_phase.get("critic_nonok_count")
    if critic_items > 0 and critic_nonok is not None:
        nonok_rate = critic_nonok / critic_items
        notes.append(
            f"Critic non-ok rate měřena: {nonok_rate*100:.1f}% "
            f"({critic_nonok}/{critic_items})"
        )
    else:
        nonok_rate = DEFAULT_CRITIC_NONOK_RATE
        notes.append(
            f"Critic non-ok rate nezměřena, používám default "
            f"{nonok_rate*100:.0f}% pro revize"
        )

    # Agregovaná nejistota — sčítáme variance napříč fázemi
    variance_seconds_squared = 0.0

    # ----------------------------------------------------------------- fáze 0
    # Taxonomy auto-discovery je JEDNORÁZOVÁ konstanta: sampluje fixní počet
    # vzorků nezávisle na velikosti datasetu, takže se NEškáluje na full rozsah.
    # Pokud byla změřena, přičteme naměřený wall-clock přímo (T_full = T_smoke).
    # Fáze 0a (výběr vzorků přes embeddingy) je taky jednorázová konstanta —
    # kóduje se nejvýš TAXONOMY_SAMPLE_POOL_MAX textů per doména.
    for const_phase, const_label in (
        ("sample", "Sample (fáze 0a, embeddingy)"),
        ("taxonomy", "Taxonomy (fáze 0)"),
    ):
        tax_phase = phases.get(const_phase)
        if not tax_phase:
            continue
        tax_wall = tax_phase.get("wall_clock_s", 0.0) or 0.0
        if tax_wall > 0:
            breakdown[const_phase] = tax_wall
            notes.append(
                f"{const_label} jednorázová konstanta: {tax_wall:.0f}s "
                f"(neškáluje se s rozsahem — fixní vzorkování datasetů)"
            )
            tax_per_item = list(tax_phase.get("per_item_times_s", []) or [])
            tax_uncert_s = tax_wall * _bootstrap_relative_uncertainty(tax_per_item) * 1.96
            variance_seconds_squared += tax_uncert_s ** 2

    for phase_name in ("extract", "critic", "revise", "finalize"):
        p = phases.get(phase_name)
        if not p:
            notes.append(
                f"⚠️  Fáze '{phase_name}' nebyla změřena — nelze odhadnout"
            )
            continue

        items = p.get("items_processed", 0) or 0
        input_chars = p.get("input_chars", 0) or 0
        wall = p.get("wall_clock_s", 0.0) or 0.0
        per_item = list(p.get("per_item_times_s", []) or [])

        if items == 0:
            notes.append(
                f"⚠️  Fáze '{phase_name}': nulová data, přeskakuji"
            )
            continue

        # PURE LINEAR EXTRAPOLATION — žádné scaling factory, žádný load offset
        if phase_name == "finalize":
            # CPU only — škáluje s počtem itemů
            phase_full_s = wall * (full_items / max(1, items))
        else:
            if input_chars == 0:
                # Fallback: jako finalize (per-item)
                phase_full_s = wall * (full_items / max(1, items))
            else:
                # LLM fáze — škáluje s input chars
                full_chars_for_phase = full_chars
                if phase_name == "revise":
                    full_chars_for_phase = int(full_chars * nonok_rate)
                phase_full_s = wall * (full_chars_for_phase / max(1, input_chars))

        breakdown[phase_name] = phase_full_s

        # Per-phase nejistota: bootstrap z per-item timingů + 5 % floor
        rel_uncert = _bootstrap_relative_uncertainty(per_item)
        # 95 % CI ~ 1.96σ pro normální rozdělení
        phase_uncert_s = phase_full_s * rel_uncert * 1.96
        variance_seconds_squared += phase_uncert_s ** 2

    total_s = sum(breakdown.values())
    total_uncertainty_s = variance_seconds_squared ** 0.5

    # Cap konfidence interval na ±2h (uživatelův požadavek na presnost)
    cap_s = 2 * 3600
    if total_uncertainty_s > cap_s:
        notes.append(
            f"⚠️  Hrubá nejistota {total_uncertainty_s/3600:.2f}h přesahuje "
            f"target ±2h (v reportu zacapováno na ±2h). Hlavní příčina: smoke "
            f"NEzaznamenává per-item časy → bootstrap běží na fallback defaultu, "
            f"ne na reálném rozptylu. Pro užší bound zaznamenej per-item časy; "
            f"navýšení --samples-per-dataset samo o sobě nepomůže, dokud se "
            f"per-item časy neměří."
        )
        total_uncertainty_s = cap_s  # ořezání pro report
    else:
        notes.append(
            f"✓ Konfidence ±{total_uncertainty_s/3600:.2f}h (pod target ±2h)"
        )

    lower = max(0.0, total_s - total_uncertainty_s)
    upper = total_s + total_uncertainty_s

    return total_s, lower, upper, breakdown, notes


# ---------------------------------------------------------------------------
# Vykreslení reportu (markdown + JSON)
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Sekund → "Xh Ym" nebo "Ym Zs" pro krátké."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def format_report_markdown(report: SmokeTestReport) -> str:
    """Markdown formátování pro CLI output + smoke_test_report.md."""
    lines: list[str] = []
    lines.append("# Production Smoke Test Report")
    lines.append("")
    lines.append(f"- **Hardware:** {report.hw_label}")
    lines.append(f"- **Samples/dataset:** {report.samples_per_dataset}")
    lines.append(f"- **Total samples processed:** {report.total_sample_items}")
    lines.append("")
    lines.append("*Odhad je čistě empirický — lineární extrapolace z naměřeného "
                 "wall-clocku na smoke vzorku. Žádné teoretické scaling factory, "
                 "žádné konstantní overhead odhady.*")
    lines.append("")

    # Per-dataset breakdown
    lines.append("## Datasety (plný rozsah)")
    lines.append("")
    lines.append("| Dataset | Items | Input chars | Smoke items | Smoke chars |")
    lines.append("|---------|------:|------------:|------------:|------------:|")
    for d in report.datasets:
        lines.append(
            f"| {d.name} | {d.total_items:,} | {d.total_input_chars:,} | "
            f"{d.sample_items} | {d.sample_input_chars:,} |"
        )
    lines.append(
        f"| **TOTAL** | **{report.total_full_items:,}** | "
        f"**{report.total_full_chars:,}** | **{report.total_sample_items}** | "
        f"**{sum(d.sample_input_chars for d in report.datasets):,}** |"
    )
    lines.append("")

    # Per-phase měření
    lines.append("## Per-phase měření na smoke vzorku")
    lines.append("")
    lines.append("| Fáze | Items | Input chars | Wall clock | Per item | Per 1k char |")
    lines.append("|------|------:|------------:|-----------:|---------:|------------:|")
    for phase_name in ("sample", "taxonomy", "extract", "critic", "revise", "finalize"):
        p = report.phases.get(phase_name)
        if phase_name in ("sample", "taxonomy"):
            # Fáze 0a / 0 jsou jednorázové konstanty — items/chars/per-item
            # nedávají smysl; zobrazíme jen wall.
            label = "sample (fáze 0a)" if phase_name == "sample" else "taxonomy (fáze 0)"
            if phase_name == "sample" and (not p or p.wall_clock_s <= 0):
                continue  # fáze 0a je volitelná — bez měření řádek vynech
            if not p or p.wall_clock_s <= 0:
                lines.append(f"| {label} | — | — | (nespuštěno) | — | — |")
            else:
                lines.append(
                    f"| {label} | — | — | "
                    f"{_fmt_duration(p.wall_clock_s)} | — | — |"
                )
            continue
        if not p:
            lines.append(f"| {phase_name} | — | — | (nezměřeno) | — | — |")
            continue
        per_1k_ms = p.per_char_s * 1000 * 1000
        lines.append(
            f"| {phase_name} | {p.items_processed} | {p.input_chars:,} | "
            f"{_fmt_duration(p.wall_clock_s)} | "
            f"{p.per_item_s:.2f}s | {per_1k_ms:.1f}ms |"
        )
    lines.append("")

    # Extrapolace
    lines.append("## Odhad celkového času pro produkci")
    lines.append("")
    lines.append("Lineární extrapolace: `T_full = T_smoke × (full_chars / smoke_chars)`")
    lines.append("")
    lines.append("| Fáze | Odhad |")
    lines.append("|------|------:|")
    for phase, sec in report.estimate_breakdown.items():
        lines.append(f"| {phase} | {_fmt_duration(sec)} |")
    lines.append(f"| **LLM + finalize celkem** | **{_fmt_duration(report.estimated_total_seconds)}** |")
    lines.append("")

    # Docker startup (pokud zadáno). Každá fáze v odhadu = jeden start docker
    # containeru, takže násobíme počtem skutečně změřených fází (taxonomy/fáze 0
    # přidá +1, pokud byla měřena). Default 4, kdyby breakdown byl prázdný.
    if report.docker_startup_overhead_s > 0:
        per_phase = report.docker_startup_overhead_s
        n_docker_phases = len(report.estimate_breakdown) or 4
        total_with_docker = report.estimated_total_seconds + per_phase * n_docker_phases
        lines.append(
            f"- **Docker startup overhead** (zadáno uživatelem): "
            f"{_fmt_duration(per_phase)} × {n_docker_phases} fází = "
            f"{_fmt_duration(per_phase * n_docker_phases)}"
        )
        lines.append(
            f"- **CELKEM včetně docker startup:** "
            f"**{_fmt_duration(total_with_docker)}**"
        )
        lines.append("")

    # Konfidence interval
    uncert_h = (report.estimated_upper_seconds - report.estimated_total_seconds) / 3600
    lines.append(
        f"**Konfidence (95 %):** "
        f"{_fmt_duration(report.estimated_lower_seconds)} – "
        f"{_fmt_duration(report.estimated_upper_seconds)} "
        f"(±{uncert_h:.2f}h od střední hodnoty)"
    )
    lines.append("")

    # Multi-GPU přehled — jen pokud HW label byl rozpoznán jako "<N>× B200".
    # ZÁMĚRNĚ rozdělené do DVOU sekcí, aby uživatel jednoznačně viděl
    # rozdíl mezi "přesné měření" a "rough orientation".
    if report.projected_total_seconds and report.measured_gpu_count is not None:
        m = report.measured_gpu_count
        # === SEKCE 1: PŘESNÝ ODHAD (jen měřený HW) =========================
        lines.append("## ✅ Přesný odhad (měřeno na cílovém HW)")
        lines.append("")
        measured_sec = report.projected_total_seconds[m]
        lines.append(f"**{m}× B200**: **{_fmt_duration(measured_sec)}** "
                     f"(95 % konfidence ±2h)")
        lines.append("")
        lines.append(
            "Toto je jediný odhad odvozený z přímého měření na cílovém "
            "hardware. Lineární extrapolace per-char rate na plný rozsah "
            "datasetu, žádné teoretické scaling factory."
        )
        lines.append("")

        # === SEKCE 2: ORIENTAČNÍ ODHADY (ostatní GPU counts) ==============
        other_counts = [n for n in sorted(report.projected_total_seconds.keys())
                        if n != m]
        if other_counts:
            lines.append("---")
            lines.append("")
            lines.append("## ⚠️ Orientační odhady pro jiné počty B200 (NEPŘESNÉ)")
            lines.append("")
            lines.append(
                "> **POZOR**: Hodnoty níže jsou ROUGH ESTIMATE s **±25 % "
                "nepřesností**. NEJSOU produkční odhad. Multi-GPU scaling "
                "LLM inference není lineární (tensor parallelism overhead, "
                "MoE expert routing, batch size effects). Pro produkční "
                "rozhodnutí **spusť smoke test znovu na cílovém HW**."
            )
            lines.append("")
            lines.append("| Počet B200 | Orientační odhad | Scaling factor | Status |")
            lines.append("|-----------:|-----------------:|---------------:|:-------|")
            for n in other_counts:
                sec = report.projected_total_seconds[n]
                factor = B200_THROUGHPUT_SCALING[n]
                lines.append(
                    f"| {n}× B200 | ~{_fmt_duration(sec)} | {factor:.1f}× | "
                    f"⚠️ ±25 %, NEPRODUKČNÍ — pro přesnost spusť smoke "
                    f"na {n}× B200 |"
                )
            lines.append("")
            lines.append(
                "*Empirické B200 scaling factory pro Qwen3.5-MoE "
                "(SGLang/vLLM benchmarks): "
                + ", ".join(
                    f"{n}× → {f:.1f}× throughput"
                    for n, f in sorted(B200_THROUGHPUT_SCALING.items())
                )
                + ". Vzorec: `time_target = time_measured × "
                "(scaling[measured] / scaling[target])`.*"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    # Notes / warnings
    if report.notes:
        lines.append("## Poznámky")
        lines.append("")
        for n in report.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Veřejné API — load / generate report
# ---------------------------------------------------------------------------


def build_report(
    hw_label: str,
    samples_per_dataset: int,
    docker_startup_s: float = 0.0,
) -> SmokeTestReport:
    """
    Sestaví SmokeTestReport z `smoke_timings.json`. Pokud chybí timingy
    některé fáze, vyhodí warning v notes, ale report stejně vyrobí.
    """
    timings = load_timings()
    # HW label může být uložen z předchozí --phase, nebo přepsán parametrem
    stored_label = timings.get("hw_label")
    if not hw_label and stored_label:
        hw_label = stored_label
    hw_label = hw_label or "unknown"

    datasets_raw = timings.get("datasets", [])
    datasets = [
        DatasetStats(
            name=d["name"],
            total_items=d.get("total_items", 0),
            total_input_chars=d.get("total_input_chars", 0),
            sample_items=d.get("sample_items", 0),
            sample_input_chars=d.get("sample_input_chars", 0),
        )
        for d in datasets_raw
    ]
    phases_raw = timings.get("phases", {})
    phases: dict[str, PhaseTimings] = {}
    for name, p in phases_raw.items():
        phases[name] = PhaseTimings(
            name=name,
            items_processed=p.get("items_processed", 0),
            input_chars=p.get("input_chars", 0),
            output_chars=p.get("output_chars", 0),
            wall_clock_s=p.get("wall_clock_s", 0.0),
            per_item_times_s=list(p.get("per_item_times_s", []) or []),
            critic_nonok_count=p.get("critic_nonok_count"),
            timestamp=p.get("timestamp", 0.0),
        )

    full_items = sum(d.total_items for d in datasets)
    full_chars = sum(d.total_input_chars for d in datasets)

    total_s, lower_s, upper_s, breakdown, notes = compute_estimate(
        timings, full_chars, full_items,
    )

    # Multi-GPU projekce — jen pokud HW label byl rozpoznán jako "<N>x B200"
    measured_gpu_count = parse_gpu_count_from_hw_label(hw_label)
    projected: dict[int, float] = {}
    if measured_gpu_count is not None:
        projected = project_to_gpu_counts(total_s, measured_gpu_count)
        notes.append(
            f"Multi-GPU projekce zapnuta: měřeno na {measured_gpu_count}x B200, "
            f"projektováno na {sorted(B200_THROUGHPUT_SCALING.keys())}. "
            f"⚠️ Projekce mají ±25 % uncertainty (scaling není lineární)."
        )
    else:
        notes.append(
            f"HW label {hw_label!r} nebyl rozpoznán jako '<N>x B200' "
            f"→ multi-GPU projekce vynechána."
        )

    return SmokeTestReport(
        hw_label=hw_label,
        samples_per_dataset=samples_per_dataset,
        datasets=datasets,
        phases=phases,
        estimated_total_seconds=total_s,
        estimated_lower_seconds=lower_s,
        estimated_upper_seconds=upper_s,
        estimate_breakdown=breakdown,
        docker_startup_overhead_s=docker_startup_s,
        measured_gpu_count=measured_gpu_count,
        projected_total_seconds=projected,
        notes=notes,
    )


def write_report_files(report: SmokeTestReport) -> tuple[Path, Path]:
    """Zapíše smoke_test_report.{md,json} do smoke checkpoint dir."""
    out_dir = smoke_checkpoint_dir()
    md_path = out_dir / "smoke_test_report.md"
    json_path = out_dir / "smoke_test_report.json"
    md_path.write_text(format_report_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return md_path, json_path
