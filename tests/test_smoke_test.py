"""
Unit testy pro labeler.smoke_test.

Pokrývá pure empirickou extrapolaci:
  - compute_estimate() — lineární extrapolace per-char rate × full chars,
    žádné teoretické scaling factory
  - critic non-ok rate měřený z dat (s fallback default)
  - missing phase fallback (graceful degradation)
  - Konfidence interval — bootstrap z per-item timingů, 5 % floor, ±2h cap
  - Bootstrap relative uncertainty
  - Round-trip přes save/load timings
  - Markdown report obsahuje všechny klíčové sekce

Pipeline funkcí (extract/critic/revise/finalize) se ve smoke testu NETESTUJE
přímo — to vyžaduje běžící LLM servery. Testujeme jen čistě výpočetní vrstvu
extrapolace, kterou jde validovat na syntetických datech.

Spuštění: python tests/test_smoke_test.py  (nebo pytest)
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.smoke_test import (
    B200_THROUGHPUT_SCALING,
    DEFAULT_CRITIC_NONOK_RATE,
    DEFAULT_SAMPLES_PER_DATASET,
    DatasetStats,
    PhaseTimings,
    PROJECTION_UNCERTAINTY,
    SmokeTestReport,
    _bootstrap_relative_uncertainty,
    build_report,
    compute_estimate,
    format_report_markdown,
    load_timings,
    parse_gpu_count_from_hw_label,
    project_to_gpu_counts,
    record_hw_label,
    save_timings,
    smoke_checkpoint_dir,
    smoke_timings_path,
)


# === Bootstrap uncertainty ================================================

def test_bootstrap_empty_returns_fallback():
    # Žádné měření → fallback 0.15
    assert _bootstrap_relative_uncertainty([]) == 0.15
    assert _bootstrap_relative_uncertainty([1.0] * 5) == 0.15  # < 10 vzorků


def test_bootstrap_constant_values_at_floor():
    # Konstantní vstup → bootstrap dá 0, ale floor 5 % se uplatní
    times = [5.0] * 100
    rel = _bootstrap_relative_uncertainty(times)
    assert abs(rel - 0.05) < 1e-6, f"Expected 5% floor, got {rel}"


def test_bootstrap_variable_values_within_bounds():
    import random
    rng = random.Random(123)
    times = [rng.uniform(2.0, 8.0) for _ in range(200)]
    rel = _bootstrap_relative_uncertainty(times)
    assert 0.05 <= rel <= 0.30  # floor a cap


# === compute_estimate — pure empirical ===================================

def _make_timings(
    *,
    extract_wall: float = 600.0,
    extract_items: int = 200,
    extract_chars: int = 1_000_000,
    critic_wall: float = 400.0,
    critic_items: int = 200,
    critic_chars: int = 2_000_000,
    critic_nonok: int = 60,
    revise_wall: float = 200.0,
    revise_items: int = 60,
    revise_chars: int = 600_000,
    finalize_wall: float = 10.0,
    finalize_items: int = 200,
    finalize_chars: int = 1_000_000,
    per_item_var: float = 0.0,
) -> dict:
    """Postaví syntetický smoke_timings.json dict pro test."""
    import random
    rng = random.Random(42)

    def per_item_list(wall, items):
        if items == 0:
            return []
        base = wall / items
        if per_item_var <= 0:
            return [base] * items
        return [base * (1 + per_item_var * (rng.random() - 0.5)) for _ in range(items)]

    return {
        "datasets": [],
        "phases": {
            "extract": {
                "items_processed": extract_items,
                "input_chars": extract_chars,
                "wall_clock_s": extract_wall,
                "per_item_times_s": per_item_list(extract_wall, extract_items),
            },
            "critic": {
                "items_processed": critic_items,
                "input_chars": critic_chars,
                "wall_clock_s": critic_wall,
                "per_item_times_s": per_item_list(critic_wall, critic_items),
                "critic_nonok_count": critic_nonok,
            },
            "revise": {
                "items_processed": revise_items,
                "input_chars": revise_chars,
                "wall_clock_s": revise_wall,
                "per_item_times_s": per_item_list(revise_wall, revise_items),
            },
            "finalize": {
                "items_processed": finalize_items,
                "input_chars": finalize_chars,
                "wall_clock_s": finalize_wall,
                "per_item_times_s": per_item_list(finalize_wall, finalize_items),
            },
        },
    }


def test_compute_estimate_linear_scaling():
    """Lineární scaling: 100x víc chars → 100x dlouhá extrakce."""
    timings = _make_timings()
    # full = same as smoke
    _, _, _, brk1x, _ = compute_estimate(
        timings, full_chars=1_000_000, full_items=200,
    )
    # full = 100x smoke
    _, _, _, brk100x, _ = compute_estimate(
        timings, full_chars=100_000_000, full_items=20_000,
    )
    ratio = brk100x["extract"] / brk1x["extract"]
    assert 99 < ratio < 101, f"Linear scaling broken: ratio={ratio}"


def test_compute_estimate_no_theoretical_scaling():
    """Smoke wall × full/smoke ratio — žádný GPU scaling factor v rovnici."""
    timings = _make_timings(
        extract_wall=600.0,
        extract_items=200, extract_chars=1_000_000,
    )
    # 10x víc chars → 10x víc času
    _, _, _, brk, _ = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    # extract: 600s × (10M / 1M) = 6000s
    assert abs(brk["extract"] - 6000.0) < 1.0, (
        f"Expected pure linear 6000s for extract, got {brk['extract']}"
    )


def test_compute_estimate_uses_measured_critic_nonok():
    """Měřená non-ok rate (60/200=30%) ovlivní revize."""
    timings = _make_timings(critic_nonok=60)
    _, _, _, brk_30, notes_30 = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    timings_60 = _make_timings(critic_nonok=120)
    _, _, _, brk_60, _ = compute_estimate(
        timings_60, full_chars=10_000_000, full_items=2000,
    )
    # 60% non-ok rate → revize 2x delší než 30%
    assert brk_60["revise"] > 1.8 * brk_30["revise"]
    assert any("non-ok rate měřena" in n for n in notes_30)


def test_compute_estimate_uses_default_nonok_when_missing():
    timings = _make_timings()
    timings["phases"]["critic"]["critic_nonok_count"] = None
    _, _, _, _, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    assert any("default" in n.lower() for n in notes)


def test_compute_estimate_missing_phase_warns():
    timings = _make_timings()
    del timings["phases"]["revise"]
    total, _, _, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    assert "revise" not in brk
    assert any(
        "revise" in n.lower() and ("nezměřena" in n.lower() or "nebyla" in n.lower())
        for n in notes
    ), f"Expected warning about missing 'revise' phase; notes={notes}"
    assert "extract" in brk and "critic" in brk and "finalize" in brk


def test_compute_estimate_breakdown_sums_to_total():
    timings = _make_timings()
    total, _, _, brk, _ = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    assert abs(total - sum(brk.values())) < 1e-6


def test_compute_estimate_finalize_uses_items_not_chars():
    """Finalize je CPU only — škáluje s items, ne chars."""
    timings = _make_timings(
        finalize_wall=10.0, finalize_items=200, finalize_chars=1_000_000,
    )
    # full_items 10x, full_chars 100x
    _, _, _, brk, _ = compute_estimate(
        timings, full_chars=100_000_000, full_items=2000,
    )
    # finalize: 10s × (2000 / 200) = 100s, NE 10s × (100M / 1M) = 1000s
    assert abs(brk["finalize"] - 100.0) < 1.0, (
        f"Finalize should scale with items (10x), got {brk['finalize']}"
    )


# === fáze 0 (taxonomy) — jednorázová konstanta ===========================

def _add_taxonomy(timings: dict, wall: float = 120.0) -> dict:
    """Přidá změřenou taxonomy fázi (fáze 0) do timings dictu.

    items/chars = 0 (sampluje fixně, nezávisle na rozsahu datasetu)."""
    timings["phases"]["taxonomy"] = {
        "items_processed": 0,
        "input_chars": 0,
        "wall_clock_s": wall,
        "per_item_times_s": [],
    }
    return timings


def _make_report_from(timings: dict, **overrides) -> SmokeTestReport:
    """Pomocník: postaví SmokeTestReport z timings + compute_estimate."""
    total, lower, upper, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    kwargs = dict(
        hw_label="1x B200",
        samples_per_dataset=50,
        datasets=[
            DatasetStats(name="x", total_items=2000, total_input_chars=10_000_000,
                         sample_items=200, sample_input_chars=1_000_000),
        ],
        phases={
            name: PhaseTimings(
                name=name, items_processed=p["items_processed"],
                input_chars=p["input_chars"], wall_clock_s=p["wall_clock_s"],
                per_item_times_s=p.get("per_item_times_s", []),
                critic_nonok_count=p.get("critic_nonok_count"),
            )
            for name, p in timings["phases"].items()
        },
        estimated_total_seconds=total,
        estimated_lower_seconds=lower,
        estimated_upper_seconds=upper,
        estimate_breakdown=brk,
        notes=notes,
    )
    kwargs.update(overrides)
    return SmokeTestReport(**kwargs)


def test_taxonomy_is_constant_not_scaled():
    """Taxonomy (fáze 0) se přičte jako konstanta — NEškáluje s rozsahem."""
    small = compute_estimate(
        _add_taxonomy(_make_timings(), wall=120.0),
        full_chars=10_000_000, full_items=2000,
    )
    big = compute_estimate(
        _add_taxonomy(_make_timings(), wall=120.0),
        full_chars=1_000_000_000, full_items=200_000,
    )
    # breakdown["taxonomy"] je v obou případech přesně naměřený wall (120s)
    assert abs(small[3]["taxonomy"] - 120.0) < 1e-6
    assert abs(big[3]["taxonomy"] - 120.0) < 1e-6, (
        "Taxonomy se nesmí škálovat s full_chars/full_items"
    )
    # Příspěvek k totalu = přesně konstanta (rozdíl total s/bez taxonomy)
    without = compute_estimate(_make_timings(), full_chars=10_000_000, full_items=2000)
    assert abs((small[0] - without[0]) - 120.0) < 1.0


def test_taxonomy_breakdown_sums_to_total():
    timings = _add_taxonomy(_make_timings(), wall=200.0)
    total, _, _, brk, _ = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    assert "taxonomy" in brk
    assert abs(total - sum(brk.values())) < 1e-6


def test_taxonomy_zero_wall_not_added():
    """Nulový wall (taxonomy nezměřena reálně) se do breakdownu nepřidá."""
    timings = _add_taxonomy(_make_timings(), wall=0.0)
    _, _, _, brk, _ = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    assert "taxonomy" not in brk


def test_format_report_taxonomy_row_present_when_measured():
    md = format_report_markdown(_make_report_from(_add_taxonomy(_make_timings(), wall=120.0)))
    assert "taxonomy (fáze 0)" in md
    assert "(nespuštěno)" not in md  # byla změřena


def test_format_report_taxonomy_row_marked_not_run_when_absent():
    """Bez taxonomy v timings řádek říká '(nespuštěno)'."""
    md = format_report_markdown(_make_report_from(_make_timings()))
    assert "taxonomy (fáze 0)" in md
    assert "(nespuštěno)" in md


def test_docker_startup_multiplier_includes_taxonomy():
    """Docker startup overhead × počet fází (taxonomy přidá +1 → 5)."""
    timings = _add_taxonomy(_make_timings(), wall=120.0)
    report = _make_report_from(timings, docker_startup_overhead_s=60.0)
    assert len(report.estimate_breakdown) == 5
    md = format_report_markdown(report)
    assert "× 5 fází" in md


# === ±2h konfidence požadavek ============================================

def test_confidence_interval_under_2h_for_typical_run():
    """Pro typický rozsah ~5-20h musí být konfidence ±2h."""
    timings = _make_timings(per_item_var=0.05)
    total, lower, upper, _, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    uncert_h = (upper - total) / 3600
    assert 0 < total / 3600 < 100
    assert uncert_h <= 2.0, (
        f"Konfidence ±{uncert_h:.2f}h přesahuje 2h target. "
        f"Total={total/3600:.2f}h"
    )


def test_confidence_caps_at_2h_with_warning():
    """Pro obří rozsah (1B chars) i s 5% floor uncertainty > 2h → cap + warn."""
    timings = _make_timings(per_item_var=0.1)
    total, lower, upper, _, notes = compute_estimate(
        timings, full_chars=1_000_000_000, full_items=200_000,
    )
    uncert_h = (upper - total) / 3600
    assert total / 3600 > 25
    assert uncert_h <= 2.0 + 1e-6, f"Cap selhal: {uncert_h:.2f}h > 2h"
    assert any(
        "přesahuje" in n.lower() or "doporu" in n.lower()
        for n in notes
    ), f"Chybí warning při překročení 2h. Notes: {notes}"


# === Round-trip přes save/load timings ====================================

def test_save_load_timings_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            data = {
                "datasets": [{"name": "x", "total_items": 100, "total_input_chars": 1000}],
                "phases": {"extract": {"items_processed": 50, "wall_clock_s": 30.0}},
            }
            save_timings(data)
            loaded = load_timings()
            assert loaded == data
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


def test_record_hw_label_persists():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            record_hw_label("1x B200")
            loaded = load_timings()
            assert loaded.get("hw_label") == "1x B200"
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


def test_load_timings_missing_file_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            assert load_timings() == {}
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


def test_build_report_uses_stored_hw_label_when_not_supplied():
    """build_report použije uložený HW label, pokud parametr je prázdný."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            save_timings({
                "hw_label": "stored 2x B200",
                "datasets": [{"name": "x", "total_items": 100, "total_input_chars": 1000,
                              "sample_items": 5, "sample_input_chars": 50}],
                "phases": {},
            })
            r = build_report("", samples_per_dataset=5)
            assert r.hw_label == "stored 2x B200"
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


# === Markdown report ======================================================

def test_format_report_markdown_contains_essential_sections():
    timings = _make_timings()
    total, lower, upper, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    report = SmokeTestReport(
        hw_label="1x B200",
        samples_per_dataset=50,
        datasets=[
            DatasetStats(name="alfa", total_items=1000, total_input_chars=2_000_000,
                         sample_items=50, sample_input_chars=100_000),
            DatasetStats(name="beta", total_items=1000, total_input_chars=2_000_000,
                         sample_items=50, sample_input_chars=100_000),
        ],
        phases={
            name: PhaseTimings(
                name=name,
                items_processed=p["items_processed"],
                input_chars=p["input_chars"],
                wall_clock_s=p["wall_clock_s"],
                per_item_times_s=p.get("per_item_times_s", []),
                critic_nonok_count=p.get("critic_nonok_count"),
            )
            for name, p in timings["phases"].items()
        },
        estimated_total_seconds=total,
        estimated_lower_seconds=lower,
        estimated_upper_seconds=upper,
        estimate_breakdown=brk,
        notes=notes,
    )
    md = format_report_markdown(report)
    # Klíčové sekce
    assert "# Production Smoke Test Report" in md
    assert "Hardware" in md
    assert "1x B200" in md
    assert "Datasety" in md
    assert "Per-phase měření" in md
    assert "Odhad celkového času" in md
    assert "Konfidence" in md
    # Lineární extrapolace vzorec v reportu
    assert "T_full" in md or "Lineární extrapolace" in md
    # Žádné teoretické multi-GPU srovnání už NEMÁ být v reportu
    assert "Srovnání multi-GPU variant" not in md
    # Report nesmí prezentovat žádný odhad pro 2x/4x B200, dokud na něm
    # neproběhl smoke. "1x B200" je OK (hw_label v hlavičce), ale tabulka
    # multi-GPU porovnání NE.
    md_lower = md.lower()
    # Pokud HW label není 2xB200, nesmí být v tabulkách
    assert "| 2x b200 |" not in md_lower
    assert "| 4x b200 |" not in md_lower


def test_format_report_includes_docker_startup_when_provided():
    """Pokud uživatel zadá --docker-startup-s, report ho zmíní."""
    timings = _make_timings()
    total, lower, upper, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    report = SmokeTestReport(
        hw_label="1x B200",
        samples_per_dataset=50,
        datasets=[
            DatasetStats(name="x", total_items=2000, total_input_chars=10_000_000,
                         sample_items=200, sample_input_chars=1_000_000),
        ],
        phases={
            name: PhaseTimings(
                name=name, items_processed=p["items_processed"],
                input_chars=p["input_chars"], wall_clock_s=p["wall_clock_s"],
            )
            for name, p in timings["phases"].items()
        },
        estimated_total_seconds=total,
        estimated_lower_seconds=lower,
        estimated_upper_seconds=upper,
        estimate_breakdown=brk,
        docker_startup_overhead_s=90.0,
        notes=notes,
    )
    md = format_report_markdown(report)
    assert "Docker startup overhead" in md
    assert "90.0s" in md or "1m 30s" in md


# === Multi-GPU projection ================================================

def test_parse_gpu_count_recognizes_basic_b200_labels():
    assert parse_gpu_count_from_hw_label("1x B200") == 1
    assert parse_gpu_count_from_hw_label("2x B200") == 2
    assert parse_gpu_count_from_hw_label("4x B200") == 4
    assert parse_gpu_count_from_hw_label("8x B200") == 8


def test_parse_gpu_count_handles_variations():
    """Spaces, capitalization, unicode times sign atd."""
    assert parse_gpu_count_from_hw_label("1 x B200") == 1
    assert parse_gpu_count_from_hw_label("1xB200") == 1
    assert parse_gpu_count_from_hw_label("2× B200") == 2  # unicode times
    assert parse_gpu_count_from_hw_label("2X b200") == 2  # case insensitive


def test_parse_gpu_count_unknown_label_returns_none():
    assert parse_gpu_count_from_hw_label("RTX PRO 6000") is None
    assert parse_gpu_count_from_hw_label("") is None
    assert parse_gpu_count_from_hw_label("3x B200") is None  # 3 není v scaling table
    assert parse_gpu_count_from_hw_label("super HW") is None


def test_project_returns_measured_count_unchanged():
    """Projekce na měřený GPU count vrátí stejný čas."""
    projected = project_to_gpu_counts(10_000.0, measured_gpu_count=1)
    assert projected[1] == 10_000.0


def test_project_1x_to_2x_uses_scaling():
    """1x B200 → 2x B200 musí použít scaling factor."""
    projected = project_to_gpu_counts(10_000.0, measured_gpu_count=1)
    # 2x B200: factor 1.8x throughput → time / 1.8
    expected = 10_000.0 / 1.8
    assert abs(projected[2] - expected) < 1.0


def test_project_2x_to_1x_and_4x_consistent():
    """Měřeno na 2x → projekce na 1x a 4x musí být konzistentní s factory."""
    projected = project_to_gpu_counts(5_000.0, measured_gpu_count=2)
    # 2x baseline (1.8x throughput): time_2x = 5000s
    # 1x (1.0x throughput): time_1x = 5000 × (1.8 / 1.0) = 9000s
    assert abs(projected[1] - 9000.0) < 1.0
    # 4x (3.3x throughput): time_4x = 5000 × (1.8 / 3.3) ≈ 2727s
    expected_4x = 5000.0 * (1.8 / 3.3)
    assert abs(projected[4] - expected_4x) < 1.0


def test_project_unknown_gpu_count_returns_empty():
    """Neznámý measured count → prázdná projekce."""
    projected = project_to_gpu_counts(5_000.0, measured_gpu_count=3)
    assert projected == {}


def test_projection_uncertainty_is_25_percent():
    assert PROJECTION_UNCERTAINTY == 0.25


def _timings_with_datasets(**kw):
    """Helper: timings + dataset stats potřebné pro non-zero estimate."""
    t = _make_timings(**kw)
    t["datasets"] = [
        {"name": "x", "total_items": 2000, "total_input_chars": 10_000_000,
         "sample_items": 200, "sample_input_chars": 1_000_000},
    ]
    return t


def test_build_report_with_b200_label_emits_projection():
    """build_report rozpozná HW label a doplní multi-GPU projekci."""
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            save_timings(_timings_with_datasets())
            r = build_report("1x B200", samples_per_dataset=50)
            assert r.measured_gpu_count == 1
            assert r.estimated_total_seconds > 0
            assert r.projected_total_seconds
            assert 1 in r.projected_total_seconds
            assert 2 in r.projected_total_seconds
            assert 4 in r.projected_total_seconds
            # 4x B200 odhad < 2x B200 odhad < 1x B200 odhad
            assert (r.projected_total_seconds[4]
                    < r.projected_total_seconds[2]
                    < r.projected_total_seconds[1])
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


def test_build_report_with_unknown_hw_skips_projection():
    """Neznámý HW label → žádná projekce + warning v notes."""
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE_CHECKPOINT_DIR"] = tmp
        try:
            save_timings(_timings_with_datasets())
            r = build_report("2x RTX PRO 6000", samples_per_dataset=50)
            assert r.measured_gpu_count is None
            assert r.projected_total_seconds == {}
            assert any(
                "nebyl rozpoznán" in n.lower() or "vynechána" in n.lower()
                for n in r.notes
            )
        finally:
            os.environ.pop("SMOKE_CHECKPOINT_DIR", None)


def test_format_report_visually_separates_precise_from_orientational():
    """
    Markdown report ROZDĚLUJE měřený HW (přesný) od ostatních (orientační)
    do DVOU sekcí, aby si je uživatel nepletl.
    """
    timings = _make_timings()
    total, lower, upper, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    report = SmokeTestReport(
        hw_label="1x B200",
        samples_per_dataset=50,
        datasets=[
            DatasetStats(name="x", total_items=2000, total_input_chars=10_000_000,
                         sample_items=200, sample_input_chars=1_000_000),
        ],
        phases={
            name: PhaseTimings(
                name=name, items_processed=p["items_processed"],
                input_chars=p["input_chars"], wall_clock_s=p["wall_clock_s"],
            )
            for name, p in timings["phases"].items()
        },
        estimated_total_seconds=total,
        estimated_lower_seconds=lower,
        estimated_upper_seconds=upper,
        estimate_breakdown=brk,
        measured_gpu_count=1,
        projected_total_seconds=project_to_gpu_counts(total, 1),
        notes=notes,
    )
    md = format_report_markdown(report)

    # DVĚ oddělené sekce
    assert "Přesný odhad" in md
    assert "Orientační odhady" in md

    # Přesný odhad: explicit ±2h, jen měřený HW (1×)
    precise_start = md.find("Přesný odhad")
    orient_start = md.find("Orientační odhady")
    precise_section = md[precise_start:orient_start]
    assert "±2h" in precise_section
    assert "1× B200" in precise_section
    # V přesné sekci NESMÍ být zmínka o 2× nebo 4× s odhadem (jen jako reference)
    # Ne-měřené counts patří do orient sekce

    # Orientační sekce: ±25 %, ostatní counts s rough odhadem
    orient_section = md[orient_start:]
    assert "±25" in orient_section or "25 %" in orient_section
    assert "2× B200" in orient_section
    assert "4× B200" in orient_section
    assert "NEPŘESNÉ" in orient_section.upper() or "NEPRODUK" in orient_section.upper()
    # Explicit "spusť smoke" výzva
    assert "spusť smoke" in orient_section.lower()


def test_format_report_omits_projection_section_when_unknown_hw():
    """Bez měřeného B200 počtu sekce v reportu vůbec není."""
    timings = _make_timings()
    total, lower, upper, brk, notes = compute_estimate(
        timings, full_chars=10_000_000, full_items=2000,
    )
    report = SmokeTestReport(
        hw_label="2x RTX PRO 6000",
        samples_per_dataset=50,
        datasets=[
            DatasetStats(name="x", total_items=2000, total_input_chars=10_000_000,
                         sample_items=200, sample_input_chars=1_000_000),
        ],
        phases={},
        estimated_total_seconds=total,
        estimated_lower_seconds=lower,
        estimated_upper_seconds=upper,
        estimate_breakdown=brk,
        measured_gpu_count=None,
        projected_total_seconds={},
        notes=notes,
    )
    md = format_report_markdown(report)
    assert "Přesný odhad" not in md
    assert "Orientační odhady" not in md


def test_json_export_separates_precise_from_orientational():
    """JSON má precise_hours (měřeno) vs orientational_hours (rough estimate)."""
    projected = project_to_gpu_counts(10_000.0, measured_gpu_count=1)
    report = SmokeTestReport(
        hw_label="1x B200", samples_per_dataset=50,
        datasets=[],
        phases={},
        estimated_total_seconds=10_000.0,
        measured_gpu_count=1,
        projected_total_seconds=projected,
    )
    d = report.to_dict()
    assert "projection" in d
    p = d["projection"]
    assert "warning" in p
    # Měřené musí být explicitně "precise"
    assert p["precise_seconds"] == 10_000.0
    assert abs(p["precise_hours"] - 10_000.0 / 3600) < 1e-6
    # Ostatní jsou orientational
    assert "2" in p["orientational_hours"]
    assert "4" in p["orientational_hours"]
    # Měřené 1× NESMÍ být v orientational
    assert "1" not in p["orientational_hours"]
    # Scaling factory přejmenované na "_orientational"
    assert "scaling_factors_orientational" in p


def test_json_export_no_precise_when_hw_unknown():
    """Pokud HW není B200, precise_* je None a orientational je prázdné."""
    report = SmokeTestReport(
        hw_label="2x RTX PRO 6000", samples_per_dataset=50,
        datasets=[], phases={},
        estimated_total_seconds=10_000.0,
        measured_gpu_count=None,
        projected_total_seconds={},
    )
    d = report.to_dict()
    assert d["projection"]["precise_seconds"] is None
    assert d["projection"]["precise_hours"] is None
    assert d["projection"]["orientational_hours"] == {}


# === Runner ===============================================================

if __name__ == "__main__":
    import inspect
    # Windows cp1250 console se dusí na Unicode glyfech v assertion messages.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
            msg = repr(str(e)) if not str(e).isascii() else str(e)
            print(f"  FAIL  {name}: {msg}")
        except Exception as e:
            failed += 1
            msg = repr(str(e)) if not str(e).isascii() else str(e)
            print(f"  ERROR  {name}: {type(e).__name__}: {msg}")
    print(f"\nTotal: {len(tests)} tests, {failed} failed.")
    sys.exit(1 if failed else 0)
