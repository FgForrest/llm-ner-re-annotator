"""Fáze 0a: kódování hlásí postup a embedder umí víc GPU."""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from labeler.taxonomy_sampling import EmbeddingServerClient


def _client(batch_size: int = 4) -> EmbeddingServerClient:
    c = EmbeddingServerClient(base_url="http://x/v1", model="bge-m3", batch_size=batch_size)
    # žádný server — dávku nahradíme deterministickým vektorem
    c._post = lambda batch: [[float(len(t)), 1.0] for t in batch]  # type: ignore[assignment]
    return c


def test_embed_returns_vector_per_text():
    out = _client().embed([f"text {i}" for i in range(10)])
    assert out.shape == (10, 2)
    assert out.dtype == np.float32


def test_embed_reports_progress(caplog):
    with caplog.at_level(logging.INFO, logger="labeler.taxonomy_sampling"):
        _client().embed([f"text {i}" for i in range(12)], progress_every_s=0.0)
    msgs = [r.message for r in caplog.records if "embedding:" in r.message]
    assert msgs, "kódování nehlásí postup"
    assert "12/12" in msgs[-1], "poslední hlášení má být o dokončení"
    assert "textů/s" in msgs[-1] and "zbývá" in msgs[-1]


def test_progress_is_throttled_by_default(caplog):
    """Default se nehlásí každou dávku — jen dokončení."""
    with caplog.at_level(logging.INFO, logger="labeler.taxonomy_sampling"):
        _client(batch_size=1).embed([f"t{i}" for i in range(20)])
    msgs = [r.message for r in caplog.records if "embedding:" in r.message]
    assert len(msgs) == 1 and "20/20" in msgs[0]


def test_embedder_service_scales_with_gpu_count():
    """Embedder nesmí mít počet GPU natvrdo — jinak ostatní karty stojí."""
    compose = pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "count: ${EMBEDDING_GPU_COUNT:-1}" in compose
    assert "--data-parallel-size" in compose


def test_sample_phase_runs_detached_like_the_others():
    sh = pathlib.Path("scripts/run_production.sh").read_text(encoding="utf-8")
    block = sh[sh.index("run_sample() {"):sh.index("run_taxonomy() {")]
    assert "up -d" in block, "fáze 0a má běžet detached jako ostatní"
    assert "_labeler_exit_code" in block, "exit code se má číst stejně jako jinde"
