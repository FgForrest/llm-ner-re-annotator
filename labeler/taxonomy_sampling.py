"""
Fáze 0a — výběr vzorků pro návrh taxonomie přes embeddingy (BAAI/bge-m3 na vLLM).

Místo náhodného výběru pošle fáze 0 LLM ukázky, které pokrývají CO NEJRŮZNĚJŠÍ
témata každé domény. Počet řídí TAXONOMY_SAMPLES_PER_DATASET:

  auto (default) — každá doména si počet spočítá sama: ukázky se přidávají, dokud
                   nemá KAŽDÝ text domény ve vzorku zástupce s kosinovou
                   podobností >= TAXONOMY_SAMPLE_COVERAGE (mez MIN..MAX).
                   Jednotvárná doména dostane pár ukázek, pestrá desítky.
  <číslo>        — přesně tolik ukázek per doména (nejodlišnější texty).

Algoritmus (TAXONOMY_SAMPLE_SCOPE):
  global (default) — přes CELÝ dataset najednou: texty se seskupí do témat
                     (farthest-point centra + přiřazení k nejbližšímu), z
                     nejodlišnějších témat se vezme NEJTYPIČTĚJŠÍ text; osamocené
                     odlehlé texty (šum) se nevybírají; stejný obsah ve více
                     doménách se vybere jednou; každá doména má min. zastoupení
                     (select_samples_global).
  domain           — per doména zvlášť: greedy farthest-point od medoidu domény
                     (select_samples).
Deterministické (seed jen pro podvýběr velkých domén), bez sklearn.

Globální strop TAXONOMY_SAMPLE_MAX_TOTAL drží prompt fáze 0 v kontextu Qwenu;
při překročení se ubere poměrně všem doménám (od konce pořadí — nejodlišnější
ukázky zůstávají).

Běh: samostatný krok PŘED fází 0 (na GPU je jen embedding server, žádný Qwen):
  docker   profil embed-vllm (embedder-vllm + labeler-embed-vllm)
  native   vLLM --runner pooling na portu 8002 (scripts/start_native_embedder.*)
  CLI      python -m labeler.main taxonomy-sample
Výstup: CHECKPOINT_DIR/taxonomy_sampling.json. Fáze 0 ho načte; když chybí,
neodpovídá aktuálnímu nastavení / datům, nebo krok selhal, spadne na náhodný
výběr (best-effort — běh se nezastaví).
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import httpx
import numpy as np

from .config import CONFIG
from .datasets import LabelItem, scan_data_dir
from .io_utils import atomic_write_json

logger = logging.getLogger(__name__)

SAMPLING_FILE = "taxonomy_sampling.json"
SCHEMA_VERSION = 1
# Stejná délka jako náhled v promptu fáze 0 (_format_samples_for_prompt).
PREVIEW_CHARS = 600
# Podíl ukázek, které musí v aktuálních datech existovat, aby soubor platil.
_MIN_FOUND_SHARE = 0.8


class EmbeddingServerError(Exception):
    """Embedding server nedostupný / vrátil nevalidní odpověď."""


# ---------------------------------------------------------------------------
# HTTP klient na OpenAI-kompatibilní /v1/embeddings (vLLM --runner pooling)
# ---------------------------------------------------------------------------


class EmbeddingServerClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        batch_size: int = 64,
        timeout: float = 300.0,
        max_retries: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "EmbeddingServerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def wait_ready(self, max_wait_s: float = 1200.0, interval_s: float = 5.0) -> None:
        deadline = time.monotonic() + max_wait_s
        while True:
            try:
                if self._http.get(f"{self.base_url}/models").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise EmbeddingServerError(
                    f"Embedding server {self.base_url} neodpověděl do {max_wait_s:.0f}s"
                )
            time.sleep(interval_s)

    def _post(self, batch: list[str]) -> list[list[float]]:
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._http.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch, "encoding_format": "float"},
                )
                if r.status_code >= 500:
                    raise EmbeddingServerError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    # 4xx se opakováním nespraví
                    raise EmbeddingServerError(f"HTTP {r.status_code}: {r.text[:200]}")
                data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
                vecs = [d["embedding"] for d in data]
                if len(vecs) != len(batch):
                    raise EmbeddingServerError(
                        f"server vrátil {len(vecs)} embeddingů pro {len(batch)} textů"
                    )
                return vecs
            except (httpx.HTTPError, EmbeddingServerError, KeyError, ValueError) as e:
                last = e
                if isinstance(e, EmbeddingServerError) and "HTTP 4" in str(e):
                    break
                time.sleep(min(2 ** attempt, 10))
        raise EmbeddingServerError(f"embedding request selhal: {last}")

    def embed(self, texts: Sequence[str], *, progress_every_s: float = 10.0) -> np.ndarray:
        """
        Zakóduje texty po dávkách.

        Kódování celého korpusu trvá minuty až desítky minut, takže průběžně
        hlásí postup (kolik hotovo, rychlost, odhad zbývajícího času) — jinak
        fáze 0a vypadá, že stojí.
        """
        out: list[list[float]] = []
        total = len(texts)
        t0 = time.monotonic()
        last_report = t0
        for i in range(0, total, self.batch_size):
            out.extend(self._post(list(texts[i:i + self.batch_size])))
            now = time.monotonic()
            done = len(out)
            if done < total and now - last_report < progress_every_s:
                continue
            last_report = now
            elapsed = max(1e-6, now - t0)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0.0
            logger.info(
                f"embedding: {done}/{total} textů ({100.0 * done / max(1, total):.0f} %), "
                f"{rate:.0f} textů/s, zbývá ~{eta / 60:.1f} min"
            )
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Výběr (pure — testovatelné bez serveru)
# ---------------------------------------------------------------------------


def _normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def farthest_point_order(emb: np.ndarray, k_max: int) -> tuple[list[int], list[float]]:
    """
    Greedy k-center nad NORMALIZOVANÝMI embeddingy. Start v medoidu (nejvyšší
    průměrná podobnost ke všem), pak vždy bod s nejnižší podobností ke svému
    nejbližšímu vybranému. Vrací (pořadí indexů, pokrytí po každém kroku);
    pokrytí = min přes všechny body jejich max podobnosti k vybraným.
    O(n·k) paměti i času — bez n×n matice.
    """
    n = emb.shape[0]
    if n == 0 or k_max <= 0:
        return [], []
    centroid = emb.mean(axis=0)
    first = int(np.argmax(emb @ centroid))
    order = [first]
    best = emb @ emb[first]
    coverage = [float(best.min())]
    while len(order) < min(k_max, n):
        best[order] = np.inf            # vybrané už nekandidují
        cand = int(np.argmin(best))
        best[order] = 1.0
        order.append(cand)
        best = np.maximum(best, emb @ emb[cand])
        coverage.append(float(best.min()))
    return order, coverage


def choose_count(
    coverage: list[float], *, auto: bool, fixed_n: int,
    min_k: int, max_k: int, target: float,
) -> int:
    """Počet ukázek: fixní N, nebo (auto) nejmenší k s pokrytím >= target v mezích."""
    n = len(coverage)
    if n == 0:
        return 0
    if not auto:
        return min(max(1, fixed_n), n)
    k = next((i + 1 for i, c in enumerate(coverage) if c >= target), n)
    return max(min(min_k, n), min(k, max_k, n))


def apply_total_budget(counts: dict[str, int], max_total: int, min_k: int) -> dict[str, int]:
    """Když součet překročí max_total, ubere poměrně všem doménám (min. min_k)."""
    total = sum(counts.values())
    if max_total <= 0 or total <= max_total:
        return dict(counts)
    scale = max_total / total
    out = {d: max(min(min_k, c), int(c * scale)) for d, c in counts.items()}
    # dorovnání (zaokrouhlení / podlahy): ubírej z největších, přidávej největším
    while sum(out.values()) > max_total and any(
        out[d] > min(min_k, counts[d]) for d in out
    ):
        d = max(out, key=lambda x: (out[x] - min(min_k, counts[x]), x))
        out[d] -= 1
    return out


def select_samples(
    items: list[LabelItem],
    embed_fn,
    *,
    auto: bool,
    fixed_n: int,
    min_k: int,
    max_k: int,
    coverage_target: float,
    max_total: int,
    pool_max: int,
    max_chars: int,
    seed: int = 42,
) -> dict:
    """
    Vybere ukázky per doména (LabelItem.domain — stejné seskupení jako fáze 0).
    `embed_fn(texts) -> np.ndarray`. Vrací dict pro taxonomy_sampling.json.
    """
    by_domain: dict[str, list[LabelItem]] = {}
    for it in items:
        if it.text and it.text.strip():
            by_domain.setdefault(it.domain, []).append(it)

    rng = random.Random(seed)
    orders: dict[str, list[LabelItem]] = {}
    coverages: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    stats: dict[str, dict] = {}
    k_cap = max(max_k, fixed_n)
    for domain in sorted(by_domain):
        dom_items = by_domain[domain]
        pool = dom_items
        if pool_max > 0 and len(pool) > pool_max:
            pool = [dom_items[i] for i in sorted(rng.sample(range(len(dom_items)), pool_max))]
        emb = _normalize(np.asarray(
            embed_fn([it.text[:max_chars] for it in pool]), dtype=np.float32,
        ))
        order, cov = farthest_point_order(emb, k_cap)
        orders[domain] = [pool[i] for i in order]
        coverages[domain] = cov
        counts[domain] = choose_count(
            cov, auto=auto, fixed_n=fixed_n, min_k=min_k, max_k=max_k,
            target=coverage_target,
        )
        stats[domain] = {"total_items": len(dom_items), "pool_size": len(pool)}

    budgeted = apply_total_budget(counts, max_total, min_k)
    if budgeted != counts:
        logger.warning(
            f"taxonomy-sample: součet ukázek {sum(counts.values())} > "
            f"TAXONOMY_SAMPLE_MAX_TOTAL={max_total} — poměrně ubráno na "
            f"{sum(budgeted.values())}"
        )

    domains_out: dict[str, dict] = {}
    for domain in sorted(orders):
        k = budgeted[domain]
        chosen = orders[domain][:k]
        domains_out[domain] = {
            **stats[domain],
            "wanted": counts[domain],
            "selected": k,
            "coverage": round(coverages[domain][k - 1], 4) if k else None,
            "samples": [
                {"item_id": it.item_id, "preview": it.text[:PREVIEW_CHARS]}
                for it in chosen
            ],
        }
    return {
        "domains": domains_out,
        "total_selected": sum(d["selected"] for d in domains_out.values()),
    }


def _max_sim_to(emb: np.ndarray, ref: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Pro každý bod max. kosinová podobnost k libovolnému bodu z `ref` (po dávkách)."""
    out = np.empty(emb.shape[0], dtype=np.float32)
    if ref.shape[0] == 0:
        out.fill(-1.0)
        return out
    for i in range(0, emb.shape[0], chunk):
        out[i:i + chunk] = (emb[i:i + chunk] @ ref.T).max(axis=1)
    return out


def _nearest_center(emb: np.ndarray, centers: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Index nejbližšího centra pro každý bod (po dávkách — bez n×k matice v paměti)."""
    out = np.empty(emb.shape[0], dtype=np.int64)
    for i in range(0, emb.shape[0], chunk):
        out[i:i + chunk] = (emb[i:i + chunk] @ centers.T).argmax(axis=1)
    return out


def select_samples_global(
    items: list[LabelItem],
    embed_fn,
    *,
    auto: bool,
    fixed_n: int,
    min_k: int,
    max_k: int,
    coverage_target: float,
    max_total: int,
    pool_max: int,
    max_chars: int,
    min_cluster: int,
    seed: int = 42,
) -> dict:
    """
    Výběr ukázek přes CELÝ DATASET najednou (TAXONOMY_SAMPLE_SCOPE=global).

      1. Zakóduje všechny texty (pool_max > 0 = strop per doména, 0 = vše).
      2. Témata: greedy farthest-point centra přes všechny domény → každý text
         se přiřadí k nejbližšímu centru (shluk = téma).
      3. Šum: shluk s < min_cluster texty je odlehlý útržek (odkazy, kódy,
         navigace) — jeho text se jako zástupce NEVYBÍRÁ a nepočítá se do
         pokrytí.
      4. Zástupci v pořadí od nejodlišnějšího tématu; zástupce = NEJTYPIČTĚJŠÍ
         text shluku (nejbližší jeho průměru), ne ten nejpodivnější. Stejný
         obsah ve více doménách (patičky, bannery) padne do jednoho shluku →
         vybere se jednou.
      5. Počet: auto = dokud nemá každý text (mimo šum) zástupce s podobností
         >= coverage_target; číslo N = N × počet domén (rozdělení podle
         rozmanitosti, ne rovnoměrně).
      6. Každá doména aspoň min(min_k, velikost), nejvýš max_k; celkem max
         max_total (ubírá se od nejméně odlišných).
    Výstup má stejný tvar jako select_samples (ukázky seskupené per doména).
    """
    by_domain: dict[str, list[LabelItem]] = {}
    for it in items:
        if it.text and it.text.strip():
            by_domain.setdefault(it.domain, []).append(it)
    domains = sorted(by_domain)
    if not domains:
        return {"domains": {}, "total_selected": 0, "global": {}}

    rng = random.Random(seed)
    pool: list[LabelItem] = []
    dom_of: list[int] = []
    stats: dict[str, dict] = {}
    for di, d in enumerate(domains):
        dom_items = by_domain[d]
        chosen = dom_items
        if pool_max > 0 and len(dom_items) > pool_max:
            chosen = [dom_items[i] for i in sorted(rng.sample(range(len(dom_items)), pool_max))]
        pool.extend(chosen)
        dom_of.extend([di] * len(chosen))
        stats[d] = {"total_items": len(dom_items), "pool_size": len(chosen)}
    dom_arr = np.asarray(dom_of)
    emb = _normalize(np.asarray(
        embed_fn([it.text[:max_chars] for it in pool]), dtype=np.float32,
    ))
    n, n_dom = emb.shape[0], len(domains)
    total_cap = max_total if max_total > 0 else n
    dom_size = np.bincount(dom_arr, minlength=n_dom)
    dom_min = np.minimum(dom_size, max(0, min_k))
    dom_max = np.maximum(dom_min, np.minimum(dom_size, max_k if max_k > 0 else n))

    # 2) témata přes celý dataset: téma = oblast, kde je každý text podobný
    #    svému centru aspoň na coverage_target → počet témat určí pokrytí
    #    greedy farthest-point (odlehlý text dostane vlastní centrum → šum).
    #    Fixní N potřebuje aspoň tolik témat, kolik ukázek chce.
    target = total_cap if auto else min(total_cap, max(1, fixed_n) * n_dom)
    k_cand = int(min(n, max(3 * total_cap, target, int(dom_min.sum()) + 1)))
    fps_order, fps_cov = farthest_point_order(emb, k_cand)
    k_cov = next((i + 1 for i, c in enumerate(fps_cov) if c >= coverage_target), len(fps_cov))
    k_topics = k_cov if auto else min(len(fps_order), max(k_cov, target))
    centers = fps_order[:k_topics]
    labels = _nearest_center(emb, emb[centers])
    sizes = np.bincount(labels, minlength=len(centers))
    noise_cluster = sizes < max(1, min_cluster)
    eligible = ~noise_cluster[labels]            # body mimo šum
    if not eligible.any():                        # vše šum (malý dataset) → bez filtru
        noise_cluster[:] = False
        eligible[:] = True

    members_of: dict[int, np.ndarray] = {}
    order_idx = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order_idx], np.arange(len(centers) + 1))
    for ci in range(len(centers)):
        members_of[ci] = order_idx[bounds[ci]:bounds[ci + 1]]

    def _typical(members: np.ndarray) -> int:
        m = emb[members].mean(axis=0)
        return int(members[int(np.argmax(emb[members] @ m))])

    # 4) + 5) zástupci od nejodlišnějšího tématu
    selected: list[int] = []
    per_dom = np.zeros(n_dom, dtype=int)
    best = np.full(n, -1.0, dtype=np.float32)
    for ci in range(len(centers)):
        if len(selected) >= target:
            break
        if noise_cluster[ci]:
            continue
        rep = _typical(members_of[ci])
        if per_dom[dom_arr[rep]] >= dom_max[dom_arr[rep]]:
            continue
        selected.append(rep)
        per_dom[dom_arr[rep]] += 1
        if auto:
            best = np.maximum(best, emb @ emb[rep])
            if float(best[eligible].min()) >= coverage_target and len(selected) >= int(dom_min.sum()):
                break

    # 6) doplnění domén pod minimem (nejodlišnější texty dané domény mimo šum)
    chosen_set = set(selected)
    for di in range(n_dom):
        need = int(dom_min[di] - per_dom[di])
        if need <= 0:
            continue
        idx = np.flatnonzero((dom_arr == di) & eligible)
        if idx.size < need:
            idx = np.flatnonzero(dom_arr == di)
        sub_order, _ = farthest_point_order(emb[idx], min(idx.size, need + len(chosen_set)))
        for j in sub_order:
            g = int(idx[j])
            if g in chosen_set:
                continue
            selected.append(g)
            chosen_set.add(g)
            per_dom[di] += 1
            need -= 1
            if need == 0:
                break

    # strop celkem: ubírej od konce (nejméně odlišné), minima domén drž
    while len(selected) > total_cap:
        for pos in range(len(selected) - 1, -1, -1):
            di = dom_arr[selected[pos]]
            if per_dom[di] > dom_min[di]:
                per_dom[di] -= 1
                del selected[pos]
                break
        else:
            break

    sel_arr = np.asarray(selected, dtype=np.int64)
    cover = _max_sim_to(emb, emb[sel_arr])
    domains_out: dict[str, dict] = {}
    for di, d in enumerate(domains):
        mine = [i for i in selected if dom_of[i] == di]
        in_dom = dom_arr == di
        elig_dom = in_dom & eligible
        domains_out[d] = {
            **stats[d],
            "wanted": int(per_dom[di]),
            "selected": len(mine),
            # pokrytí domény ukázkami z CELÉHO výběru (i z jiných domén)
            "coverage": round(float(cover[elig_dom].min()), 4) if elig_dom.any() else None,
            "samples": [
                {"item_id": pool[i].item_id, "preview": pool[i].text[:PREVIEW_CHARS]}
                for i in mine
            ],
        }
    return {
        "domains": domains_out,
        "total_selected": len(selected),
        "global": {
            "embedded_items": int(n),
            "topic_clusters": int(len(centers)),
            "noise_clusters": int(noise_cluster.sum()),
            "noise_items": int((~eligible).sum()),
            "coverage": round(float(cover[eligible].min()), 4) if eligible.any() else None,
        },
    }


# ---------------------------------------------------------------------------
# Nastavení, zápis, načtení pro fázi 0
# ---------------------------------------------------------------------------


def current_settings() -> dict:
    """Nastavení, se kterým byl výběr spočítán — při změně je soubor neplatný."""
    return {
        "mode": "auto" if CONFIG.taxonomy_samples_auto else "fixed",
        "scope": CONFIG.taxonomy_sample_scope,
        "min_cluster": CONFIG.taxonomy_sample_min_cluster,
        "fixed_n": None if CONFIG.taxonomy_samples_auto else CONFIG.taxonomy_samples_per_dataset,
        "min": CONFIG.taxonomy_sample_min,
        "max": CONFIG.taxonomy_sample_max,
        "coverage": CONFIG.taxonomy_sample_coverage,
        "max_total": CONFIG.taxonomy_sample_max_total,
        "pool_max": CONFIG.taxonomy_sample_pool_max,
        "max_chars": CONFIG.embedding_max_chars,
        "model": CONFIG.taxonomy_embedding_model,
        "locales": CONFIG.data_locales,
    }


def sampling_path(checkpoint_dir: Optional[Path] = None) -> Path:
    return (checkpoint_dir or CONFIG.checkpoint_dir) / SAMPLING_FILE


def run_taxonomy_sampling(client: Optional[EmbeddingServerClient] = None) -> Path:
    """
    Fáze 0a: zakóduje texty z DATA_DIR na embedding serveru, vybere ukázky a
    zapíše taxonomy_sampling.json do CHECKPOINT_DIR. Chyby propaguje (volající
    rozhoduje, zda je best-effort).
    """
    items = list(scan_data_dir(CONFIG.data_dir))
    if not items:
        raise FileNotFoundError(f"V {CONFIG.data_dir} nejsou žádné label itemy")
    own = client is None
    if own:
        client = EmbeddingServerClient(
            CONFIG.embedding_base_url, CONFIG.taxonomy_embedding_model,
            batch_size=CONFIG.embedding_batch_size,
        )
    settings = current_settings()
    try:
        client.wait_ready()
        t0 = time.monotonic()
        common = dict(
            auto=CONFIG.taxonomy_samples_auto,
            fixed_n=CONFIG.taxonomy_samples_per_dataset,
            min_k=CONFIG.taxonomy_sample_min,
            max_k=CONFIG.taxonomy_sample_max,
            coverage_target=CONFIG.taxonomy_sample_coverage,
            max_total=CONFIG.taxonomy_sample_max_total,
            pool_max=CONFIG.taxonomy_sample_pool_max,
            max_chars=CONFIG.embedding_max_chars,
        )
        if CONFIG.taxonomy_sample_scope == "domain":
            result = select_samples(items, client.embed, **common)
        else:
            result = select_samples_global(
                items, client.embed, min_cluster=CONFIG.taxonomy_sample_min_cluster,
                **common,
            )
        elapsed = time.monotonic() - t0
    finally:
        if own:
            client.close()

    out = {
        "version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "embedding_seconds": round(elapsed, 2),
        **result,
    }
    path = sampling_path()
    atomic_write_json(path, out)
    per_dom = ", ".join(
        f"{d}={v['selected']}" for d, v in out["domains"].items()
    )
    g = out.get("global") or {}
    extra = (
        f"; celý dataset: {g['embedded_items']} textů, {g['topic_clusters']} témat, "
        f"šum {g['noise_items']} textů, pokrytí {g['coverage']}"
        if g else ""
    )
    logger.info(
        f"taxonomy-sample: {out['total_selected']} ukázek ({settings['mode']}, "
        f"{settings['scope']}) za {elapsed:.1f}s → {path} [{per_dom}]{extra}"
    )
    return path


def load_samples_for_discovery(
    items: list[LabelItem], checkpoint_dir: Optional[Path] = None,
) -> Optional[tuple[dict[str, list[str]], str]]:
    """
    Načte výběr z fáze 0a pro fázi 0. Vrací ({doména: [texty]}, poznámka),
    nebo None, když soubor chybí / neodpovídá aktuálnímu nastavení / datům
    (pak fáze 0 použije náhodný výběr).
    """
    import json

    path = sampling_path(checkpoint_dir)
    if not path.exists():
        logger.warning(
            f"taxonomy: {path.name} neexistuje (fáze 0a 'sample' neproběhla) — "
            f"použiji náhodný výběr ukázek"
        )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"taxonomy: {path.name} nejde načíst ({e}) — náhodný výběr")
        return None
    if data.get("settings") != current_settings():
        logger.warning(
            f"taxonomy: {path.name} byl spočítán s jiným nastavením "
            f"({data.get('settings')}) — náhodný výběr. Spusť fázi 'sample' znovu."
        )
        return None

    text_by_id = {it.item_id: it.text for it in items}
    wanted = found = 0
    samples: dict[str, list[str]] = {}
    for domain, d in (data.get("domains") or {}).items():
        texts = []
        for s in d.get("samples", []):
            wanted += 1
            if s.get("item_id") in text_by_id:
                found += 1
                texts.append(text_by_id[s["item_id"]])
        if texts:
            samples[domain] = texts
    if not wanted or found / wanted < _MIN_FOUND_SHARE:
        logger.warning(
            f"taxonomy: {path.name} neodpovídá aktuálním datům "
            f"({found}/{wanted} ukázek nalezeno) — náhodný výběr"
        )
        return None

    counts = [len(v) for v in samples.values()]
    s = data["settings"]
    how = (
        f"auto (pokrytí {s['coverage']}, {min(counts)}–{max(counts)} per doména)"
        if s["mode"] == "auto" else f"{s['fixed_n']} per doména"
    )
    if s.get("scope") == "global":
        how += ", nejodlišnější témata z celého datasetu"
    note = f"vzorky vybrané embeddingy {s['model']}: {how}, {sum(counts)} celkem"
    logger.info(f"taxonomy: {note}")
    return samples, note
