"""
Unit testy pro labeler.taxonomy_sampling (fáze 0a — výběr vzorků embeddingy).

Bez GPU i bez serveru: embeddingy jsou syntetické (shluky kolem os), SGLang
/v1/embeddings je nahrazený httpx.MockTransport.

Pokrývá:
  - farthest-point výběr: první ukázky pokryjí různé shluky, pokrytí roste
  - auto počet: pestrá doména dostane víc ukázek než jednotvárná, mez MIN/MAX
  - fixní počet a globální strop (MAX_TOTAL)
  - klient /v1/embeddings: dávkování, pořadí podle index, chyba serveru
  - run → taxonomy_sampling.json → načtení pro fázi 0 (i zastaralý soubor)

Spuštění: python tests/test_taxonomy_sampling.py
"""

import json
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeler.config import CONFIG
from labeler.datasets import LabelItem
from labeler.taxonomy_sampling import (
    EmbeddingServerClient,
    EmbeddingServerError,
    apply_total_budget,
    choose_count,
    farthest_point_order,
    load_samples_for_discovery,
    select_samples,
)
import labeler.taxonomy_sampling as ts


def _cluster_embed(texts):
    """Text 'c<k>-...' → vektor kolem osy k (+ malý deterministický šum)."""
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        k = int(t.split("-")[0][1:])
        out[i, k] = 1.0
        out[i, (k + 1 + sum(map(ord, t)) % 15) % 16] += 0.05  # šum podle textu
    return out


def _items(domain, n_clusters, per_cluster, prefix):
    return [
        LabelItem(item_id=f"{prefix}{c}_{j}", source=prefix, domain=domain,
                  text=f"c{c}-{prefix} text {j} " + "x" * 50)
        for c in range(n_clusters) for j in range(per_cluster)
    ]


def _select(items, **kw):
    args = dict(auto=True, fixed_n=10, min_k=2, max_k=20, coverage_target=0.9,
                max_total=200, pool_max=1000, max_chars=2000)
    args.update(kw)
    return select_samples(items, _cluster_embed, **args)


# ---------- pure výběr -----------------------------------------------------

def test_farthest_point_covers_distinct_clusters_first():
    texts = [f"c{c}-t{j}" for c in range(4) for j in range(5)]
    emb = _cluster_embed(texts)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    order, cov = farthest_point_order(emb, 8)
    first4 = {texts[i].split("-")[0] for i in order[:4]}
    assert first4 == {"c0", "c1", "c2", "c3"}
    assert all(b >= a - 1e-6 for a, b in zip(cov, cov[1:]))  # pokrytí neklesá
    assert cov[3] > 0.9


def test_choose_count_auto_and_fixed():
    cov = [0.1, 0.4, 0.7, 0.95, 0.97]
    assert choose_count(cov, auto=True, fixed_n=0, min_k=1, max_k=10, target=0.9) == 4
    assert choose_count(cov, auto=True, fixed_n=0, min_k=1, max_k=2, target=0.9) == 2
    assert choose_count(cov, auto=True, fixed_n=0, min_k=5, max_k=10, target=0.1) == 5
    assert choose_count(cov, auto=False, fixed_n=3, min_k=1, max_k=10, target=0.9) == 3


def test_auto_gives_diverse_domain_more_samples():
    items = _items("pestrá", 8, 5, "p") + _items("jednotvárná", 2, 20, "j")
    res = _select(items)
    assert res["domains"]["pestrá"]["selected"] > res["domains"]["jednotvárná"]["selected"]
    assert res["domains"]["pestrá"]["selected"] >= 8          # každý shluk zástupce
    assert res["domains"]["jednotvárná"]["selected"] >= 2
    assert res["domains"]["pestrá"]["coverage"] >= 0.9


def test_fixed_count_and_total_budget():
    items = _items("a", 6, 5, "a") + _items("b", 6, 5, "b")
    res = _select(items, auto=False, fixed_n=5)
    assert [d["selected"] for d in res["domains"].values()] == [5, 5]
    res = _select(items, auto=False, fixed_n=10, max_total=12)
    assert res["total_selected"] <= 12
    assert apply_total_budget({"a": 30, "b": 10}, 20, 2) == {"a": 15, "b": 5}


# ---------- výběr přes celý dataset (scope=global) ---------------------------

def _topic_embed(texts):
    """Text 't<k>-s<j>-...' → vektor na ose k s odchylkou j*0.1 na ose 15
    (vzdálenost od 'typického' středu tématu); 'x<k>-...' = odlehlý šum na ose k."""
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        head = t.split(" ")[0]
        if head.startswith("x"):
            out[i, int(head[1:].split("-")[0])] = 1.0
            continue
        k, j = head[1:].split("-s")
        out[i, int(k)] = 1.0
        out[i, 15] = int(j) * 0.1
    return out


def _it(domain, iid, text):
    return LabelItem(item_id=iid, source=domain, domain=domain, text=text + " " + "y" * 40)


def _global(items, **kw):
    args = dict(auto=True, fixed_n=10, min_k=1, max_k=50, coverage_target=0.9,
                max_total=200, pool_max=0, max_chars=2000, min_cluster=3)
    args.update(kw)
    return ts.select_samples_global(items, _topic_embed, **args)


def _chosen(res):
    return [s["item_id"] for d in res["domains"].values() for s in d["samples"]]


def test_global_shared_topic_selected_once_and_noise_skipped():
    items = []
    # téma 0 = sdílená patička v obou doménách, témata 1-2 jen v A, 3-4 jen v B
    for dom, topics in (("A", (0, 1, 2)), ("B", (0, 3, 4))):
        for k in topics:
            for j in range(5):
                items.append(_it(dom, f"{dom}-t{k}-{j}", f"t{k}-s{j}"))
    items.append(_it("A", "A-noise", "x9-odkazy"))        # osamocený šum
    res = _global(items)
    chosen = _chosen(res)
    topics = [cid.split("-")[1] for cid in chosen]
    assert sorted(set(topics)) == ["t0", "t1", "t2", "t3", "t4"]
    assert topics.count("t0") == 1                         # sdílený obsah jednou
    assert "A-noise" not in chosen                          # šum se nevybírá
    assert res["global"]["noise_items"] == 1
    assert res["global"]["coverage"] >= 0.9


def test_global_representative_is_typical_not_extreme():
    items = [_it("A", f"A-{j}", f"t1-s{j}") for j in range(5)]
    items += [_it("A", f"B-{j}", f"t2-s{j}") for j in range(5)]
    res = _global(items)
    chosen = set(_chosen(res))
    assert "A-4" not in chosen and "B-4" not in chosen     # ne nejvzdálenější okraj
    assert {"A-2", "B-2"} <= chosen                         # nejblíž průměru tématu


def test_global_every_domain_represented_and_total_cap():
    items = [_it("A", f"A-{k}-{j}", f"t{k}-s{j}") for k in range(10) for j in range(4)]
    items += [_it("B", f"B-{j}", f"t1-s{j}") for j in range(4)]   # B = kopie tématu z A
    res = _global(items, min_k=2, max_total=6)
    assert res["domains"]["B"]["selected"] >= 2
    assert res["total_selected"] <= 6


# ---------- klient /v1/embeddings -------------------------------------------

def _mock_server(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "BAAI/bge-m3"}]})
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        vecs = _cluster_embed(body["input"]).tolist()
        # schválně obrácené pořadí — klient musí řadit podle index
        data = [{"index": i, "embedding": v} for i, v in enumerate(vecs)][::-1]
        return httpx.Response(200, json={"data": data})
    return httpx.MockTransport(handler)


def test_client_batches_and_orders_by_index():
    calls = []
    client = EmbeddingServerClient("http://x/v1", "BAAI/bge-m3", batch_size=4,
                                   transport=_mock_server(calls))
    texts = [f"c{i % 3}-t{i}" for i in range(10)]
    emb = client.embed(texts)
    assert calls == [4, 4, 2]
    assert np.allclose(emb, _cluster_embed(texts))


def test_client_raises_on_client_error():
    client = EmbeddingServerClient(
        "http://x/v1", "m", transport=httpx.MockTransport(
            lambda r: httpx.Response(400, text="bad model")),
    )
    try:
        client.embed(["a"])
        raise AssertionError("čekal jsem EmbeddingServerError")
    except EmbeddingServerError:
        pass


# ---------- run → soubor → fáze 0 -------------------------------------------

def test_run_and_load_for_discovery_roundtrip_and_staleness():
    items = _items("dom", 4, 5, "d")
    orig_ckpt, orig_scan = CONFIG.checkpoint_dir, ts.scan_data_dir
    ts.scan_data_dir = lambda _dir: items
    try:
        with tempfile.TemporaryDirectory() as tmp:
            object.__setattr__(CONFIG, "checkpoint_dir", Path(tmp))
            client = EmbeddingServerClient("http://x/v1", "BAAI/bge-m3",
                                           transport=_mock_server([]))
            path = ts.run_taxonomy_sampling(client)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["settings"]["model"] == CONFIG.taxonomy_embedding_model
            loaded = load_samples_for_discovery(items)
            assert loaded is not None
            samples, note = loaded
            assert set(samples) == {"dom"} and len(samples["dom"]) >= 4
            assert "embeddingy" in note

            # jiné nastavení → soubor neplatí → fáze 0 spadne na náhodný výběr
            data["settings"]["coverage"] = 0.123
            path.write_text(json.dumps(data), encoding="utf-8")
            assert load_samples_for_discovery(items) is None

            # jiná data (item_id nesedí) → neplatí
            path.unlink()
            ts.run_taxonomy_sampling(EmbeddingServerClient(
                "http://x/v1", "m", transport=_mock_server([])))
            other = _items("dom", 4, 5, "zz")
            assert load_samples_for_discovery(other) is None
    finally:
        object.__setattr__(CONFIG, "checkpoint_dir", orig_ckpt)
        ts.scan_data_dir = orig_scan


def test_discovery_prompt_uses_embedding_samples():
    """Fáze 0 pošle LLM ukázky z taxonomy_sampling.json a zapíše to do notes."""
    import asyncio
    import labeler.taxonomy as tax_mod
    from test_taxonomy import _FakeResp, _good_labels_payload

    items = _items("dom", 4, 5, "d")
    seen = {}

    class FakeClient:
        model = "fake-model"

        async def chat(self, messages, **kw):
            seen["prompt"] = messages[-1]["content"]
            return _FakeResp(_good_labels_payload())

    orig_ckpt = CONFIG.checkpoint_dir
    orig_ts_scan, orig_tax_scan = ts.scan_data_dir, tax_mod.scan_data_dir
    ts.scan_data_dir = tax_mod.scan_data_dir = lambda _dir: items
    try:
        with tempfile.TemporaryDirectory() as tmp:
            object.__setattr__(CONFIG, "checkpoint_dir", Path(tmp))
            path = ts.run_taxonomy_sampling(EmbeddingServerClient(
                "http://x/v1", "m", transport=_mock_server([])))
            chosen = json.loads(path.read_text(encoding="utf-8"))["domains"]["dom"]["samples"]
            tax = asyncio.run(tax_mod.discover_taxonomy_auto(extractor_client=FakeClient()))
    finally:
        object.__setattr__(CONFIG, "checkpoint_dir", orig_ckpt)
        ts.scan_data_dir, tax_mod.scan_data_dir = orig_ts_scan, orig_tax_scan
    # každý vybraný text je v promptu, a žádný jiný z domény (výběr, ne náhoda)
    by_id = {it.item_id: it.text for it in items}
    for s in chosen:
        assert by_id[s["item_id"]][:600] in seen["prompt"]
    assert seen["prompt"].count("  [") == len(chosen)
    assert "embeddingy" in tax.notes


# ---------- Runner ---------------------------------------------------------

if __name__ == "__main__":
    import inspect
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
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR  {name}: {type(e).__name__}: {e}")
    print(f"\nTotal: {len(tests)} tests, {failed} failed.")
    sys.exit(1 if failed else 0)
