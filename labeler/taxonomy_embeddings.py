"""
Fáze 0b — Embedding analýza sémantické disjunktnosti taxonomie.

Běží AUTOMATICKY po každé tvorbě taxonomie (smoke test i produkce, viz
taxonomy._run_embedding_analysis_best_effort); vypnutí TAXONOMY_EMBEDDING_ENABLED=false.

Pro třídy z `entity_taxonomy.json` vytvoří sentence embeddingy z jejich
textových profilů (kód, popis, aliasy, domény, příklady), spočítá kosinovou
podobnost všech dvojic, provede deterministický aglomerativní clustering a
klasifikuje dvojice jako:

    possible_duplicate   — třídy jsou natolik podobné, že může jít o duplicity
    possible_overlap     — třídy mohou popisovat překrývající se koncepty
    disjoint_candidate   — třídy jsou v embedding prostoru daleko od sebe
    uncertain            — nejednoznačné, vyžaduje ruční kontrolu

DŮLEŽITÉ: embeddingová vzdálenost NENÍ formální logický důkaz disjunktnosti.
Všechny výstupy nesou `formal_disjointness_proof: false` a heuristickou
terminologii (disjoint_candidate, possible_overlap, …). `confidence` je
heuristické skóre, NE kalibrovaná pravděpodobnost.

Architektura:
  - Samostatná CPU/GPU fáze — ŽÁDNÝ zásah do extractor/critic/revise.
  - Deterministické při stejných datech, konfiguraci a seedu (řazení podle
    kódů tříd, stabilní přemapování clusterů, fixní seed).
  - Best-effort vůči checkpointům: čte taxonomii, zapisuje POUZE vlastní
    výstupy (atomic write) — nikdy nesahá na extractions/critic/final soubory.
  - Embedding backend je vyměnitelná abstrakce (Protocol); default
    Sentence Transformers (v requirements.txt) — importuje se lazy,
    nikdy při importu tohoto modulu.
  - Clustering: vlastní deterministická average-linkage aglomerace v numpy.
    Projekt scikit-learn NEPOUŽÍVÁ — přidávat ho kvůli ~30 řádkům by byla
    zbytečná závislost; vlastní implementace je pro nižší stovky tříd
    dostatečná (O(n^3)) a plně deterministická.

Výstupy (checkpoint dir, atomicky):
    taxonomy_embedding_report.json   — strojový report (Pydantic modely)
    taxonomy_embedding_report.md     — lidsky čitelný report
    taxonomy_embeddings.json         — metadata + profily (+ volitelně vektory);
                                       slouží zároveň jako cache
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .io_utils import atomic_write_json, atomic_write_text, text_hash
from .taxonomy import Taxonomy, TaxonomyLabel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Verze způsobu sestavení profilu — součást cache klíče: změna formátu
# profilu invaliduje cache embeddingů.
PROFILE_BUILDER_VERSION = 1
METHOD_NAME = "semantic_embedding_clustering"
HEURISTIC_WARNING = (
    "Embedding similarity is heuristic and is not a formal OWL disjointness proof."
)

EMBEDDINGS_FILE = "taxonomy_embeddings.json"
REPORT_JSON_FILE = "taxonomy_embedding_report.json"
REPORT_MD_FILE = "taxonomy_embedding_report.md"

CLASSIFICATION_DUPLICATE = "possible_duplicate"
CLASSIFICATION_OVERLAP = "possible_overlap"
CLASSIFICATION_DISJOINT = "disjoint_candidate"
CLASSIFICATION_UNCERTAIN = "uncertain"


class TaxonomyEmbeddingError(Exception):
    """Srozumitelná chyba embedding analýzy (chybějící taxonomie, nevalidní
    embeddingy, špatné prahy…). Nezakrývá původní výjimku (raise ... from)."""


# ---------------------------------------------------------------------------
# 1) Konfigurace (Pydantic — validace prahů PŘED načtením modelu)
# ---------------------------------------------------------------------------


class TaxonomyEmbeddingConfig(BaseModel):
    """Konfigurace analýzy. Validuje prahy při konstrukci — tj. dřív, než se
    sahá na embedding model (CLI s neplatnými prahy skončí okamžitě)."""

    backend: str = "sentence_transformers"
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = Field(32, ge=1)
    normalize: bool = True
    # Mean-centering PŘED normalizací — odstraní anizotropii (společnou
    # komponentu všech vektorů). Bez něj se všechny podobnosti tísní ve
    # vysokém pásmu (typicky 0.5-0.85) a absolutní prahy nedávají smysl.
    # Vyžaduje >= 3 třídy (u 1-2 tříd je centrování degenerované → skipuje se).
    center: bool = True
    seed: int = 42
    # --- Klasifikace: 'relative' (default) vs 'absolute' -------------------
    # relative — prahy v jednotkách robustního z-skóre nad DISTRIBUCÍ reálných
    #            podobností dané taxonomie (median + škálovaná MAD). Funguje
    #            napříč modely a jazyky bez ručního ladění.
    # absolute — pevné prahy kosinové podobnosti (legacy chování).
    threshold_mode: str = "relative"
    duplicate_threshold: float = Field(0.90, ge=-1.0, le=1.0)
    overlap_threshold: float = Field(0.72, ge=-1.0, le=1.0)
    disjoint_threshold: float = Field(0.30, ge=-1.0, le=1.0)
    duplicate_z: float = 2.5
    overlap_z: float = 1.0
    disjoint_z: float = -0.5
    clustering_algorithm: str = "agglomerative"
    cluster_distance_threshold: float = Field(0.35, ge=0.0, le=2.0)
    write_vectors: bool = False

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> "TaxonomyEmbeddingConfig":
        if self.threshold_mode not in ("relative", "absolute"):
            raise ValueError(
                f"Neplatný threshold_mode {self.threshold_mode!r} "
                f"(podporované: 'relative', 'absolute')."
            )
        if not (
            self.disjoint_threshold
            < self.overlap_threshold
            <= self.duplicate_threshold
        ):
            raise ValueError(
                f"Neplatné prahy: musí platit disjoint ({self.disjoint_threshold}) "
                f"< overlap ({self.overlap_threshold}) <= duplicate "
                f"({self.duplicate_threshold})."
            )
        if not (self.disjoint_z < self.overlap_z <= self.duplicate_z):
            raise ValueError(
                f"Neplatné z-prahy: musí platit disjoint_z ({self.disjoint_z}) "
                f"< overlap_z ({self.overlap_z}) <= duplicate_z "
                f"({self.duplicate_z})."
            )
        if self.threshold_mode == "relative" and not (
            0.0 <= self.cluster_distance_threshold <= 1.0
        ):
            raise ValueError(
                f"V 'relative' módu je cluster_distance_threshold KVANTIL a "
                f"musí být v <0, 1>; dostal jsem "
                f"{self.cluster_distance_threshold}."
            )
        if self.clustering_algorithm != "agglomerative":
            raise ValueError(
                f"Nepodporovaný clustering algoritmus "
                f"{self.clustering_algorithm!r} (podporován: 'agglomerative')."
            )
        return self

    @classmethod
    def from_env(cls) -> "TaxonomyEmbeddingConfig":
        """Konfigurace z CONFIG (env) — rozšiřuje existující env systém
        projektu, žádný paralelní loader."""
        from .config import CONFIG
        return cls(
            backend=CONFIG.taxonomy_embedding_backend,
            model=CONFIG.taxonomy_embedding_model,
            device=CONFIG.taxonomy_embedding_device,
            batch_size=CONFIG.taxonomy_embedding_batch_size,
            normalize=CONFIG.taxonomy_embedding_normalize,
            center=CONFIG.taxonomy_embedding_center,
            seed=CONFIG.taxonomy_embedding_seed,
            threshold_mode=CONFIG.taxonomy_threshold_mode,
            duplicate_threshold=CONFIG.taxonomy_duplicate_similarity_threshold,
            overlap_threshold=CONFIG.taxonomy_overlap_similarity_threshold,
            disjoint_threshold=CONFIG.taxonomy_disjoint_similarity_threshold,
            duplicate_z=CONFIG.taxonomy_duplicate_z,
            overlap_z=CONFIG.taxonomy_overlap_z,
            disjoint_z=CONFIG.taxonomy_disjoint_z,
            clustering_algorithm=CONFIG.taxonomy_clustering_algorithm,
            cluster_distance_threshold=CONFIG.taxonomy_cluster_distance_threshold,
            write_vectors=CONFIG.taxonomy_embedding_write_vectors,
        )


# ---------------------------------------------------------------------------
# Report modely (Pydantic — žádné ad hoc slovníky)
# ---------------------------------------------------------------------------


class PairResult(BaseModel):
    """Výsledek jedné dvojice tříd (i < j; každá dvojice právě jednou)."""

    class_a: str
    class_b: str
    cosine_similarity: float
    cosine_distance: float
    # Robustní z-skóre podobnosti v rámci TÉTO taxonomie: (sim - median) /
    # (1.4826 * MAD). Kladné = pár je podobnější než typický pár, záporné =
    # vzdálenější. V 'relative' módu na něm stojí klasifikace.
    similarity_z: float = 0.0
    same_cluster: bool
    classification: str
    # Heuristické skóre 0-1 (vzdálenost od prahu) — NENÍ kalibrovaná
    # pravděpodobnost.
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class SimilarityDistribution(BaseModel):
    """Statistiky distribuce párových podobností (audit + interpretace
    z-skóre v reportu)."""

    count: int
    min: float
    max: float
    mean: float
    std: float
    median: float
    # Škálovaná MAD (1.4826 × medián absolutních odchylek) — robustní
    # obdoba směrodatné odchylky; jmenovatel z-skóre.
    scaled_mad: float


class ClusterInfo(BaseModel):
    """Jeden cluster — ID je stabilně přemapované (lexikograficky nejmenší
    kód třídy určuje pořadí), labels seřazené."""

    cluster_id: int
    labels: list[str]


class ReportSummary(BaseModel):
    label_count: int
    pair_count: int
    possible_duplicate_count: int
    possible_overlap_count: int
    disjoint_candidate_count: int
    uncertain_count: int


class TaxonomyEmbeddingReport(BaseModel):
    """Kompletní auditní report analýzy (taxonomy_embedding_report.json)."""

    version: int = SCHEMA_VERSION
    method: str = METHOD_NAME
    formal_disjointness_proof: bool = False
    taxonomy_path: str
    taxonomy_hash: str
    backend: str
    model: str
    embedding_dimension: int
    # 'relative' (z-skóre nad distribucí) nebo 'absolute' (pevné prahy)
    threshold_mode: str = "relative"
    centered: bool = False
    thresholds: dict[str, float]
    cluster_distance_threshold: float
    similarity_distribution: Optional[SimilarityDistribution] = None
    clusters: list[ClusterInfo]
    pairs: list[PairResult]
    summary: ReportSummary
    warnings: list[str] = Field(default_factory=lambda: [HEURISTIC_WARNING])


# ---------------------------------------------------------------------------
# 2) Načtení a validace taxonomie
# ---------------------------------------------------------------------------


def load_taxonomy_for_analysis(taxonomy_path: Path) -> Taxonomy:
    """Načte a zvaliduje taxonomii pro analýzu. Labely vrací seřazené podle
    kódu (determinismus všech navazujících kroků)."""
    if not taxonomy_path.exists():
        raise TaxonomyEmbeddingError(
            f"Taxonomie neexistuje: {taxonomy_path}. Vygeneruj ji přes "
            f"`python -m labeler.main taxonomy --mode auto|user`."
        )
    try:
        raw = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as e:
        raise TaxonomyEmbeddingError(
            f"Taxonomii {taxonomy_path} nelze načíst: {e}"
        ) from e
    tax = Taxonomy.from_dict(raw)
    if not tax.labels:
        raise TaxonomyEmbeddingError(
            f"Taxonomie {taxonomy_path} neobsahuje žádné labely."
        )
    codes = [lb.code for lb in tax.labels]
    dup = sorted({c for c in codes if codes.count(c) > 1})
    if dup:
        raise TaxonomyEmbeddingError(
            f"Taxonomie {taxonomy_path} má duplicitní kódy tříd: {dup}. "
            f"Oprav taxonomii (aliasová deduplikace je v labeler/taxonomy.py)."
        )
    tax.labels = sorted(tax.labels, key=lambda lb: lb.code)
    return tax


# Pole, která do taxonomie zapisuje SAMA embedding analýza (anotace) — do hashe
# obsahu nepatří, jinak by anotace invalidovala cache a report by odkazoval na
# jiný hash než anotovaný soubor.
_ANNOTATION_LABEL_KEYS = ("confusable_with", "disjoint_candidates", "embedding_cluster")


def taxonomy_content_hash(tax: Taxonomy) -> str:
    """Deterministický hash obsahu taxonomie (pro cache a audit). Ignoruje
    embeddingové anotace (výstup této analýzy, ne její vstup)."""
    d = tax.to_dict()
    d.pop("embedding_analysis", None)
    for lb in d.get("labels", []):
        for k in _ANNOTATION_LABEL_KEYS:
            lb.pop(k, None)
    canonical = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return text_hash(canonical)


# ---------------------------------------------------------------------------
# 3) Textový profil třídy
# ---------------------------------------------------------------------------


def build_taxonomy_label_profile(label: TaxonomyLabel) -> str:
    """
    Sestaví stabilní textový profil třídy pro embedding.

    Pravidla: jen dostupná pole, deterministické pořadí, seznamy seřazené,
    prázdné hodnoty vynechané, čeština s diakritikou zachovaná. Pole
    force_text_match a unsafe_texts se ZÁMĚRNĚ nepoužívají — jsou to
    normalizační heuristiky, ne významová definice třídy.
    """
    lines = [f"Kód: {label.code}"]
    desc = (label.description or "").strip()
    if desc:
        lines.append(f"Popis: {desc}")
    aliases = sorted({a.strip() for a in label.aliases if a.strip()})
    if aliases:
        lines.append(f"Alias: {', '.join(aliases)}")
    domains = sorted({d.strip() for d in label.domains if d.strip()})
    if domains:
        lines.append(f"Domény: {', '.join(domains)}")
    examples = sorted({e.strip() for e in label.examples if e.strip()})
    if examples:
        lines.append(f"Příklady: {'; '.join(examples)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4) Embedding backend (vyměnitelná abstrakce)
# ---------------------------------------------------------------------------


class EmbeddingBackend(Protocol):
    """Protokol backendu: encode() vrací float32 matici
    (počet_textů, dimenze). Implementace nesmí stahovat model při importu
    modulu — až při konstrukci/prvním použití."""

    name: str
    model_name: str

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        ...


class SentenceTransformersBackend:
    """Lokální Sentence Transformers backend.

    Import knihovny probíhá až v konstruktoru — chybějící závislost dá
    srozumitelnou chybu s odkazem na requirements.txt.
    """

    name = "sentence_transformers"

    def __init__(self, model_name: str, device: str = "auto", seed: int = 42):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise TaxonomyEmbeddingError(
                "Závislost 'sentence-transformers' není nainstalovaná. "
                "Instaluj: pip install -r requirements.txt"
            ) from e
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        resolved_device = None if device == "auto" else device
        try:
            self._model = SentenceTransformer(
                model_name, device=resolved_device,
            )
        except Exception as e:
            raise TaxonomyEmbeddingError(
                f"Embedding model {model_name!r} nelze načíst "
                f"({type(e).__name__}: {e}). Zkontroluj název modelu, "
                f"připojení a paměť."
            ) from e

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """Dávkové embeddingy jako float32 matice (n_texts, dim)."""
        emb = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


# Načtené backendy per (backend, model, device, seed) — fáze taxonomie používá
# embeddingy v několika krocích (audit, zostření, finální analýza) a model se
# má načíst jen jednou za proces.
_BACKEND_CACHE: dict[tuple, EmbeddingBackend] = {}


def create_backend(config: TaxonomyEmbeddingConfig) -> EmbeddingBackend:
    """Vytvoří (nebo vrátí už načtený) backend podle konfigurace."""
    if config.backend == "sentence_transformers":
        key = (config.backend, config.model, config.device, config.seed)
        if key not in _BACKEND_CACHE:
            _BACKEND_CACHE[key] = SentenceTransformersBackend(
                model_name=config.model, device=config.device, seed=config.seed,
            )
        return _BACKEND_CACHE[key]
    raise TaxonomyEmbeddingError(
        f"Neznámý embedding backend {config.backend!r} "
        f"(podporován: 'sentence_transformers')."
    )


# ---------------------------------------------------------------------------
# 5) Validace + normalizace embeddingů
# ---------------------------------------------------------------------------


def validate_embeddings(
    embeddings: np.ndarray, expected_rows: int, *, normalize: bool,
) -> None:
    """Validuje tvar a numerickou zdravost embedding matice."""
    if embeddings.ndim != 2:
        raise TaxonomyEmbeddingError(
            f"Embeddingy musí být 2D matice, dostal jsem ndim={embeddings.ndim}."
        )
    if embeddings.shape[0] != expected_rows:
        raise TaxonomyEmbeddingError(
            f"Špatný počet embeddingů: {embeddings.shape[0]} řádků pro "
            f"{expected_rows} tříd."
        )
    if embeddings.shape[1] == 0:
        raise TaxonomyEmbeddingError("Embedding dimenze je 0.")
    if np.isnan(embeddings).any():
        raise TaxonomyEmbeddingError("Embeddingy obsahují NaN hodnoty.")
    if np.isinf(embeddings).any():
        raise TaxonomyEmbeddingError("Embeddingy obsahují nekonečné hodnoty.")
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        if (norms == 0).any():
            zero_idx = np.where(norms == 0)[0].tolist()
            raise TaxonomyEmbeddingError(
                f"Nulové embedding vektory na indexech {zero_idx} — "
                f"nelze normalizovat."
            )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalizace řádků (float32)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


# Minimální počet tříd, aby mělo centrování smysl. Pro n=1 je centrovaný
# vektor nulový (nelze normalizovat), pro n=2 vyjdou vektory přesně opačné
# (podobnost vždy -1) — v obou případech se centrování přeskakuje.
_MIN_LABELS_FOR_CENTERING = 3


def center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Odečte průměrný vektor (mean removal / "all-but-the-mean").

    Sentence embeddingy jsou ANIZOTROPNÍ — leží v úzkém kuželu se silnou
    společnou komponentou (u profilů taxonomie ji navíc posiluje sdílený
    boilerplate "Kód:/Popis:/Příklady:"). Bez centrování se všechny párové
    podobnosti tísní ve vysokém pásmu (typicky 0.5-0.85) a rozdíly mezi
    sémanticky blízkými a vzdálenými třídami se ztrácejí v šumu.

    Po centrování je znaménko podobnosti interpretovatelné: kladná = pár je
    podobnější než průměrný pár, záporná = vzdálenější.
    """
    return (embeddings - embeddings.mean(axis=0, keepdims=True)).astype(np.float32)


def apply_embedding_transform(
    embeddings: np.ndarray, *, center: bool, normalize: bool,
) -> np.ndarray:
    """
    Deterministická transformace embeddingů: centrování → L2 normalizace.

    Centrování se přeskočí pro < 3 třídy (degenerované, viz
    `_MIN_LABELS_FOR_CENTERING`).
    """
    out = np.asarray(embeddings, dtype=np.float32)
    if center:
        if out.shape[0] >= _MIN_LABELS_FOR_CENTERING:
            out = center_embeddings(out)
        else:
            logger.info(
                f"taxonomy-embeddings: < {_MIN_LABELS_FOR_CENTERING} tříd — "
                f"centrování přeskočeno (degenerované)"
            )
    if normalize:
        norms = np.linalg.norm(out, axis=1)
        if (norms == 0).any():
            raise TaxonomyEmbeddingError(
                "Po centrování vznikl nulový vektor — třída má embedding "
                "shodný s průměrem. Zkus TAXONOMY_EMBEDDING_CENTER=false."
            )
        out = normalize_embeddings(out)
    return out


def robust_similarity_z_scores(
    values: np.ndarray,
) -> tuple[np.ndarray, SimilarityDistribution]:
    """
    Robustní z-skóre párových podobností: (x - median) / (1.4826 * MAD).

    Medián + MAD (ne průměr + std), aby pár outlierů neroztáhl škálu.

    Degenerované případy: když je většina podobností (skoro) shodná, MAD klesá
    k nule a naivní dělení by z-skóre vyhnalo do milionů (a zesílilo by
    float šum na nesmyslné rozdíly). Měřítko má proto podlahu RELATIVNÍ
    k rozpětí dat: MAD → std → rozpětí. Když je konstantní i rozpětí, vrátí se
    samé nuly (vše skončí jako 'uncertain' — správně, není co rozlišovat).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        empty = SimilarityDistribution(
            count=0, min=0.0, max=0.0, mean=0.0, std=0.0,
            median=0.0, scaled_mad=0.0,
        )
        return np.zeros(0, dtype=np.float64), empty
    median = float(np.median(v))
    scaled_mad = float(1.4826 * np.median(np.abs(v - median)))
    spread = float(v.max() - v.min())
    std = float(v.std())
    # Podlaha měřítka relativní k rozpětí — brání explozi z-skóre při MAD≈0.
    floor = max(1e-12, spread * 1e-6)
    if scaled_mad > floor:
        scale = scaled_mad
    elif std > floor:
        scale = std
    elif spread > 1e-12:
        scale = spread
    else:
        scale = 0.0
    z = (v - median) / scale if scale > 0.0 else np.zeros_like(v)
    dist = SimilarityDistribution(
        count=int(v.size),
        min=round(float(v.min()), 6),
        max=round(float(v.max()), 6),
        mean=round(float(v.mean()), 6),
        std=round(float(v.std()), 6),
        median=round(median, 6),
        scaled_mad=round(scaled_mad, 6),
    )
    return z, dist


# ---------------------------------------------------------------------------
# 6) Kosinová podobnost
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(normalized_embeddings: np.ndarray) -> np.ndarray:
    """Symetrická matice kosinových podobností s diagonálou 1.0, ořezaná
    do [-1, 1]. Vstup MUSÍ být L2-normalizovaný (maticový součin, žádná
    Python smyčka)."""
    sim = normalized_embeddings @ normalized_embeddings.T
    sim = ((sim + sim.T) / 2.0).astype(np.float32)   # numerická symetrizace
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)
    return sim


# ---------------------------------------------------------------------------
# 7) Clustering — deterministická average-linkage aglomerace (numpy)
# ---------------------------------------------------------------------------


def resolve_cluster_distance_threshold(
    similarity: np.ndarray, threshold: float, threshold_mode: str,
) -> float:
    """
    Přeloží konfigurovaný práh clusteringu na ABSOLUTNÍ kosinovou vzdálenost.

    V 'relative' módu je `threshold` KVANTIL (0-1) pozorovaných párových
    vzdáleností — slučuje se v rámci nejbližších X % dvojic. Absolutní práh
    tu nefunguje: po centrování jsou typické vzdálenosti kolem 1.1 (podobnosti
    jsou z definice většinou záporné), takže pevná 0.35 by nesloučila nic a
    všechny třídy by zůstaly samostatné.

    V 'absolute' módu se `threshold` použije přímo.
    """
    if threshold_mode != "relative":
        return threshold
    n = similarity.shape[0]
    if n < 2:
        return threshold
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - similarity[iu].astype(np.float64)
    q = min(max(threshold, 0.0), 1.0)
    return float(np.quantile(distances, q))


def agglomerative_average_cosine(
    similarity: np.ndarray, codes: list[str], distance_threshold: float,
) -> list[int]:
    """
    Aglomerativní clustering s kosinovou vzdáleností (1 - podobnost) a
    average linkage. Vrací cluster ID per index (pořadí = pořadí `codes`).

    Determinismus: kandidátní merge s nejmenší průměrnou vzdáleností;
    při shodě vyhrává dvojice s lexikograficky nejmenšími kódy. Výsledná
    cluster ID jsou přemapovaná podle lexikograficky nejmenšího kódu člena
    (nezávislé na pořadí merge kroků).

    Pro n < 2 vrací triviální přiřazení bez clusteringu.
    """
    n = similarity.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    dist = 1.0 - similarity.astype(np.float64)
    clusters: list[set[int]] = [{i} for i in range(n)]

    def avg_dist(a: set[int], b: set[int]) -> float:
        return float(dist[np.ix_(sorted(a), sorted(b))].mean())

    def merge_sort_key(ai: int, bi: int) -> tuple:
        ca = min(codes[i] for i in clusters[ai])
        cb = min(codes[i] for i in clusters[bi])
        return tuple(sorted((ca, cb)))

    while len(clusters) > 1:
        best: Optional[tuple[float, tuple, int, int]] = None
        for ai in range(len(clusters)):
            for bi in range(ai + 1, len(clusters)):
                d = avg_dist(clusters[ai], clusters[bi])
                key = (round(d, 12), merge_sort_key(ai, bi), ai, bi)
                if best is None or key < best:
                    best = key
        d_best, _key, ai, bi = best[0], best[1], best[2], best[3]
        if d_best > distance_threshold:
            break
        clusters[ai] = clusters[ai] | clusters[bi]
        del clusters[bi]

    # Stabilní přemapování ID: clustery seřazené podle nejmenšího kódu člena.
    ordered = sorted(clusters, key=lambda c: min(codes[i] for i in c))
    assignment = [0] * n
    for cluster_id, members in enumerate(ordered):
        for i in members:
            assignment[i] = cluster_id
    return assignment


# ---------------------------------------------------------------------------
# 8) Klasifikace dvojic
# ---------------------------------------------------------------------------


def classify_pair(
    similarity: float,
    same_cluster: bool,
    config: TaxonomyEmbeddingConfig,
    similarity_z: float = 0.0,
) -> tuple[str, float, list[str]]:
    """
    Klasifikuje dvojici. Cluster je jen DOPLŇKOVÝ signál (přidá reason),
    klasifikace stojí na podobnosti.

    threshold_mode='relative' (default): rozhoduje robustní z-skóre nad
    distribucí podobností dané taxonomie + ZNAMÉNKOVÁ POJISTKA — overlap
    vyžaduje podobnost nad mediánem, disjoint pod ním. Bez ní by percentilová
    logika vždy něco označila i v dokonale disjunktní taxonomii.

    threshold_mode='absolute': pevné prahy kosinové podobnosti (legacy).

    Vrací (classification, confidence, reasons). Confidence je heuristické
    skóre 0.5-1.0 (jak hluboko za prahem hodnota je) — NE kalibrovaná
    pravděpodobnost; pro 'uncertain' je fixně nízké (0.25).
    """
    eps = 1e-9
    reasons: list[str] = []
    classification = CLASSIFICATION_UNCERTAIN
    confidence = 0.25

    if config.threshold_mode == "relative":
        # Znaménková pojistka: 'similarity_z' je centrované na medián, takže
        # z > 0 == podobnější než typický pár, z < 0 == vzdálenější.
        if similarity_z >= config.duplicate_z:
            classification = CLASSIFICATION_DUPLICATE
            depth = (similarity_z - config.duplicate_z) / max(
                config.duplicate_z, 1.0
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("very_high_relative_similarity")
        elif similarity_z >= config.overlap_z and similarity_z > 0:
            classification = CLASSIFICATION_OVERLAP
            depth = (similarity_z - config.overlap_z) / max(
                config.duplicate_z - config.overlap_z, eps
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("high_relative_similarity")
        elif similarity_z <= config.disjoint_z and similarity_z < 0:
            classification = CLASSIFICATION_DISJOINT
            depth = (config.disjoint_z - similarity_z) / max(
                abs(config.disjoint_z) + 1.0, eps
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("low_relative_similarity")
        else:
            reasons.append("mid_relative_similarity")
    else:  # absolute
        if similarity >= config.duplicate_threshold:
            classification = CLASSIFICATION_DUPLICATE
            depth = (similarity - config.duplicate_threshold) / max(
                1.0 - config.duplicate_threshold, eps
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("very_high_semantic_similarity")
        elif similarity >= config.overlap_threshold:
            classification = CLASSIFICATION_OVERLAP
            depth = (similarity - config.overlap_threshold) / max(
                config.duplicate_threshold - config.overlap_threshold, eps
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("high_semantic_similarity")
        elif similarity <= config.disjoint_threshold:
            classification = CLASSIFICATION_DISJOINT
            depth = (config.disjoint_threshold - similarity) / max(
                config.disjoint_threshold + 1.0, eps
            )
            confidence = 0.5 + 0.5 * min(depth, 1.0)
            reasons.append("low_semantic_similarity")
        else:
            reasons.append("mid_semantic_similarity")

    if same_cluster:
        reasons.append("same_cluster")
    else:
        reasons.append("different_cluster")
    return classification, round(confidence, 4), reasons


def build_pair_results(
    codes: list[str],
    similarity: np.ndarray,
    cluster_assignment: list[int],
    config: TaxonomyEmbeddingConfig,
) -> tuple[list[PairResult], Optional[SimilarityDistribution]]:
    """
    Vygeneruje každou dvojici právě jednou (i < j), stabilně seřazenou podle
    (class_a, class_b). Diagonála se nikdy nedostane do výsledků.

    Vrací (pairs, distribution). Distribuce je None, když nejsou žádné páry.
    """
    n = len(codes)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not idx:
        return [], None
    sims = np.array([float(similarity[i, j]) for i, j in idx], dtype=np.float64)
    z_scores, distribution = robust_similarity_z_scores(sims)

    pairs: list[PairResult] = []
    for (i, j), sim, z in zip(idx, sims, z_scores):
        same_cluster = cluster_assignment[i] == cluster_assignment[j]
        classification, confidence, reasons = classify_pair(
            float(sim), same_cluster, config, similarity_z=float(z),
        )
        pairs.append(PairResult(
            class_a=codes[i],
            class_b=codes[j],
            cosine_similarity=round(float(sim), 6),
            cosine_distance=round(1.0 - float(sim), 6),
            similarity_z=round(float(z), 4),
            same_cluster=same_cluster,
            classification=classification,
            confidence=confidence,
            reasons=reasons,
        ))
    pairs.sort(key=lambda p: (p.class_a, p.class_b))
    return pairs, distribution


# ---------------------------------------------------------------------------
# Měření konfuze pro disjointness audit (fáze 0) — "embedding ruler"
# ---------------------------------------------------------------------------
# Audit ve fázi 0 (labeler.taxonomy.audit_and_refine_taxonomy) potřebuje
# METRIKU, jak moc jsou třídy zaměnitelné, aby mohl iterativně zostřovat
# hranice. Původní ruler byl LLM self-consistency probe (drahý a hlučný);
# tohle je jeho deterministická a levná náhrada nad embeddingy.
#
# ŽÁDNÝ zápis na disk, ŽÁDNÁ cache — profily se mezi koly auditu mění
# (zostřování přepisuje description), takže se embedduje pokaždé znovu.

# Podíl nejpodobnějších dvojic, ze kterých se počítá skóre konfuze.
# Metrika cílí na "nejhorší dvojice", ne na průměr přes všechny (ten je
# rozmělněný množstvím evidentně nesouvisejících párů).
AUDIT_TOP_SHARE = 0.2


@dataclass
class TaxonomyConfusionMeasurement:
    """Výsledek jednoho měření zaměnitelnosti tříd nad embeddingy."""

    # Skóre konfuze: průměrná kosinová podobnost nejpodobnějších
    # AUDIT_TOP_SHARE dvojic, oříznutá zdola nulou. 0 = i nejbližší třídy
    # spolu nekorelují (dobře oddělené); vyšší = horší. Absolutní (ne
    # percentilové) — jen tak může audit měřit ZLEPŠENÍ mezi koly.
    score: float
    # (kód_a, kód_b, kosinová podobnost, z-skóre) — seřazeno od nejhoršího
    pairs: list[tuple[str, str, float, float]] = field(default_factory=list)
    distribution: Optional[SimilarityDistribution] = None


@dataclass
class _EmbeddingRulerState:
    """Držák backendu mezi koly auditu — model se načítá JEN JEDNOU."""

    config: TaxonomyEmbeddingConfig
    backend: Optional[EmbeddingBackend] = None

    def ensure_backend(self) -> EmbeddingBackend:
        if self.backend is None:
            self.backend = create_backend(self.config)
        return self.backend


def measure_taxonomy_confusion(
    tax: Taxonomy,
    config: TaxonomyEmbeddingConfig,
    backend: Optional[EmbeddingBackend] = None,
    *,
    top_share: float = AUDIT_TOP_SHARE,
) -> TaxonomyConfusionMeasurement:
    """
    Změří zaměnitelnost tříd taxonomie přes embeddingy jejich profilů.

    Vrací absolutní skóre (viz TaxonomyConfusionMeasurement.score) + dvojice
    seřazené od nejpodobnější. Deterministické; `backend` lze injektovat
    (testy / sdílení modelu mezi koly auditu).

    Taxonomie s < 2 třídami vrací skóre 0 a prázdný seznam dvojic.
    """
    labels = sorted(tax.labels, key=lambda lb: lb.code)
    codes = [lb.code for lb in labels]
    if len(codes) < 2:
        return TaxonomyConfusionMeasurement(score=0.0)

    profiles = [build_taxonomy_label_profile(lb) for lb in labels]
    if backend is None:
        backend = create_backend(config)
    raw = np.asarray(
        backend.encode(profiles, batch_size=config.batch_size), dtype=np.float32
    )
    validate_embeddings(raw, len(codes), normalize=config.normalize)
    emb = apply_embedding_transform(
        raw, center=config.center, normalize=config.normalize
    )
    sim = cosine_similarity_matrix(emb)

    n = len(codes)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sims = np.array([float(sim[i, j]) for i, j in idx], dtype=np.float64)
    z_scores, distribution = robust_similarity_z_scores(sims)

    k = max(1, int(round(len(sims) * min(max(top_share, 0.0), 1.0))))
    top_mean = float(np.sort(sims)[-k:].mean())
    score = max(0.0, top_mean)

    pairs = [
        (codes[i], codes[j], round(float(s), 6), round(float(z), 4))
        for (i, j), s, z in zip(idx, sims, z_scores)
    ]
    # Nejhorší (nejpodobnější) první; tie-break kódy pro determinismus.
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    return TaxonomyConfusionMeasurement(
        score=round(score, 6), pairs=pairs, distribution=distribution,
    )


# ---------------------------------------------------------------------------
# Cache (taxonomy_embeddings.json — bezpečná vůči poškozenému souboru)
# ---------------------------------------------------------------------------


def _cache_key(config: TaxonomyEmbeddingConfig, taxonomy_hash: str) -> dict:
    return {
        "taxonomy_hash": taxonomy_hash,
        "model": config.model,
        "backend": config.backend,
        "profile_builder_version": PROFILE_BUILDER_VERSION,
        "normalize_embeddings": config.normalize,
        # Centrování mění ULOŽENÉ vektory (aplikuje se před zápisem), takže
        # jeho změna musí cache invalidovat.
        "center_embeddings": config.center,
    }


def load_cached_embeddings(
    embeddings_path: Path, config: TaxonomyEmbeddingConfig, taxonomy_hash: str,
) -> Optional[np.ndarray]:
    """Vrátí cachované embeddingy, pokud cache klíč sedí a vektory jsou
    v souboru. Poškozený/nekompatibilní soubor → None (žádná výjimka)."""
    if not embeddings_path.exists():
        return None
    try:
        data = json.loads(embeddings_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cache {embeddings_path} nejde načíst ({e}) — přepočítám.")
        return None
    key = _cache_key(config, taxonomy_hash)
    stored_key = {
        "taxonomy_hash": data.get("taxonomy_hash"),
        "model": data.get("model"),
        "backend": data.get("backend"),
        "profile_builder_version": data.get("profile_builder_version"),
        "normalize_embeddings": data.get("normalize_embeddings"),
        "center_embeddings": data.get("center_embeddings"),
    }
    if stored_key != key:
        return None
    vectors = data.get("embeddings")
    if not vectors:
        return None
    try:
        emb = np.asarray(vectors, dtype=np.float32)
    except (ValueError, TypeError) as e:
        logger.warning(f"Cache vektory nejsou validní ({e}) — přepočítám.")
        return None
    if emb.ndim != 2 or emb.shape[1] != data.get("embedding_dimension"):
        return None
    return emb


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _md_pair_table(pairs: list[PairResult]) -> list[str]:
    if not pairs:
        return ["_žádné dvojice_", ""]
    out = [
        "| Třída A | Třída B | Podobnost | z-skóre | Stejný cluster | Confidence |",
        "|---|---|---:|---:|:---:|---:|",
    ]
    for p in pairs:
        out.append(
            f"| {p.class_a} | {p.class_b} | {p.cosine_similarity:.4f} "
            f"| {p.similarity_z:+.2f} "
            f"| {'ano' if p.same_cluster else 'ne'} | {p.confidence:.2f} |"
        )
    out.append("")
    return out


def build_markdown_report(report: TaxonomyEmbeddingReport) -> str:
    """Lidsky čitelný Markdown report se stabilním řazením tabulek."""
    by_cls = {
        CLASSIFICATION_DUPLICATE: [],
        CLASSIFICATION_OVERLAP: [],
        CLASSIFICATION_DISJOINT: [],
        CLASSIFICATION_UNCERTAIN: [],
    }
    for p in report.pairs:
        by_cls[p.classification].append(p)

    # Duplicity/překryvy: sestupně dle podobnosti; disjoint: vzestupně;
    # tie-break (class_a, class_b) drží stabilitu.
    dup = sorted(by_cls[CLASSIFICATION_DUPLICATE],
                 key=lambda p: (-p.cosine_similarity, p.class_a, p.class_b))
    ovl = sorted(by_cls[CLASSIFICATION_OVERLAP],
                 key=lambda p: (-p.cosine_similarity, p.class_a, p.class_b))
    dis = sorted(by_cls[CLASSIFICATION_DISJOINT],
                 key=lambda p: (p.cosine_similarity, p.class_a, p.class_b))
    unc = sorted(by_cls[CLASSIFICATION_UNCERTAIN],
                 key=lambda p: (-p.cosine_similarity, p.class_a, p.class_b))

    s = report.summary
    if report.threshold_mode == "relative":
        thresholds_line = (
            f"- Prahy (z-skóre, relativní): duplicate >= "
            f"{report.thresholds['possible_duplicate_z']}, overlap >= "
            f"{report.thresholds['possible_overlap_z']}, disjoint <= "
            f"{report.thresholds['disjoint_candidate_z']}"
        )
    else:
        thresholds_line = (
            f"- Prahy (absolutní podobnost): duplicate >= "
            f"{report.thresholds['possible_duplicate']}, overlap >= "
            f"{report.thresholds['possible_overlap']}, disjoint <= "
            f"{report.thresholds['disjoint_candidate']}"
        )
    lines = [
        "# Taxonomy Embedding Analysis",
        "",
        "> **Upozornění:** Výsledky jsou HEURISTICKÉ návrhy založené na "
        "sémantické podobnosti embeddingů. NEJDE o formální důkaz OWL "
        "disjunktnosti. `confidence` je heuristické skóre, ne kalibrovaná "
        "pravděpodobnost. Před použitím v ontologii proveď ruční validaci.",
        "",
        "## Konfigurace",
        "",
        f"- Backend: `{report.backend}`",
        f"- Model: `{report.model}`",
        f"- Embedding dimenze: {report.embedding_dimension}",
        f"- Taxonomie: `{report.taxonomy_path}` (hash `{report.taxonomy_hash}`)",
        f"- Centrování (mean removal): {'ano' if report.centered else 'ne'}",
        thresholds_line,
        f"- Cluster distance threshold: {report.cluster_distance_threshold}",
        "",
    ]
    if report.similarity_distribution is not None:
        d = report.similarity_distribution
        lines += [
            "### Distribuce párových podobností",
            "",
            f"- Rozsah: {d.min:.4f} … {d.max:.4f} (rozpětí {d.max - d.min:.4f})",
            f"- Medián: {d.median:.4f}, průměr: {d.mean:.4f}",
            f"- Škálovaná MAD: {d.scaled_mad:.4f} (jmenovatel z-skóre), "
            f"std: {d.std:.4f}",
            "",
            "z-skóre = (podobnost − medián) / škálovaná MAD. Kladné = dvojice je "
            "podobnější než typická dvojice v této taxonomii, záporné = "
            "vzdálenější. Široké rozpětí znamená dobře rozlišené třídy; "
            "úzké rozpětí značí, že model třídy nerozlišuje (zvaž centrování "
            "nebo jiný model).",
            "",
        ]
    lines += [
        "## Clustery",
        "",
    ]
    for c in report.clusters:
        lines.append(f"- **Cluster {c.cluster_id}**: {', '.join(c.labels)}")
    lines += [
        "",
        f"## Možné duplicity ({len(dup)})",
        "",
        "Třídy natolik podobné, že může jít o duplicitní definice.",
        "",
        *_md_pair_table(dup),
        f"## Možné překryvy ({len(ovl)})",
        "",
        "Třídy mohou popisovat překrývající se koncepty.",
        "",
        *_md_pair_table(ovl),
        f"## Kandidáti na disjunktnost ({len(dis)})",
        "",
        "Třídy vzdálené v embedding prostoru — výsledek NENÍ logický důkaz.",
        "",
        *_md_pair_table(dis),
        f"## Nejisté dvojice ({len(unc)})",
        "",
        "Nejednoznačné — vyžadují ruční kontrolu.",
        "",
        *_md_pair_table(unc),
        "## Doporučení",
        "",
        "1. Ručně projdi možné duplicity — zvaž sloučení tříd nebo zostření "
        "popisů (viz disjointness audit ve fázi 0).",
        "2. U možných překryvů zkontroluj rozhodovací pravidla "
        "(decision_rule) a counter_examples.",
        "3. Kandidáty na disjunktnost považuj za NÁVRHY pro ruční schválení — "
        "OWL `disjointWith` přidávej jen po lidské validaci "
        "(TAXONOMY_OWL_DISJOINT_MODE=assert).",
        "",
        "## Souhrn",
        "",
        f"- Tříd: {s.label_count}",
        f"- Dvojic: {s.pair_count}",
        f"- possible_duplicate: {s.possible_duplicate_count}",
        f"- possible_overlap: {s.possible_overlap_count}",
        f"- disjoint_candidate: {s.disjoint_candidate_count}",
        f"- uncertain: {s.uncertain_count}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9) Orchestrace
# ---------------------------------------------------------------------------


def run_taxonomy_embedding_analysis(
    taxonomy_path: Path,
    output_dir: Path,
    config: TaxonomyEmbeddingConfig,
    backend: Optional[EmbeddingBackend] = None,
) -> TaxonomyEmbeddingReport:
    """
    Veřejný entry point — kompletní analýza + zápis reportů.

    Synchronní (Sentence Transformers je sync; žádné LLM/HTTP volání).
    `backend` lze injektovat (testy — fake backend bez stahování modelu);
    default se vytvoří z konfigurace až TADY (žádné stahování při importu).

    Zapisuje atomicky do `output_dir`:
      taxonomy_embedding_report.json / .md, taxonomy_embeddings.json

    Nikdy nemodifikuje existující pipeline checkpointy.
    """
    tax = load_taxonomy_for_analysis(taxonomy_path)
    codes = [lb.code for lb in tax.labels]
    n = len(codes)
    tax_hash = taxonomy_content_hash(tax)
    profiles = [build_taxonomy_label_profile(lb) for lb in tax.labels]
    logger.info(
        f"taxonomy-embeddings: {n} tříd, backend={config.backend}, "
        f"model={config.model}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / EMBEDDINGS_FILE

    # --- Embeddingy (cache → backend) --------------------------------------
    t0 = time.monotonic()
    cached = load_cached_embeddings(embeddings_path, config, tax_hash)
    cache_used = cached is not None
    if cache_used:
        # Cache drží už transformované (centrované + normalizované) vektory —
        # NEtransformovat znovu (centrování není idempotentní).
        embeddings = cached
        validate_embeddings(embeddings, n, normalize=config.normalize)
        logger.info("taxonomy-embeddings: použita cache (embeddingy nepřepočítávám)")
    else:
        if backend is None:
            backend = create_backend(config)
        raw = np.asarray(
            backend.encode(profiles, batch_size=config.batch_size),
            dtype=np.float32,
        )
        validate_embeddings(raw, n, normalize=config.normalize)
        embeddings = apply_embedding_transform(
            raw, center=config.center, normalize=config.normalize,
        )
    embed_s = time.monotonic() - t0
    dim = int(embeddings.shape[1])
    centering_applied = bool(config.center and n >= _MIN_LABELS_FOR_CENTERING)
    logger.info(
        f"taxonomy-embeddings: dimenze={dim}, cache={'ano' if cache_used else 'ne'}, "
        f"centrování={'ano' if centering_applied else 'ne'}, "
        f"embedding {embed_s:.2f}s"
    )

    # --- Podobnosti + clustering -------------------------------------------
    sim = cosine_similarity_matrix(embeddings)
    t1 = time.monotonic()
    effective_cluster_threshold = config.cluster_distance_threshold
    if n < 2:
        cluster_assignment = [0] * n
        logger.info("taxonomy-embeddings: < 2 třídy — clustering přeskočen")
    else:
        effective_cluster_threshold = resolve_cluster_distance_threshold(
            sim, config.cluster_distance_threshold, config.threshold_mode,
        )
        if config.threshold_mode == "relative":
            logger.info(
                f"taxonomy-embeddings: cluster práh {config.cluster_distance_threshold} "
                f"(kvantil) → absolutní vzdálenost {effective_cluster_threshold:.4f}"
            )
        cluster_assignment = agglomerative_average_cosine(
            sim, codes, effective_cluster_threshold,
        )
    cluster_s = time.monotonic() - t1
    logger.info(
        f"taxonomy-embeddings: clustering {cluster_s:.2f}s, "
        f"{len(set(cluster_assignment)) if cluster_assignment else 0} clusterů"
    )

    clusters_map: dict[int, list[str]] = {}
    for code, cid in zip(codes, cluster_assignment):
        clusters_map.setdefault(cid, []).append(code)
    clusters = [
        ClusterInfo(cluster_id=cid, labels=sorted(members))
        for cid, members in sorted(clusters_map.items())
    ]

    pairs, distribution = build_pair_results(codes, sim, cluster_assignment, config)
    if distribution is not None:
        logger.info(
            f"taxonomy-embeddings: distribuce podobností median="
            f"{distribution.median:.3f}, rozsah [{distribution.min:.3f}, "
            f"{distribution.max:.3f}], scaled MAD={distribution.scaled_mad:.3f}, "
            f"mód prahů={config.threshold_mode}"
        )
    counts = {
        CLASSIFICATION_DUPLICATE: 0,
        CLASSIFICATION_OVERLAP: 0,
        CLASSIFICATION_DISJOINT: 0,
        CLASSIFICATION_UNCERTAIN: 0,
    }
    for p in pairs:
        counts[p.classification] += 1
    logger.info(
        f"taxonomy-embeddings: dvojic={len(pairs)} "
        f"(duplicate={counts[CLASSIFICATION_DUPLICATE]}, "
        f"overlap={counts[CLASSIFICATION_OVERLAP]}, "
        f"disjoint_candidate={counts[CLASSIFICATION_DISJOINT]}, "
        f"uncertain={counts[CLASSIFICATION_UNCERTAIN]})"
    )

    report = TaxonomyEmbeddingReport(
        taxonomy_path=str(taxonomy_path),
        taxonomy_hash=tax_hash,
        backend=config.backend,
        model=config.model,
        embedding_dimension=dim,
        threshold_mode=config.threshold_mode,
        centered=centering_applied,
        thresholds=(
            {
                "possible_duplicate_z": config.duplicate_z,
                "possible_overlap_z": config.overlap_z,
                "disjoint_candidate_z": config.disjoint_z,
            }
            if config.threshold_mode == "relative"
            else {
                "possible_duplicate": config.duplicate_threshold,
                "possible_overlap": config.overlap_threshold,
                "disjoint_candidate": config.disjoint_threshold,
            }
        ),
        cluster_distance_threshold=round(effective_cluster_threshold, 6),
        similarity_distribution=distribution,
        clusters=clusters,
        pairs=pairs,
        summary=ReportSummary(
            label_count=n,
            pair_count=len(pairs),
            possible_duplicate_count=counts[CLASSIFICATION_DUPLICATE],
            possible_overlap_count=counts[CLASSIFICATION_OVERLAP],
            disjoint_candidate_count=counts[CLASSIFICATION_DISJOINT],
            uncertain_count=counts[CLASSIFICATION_UNCERTAIN],
        ),
    )

    # --- Výstupy (atomicky) -------------------------------------------------
    from datetime import datetime, timezone
    embeddings_doc = {
        "version": SCHEMA_VERSION,
        "model": config.model,
        "backend": config.backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_hash": tax_hash,
        "profile_builder_version": PROFILE_BUILDER_VERSION,
        "normalize_embeddings": config.normalize,
        "center_embeddings": config.center,
        "centering_applied": centering_applied,
        "labels": codes,
        "profiles": profiles,
        "embedding_dimension": dim,
        # Vektory jen na explicitní přání (velikost/citlivost). Bez nich
        # cache nefunguje napříč běhy, report ale funguje vždy.
        "embeddings": (
            [[round(float(x), 7) for x in row] for row in embeddings]
            if config.write_vectors else None
        ),
    }
    report_json_path = output_dir / REPORT_JSON_FILE
    report_md_path = output_dir / REPORT_MD_FILE
    atomic_write_json(embeddings_path, embeddings_doc)
    atomic_write_json(report_json_path, report.model_dump(mode="json"))
    atomic_write_text(report_md_path, build_markdown_report(report))
    logger.info(
        f"taxonomy-embeddings: výstupy: {report_json_path}, "
        f"{report_md_path}, {embeddings_path}"
    )
    return report
