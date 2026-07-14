"""
Fáze 3 (finalize) — Korpusové hlasování o kategoriích entit.

Finalize je jediné místo v pipeline, které vidí CELÝ korpus najednou —
extractor i critic rozhodují o kategorii vždy v izolaci jednoho dokumentu.
Tenhle modul z toho těží: posbírá rozdělení kategorií pro každou entitu
(klíčovanou přes normalizované lemma) napříč všemi dokumenty a menšinové
výskyty přelabeluje na dominantní kategorii.

PRINCIPY (viz zadání projektu):
  - ŽÁDNÁ hard-coded slova. Jediné konstanty jsou poměrové prahy
    (dominance, min. support, max. document frequency) — čísla, ne slova.
  - ŽÁDNÝ člověk. Vše je deterministická statistika nad korpusem.
  - Univerzální: funguje s libovolnou taxonomií (default 16, custom z fáze 0)
    i libovolným jazykem.
  - Entropy guard: když si korpus není jistý (žádná kategorie nedominuje),
    NIC se nemění — legitimní polysémie se nesmí přepsat globální většinou.

Tři použití:

  1. apply_corpus_votes() — přelabeluje menšinové výskyty na dominantní
     kategorii (dominance >= práh, support >= minimum).
  2. CorpusCategoryStats.preferred_category() — datový rozhodčí pro
     within-document konflikty labelů v normalize (nahrazuje smazanou
     hard-coded _LABEL_PRIORITY).
  3. CorpusCategoryStats.is_generic_surface() / is_safe_expand() — datová
     bezpečnost auto-expandu v normalize (nahrazuje smazané seznamy
     _SAFE_FOR_AUTO_EXPAND_LABELS a _UNSAFE_TEXT_FOR_AUTO_EXPAND).
     Funkční slova a doménová generika se poznají podle document frequency:
     výraz, který se vyskytuje ve většině dokumentů korpusu, není bezpečný
     kandidát na expanzi — žádný seznam slov není potřeba.

Determinismus: stejný korpus → stejné výsledky. Remíza = žádný signál
(vrací None), rozhodne až další stupeň (dokumentová většina → první výskyt).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Klíčování — identická normalizace jako finalize._norm_entity_text, aby
# hlasovací klíč a chunk-merge klíč byly konzistentní.
# ---------------------------------------------------------------------------


def norm_key(value: str) -> str:
    """Normalizace textu/lemmatu na hlasovací klíč (lowercase, bez interpunkce)."""
    return " ".join(
        (value or "")
        .lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .replace(":", " ")
        .split()
    )


def entity_key(entity: dict) -> str:
    """Hlasovací klíč entity (dict tvar z final_records): lemma, fallback text."""
    return norm_key(entity.get("lemma") or entity.get("text") or "")


def _contains_word(text_cf: str, needle_cf: str) -> bool:
    """
    True pokud casefoldnutý text obsahuje needle jako celé slovo (word
    boundary na obou stranách — "auto" se nenajde uvnitř "automobil").
    """
    if not needle_cf:
        return False
    n = len(text_cf)
    m = len(needle_cf)
    pos = 0
    while True:
        idx = text_cf.find(needle_cf, pos)
        if idx == -1:
            return False
        before_ok = idx == 0 or not (
            text_cf[idx - 1].isalnum() and needle_cf[0].isalnum()
        )
        end = idx + m
        after_ok = end >= n or not (
            text_cf[end].isalnum() and needle_cf[-1].isalnum()
        )
        if before_ok and after_ok:
            return True
        pos = idx + 1


# ---------------------------------------------------------------------------
# Korpusové statistiky
# ---------------------------------------------------------------------------


@dataclass
class CorpusCategoryStats:
    """
    Rozdělení kategorií per klíč (normalizované lemma) přes celý korpus
    + document frequency povrchových tvarů (lazy, cache).

    Prahy jsou uložené v instanci (nastaví je finalize z CONFIG), aby
    normalize nemusel importovat config — jen se ptá stats objektu.
    """
    dominance: float = 0.8        # podíl dominantní kategorie, aby se hlasovalo
    min_support: int = 3          # min. výskytů klíče, aby hlas platil
    max_df_ratio: float = 0.5     # max. podíl dokumentů obsahujících výraz
                                  # (víc = funkční slovo/generikum → unsafe)

    counts: dict[str, Counter] = field(default_factory=dict)
    n_docs: int = 0
    _texts_cf: list[str] = field(default_factory=list)
    _df_cache: dict[str, float] = field(default_factory=dict)

    # --- stavba -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        records: list[dict],
        *,
        dominance: float = 0.8,
        min_support: int = 3,
        max_df_ratio: float = 0.5,
    ) -> "CorpusCategoryStats":
        """
        Jeden průchod přes final_records (dict tvar po merge chunků).
        Reject záznamy se NEpočítají — jejich grafy critic označil za vadné,
        hlasovat s nimi by zafixovalo chyby.
        """
        stats = cls(
            dominance=dominance,
            min_support=min_support,
            max_df_ratio=max_df_ratio,
        )
        for rec in records:
            if rec.get("critic_status") == "reject":
                continue
            stats.n_docs += 1
            stats._texts_cf.append(str(rec.get("input_text") or "").casefold())
            for e in (rec.get("graph") or {}).get("entities") or []:
                k = entity_key(e)
                cat = str(e.get("category") or "")
                if not k or not cat:
                    continue
                stats.counts.setdefault(k, Counter())[cat] += 1
        return stats

    # --- dotazy -------------------------------------------------------------

    def support(self, key: str) -> int:
        c = self.counts.get(key)
        return sum(c.values()) if c else 0

    def dominant(self, key: str) -> Optional[tuple[str, float]]:
        """(dominantní kategorie, její podíl) nebo None. Remíza na 1. místě
        = žádný jasný vítěz → None (entropy guard)."""
        c = self.counts.get(key)
        if not c:
            return None
        total = sum(c.values())
        ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        cat, n = ranked[0]
        return cat, n / total

    def preferred_category(self, key: str, candidates: list[str]) -> Optional[str]:
        """
        Datový rozhodčí pro within-document konflikt: z kandidátních kategorií
        vyber tu s nejvyšším korpusovým počtem pro daný klíč. None = korpus
        nemá signál (klíč nezná, kandidáty nezná, nebo remíza).
        """
        c = self.counts.get(key)
        if not c:
            return None
        scored = sorted(
            ((c.get(cat, 0), cat) for cat in set(candidates)),
            key=lambda t: (-t[0], t[1]),
        )
        if not scored or scored[0][0] == 0:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None  # remíza → žádný signál
        return scored[0][1]

    def mention_weight(self, key: str, category: str) -> int:
        """Korpusový počet (klíč, kategorie) — tie-break pro překryvy."""
        c = self.counts.get(key)
        return c.get(category, 0) if c else 0

    def text_doc_ratio(self, surface: str) -> float:
        """
        Podíl dokumentů, jejichž input_text obsahuje výraz jako celé slovo.
        Lazy + cache (počítá se jen pro dotazované výrazy). Funkční slova
        ("se", "na") a doménová generika ("pojištění" v pojišťovnickém
        korpusu) mají poměr blízko 1.0.
        """
        cf = (surface or "").strip().casefold()
        if not cf or not self.n_docs:
            return 0.0
        cached = self._df_cache.get(cf)
        if cached is not None:
            return cached
        hits = sum(1 for t in self._texts_cf if _contains_word(t, cf))
        ratio = hits / self.n_docs
        self._df_cache[cf] = ratio
        return ratio

    def is_generic_surface(self, surface: str) -> bool:
        """True = výraz je statistické generikum/funkční slovo (vysoká DF)
        → nebezpečný pro auto-expand. Nahrazuje hard-coded stopword seznam."""
        return self.text_doc_ratio(surface) > self.max_df_ratio

    def is_safe_expand(self, key: str, category: str, surface: str) -> bool:
        """
        True = korpus potvrzuje, že výraz je konzistentně tahle kategorie
        (dost výskytů, jasná dominance, není to generikum) → auto-expand
        bezpečný i pro krátké texty. Nahrazuje hard-coded safe-label seznam.
        """
        if self.support(key) < self.min_support:
            return False
        dom = self.dominant(key)
        if dom is None:
            return False
        dom_cat, share = dom
        if dom_cat != category or share < self.dominance:
            return False
        return not self.is_generic_surface(surface)


# ---------------------------------------------------------------------------
# Hlasování — přelabelování menšinových výskytů
# ---------------------------------------------------------------------------


@dataclass
class VoteReport:
    """Auditovatelný výsledek hlasování (jde do category_vote_report.json)."""
    n_docs: int = 0
    entities_seen: int = 0
    relabeled: int = 0
    decisions: int = 0                 # kolik klíčů mělo jasnou dominanci
    ambiguous: int = 0                 # kolik klíčů zůstalo nejednoznačných
    relabel_pairs: Counter = field(default_factory=Counter)  # "FROM→TO" → počet
    ambiguous_sample: list[str] = field(default_factory=list)
    changed_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_docs": self.n_docs,
            "entities_seen": self.entities_seen,
            "relabeled": self.relabeled,
            "decision_keys": self.decisions,
            "ambiguous_keys": self.ambiguous,
            "relabel_pairs": dict(
                sorted(self.relabel_pairs.items(), key=lambda kv: -kv[1])
            ),
            "ambiguous_sample": self.ambiguous_sample[:50],
            "changed_sample": self.changed_sample[:50],
        }


def apply_corpus_votes(
    records: list[dict], stats: CorpusCategoryStats,
) -> VoteReport:
    """
    Přelabeluje menšinové výskyty na korpusově dominantní kategorii.

    Mění records IN-PLACE (dict tvar graph.entities[].category). Mění se JEN
    entity, jejichž klíč má support >= min_support a dominanci >= prahu.
    Nejednoznačné klíče (žádná dominance) se NEmění — jen jdou do reportu
    (a jsou zároveň přesný vstup pro disjointness audit v příštím běhu).

    Reject záznamy se přeskakují (nejsou v hlavním datasetu).
    """
    rep = VoteReport(n_docs=stats.n_docs)

    # Předpočítej rozhodnutí per klíč — deterministicky (sorted).
    decisions: dict[str, str] = {}
    for key in sorted(stats.counts):
        if stats.support(key) < stats.min_support:
            continue
        dom = stats.dominant(key)
        if dom is None:
            rep.ambiguous += 1
            if len(rep.ambiguous_sample) < 50:
                dist = dict(stats.counts[key])
                rep.ambiguous_sample.append(f"{key!r}: {dist}")
            continue
        cat, share = dom
        if share >= stats.dominance:
            decisions[key] = cat
            rep.decisions += 1
        else:
            rep.ambiguous += 1
            if len(rep.ambiguous_sample) < 50:
                dist = dict(stats.counts[key])
                rep.ambiguous_sample.append(f"{key!r}: {dist}")

    for rec in records:
        if rec.get("critic_status") == "reject":
            continue
        for e in (rec.get("graph") or {}).get("entities") or []:
            rep.entities_seen += 1
            target = decisions.get(entity_key(e))
            current = str(e.get("category") or "")
            if target and current and current != target:
                # ASCII '->' (ne '→') — bezpečné pro Windows konzole (cp1250)
                pair = f"{current}->{target}"
                rep.relabel_pairs[pair] += 1
                if len(rep.changed_sample) < 50:
                    rep.changed_sample.append(f"{e.get('text', '')!r}: {pair}")
                e["category"] = target
                rep.relabeled += 1

    if rep.relabeled:
        logger.info(
            f"category vote: přelabelováno {rep.relabeled}/{rep.entities_seen} "
            f"entit podle korpusové dominance; top páry: "
            f"{dict(rep.relabel_pairs.most_common(5))}"
        )
    return rep
