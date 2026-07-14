"""
Fáze 3 (finalize) — Deterministická lemmatizace entit přes Stanza.

PRINCIP DĚLBY PRÁCE (viz positions.py — stejná filozofie):
  LLM vrací jen to, co vyžaduje porozumění textu: entity (spany + kategorie)
  a relace. Všechno mechanické dopočítá kód — pozice počítá positions.py,
  LEMMA počítá tento modul přes Stanza. Lemmatizace češtiny (i jiných jazyků)
  je vyřešený deterministický problém; LLM lemma je šum (jednou lemmatizuje,
  jednou ne, jednou si vymyslí), který tříští klíče korpusového hlasování
  (category_vote), chunk-merge deduplikaci i triples.

OMEZENÍ (zadání projektu): Stanza se používá VÝHRADNĚ na lemma. Pipeline
interně potřebuje POS tagger (lemmatizer na něm závisí), ale POS tagy se
NIKDE v kódu nepoužívají — žádné POS filtry, žádná morfologie.

Chování:
  - Zpracuje se každý dokument JEDNOU (lemma v kontextu věty je přesnější
    než izolovaně); entity se mapují přes start_char/end_char na tokeny.
  - Víceslovná entita = join lemmat tokenů v jejím spanu ("rodiny s dětmi"
    → "rodina s dítě"). Gramaticky kostrbaté, ale KONZISTENTNÍ — a klíče
    (hlasování, dedup) potřebují konzistenci, ne krásu.
  - Entita bez validních offsetů → standalone lemmatizace jejího textu.
  - BEST-EFFORT: pokud stanza není nainstalovaná / model chybí / inicializace
    selže, celý krok se přeskočí (ponechají se LLM lemmata) a běh POKRAČUJE.
    Determinismus: stejný text → stejná lemmata (Stanza je deterministická).

Jazyk je konfigurace (STANZA_LANG, default "cs"), ne hardcode — Stanza
podporuje ~70 jazyků, pipeline zůstává univerzální.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Token = (start_char, end_char, lemma)
Token = tuple[int, int, str]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class LemmaReport:
    """Auditovatelný výsledek lemmatizace (jde do stats.json per branch)."""
    docs_processed: int = 0
    docs_failed: int = 0
    entities_seen: int = 0
    lemmas_changed: int = 0
    entities_standalone_fallback: int = 0   # bez offsetů → lemma z textu entity
    changed_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "docs_processed": self.docs_processed,
            "docs_failed": self.docs_failed,
            "entities_seen": self.entities_seen,
            "lemmas_changed": self.lemmas_changed,
            "entities_standalone_fallback": self.entities_standalone_fallback,
            "changed_sample": self.changed_sample[:50],
        }


# ---------------------------------------------------------------------------
# Pure logika (testovatelná bez stanza)
# ---------------------------------------------------------------------------


def lemma_for_span(
    tokens: list[Token], start: int, end: int,
) -> Optional[str]:
    """
    Z tokenů dokumentu složí lemma pro span [start, end): join lemmat všech
    tokenů, které se se spanem překrývají. None = žádný token v spanu
    (caller spadne na standalone fallback). Pure — testovatelné bez stanza.
    """
    if start is None or end is None or start >= end:
        return None
    parts = [
        lemma for (ts, te, lemma) in tokens
        if lemma and not (te <= start or ts >= end)
    ]
    if not parts:
        return None
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stanza wrapper (lazy, best-effort)
# ---------------------------------------------------------------------------


class StanzaLemmatizer:
    """
    Lazy wrapper nad stanza.Pipeline. Inicializace (import + load modelu)
    proběhne až při prvním použití; selhání → available=False a krok se
    přeskočí (best-effort, běh pipeline se NIKDY nezastaví kvůli lemmatům).

    Používá POUZE processory tokenize(+mwt)+pos+lemma — pos je interní
    závislost lemmatizeru, JEHO tagy se nikde nečtou.
    """

    def __init__(self, lang: str = "cs", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._pipeline = None
        self._init_tried = False

    @property
    def available(self) -> bool:
        self._ensure_pipeline()
        return self._pipeline is not None

    def _ensure_pipeline(self) -> None:
        if self._init_tried:
            return
        self._init_tried = True
        try:
            import stanza  # lazy — volitelná závislost
        except ImportError:
            logger.warning(
                "stanza není nainstalovaná (pip install stanza) — lemmatizace "
                "se přeskakuje, ponechávají se LLM lemmata."
            )
            return
        try:
            # Model se při prvním běhu automaticky stáhne do ~/stanza_resources.
            self._pipeline = stanza.Pipeline(
                lang=self.lang,
                processors="tokenize,mwt,pos,lemma",
                use_gpu=self.use_gpu,
                verbose=False,
            )
            logger.info(
                f"Stanza lemmatizer připraven (lang={self.lang}, "
                f"gpu={self.use_gpu})."
            )
        except Exception as e:  # noqa: BLE001 — best-effort init
            logger.warning(
                f"Stanza pipeline se nepodařilo inicializovat "
                f"({type(e).__name__}: {e}) — lemmatizace se přeskakuje, "
                f"ponechávají se LLM lemmata."
            )
            self._pipeline = None

    def tokens_for(self, text: str) -> Optional[list[Token]]:
        """
        Tokeny dokumentu jako (start_char, end_char, lemma). None při
        nedostupnosti/chybě. MWT tokeny (jedno slovo → víc words) se skládají
        joinem lemmat svých words.
        """
        if not text or not self.available:
            return None
        try:
            doc = self._pipeline(text)
        except Exception as e:  # noqa: BLE001 — jeden vadný dokument ≠ konec
            logger.warning(f"Stanza selhala na dokumentu ({e}); přeskakuji")
            return None
        out: list[Token] = []
        for sent in doc.sentences:
            for token in sent.tokens:
                if token.start_char is None or token.end_char is None:
                    continue
                lemma = " ".join(
                    w.lemma for w in token.words if getattr(w, "lemma", None)
                )
                out.append((token.start_char, token.end_char, lemma))
        return out

    def lemma_of_text(self, text: str) -> Optional[str]:
        """Standalone lemma fráze (pro entity bez validních offsetů)."""
        tokens = self.tokens_for(text)
        if not tokens:
            return None
        return " ".join(lemma for (_s, _e, lemma) in tokens if lemma) or None


# Singleton — Stanza model se načítá jednou per proces (config se za běhu nemění).
_LEMMATIZER: Optional[StanzaLemmatizer] = None


def get_lemmatizer() -> StanzaLemmatizer:
    """Vrátí sdílený lemmatizer podle CONFIG (lazy init při prvním použití)."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        from .config import CONFIG
        _LEMMATIZER = StanzaLemmatizer(
            lang=CONFIG.stanza_lang, use_gpu=CONFIG.stanza_use_gpu,
        )
    return _LEMMATIZER


# ---------------------------------------------------------------------------
# Aplikace na final_records
# ---------------------------------------------------------------------------


def apply_lemmas_to_records(
    records: list[dict], lemmatizer,
) -> LemmaReport:
    """
    Přepíše entity.lemma deterministickými Stanza lemmaty. Mutuje records
    IN-PLACE (dict tvar z finalize, PŘED merge chunků — konzistentní lemmata
    pak zlepší chunk-merge dedup i klíče korpusového hlasování).

    `lemmatizer` je duck-typed (viz StanzaLemmatizer): potřebuje `available`,
    `tokens_for(text)` a `lemma_of_text(text)` — testy dodají fake.

    Entity s offsety → lemma z tokenů dokumentu (kontextové);
    bez offsetů → standalone lemma textu entity; když Stanza nedá nic,
    ponechá se původní (LLM) lemma.
    """
    rep = LemmaReport()
    if not getattr(lemmatizer, "available", False):
        return rep

    for rec in records:
        text = str(rec.get("input_text") or "")
        entities = (rec.get("graph") or {}).get("entities") or []
        if not entities:
            continue
        tokens = lemmatizer.tokens_for(text)
        if tokens is None:
            rep.docs_failed += 1
            continue
        rep.docs_processed += 1
        for e in entities:
            rep.entities_seen += 1
            new_lemma: Optional[str] = lemma_for_span(
                tokens, e.get("start_char"), e.get("end_char"),
            )
            if new_lemma is None:
                # Bez offsetů / mimo tokeny → standalone fallback
                new_lemma = lemmatizer.lemma_of_text(str(e.get("text") or ""))
                if new_lemma is not None:
                    rep.entities_standalone_fallback += 1
            if not new_lemma:
                continue  # Stanza nedala nic → ponech LLM lemma
            old = str(e.get("lemma") or "")
            if new_lemma != old:
                if len(rep.changed_sample) < 50:
                    rep.changed_sample.append(
                        f"{e.get('text', '')!r}: {old!r} -> {new_lemma!r}"
                    )
                e["lemma"] = new_lemma
                rep.lemmas_changed += 1

    if rep.lemmas_changed:
        logger.info(
            f"stanza lemma: přepsáno {rep.lemmas_changed}/{rep.entities_seen} "
            f"lemmat ({rep.docs_processed} dokumentů, "
            f"{rep.entities_standalone_fallback} standalone fallbacků)"
        )
    return rep
