"""
Fáze 0 — Taxonomy Discovery.

Volitelná fáze, která PŘED extract určí, jaké labely (entity kategorie) má
LLM používat. Tři režimy:

  - off    Žádná discovery. Pipeline použije defaultní 16 kategorií
            z labeler/schemas.py::EntityCategory (zachování existujícího chování).

  - auto   LLM (Qwen3.5) analyzuje vzorek textů ze VŠECH datasetů v DATA_DIR
            a navrhne JEDNOTNOU taxonomii pokrývající všechny domény. Garantuje
            konzistenci: stejná entita (např. produkt) = stejný název napříč
            doménami; žádné duplikáty pro stejný koncept (PRODUKT vs ITEM apod.).

  - user   Uživatel poskytne JSON s labely. Pipeline ho validuje (UPPERCASE_SNAKE
            pattern + min description) a použije.

Výstup je VŽDY `entity_taxonomy.json` v CHECKPOINT_DIR — později ho extract
přebere a doplní do promptu místo defaultu.

Konzistence napříč datasety v auto módu:
  1. Pro každou doménu vezmi ukázky vybrané embeddingy ve fázi 0a
     (taxonomy_sampling.json), jinak N náhodných (default 10).
  2. Pošli LLM jeden velký prompt: "Tady máš ukázky z domén X, Y, Z. Navrhni
     UNIVERZÁLNÍ taxonomii NER labelů. Stejný koncept = stejný label, i když
     se v různých doménách projevuje jinak."
  3. Validuj výstup (pattern, no duplicates, mín. description).
  4. Ulož do entity_taxonomy.json.

Při znovuspuštění auto se přepíše. User mode soubor jen načte (kopíruje na
zápis pro audit trail).
"""

from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from pydantic import ValidationError

from .config import CONFIG
from .datasets import LabelItem, scan_data_dir
from .io_utils import atomic_write_json
from .llm_client import LLMClient, LLMFatalError, LLMTransientError
from .schemas import _LABEL_NAME_RE  # type: ignore  # reuse pattern

logger = logging.getLogger(__name__)


class TaxonomyQualityError(Exception):
    """Auto-discovered taxonomie neprošla kvalitativní bránou ani po retry.

    Signalizuje volajícímu (run_taxonomy_discovery), že má spadnout na
    defaultní 16 kategorií — NE celý běh shodit. Pipeline pak pokračuje,
    jako kdyby byl taxonomy mode = off.
    """


# ---------------------------------------------------------------------------
# Datové struktury
# ---------------------------------------------------------------------------


class TaxonomyMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    USER = "user"


@dataclass
class TaxonomyLabel:
    """
    Jeden label v taxonomii.

    Volitelná data-driven pravidla pro normalizaci (ŽÁDNÉ hard-coding v kódu —
    všechny doménové heuristiky se konfigurují tady v taxonomy.json):

    force_text_match:
      Texty (surface forms), které VŽDY spadají pod tento label, i když LLM
      navrhl jiný. Použij pro pevné mapování (např. v pojišťovnické taxonomii
      přidat "ALFA", "BETA", "GAMA" do PRODUKT.force_text_match —
      pak je heuristika přepíše z KATEGORIE/SUBJEKT na PRODUKT).

    unsafe_texts:
      Texty, které se NIKDY nemají dostat do tohoto labelu. Pokud LLM označí
      "pojištění" jako PRODUKT, ale "pojištění" je v PRODUKT.unsafe_texts,
      heuristika ho downgraduje na KEYPHRASE_SUBJECT (= generic fallback,
      který není v žádné target množině v relation rules → tím se chrání i
      core NER export a relation extraction).

    case_sensitive:
      Pokud True, force_text_match a unsafe_texts se matchují case-sensitive.
      Default False (case-insensitive — typicky se hodí pro generická slova).
    """
    code: str                          # UPPERCASE_SNAKE
    description: str                   # 1-2 věty popisu
    examples: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    # === Volitelné data-driven heuristiky (žádné hard-coding v Python kódu) ===
    force_text_match: list[str] = field(default_factory=list)
    unsafe_texts: list[str] = field(default_factory=list)
    case_sensitive: bool = False
    # === Auto-generovaná hraniční vodítka (Pilíř A / disjointness audit) ===
    # NIKDY nepsané člověkem — vytváří je LLM ve fázi 0, když audit najde
    # zaměňovaný pár labelů a zostří jejich hranici. Renderují se do promptu
    # (labeler.prompts._taxonomy_block), aby model rozlišoval A vs B přesněji.
    #   decision_rule    jak rozhodnout, když entita padá na hranici s jiným labelem
    #   counter_examples výrazy, které vypadají jako tenhle label, ale patří jinam
    decision_rule: str = ""
    counter_examples: list[str] = field(default_factory=list)
    # === Embeddingové vztahy (fáze 0b) =======================================
    # Vyplňuje je AUTOMATICKY embedding analýza po tvorbě taxonomie (viz
    # annotate_taxonomy_with_embeddings). Heuristika, NE logický důkaz.
    #   confusable_with      třídy, se kterými je tahle v embedding prostoru
    #                        zaměnitelná (possible_duplicate / possible_overlap),
    #                        od nejpodobnější — renderují se do promptu extractoru
    #   disjoint_candidates  třídy daleko v embedding prostoru (disjoint_candidate)
    #   embedding_cluster    id clusteru z aglomerativního clusteringu (-1 = neznámý)
    confusable_with: list[str] = field(default_factory=list)
    disjoint_candidates: list[str] = field(default_factory=list)
    embedding_cluster: int = -1
    # === Normalizační metadata (nahrazují hard-coded DEFAULT_LABEL_MAPPING) ===
    # kind:      "ner" = tvrdá NER kategorie (krátké pojmenovatelné spany —
    #            produkty, organizace, místa, data, hodnoty); "keyphrase" =
    #            delší popisné fráze (benefity, použití) — mimo core NER
    #            trénink, po normalizaci dostanou prefix KEYPHRASE_.
    # remap_to:  cílový label po normalizaci. Prázdné = odvodí se z kind
    #            (keyphrase → KEYPHRASE_<code>, ner → code beze změny).
    # Mapping labelů je tím pádem VLASTNOST TAXONOMIE (dat), ne Python kódu —
    # viz Taxonomy.label_mapping(). V auto módu kind určuje LLM ve fázi 0.
    kind: str = "ner"
    remap_to: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaxonomyLabel":
        return cls(
            code=str(d["code"]).strip(),
            description=str(d.get("description", "")).strip(),
            examples=[str(x) for x in (d.get("examples") or [])],
            domains=[str(x) for x in (d.get("domains") or [])],
            aliases=[str(x) for x in (d.get("aliases") or [])],
            force_text_match=[str(x) for x in (d.get("force_text_match") or [])],
            unsafe_texts=[str(x) for x in (d.get("unsafe_texts") or [])],
            case_sensitive=bool(d.get("case_sensitive", False)),
            decision_rule=str(d.get("decision_rule", "")).strip(),
            counter_examples=[str(x) for x in (d.get("counter_examples") or [])],
            confusable_with=[str(x) for x in (d.get("confusable_with") or [])],
            disjoint_candidates=[str(x) for x in (d.get("disjoint_candidates") or [])],
            embedding_cluster=int(d.get("embedding_cluster", -1)),
            kind=str(d.get("kind", "ner")).strip().lower() or "ner",
            remap_to=str(d.get("remap_to", "")).strip(),
        )

    def validate(self) -> list[str]:
        """Vrátí seznam problémů (prázdný = OK)."""
        errs: list[str] = []
        if not _LABEL_NAME_RE.match(self.code):
            errs.append(
                f"label.code {self.code!r} musí být UPPERCASE_SNAKE "
                f"(^[A-Z][A-Z0-9_]{{0,49}}$)"
            )
        if not self.description or len(self.description) < 10:
            errs.append(
                f"label.description pro {self.code!r} musí mít alespoň 10 znaků"
            )
        if self.kind not in ("ner", "keyphrase"):
            errs.append(
                f"label.kind pro {self.code!r} musí být 'ner' nebo 'keyphrase', "
                f"dostal jsem {self.kind!r}"
            )
        if self.remap_to and not _LABEL_NAME_RE.match(self.remap_to):
            errs.append(
                f"label.remap_to {self.remap_to!r} pro {self.code!r} musí být "
                f"UPPERCASE_SNAKE"
            )
        return errs

    def mapped_target(self) -> str:
        """Cílový label po normalizaci: explicitní remap_to, jinak odvození
        z kind (keyphrase → KEYPHRASE_<code>, ner → code beze změny)."""
        if self.remap_to:
            return self.remap_to
        if self.kind == "keyphrase" and not self.code.startswith("KEYPHRASE_"):
            return f"KEYPHRASE_{self.code}"
        return self.code


@dataclass
class Taxonomy:
    """Kompletní taxonomie — seznam labelů + metadata."""
    labels: list[TaxonomyLabel] = field(default_factory=list)
    version: int = 1
    mode: str = TaxonomyMode.OFF.value  # off/auto/user
    generated_at: str = ""
    source_datasets: list[str] = field(default_factory=list)
    notes: str = ""
    # Souhrn embedding analýzy (model, distribuce podobností, počty párů…) —
    # vyplňuje annotate_taxonomy_with_embeddings. Prázdné = analýza neproběhla.
    embedding_analysis: dict = field(default_factory=dict)

    @property
    def codes(self) -> list[str]:
        return [lb.code for lb in self.labels]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "source_datasets": self.source_datasets,
            "notes": self.notes,
            "embedding_analysis": self.embedding_analysis,
            "labels": [lb.to_dict() for lb in self.labels],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Taxonomy":
        labels_raw = d.get("labels", []) or []
        labels = [TaxonomyLabel.from_dict(x) for x in labels_raw]
        return cls(
            labels=labels,
            version=int(d.get("version", 1)),
            mode=str(d.get("mode", TaxonomyMode.USER.value)),
            generated_at=str(d.get("generated_at", "")),
            source_datasets=list(d.get("source_datasets", [])),
            notes=str(d.get("notes", "")),
            embedding_analysis=dict(d.get("embedding_analysis") or {}),
        )

    def validate(self) -> list[str]:
        """Vrátí seznam problémů napříč všemi labely + duplicit kódů."""
        errs: list[str] = []
        seen_codes: set[str] = set()
        for lb in self.labels:
            errs.extend(lb.validate())
            if lb.code in seen_codes:
                errs.append(f"duplicitní label.code {lb.code!r}")
            seen_codes.add(lb.code)
        if len(self.labels) < 2:
            errs.append("taxonomy musí mít alespoň 2 labely")
        return errs

    def label_mapping(self) -> dict[str, str]:
        """
        Odvodí normalizační mapping (starý label → nový label) Z DAT taxonomie
        — nahrazuje dřívější hard-coded DEFAULT_LABEL_MAPPING v Python kódu.

        Pravidla:
          - label s remap_to → mapuje se na remap_to,
          - label kind="keyphrase" bez remap_to → KEYPHRASE_<code>,
          - label kind="ner" bez remap_to → identita (beze změny).
        Pro každý CÍLOVÝ label se doplní identita (target → target), aby byl
        mapping idempotentní při znovuzpracování už normalizovaných datasetů.
        """
        mapping: dict[str, str] = {}
        for lb in self.labels:
            mapping[lb.code] = lb.mapped_target()
        for target in list(mapping.values()):
            mapping.setdefault(target, target)
        return mapping

    def ner_label_targets(self) -> frozenset[str]:
        """
        Množina tvrdých NER labelů PO normalizaci (mapované cíle labelů
        kind="ner"). Keyphrase labely (kind i KEYPHRASE_ prefix) se vylučují.
        Nahrazuje hard-coded DEFAULT_NER_LABELS pro taxonomy-driven cestu.
        """
        return frozenset(
            lb.mapped_target()
            for lb in self.labels
            if lb.kind != "keyphrase"
            and not lb.code.startswith("KEYPHRASE_")
            and not lb.mapped_target().startswith("KEYPHRASE_")
        )


# ---------------------------------------------------------------------------
# Default taxonomie (data, ne kód)
# ---------------------------------------------------------------------------

# Default 16-kategorie fallback set žije jako DATA v labeler/default_taxonomy.json
# (jediné místo definice). Z něj se odvozuje DEFAULT_LABEL_MAPPING i
# DEFAULT_NER_LABELS v normalize.py — žádná hard-coded tabulka jmen v kódu.
_DEFAULT_TAXONOMY_PATH = Path(__file__).parent / "default_taxonomy.json"
_default_taxonomy_cache: Optional[Taxonomy] = None


def default_taxonomy() -> Taxonomy:
    """Načte (a cachuje) packaged default taxonomii z default_taxonomy.json."""
    global _default_taxonomy_cache
    if _default_taxonomy_cache is None:
        d = json.loads(_DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8-sig"))
        tax = Taxonomy.from_dict(d)
        errs = tax.validate()
        if errs:
            raise ValueError(
                f"labeler/default_taxonomy.json je nevalidní: {errs[:5]}"
            )
        _default_taxonomy_cache = tax
    return _default_taxonomy_cache


# ---------------------------------------------------------------------------
# Cesty
# ---------------------------------------------------------------------------


def taxonomy_path() -> Path:
    """Kanonická cesta k entity_taxonomy.json v aktivních checkpointech."""
    return CONFIG.checkpoint_dir / "entity_taxonomy.json"


def production_taxonomy_path() -> Optional[Path]:
    """
    Produkční cesta k entity_taxonomy.json (mimo smoke override).

    Smoke test override-uje `CONFIG.checkpoint_dir` na `checkpoints/smoke/`,
    což znamená že `taxonomy_path()` ukazuje do smoke subfolderu — ale
    uživatel obvykle generuje taxonomy přes `taxonomy --mode auto` do
    produkčního `checkpoints/`. Tato funkce vrátí path z původního
    `CHECKPOINT_DIR` env varu, aby smoke test mohl číst produkční taxonomy.

    Vrátí None pokud nelze odvodit (env var nenastavený a current path
    nevypadá jako smoke subfolder).
    """
    import os
    # Primární zdroj: env var CHECKPOINT_DIR (před smoke override)
    env_dir = os.environ.get("CHECKPOINT_DIR")
    if env_dir:
        return Path(env_dir) / "entity_taxonomy.json"
    # Fallback: pokud current checkpoint_dir končí na "smoke", vezmi parent
    cur = CONFIG.checkpoint_dir
    if cur.name == "smoke":
        return cur.parent / "entity_taxonomy.json"
    return None


def load_taxonomy_if_exists() -> Optional[Taxonomy]:
    """
    Načte entity_taxonomy.json. Zkouší v pořadí:

      1. Aktivní `CONFIG.checkpoint_dir / entity_taxonomy.json`
         (typicky `checkpoints/entity_taxonomy.json` v produkci, nebo
         `checkpoints/smoke/entity_taxonomy.json` ve smoke testu pokud
         byl tam generovaný).
      2. Produkční fallback (jen ve smoke test módu):
         `production_taxonomy_path()` — uživatelova produkční taxonomie
         vygenerovaná v hlavním checkpoint_dir.

    Bez fallbacku by smoke test používal default 16 kategorií i když má
    uživatel produkční taxonomy připravenou — to by zkreslilo time
    estimate a OWL export.

    Vrátí None pokud žádná taxonomy soubor není ani na jedné cestě.
    """
    candidate_paths: list[Path] = [taxonomy_path()]
    prod = production_taxonomy_path()
    if prod is not None and prod != candidate_paths[0]:
        candidate_paths.append(prod)

    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
            tax = Taxonomy.from_dict(d)
            errs = tax.validate()
            if errs:
                logger.warning(
                    f"Existující taxonomy {p} má {len(errs)} problémů: {errs[:3]}"
                )
            if p != taxonomy_path():
                logger.info(
                    f"Načítám taxonomy z produkční cesty {p} "
                    f"(smoke checkpoint_dir taxonomy nemá)"
                )
            return tax
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Nelze načíst {p}: {e}")
            continue
    return None


def save_taxonomy(tax: Taxonomy) -> Path:
    """Atomicky zapíše taxonomy do checkpointů."""
    p = taxonomy_path()
    if not tax.generated_at:
        tax.generated_at = datetime.now(timezone.utc).isoformat()
    atomic_write_json(p, tax.to_dict())
    logger.info(f"Taxonomy uložena: {p} ({len(tax.labels)} labelů)")
    return p


# ---------------------------------------------------------------------------
# User mode — načtení z poskytnutého souboru
# ---------------------------------------------------------------------------


def load_user_taxonomy(path: Path) -> Taxonomy:
    """
    Načte uživatelský taxonomy JSON (user mode). Validuje strukturu.

    Očekávaný formát viz examples/entity_taxonomy.example.json.
    """
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy soubor neexistuje: {path}")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    tax = Taxonomy.from_dict(raw)
    tax.mode = TaxonomyMode.USER.value
    if not tax.generated_at:
        tax.generated_at = datetime.now(timezone.utc).isoformat()
    errs = tax.validate()
    if errs:
        raise ValueError(
            f"User taxonomy {path} má problémy:\n  - " + "\n  - ".join(errs[:10])
        )
    logger.info(
        f"User taxonomy načtena: {path} ({len(tax.labels)} labelů: {tax.codes})"
    )
    return tax


# ---------------------------------------------------------------------------
# Auto mode — LLM návrh
# ---------------------------------------------------------------------------


TAXONOMY_SYSTEM_PROMPT = """Jsi expert na návrh NER (Named Entity Recognition) taxonomií pro tréninkové datasety.

Tvůj úkol: z UKÁZEK TEXTŮ z více domén navrhnout JEDNOTNOU taxonomii labelů,
která pokryje VŠECHNY domény konzistentně. Jediná správná odpověď je validní JSON.

ZÁKLADNÍ PRAVIDLA:

1. UNIVERZÁLNÍ KONZISTENCE
   Stejný koncept = STEJNÝ label, i když se v různých doménách projevuje jinak.
   Příklad: "Pojištění auta" (pojišťovnictví) a "ACME Basic" (e-shop) jsou
   oba PRODUKT, ne dva různé labely.

2. ŽÁDNÉ DUPLIKÁTY
   NEVYTVÁŘEJ samostatné labely pro stejný koncept. Špatně: PRODUKT a ITEM.
   Špatně: PRODUKT a INSURANCE_PRODUCT.

3. UPPERCASE_SNAKE jména
   Labely musí být anglické (nebo české bez diakritiky), velkými písmeny,
   slova oddělená podtržítkem. Délka 2-50 znaků. Pattern: [A-Z][A-Z0-9_]{0,49}

4. POČET LABELŮ
{LABEL_COUNT_RULE}

5. KAŽDÝ LABEL MUSÍ MÍT
   - code (UPPERCASE_SNAKE)
   - description (1-2 věty proč existuje a co zachycuje)
   - examples (2-6 konkrétních výrazů z ukázek)
   - domains (které domény label používají — z těch v inputu)
   - kind: "ner" NEBO "keyphrase"
     * "ner"       = krátký pojmenovatelný span (produkt, organizace, místo,
                     datum, hodnota, osoba/role) — tvrdá NER kategorie
     * "keyphrase" = delší popisná fráze (přínos, scénář použití, vlastnost,
                     parametr) — extrahuje se, ale netrénuje se jako tvrdé NER

6. POVINNÉ LABELY (vždy v taxonomii)
   - PRODUKT       konkrétní pojmenovaný produkt nebo služba
   - ORGANIZACE    firma, instituce, značka
   - DATUM         konkrétní datum / časový bod (NE trvání — to je HODNOTA)
   - MISTO         geografické místo
   - HODNOTA       konkrétní hodnota (peníze, množství, parametr)

7. NIKDY NEPŘIDÁVEJ pro stejný koncept více labelů:
   - PRODUKT vs PRODUCT vs ITEM vs ARTICLE  (jen PRODUKT)
   - MISTO vs LOCATION vs PLACE vs CITY     (jen MISTO)
   - DATUM vs DATE vs TIME                  (jen DATUM)
   - ORGANIZACE vs ORG vs COMPANY vs BRAND  (jen ORGANIZACE)

8. NIKDY NEVYTVÁŘEJ tyto labely — jsou to regex-detekovatelné struktury,
   ne pojmenované entity. Pipeline si je dotáhne sama přes regex nad
   input_text; netrénuje na ně NER:
   - URL / WEB / DOMAIN / LINK / WEBSITE
   - EMAIL / E_MAIL / MAIL
   - TELEFON / PHONE / TEL / PHONE_NUMBER

9. DISJUNKTNÍ SÉMANTIKA — NESLÉVEJ různé věci do jedné třídy
   Když ukázky obsahují tyhle ODLIŠNÉ koncepty, dej je do SAMOSTATNÝCH labelů
   (ne do jedné široké třídy). Každá třída = jedna jasná sémantika (ať je
   dataset použitelný pro RE):
   - OSOBA (konkrétní jmenovaná osoba, např. "Jan Novák") vs ROLE / AKTER
     (obecná role: "kupující", "spotřebitel", "poskytovatel údajů"). NEMÍCHAT
     do jednoho OSOBA_ROLE.
   - DATUM (časový bod: "1.1.2026", "rok 2013") vs trvání ("5 let", "14 dnů") —
     trvání NENÍ datum (patří do HODNOTA, nebo zvláštní DOBA/TRVANI).
   - PRODUKT / typ zboží vs KANAL / MEDIUM ("e-shop", "webové stránky",
     "mobilní telefon") vs DOKUMENT ("obchodní podmínky", "smlouva", "návod") —
     kanály ani dokumenty NEJSOU typ produktu.

VRÁTÍŠ POUZE VALIDNÍ JSON v tomto formátu (žádný markdown, žádný text mimo JSON):

{
  "labels": [
    {
      "code": "PRODUKT",
      "description": "Konkrétní pojmenovaný produkt nebo služba — to, co lze koupit, objednat nebo sjednat.",
      "examples": ["ACME Basic", "Pojištění auta OMEGA", "iPhone 15 Pro Max"],
      "domains": ["e-shop", "pojišťovnictví"],
      "kind": "ner"
    },
    ...
  ]
}
"""


# Připomínka pro případy bez stropu: neomezený seznam je přesně místo, kde
# vznikají překrývající se kategorie.
_DISTINGUISHABLE_NOTE = (
    "   Pozor: labely musí zůstat navzájem odlišitelné. Radši jeden label\n"
    "   s jasnou hranicí než dva, mezi kterými se nerozhodneš."
)


def label_count_rule(lo: int, hi: int) -> str:
    """
    Text pravidla o počtu labelů pro prompt fáze 0.

    lo <= 0 = bez spodní hranice, hi <= 0 = bez horního stropu (lze obojí).
    Normalizuje nekonzistentní vstup (0 < hi < lo → hi = lo), ať nevznikne
    nesmyslný rozsah typu "10-5" bez ohledu na to, odkud se funkce volá.
    """
    lo = max(0, lo)
    hi = max(0, hi)
    if 0 < hi < lo:
        hi = lo

    if lo <= 0 and hi <= 0:
        return (
            "   POČET LABELŮ NENÍ NIJAK OMEZENÝ — udělej jich přesně tolik,\n"
            "   kolik dataset potřebuje, aby každá entita měla přesný label\n"
            "   a nemusela padat do obecného koše.\n"
            + _DISTINGUISHABLE_NOTE
        )
    if hi <= 0:
        return (
            f"   Navrhni aspoň {lo} labelů. HORNÍ HRANICE NENÍ STANOVENÁ — udělej\n"
            f"   jich tolik, kolik dataset skutečně potřebuje, aby každá entita\n"
            f"   měla přesný label a nemusela padat do obecného koše.\n"
            + _DISTINGUISHABLE_NOTE
        )
    if lo <= 0:
        return (
            f"   Navrhni nejvýš {hi} labelů. SPODNÍ HRANICE NENÍ STANOVENÁ —\n"
            f"   když dataset pokryje míň kategorií, je to v pořádku; nevymýšlej\n"
            f"   labely jen kvůli počtu.\n"
            + _DISTINGUISHABLE_NOTE
        )
    return (
        f"   Rozumný počet je {lo}-{hi} labelů.\n"
        f"   Příliš málo = neinformativní; příliš mnoho = nekonzistentní LLM výstup."
    )


def taxonomy_system_prompt(
    min_labels: Optional[int] = None, max_labels: Optional[int] = None,
) -> str:
    """Systémový prompt fáze 0 s dosazeným počtem labelů
    (TAXONOMY_MIN_LABELS / TAXONOMY_MAX_LABELS, default 8-16).

    0 (nebo prázdné) vypne příslušnou hranici:
      TAXONOMY_MAX_LABELS=0 → bez horního stropu
      TAXONOMY_MIN_LABELS=0 → bez spodní hranice (vypne i kvalitativní bránu
                              na počet labelů, viz min_llm_labels())
    Lze obojí najednou — pak počet labelů neomezuje nic.

    Dosazuje se přes .replace() — prompt obsahuje literální složené závorky
    (regex pattern, JSON příklad), takže .format() nelze použít. Nekonzistentní
    rozsah (0 < max < min) se srovná na max = min."""
    lo = CONFIG.taxonomy_min_labels if min_labels is None else min_labels
    hi = CONFIG.taxonomy_max_labels if max_labels is None else max_labels
    lo = max(0, lo)
    hi = max(0, hi)
    if 0 < hi < lo:
        hi = lo
    return (
        TAXONOMY_SYSTEM_PROMPT
        .replace("{LABEL_COUNT_RULE}", label_count_rule(lo, hi))
        .replace("{MIN_LABELS}", str(lo) if lo > 0 else "bez minima")
        .replace("{MAX_LABELS}", str(hi) if hi > 0 else "bez stropu")
    )


TAXONOMY_USER_TEMPLATE = """DOMÉNY V DATASETU:
{domains_block}

UKÁZKY TEXTŮ Z KAŽDÉ DOMÉNY (po {samples_per_dataset} ukázek):

{samples_block}

Navrhni JEDNOTNOU taxonomii NER labelů, která pokryje všechny domény konzistentně.
Vrať validní JSON v dokumentovaném formátu."""


def _format_samples_for_prompt(
    samples_by_domain: dict[str, list[str]],
    preview_chars: Optional[int] = None,
) -> tuple[str, str]:
    """
    Vytvoří dva bloky:
      domains_block: bulletový seznam domén
      samples_block: ukázky textů per doména
    """
    cap = preview_chars or CONFIG.taxonomy_sample_preview_chars
    domains_block = "\n".join(f"  - {d}" for d in sorted(samples_by_domain.keys()))
    samples_block_parts = []
    for d in sorted(samples_by_domain.keys()):
        samples = samples_by_domain[d]
        samples_block_parts.append(f"=== {d} ===")
        for i, s in enumerate(samples, 1):
            preview = s[:cap].replace("\n", " ")
            samples_block_parts.append(f"  [{i}] {preview}")
        samples_block_parts.append("")
    return domains_block, "\n".join(samples_block_parts)


# Hrubý převod znaků na tokeny. Qwen tokenizer dělá u české diakritiky ~2,5-3
# znaky/token; 2.5 je konzervativní (radši prompt uříznu víc, než abych dostal
# HTTP 400 a spadl na default 16 kategorií).
_CHARS_PER_TOKEN = 2.5


def _prompt_char_budget(completion_tokens: int, reserve_tokens: int = 1536) -> int:
    """Kolik znaků promptu se vejde do kontextu serveru vedle odpovědi."""
    ctx = max(4096, CONFIG.extractor_context_len)
    usable = ctx - completion_tokens - reserve_tokens
    if usable < 1024:
        usable = max(512, ctx // 4)
    return int(usable * _CHARS_PER_TOKEN)


def _fit_samples_to_budget(
    samples_by_domain: dict[str, list[str]],
    char_budget: int,
    *,
    min_per_domain: int = 2,
) -> tuple[dict[str, list[str]], int, Optional[str]]:
    """
    Uřízne ukázky tak, aby se blok vešel do char_budget.

    Nejdřív zkracuje náhled jednoho textu (600 → 400 → 300 → 200 → 120 znaků),
    pak ubírá ukázky kolem dokola přes domény (každá si drží min_per_domain).
    Vrací (ukázky, preview_chars, popis uříznutí nebo None).
    """
    def rendered_len(sm: dict[str, list[str]], cap: int) -> int:
        _, block = _format_samples_for_prompt(sm, preview_chars=cap)
        return len(block)

    caps = [c for c in (CONFIG.taxonomy_sample_preview_chars, 400, 300, 200, 120)
            if c <= CONFIG.taxonomy_sample_preview_chars] or [120]
    n_before = sum(len(v) for v in samples_by_domain.values())

    cur = {d: list(v) for d, v in samples_by_domain.items()}
    cap = caps[0]
    if rendered_len(cur, cap) <= char_budget:
        return cur, cap, None

    for c in caps:
        cap = c
        if rendered_len(cur, cap) <= char_budget:
            return cur, cap, (
                f"náhled ukázky zkrácen na {cap} znaků (kontext "
                f"{CONFIG.extractor_context_len})"
            )

    # Ani nejkratší náhled nestačí → ubírej ukázky (od domén s nejvíc ukázkami).
    while rendered_len(cur, cap) > char_budget:
        biggest = max(cur, key=lambda d: len(cur[d]))
        if len(cur[biggest]) <= min_per_domain:
            break
        cur[biggest].pop()
    n_after = sum(len(v) for v in cur.values())
    return cur, cap, (
        f"prompt uříznut na kontext {CONFIG.extractor_context_len}: "
        f"{n_after}/{n_before} ukázek, náhled {cap} znaků"
    )


def _collect_samples(
    items: Iterable[LabelItem],
    samples_per_dataset: int = 10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Pro každý source dataset vyber vzorky ROZPROSTŘENÉ po celém datasetu
    (ne jen prvních N). Strategie:

      - 3 "kotvy": první, prostřední a poslední položka datasetu — garantují
        pokrytí začátku, středu i konce (datasety bývají chronologicky/tematicky
        seřazené, takže prvních N by bylo zkreslené).
      - zbytek: náhodný výběr `samples_per_dataset` položek ze zbývajících
        indexů (deterministicky přes seed).

    Výsledek tak může mít až `samples_per_dataset + 3` položek a je rozprostřený
    napříč celým datasetem. Malé datasety (<= samples_per_dataset + 3) se vezmou
    celé. Deterministické přes seed (stejný seed → identický výběr).
    """
    by_source: dict[str, list[str]] = {}
    for it in items:
        by_source.setdefault(it.domain, []).append(it.text)
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    # sorted() pro deterministické pořadí konzumace rng napříč doménami
    for domain in sorted(by_source.keys()):
        texts = by_source[domain]
        n = len(texts)
        if n <= samples_per_dataset + 3:
            # Malý dataset — vezmi vše (rozprostírání nemá smysl)
            out[domain] = list(texts)
            continue
        # 3 kotvy přes index (první / prostřední / poslední)
        anchor_idx = {0, n // 2, n - 1}
        rest_idx = [i for i in range(n) if i not in anchor_idx]
        k = min(samples_per_dataset, len(rest_idx))
        chosen_idx = sorted(anchor_idx | set(rng.sample(rest_idx, k)))
        out[domain] = [texts[i] for i in chosen_idx]
    return out


# Labely, které LLM NIKDY nesmí vytvořit v taxonomii — jsou to regex-
# detekovatelné struktury, ne NER kategorie. Prompt to říká, ale LLM
# občas instrukci ignoruje (zejména menší modely / fallback). Tady je
# bezpečnostně dropujeme po parse.
_FORBIDDEN_TAXONOMY_CODES: frozenset[str] = frozenset({
    "URL", "WEB", "DOMAIN", "LINK", "WEBSITE", "WEBOVA_ADRESA",
    "EMAIL", "E_MAIL", "MAIL", "EMAILOVA_ADRESA",
    "TELEFON", "PHONE", "TEL", "PHONE_NUMBER", "TELEFONNI_CISLO",
})


def _parse_llm_taxonomy_response(content: str, parsed: Optional[dict]) -> Taxonomy:
    """
    Z LLM odpovědi vytvoř Taxonomy nebo raise ValueError.

    Bezpečnostní filtr: regex-detekovatelné struktury (URL/EMAIL/TELEFON)
    se dropnou, i kdyby je LLM v taxonomii navrhl. Pipeline si je
    dotáhne sama přes regex nad input_text — netrénuje se na ně NER.
    """
    if parsed is None:
        # Pokus o extrakci JSON z textu (LLMClient už to zkouší, ale jistota)
        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM nevrátil validní JSON: {e}")

    # Model může vrátit buď {"labels":[...]} nebo rovnou bare list labelů.
    if isinstance(parsed, list):
        raw_labels = parsed
    elif isinstance(parsed, dict):
        raw_labels = parsed.get("labels") or parsed.get("Labels") or []
    else:
        raw_labels = []
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("LLM vrátil JSON bez 'labels' pole")

    labels: list[TaxonomyLabel] = []
    dropped: list[str] = []
    for raw in raw_labels:
        if not isinstance(raw, dict) or "code" not in raw:
            continue
        code_upper = str(raw["code"]).strip().upper()
        if code_upper in _FORBIDDEN_TAXONOMY_CODES:
            dropped.append(code_upper)
            continue
        labels.append(TaxonomyLabel.from_dict(raw))

    if dropped:
        logger.info(
            f"taxonomy: dropuji {len(dropped)} forbidden labels (regex-detekovatelné "
            f"struktury, ne NER): {dropped}"
        )

    return Taxonomy(labels=labels, mode=TaxonomyMode.AUTO.value)


def _deduplicate_aliases(tax: Taxonomy) -> Taxonomy:
    """
    Pokud LLM vrátí 2 labely s totožným code nebo se code matchuje s něčím v
    aliases jiného labelu, sloučí je (zachová pierwszy, druhý přidá do aliases).
    """
    seen: dict[str, TaxonomyLabel] = {}
    for lb in tax.labels:
        key = lb.code.upper()
        if key in seen:
            # Duplicitní — přidej do aliases
            existing = seen[key]
            for ex in lb.examples:
                if ex not in existing.examples:
                    existing.examples.append(ex)
            for d in lb.domains:
                if d not in existing.domains:
                    existing.domains.append(d)
            for a in lb.aliases:
                if a not in existing.aliases:
                    existing.aliases.append(a)
            logger.info(f"taxonomy: deduplikuji duplikát code={lb.code!r}")
            continue
        # Match přes aliases
        merged = False
        for canon in seen.values():
            if key in {a.upper() for a in canon.aliases}:
                logger.info(
                    f"taxonomy: label {lb.code!r} matched jako alias "
                    f"{canon.code!r}, slévám"
                )
                if lb.code not in canon.aliases:
                    canon.aliases.append(lb.code)
                # Merge examples, domains, aliases ze sloučeného labelu
                for ex in lb.examples:
                    if ex not in canon.examples:
                        canon.examples.append(ex)
                for d in lb.domains:
                    if d not in canon.domains:
                        canon.domains.append(d)
                for a in lb.aliases:
                    if a.upper() != canon.code.upper() and a not in canon.aliases:
                        canon.aliases.append(a)
                merged = True
                break
        if not merged:
            seen[key] = lb
    tax.labels = list(seen.values())
    return tax


# Povinné NER labely. Tyto univerzální kategorie pipeline očekává napříč
# doménami (relation rules, core NER export). Pokud je LLM v auto módu
# nenavrhne, SW si je doplní SÁM (nečeká, až to model "trefí"). Popisy jsou
# generické a >= 10 znaků (projdou TaxonomyLabel.validate()).
_MANDATORY_LABELS: dict[str, str] = {
    "PRODUKT": "Konkrétní pojmenovaný produkt, služba, tarif nebo balíček.",
    "ORGANIZACE": "Firma, instituce, značka nebo jiný organizační subjekt.",
    "DATUM": "Konkrétní datum / časový bod (termín, platnost). NE trvání ('5 let', '14 dnů') — to patří do HODNOTA.",
    "MISTO": "Místo, adresa, lokalita nebo geografický údaj.",
    "HODNOTA": "Číselná hodnota, částka, množství, sazba nebo jednotka.",
}

# Minimální počet labelů, které musí navrhnout SÁM LLM (před doplněním
# povinných), aby se auto-taxonomie považovala za kvalitní. Pod tímto prahem
# to vypadá, že model instrukci odbyl → reklamace + retry.
#
# Odvozeno z TAXONOMY_MIN_LABELS s tolerancí 3 pod instruovaným minimem:
# prompt říká "aspoň MIN", brána reklamuje až pod MIN-3 (default 8-3=5 —
# zachovává původní chování; brána nemá být přísnější než nutné, model
# občas legitimně slije 1-2 kategorie). Podlaha 2 (míň nedává smysl).
def min_llm_labels() -> int:
    """
    Kolik vlastních labelů musí LLM navrhnout, aby návrh prošel bránou.

    Tolerance 3 pod TAXONOMY_MIN_LABELS — model občas legitimně slije
    kategorie. 0 = brána na počet je vypnutá (TAXONOMY_MIN_LABELS=0).

    Čte se za běhu, ne při importu, aby šlo hodnotu měnit v .env i v testech.
    Úplně prázdný návrh projít nemůže ani s vypnutou bránou: parse odmítne
    JSON bez 'labels'.
    """
    configured = CONFIG.taxonomy_min_labels
    if configured <= 0:
        return 0
    return max(2, configured - 3)


def _ensure_mandatory_labels(tax: Taxonomy) -> list[str]:
    """
    Doplní povinné labely (_MANDATORY_LABELS), které LLM nenavrhl.

    Přítomnost se kontroluje case-insensitive proti VŠEM existujícím kódům
    i aliasům (aby se PRODUKT nepřidal duplicitně, když už tam je jako alias
    jiného labelu). Vrací seznam kódů, které byly doplněny (prázdný = nic
    nechybělo). Mutuje `tax` in-place.
    """
    present: set[str] = set()
    for lb in tax.labels:
        present.add(lb.code.upper())
        for a in lb.aliases:
            present.add(a.upper())

    added: list[str] = []
    for code, desc in _MANDATORY_LABELS.items():
        if code.upper() not in present:
            tax.labels.append(TaxonomyLabel(code=code, description=desc))
            present.add(code.upper())
            added.append(code)

    if added:
        logger.info(
            f"taxonomy: SW doplnil {len(added)} povinných labelů, které LLM "
            f"nenavrhl: {added}"
        )
    return added


def _assess_taxonomy_quality(tax: Taxonomy, n_llm_labels: int) -> list[str]:
    """
    Posoudí kvalitu auto-discovered taxonomie. Vrací seznam konkrétních výtek
    (prázdný = OK). Výtky se LLM posílají jako "reklamace" pro retry.

    n_llm_labels = počet labelů, které navrhl SÁM LLM (po dedup, PŘED doplněním
    povinných) — povinné labely doplněné SW se do kvality LLM nepočítají.
    """
    complaints: list[str] = []
    min_labels = min_llm_labels()
    if min_labels and n_llm_labels < min_labels:
        complaints.append(
            f"Navrhl jsi jen {n_llm_labels} vlastních labelů, což je málo — "
            f"potřebuju aspoň {min_labels} smysluplných NER kategorií, "
            f"které pokrývají entity ve vzorcích (osoby, produkty, organizace, "
            f"místa, částky, data, …)."
        )
    errs = tax.validate()
    if errs:
        complaints.append(
            "Některé labely nejsou validní (kód MUSÍ být UPPERCASE_SNAKE, popis "
            "aspoň 10 znaků, žádné duplicity): " + "; ".join(errs[:5])
        )
    return complaints


def _build_complaint_message(complaints: list[str]) -> str:
    """Sestaví "reklamační" user zprávu pro LLM s konkrétními výtkami a
    připomene formát. Posílá se jako další user turn pro retry."""
    bullets = "\n".join(f"- {c}" for c in complaints)
    return (
        "Tvůj návrh taxonomie má tyto problémy:\n"
        f"{bullets}\n\n"
        "Oprav je a vrať CELOU opravenou taxonomii znovu — POUZE validní JSON "
        "objekt s polem 'labels'. Každý label: 'code' v UPPERCASE_SNAKE, "
        "'description' aspoň 10 znaků, volitelně 'examples'/'domains'/'aliases'. "
        "Žádný text okolo JSON, žádné URL/EMAIL/TELEFON labely."
    )


async def discover_taxonomy_auto(
    *,
    extractor_client: LLMClient,
    samples_per_dataset: int = 10,
    seed: int = 42,
    max_tokens: int = 8192,
) -> Taxonomy:
    """
    Auto mode: vezmi vzorky ze všech datasetů v DATA_DIR a pošli LLM.

    Returns Taxonomy. Raise LLMFatalError / ValueError při selhání.
    """
    items = list(scan_data_dir(CONFIG.data_dir))
    if not items:
        raise FileNotFoundError(
            f"V {CONFIG.data_dir} nejsou žádné label itemy — taxonomy discovery "
            f"nelze spustit. Nejdřív tam dej JSON exporty."
        )

    # Fáze 0a: ukázky vybrané embeddingy (taxonomy_sampling.json), jinak
    # náhodný výběr (best-effort — chybějící / zastaralý výběr běh nezastaví).
    from .taxonomy_sampling import load_samples_for_discovery
    emb_samples = load_samples_for_discovery(items)
    if emb_samples is not None:
        samples, sampling_note = emb_samples
        counts = [len(v) for v in samples.values()] or [0]
        per_domain_label = (
            str(counts[0]) if min(counts) == max(counts)
            else f"{min(counts)}–{max(counts)}"
        )
    else:
        samples = _collect_samples(items, samples_per_dataset=samples_per_dataset, seed=seed)
        sampling_note = f"náhodný výběr {samples_per_dataset}+3 kotvy per doména"
        per_domain_label = str(samples_per_dataset)
    if not samples:
        raise ValueError("Žádné vzorky nelze sestavit (prázdné texty?)")

    # Prompt musí sednout do kontextu serveru VČETNĚ místa na odpověď — jinak
    # vrátí HTTP 400 a celá fáze 0 spadne na defaultních 16 kategorií.
    completion_tokens = max(1024, min(max_tokens, CONFIG.taxonomy_max_tokens))
    samples, preview_chars, trim_note = _fit_samples_to_budget(
        samples, _prompt_char_budget(completion_tokens),
    )
    if trim_note:
        logger.warning(f"taxonomy discovery: {trim_note}")
        sampling_note = f"{sampling_note}; {trim_note}"

    domains_block, samples_block = _format_samples_for_prompt(
        samples, preview_chars=preview_chars,
    )
    user_prompt = TAXONOMY_USER_TEMPLATE.format(
        domains_block=domains_block,
        samples_block=samples_block,
        samples_per_dataset=per_domain_label,
    )

    logger.info(
        f"taxonomy discovery: posílám LLM {sum(len(v) for v in samples.values())} "
        f"ukázek z {len(samples)} domén: {sorted(samples.keys())} "
        f"(~{len(samples_block) // 1000} kB promptu, {completion_tokens} tokenů "
        f"na odpověď, kontext {CONFIG.extractor_context_len})"
    )

    messages = [
        {"role": "system", "content": taxonomy_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    # 2 pokusy: když je kvalita nedostatečná, pošleme LLM "reklamaci" s
    # konkrétními výtkami a necháme to zkusit znovu. Když neprojde ani druhý
    # pokus, vyhodíme TaxonomyQualityError → caller spadne na default 16 kategorií.
    max_attempts = 2
    last_complaints: list[str] = ["(žádný validní výstup)"]
    for attempt in range(1, max_attempts + 1):
        try:
            # reasoning z CONFIG (TAXONOMY_REASONING, default ON) — thinking
            # zlepšuje kvalitu navržené taxonomie; jde o jednorázová volání.
            resp = await extractor_client.chat(
                messages, max_tokens=completion_tokens,
                reasoning=CONFIG.taxonomy_reasoning,
                # Bez tohohle by llm_client s reasoning=True tlačil minimum
                # 16384 tokenů na odpověď — s velkým promptem se to do
                # kontextu nevejde (HTTP 400).
                reasoning_max_tokens=completion_tokens,
            )
        except (LLMFatalError, LLMTransientError) as e:
            raise LLMFatalError(f"taxonomy discovery: LLM volání selhalo: {e}")

        # Parse — neparsovatelná odpověď je taky důvod k reklamaci + retry.
        tax: Optional[Taxonomy] = None
        try:
            tax = _parse_llm_taxonomy_response(resp.content, resp.parsed)
        except ValueError as e:
            last_complaints = [f"Tvoje odpověď nešla naparsovat jako validní JSON: {e}."]

        if tax is not None:
            tax = _deduplicate_aliases(tax)
            # Počet labelů od SAMOTNÉHO LLM (po dedup, PŘED doplněním povinných) —
            # kvalita se posuzuje z toho, co navrhl model, ne z toho co doplnil SW.
            n_llm_labels = len(tax.labels)
            _ensure_mandatory_labels(tax)  # T2: SW si povinné labely doplní sám
            complaints = _assess_taxonomy_quality(tax, n_llm_labels)
            if not complaints:
                # === KVALITA OK — hotovo ===
                tax.source_datasets = sorted(samples.keys())
                tax.notes = (
                    f"Auto-discovered ({sampling_note}) z {per_domain_label} ukázek per doména "
                    f"({sum(len(v) for v in samples.values())} celkem). LLM: "
                    f"{extractor_client.model}. Pokus {attempt}/{max_attempts}."
                )
                return tax
            last_complaints = complaints

        # Kvalita/parse selhaly — reklamuj a zkus znovu (pokud zbývá pokus).
        if attempt < max_attempts:
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {"role": "user", "content": _build_complaint_message(last_complaints)}
            )
            logger.warning(
                f"taxonomy discovery: pokus {attempt}/{max_attempts} nedostatečný "
                f"({len(last_complaints)} výtek) — reklamuji a posílám znovu: "
                f"{last_complaints}"
            )

    # Vyčerpány pokusy — fail na default (caller to odchytí a použije 16 kategorií).
    raise TaxonomyQualityError(
        f"taxonomy discovery: kvalita nedostatečná ani po {max_attempts} "
        f"pokusech. Poslední výtky: {last_complaints}"
    )


# ---------------------------------------------------------------------------
# Fáze 0 / Pilíř A — Automatický disjointness audit
# ---------------------------------------------------------------------------
#
# Po návrhu taxonomie (discover_taxonomy_auto) tenhle audit EMPIRICKY ověří,
# jestli jsou hranice labelů v praxi rozlišitelné, a zaměňované páry sám
# zostří. Plně automatické, univerzální, BEZ hard-coded slov a BEZ člověka:
#
#   ①  Probe     na held-out vzorcích si od modelu vyžádá u každého span
#                top-1 / top-2 label + jistotu → datová matice konfuze.
#   ②  Diagnóza  statistickým prahem vybere nejzaměňovanější páry.
#   ③  Oprava    "zostři" — model přepíše description + decision_rule +
#                counter_examples OBOU labelů z reálně zaměněných spanů.
#                (rozděl/sluč jsou za flagem structural_edits, default OFF.)
#   ④  Přeměř    další kolo změří efekt na čerstvém vzorku.
#   ⑤  Brána     drží se NEJLEPŠÍ viděná taxonomie; konec při dosažení cíle,
#                plateau/regresi, nebo vyčerpání kol. Nikdy nezhorší vstup.
#
# Ruler (čím se konfuze měří) je VYMĚNITELNÝ: teď LLM top-2 probe (Pilíř A),
# později stačí nahradit _measure_confusion embedding-margin verzí (Pilíř B) —
# smyčka audit_and_refine_taxonomy se nezmění.


@dataclass
class ConfusablePair:
    """Jeden zaměňovaný pár labelů z probe (a < b lexikograficky)."""
    a: str
    b: str
    score: float          # součet nejistoty přes zaměněné spany (vyšší = horší)
    support: int          # počet spanů, které tenhle pár zaměnily
    # Který ruler pár našel: "llm" | "emb" | "both" (u combined ruleru).
    sources: str = "llm"


@dataclass
class ConfusionReport:
    """Výsledek jednoho měření konfuze nad held-out vzorkem."""
    score: float                              # průměrná nejistota top-1 (0-1)
    n_spans: int
    pairs: list[ConfusablePair] = field(default_factory=list)
    evidence: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Který ruler měření provedl: "llm" (self-consistency probe nad spany)
    # nebo "embedding" (kosinová podobnost profilů tříd). Ovlivňuje výběr
    # párů — u embedding ruleru nemá `support` význam (viz
    # _select_confusable_pairs).
    ruler: str = "llm"


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _aggregate_confusion(spans: list, valid_codes: set) -> ConfusionReport:
    """
    Z odpovědi probe (seznam {text, top1, top2, confidence}) spočítá:
      - score = průměrná nejistota (1 - confidence) přes validní spany,
      - páry (top1, top2) seřazené podle sumární nejistoty,
      - evidence = reálné zaměněné spany per pár (vstup pro zostření).

    Pure/deterministické — testovatelné bez LLM. valid_codes se matchují
    case-insensitive.
    """
    upper = {str(c).upper() for c in valid_codes}
    total_amb = 0.0
    n = 0
    pair_score: dict[tuple[str, str], float] = {}
    pair_support: dict[tuple[str, str], int] = {}
    evidence: dict[tuple[str, str], list[str]] = {}

    for sp in spans:
        if not isinstance(sp, dict):
            continue
        top1 = str(sp.get("top1") or "").strip().upper()
        if top1 not in upper:
            continue
        amb = 1.0 - _clamp01(sp.get("confidence", 1.0))
        total_amb += amb
        n += 1
        top2 = str(sp.get("top2") or "").strip().upper()
        if top2 and top2 in upper and top2 != top1:
            key = tuple(sorted((top1, top2)))
            pair_score[key] = pair_score.get(key, 0.0) + amb
            pair_support[key] = pair_support.get(key, 0) + 1
            text = str(sp.get("text") or "").strip()
            if text and text not in evidence.setdefault(key, []):
                evidence[key].append(text)

    score = (total_amb / n) if n else 0.0
    pairs = [
        ConfusablePair(a=k[0], b=k[1], score=pair_score[k], support=pair_support[k])
        for k in pair_score
    ]
    pairs.sort(key=lambda p: (p.score, p.support), reverse=True)
    return ConfusionReport(score=score, n_spans=n, pairs=pairs, evidence=evidence)


def _select_confusable_pairs(
    report: ConfusionReport, max_pairs: int, min_support: int
) -> list[ConfusablePair]:
    """
    Vybere nejzaměňovanější páry. Pure.

    `min_support` (kolik spanů pár reálně zaměnilo) filtruje jen páry z LLM
    probe — chrání před zostřováním na základě jednoho náhodného výskytu.
    Embedding měření je DETERMINISTICKÉ a každá dvojice je změřena právě
    jednou, takže u ní support nenese informaci a filtr se přeskakuje;
    u combined ruleru se filtr aplikuje jen na páry, které našel POUZE LLM.
    """
    if report.ruler == "embedding":
        return report.pairs[:max_pairs]
    if report.ruler == "combined":
        eligible = [
            p for p in report.pairs
            if p.sources != "llm" or p.support >= min_support
        ]
        return eligible[:max_pairs]
    eligible = [p for p in report.pairs if p.support >= min_support]
    return eligible[:max_pairs]


def _extract_json_list(content: str, parsed, *keys: str) -> list:
    """Vytáhne seznam z LLM odpovědi — buď z parsed, nebo z content. Zkouší
    dané klíče v dict odpovědi, nebo bere bare list. Vrací [] když nic."""
    data = parsed
    if data is None:
        try:
            data = json.loads((content or "").strip())
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def _heldout_snippets(
    items,
    *,
    samples_per_dataset: int,
    seed: int,
    max_snippets: int = 24,
    max_chars: int = 500,
) -> list[str]:
    """
    Sestaví held-out textové úryvky pro probe. Reuse _collect_samples
    (rozprostření + seed), pak interleave napříč doménami (vyvážený vzorek),
    ořízne délku a nastropí počet, aby prompt nepřetekl. Různý seed per kolo
    → různé vzorky (proxy za held-out).
    """
    by_domain = _collect_samples(
        items, samples_per_dataset=samples_per_dataset, seed=seed
    )
    pools = [list(by_domain[k]) for k in sorted(by_domain.keys())]
    snippets: list[str] = []
    layer = 0
    while len(snippets) < max_snippets:
        progressed = False
        for pool in pools:
            if layer < len(pool):
                txt = " ".join(str(pool[layer]).split())[:max_chars]
                if txt:
                    snippets.append(txt)
                    progressed = True
                if len(snippets) >= max_snippets:
                    break
        if not progressed:
            break
        layer += 1
    return snippets


def _audit_labels_block(tax: Taxonomy) -> str:
    """Kompaktní výpis labelů pro audit prompty (kód + popis + hranice)."""
    lines: list[str] = []
    for lb in tax.labels:
        lines.append(f"  - {lb.code}: {lb.description}")
        if lb.decision_rule:
            lines.append(f"      (hranice: {lb.decision_rule})")
    return "\n".join(lines)


# ---- ① Probe -----------------------------------------------------------------

_CONFUSION_PROBE_SYSTEM = """Jsi auditor NER taxonomie. Dostaneš SEZNAM POVOLENÝCH LABELŮ
a několik krátkých textů. Tvůj úkol NENÍ extrahovat všechno — jen u význačných
entitních spanů posoudit, jak JISTĚ jim lze přiřadit label.

Pro každý význačný span vrať:
  - "text"        přesný výraz z textu,
  - "top1"        nejpravděpodobnější label (POUZE z povolených),
  - "top2"        druhý nejpravděpodobnější label, když span reálně padá na hranici
                  dvou labelů; jinak null,
  - "confidence"  tvá jistota v top1 (0.0-1.0). Když bys mezi top1/top2 skoro
                  házel korunou, dej nízkou (~0.5).

Zajímají nás hlavně HRANIČNÍ spany (kde váháš mezi dvěma labely) — ty odhalují
překrývající se kategorie. Nevymýšlej labely mimo povolené. Vrať POUZE validní
JSON {"spans": [...]}, žádný text okolo."""


def _build_confusion_probe_messages(tax: Taxonomy, snippets: list[str]) -> list[dict]:
    labels_block = _audit_labels_block(tax)
    texts_block = "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets, 1))
    user = (
        f"POVOLENÉ LABELY:\n{labels_block}\n\n"
        f"TEXTY:\n{texts_block}\n\n"
        'Vrať JSON {"spans": [{"text": ..., "top1": ..., "top2": ...|null, '
        '"confidence": 0.0-1.0}]} — hlavně hraniční spany.'
    )
    return [
        {"role": "system", "content": _CONFUSION_PROBE_SYSTEM},
        {"role": "user", "content": user},
    ]


async def _measure_confusion(
    client, tax: Taxonomy, snippets: list[str], *, max_tokens: int = 8192
) -> ConfusionReport:
    """Ruler (Pilíř A): LLM top-2 probe → matice konfuze. Best-effort — když
    volání/parse selže, vrátí prázdný report (audit to bere jako 'neměřím')."""
    if not snippets:
        return ConfusionReport(score=0.0, n_spans=0)
    messages = _build_confusion_probe_messages(tax, snippets)
    try:
        resp = await client.chat(
            messages, max_tokens=max_tokens,
            reasoning=CONFIG.taxonomy_reasoning,
            # Strop i pro thinking trace — jinak llm_client s reasoning=True
            # tlačí minimum 16384 a při těsném kontextu vrátí server HTTP 400.
            reasoning_max_tokens=max(1024, min(max_tokens, CONFIG.taxonomy_max_tokens)),
        )
    except (LLMFatalError, LLMTransientError) as e:
        logger.warning(f"audit: probe LLM volání selhalo ({e}); přeskakuji měření")
        return ConfusionReport(score=0.0, n_spans=0)
    spans = _extract_json_list(resp.content, resp.parsed, "spans", "Spans")
    valid: set[str] = set(tax.codes)
    for lb in tax.labels:
        valid.update(lb.aliases)
    report = _aggregate_confusion(spans, valid)
    logger.info(
        f"audit: probe změřil {report.n_spans} spanů, konfuze={report.score:.3f}, "
        f"top páry={[(p.a, p.b, round(p.score, 2)) for p in report.pairs[:3]]}"
    )
    return report


# ---- ①b Ruler přes EMBEDDINGY (Pilíř B) --------------------------------------
# Deterministická a levná alternativa LLM probe: konfuze se měří kosinovou
# podobností profilů tříd (labeler/taxonomy_embeddings.py). Výhody proti LLM
# ruleru: žádné LLM volání na měření, nulový šum, řádově rychlejší → smyčka
# může iterovat víc kol. Vyžaduje VOLITELNOU závislost sentence-transformers;
# bez ní se `create_embedding_ruler` vrátí None a audit spadne na LLM ruler.


def create_embedding_ruler():
    """
    Vytvoří stav embedding ruleru (drží backend, aby se model načetl JEN
    JEDNOU napříč koly auditu). Vrátí None, když embeddingy nejsou dostupné
    (chybí volitelná závislost / model) — volající pak použije LLM ruler.
    """
    try:
        from .taxonomy_embeddings import (
            TaxonomyEmbeddingConfig, _EmbeddingRulerState,
        )
        state = _EmbeddingRulerState(config=TaxonomyEmbeddingConfig.from_env())
        state.ensure_backend()   # fail-fast: model se načte teď, ne uprostřed smyčky
        return state
    except Exception as e:  # noqa: BLE001 — volitelná závislost, best-effort
        logger.warning(
            f"audit: embedding ruler nedostupný ({type(e).__name__}: {e}) — "
            f"použije se LLM ruler. Embeddingy vyžadují sentence-transformers "
            f"(pip install -r requirements.txt) a stažitelný model "
            f"TAXONOMY_EMBEDDING_MODEL."
        )
        return None


def measure_confusion_embedding(ruler_state, tax: Taxonomy) -> ConfusionReport:
    """
    Ruler (Pilíř B): kosinová podobnost profilů tříd → matice konfuze.

    Skóre je ABSOLUTNÍ (průměr podobností nejhorších dvojic), takže audit umí
    porovnávat kola mezi sebou — percentilová metrika by byla samonormalizační
    a zlepšení by nikdy nenaměřila.

    `evidence` zůstává prázdná: embedding ruler nepracuje se spany z korpusu.
    Zostřovací prompt má i tak k dispozici popisy a příklady obou tříd.
    Best-effort — při chybě vrátí prázdný report (audit to bere jako 'neměřím').
    """
    try:
        from .taxonomy_embeddings import measure_taxonomy_confusion
        m = measure_taxonomy_confusion(
            tax, ruler_state.config, ruler_state.ensure_backend(),
        )
    except Exception as e:  # noqa: BLE001 — měření je best-effort
        logger.warning(f"audit: embedding měření selhalo ({e}); přeskakuji kolo")
        return ConfusionReport(score=0.0, n_spans=0, ruler="embedding")

    pairs = [
        ConfusablePair(a=a, b=b, score=sim, support=1)
        for a, b, sim, _z in m.pairs
    ]
    report = ConfusionReport(
        score=m.score,
        n_spans=len(pairs),      # u embeddingů = počet změřených dvojic
        pairs=pairs,
        evidence={},
        ruler="embedding",
    )
    logger.info(
        f"audit: embedding ruler změřil {len(pairs)} dvojic, "
        f"konfuze={report.score:.3f}, nejhorší="
        f"{[(p.a, p.b, round(p.score, 2)) for p in pairs[:3]]}"
    )
    return report


# ---- ①c Kombinace obou rulerů (auto) -----------------------------------------


def _combine_confusion_reports(
    emb: Optional[ConfusionReport],
    llm: Optional[ConfusionReport],
    weight: float,
) -> ConfusionReport:
    """
    Spojí embedding a LLM měření do jednoho reportu (ruler="combined").

    Skóre = weight × embedding + (1 − weight) × LLM (obě jsou v 0-1: embedding
    = průměrná podobnost nejbližších dvojic, LLM = průměrná nejistota top-1).

    Páry: skóre každého ruleru se normalizuje jeho vlastním maximem (absolutní
    hodnoty nejsou srovnatelné) a pak se sečtou stejnými vahami → pár, který
    našly OBA rulery, má vyšší skóre než pár z jednoho. `sources` říká odkud
    (emb / llm / both), `support` a `evidence` (reálně zaměněné výrazy pro
    zostřovací prompt) přebíráme z LLM probe.

    Když jedno měření chybí nebo nic nenaměřilo, vrací se druhé beze změny.
    Pure — testovatelné bez LLM i bez embeddingů.
    """
    if emb is None or emb.n_spans == 0:
        return llm if llm is not None else ConfusionReport(score=0.0, n_spans=0)
    if llm is None or llm.n_spans == 0:
        return emb
    w = min(max(weight, 0.0), 1.0)

    def _norm(pairs: list[ConfusablePair]) -> dict[tuple[str, str], float]:
        top = max((p.score for p in pairs), default=0.0)
        if top <= 0:
            return {(p.a, p.b): 0.0 for p in pairs}
        return {(p.a, p.b): p.score / top for p in pairs}

    emb_norm, llm_norm = _norm(emb.pairs), _norm(llm.pairs)
    llm_support = {(p.a, p.b): p.support for p in llm.pairs}
    pairs: list[ConfusablePair] = []
    for key in sorted(set(emb_norm) | set(llm_norm)):
        in_emb, in_llm = key in emb_norm, key in llm_norm
        pairs.append(ConfusablePair(
            a=key[0], b=key[1],
            score=w * emb_norm.get(key, 0.0) + (1 - w) * llm_norm.get(key, 0.0),
            support=llm_support.get(key, 0),
            sources="both" if in_emb and in_llm else ("emb" if in_emb else "llm"),
        ))
    pairs.sort(key=lambda p: (p.score, p.sources == "both", p.support), reverse=True)
    return ConfusionReport(
        score=w * emb.score + (1 - w) * llm.score,
        n_spans=emb.n_spans + llm.n_spans,
        pairs=pairs,
        evidence=dict(llm.evidence),
        ruler="combined",
    )


# ---- ③ Oprava (zostři) -------------------------------------------------------

_SHARPEN_SYSTEM = """Jsi expert na návrh NER taxonomií. DVA labely se v praxi zaměňují —
model u některých entit neumí rozhodnout, který z nich použít. Dostaneš oba labely
a REÁLNÉ výrazy, které je zaměnily.

Zostři jejich HRANICI. Pro OBA labely vrať:
  - "code"             přesně stejný kód jako na vstupu (NEMĚŇ ho),
  - "description"      ostřejší popis (>= 10 znaků), aby šel od druhého labelu odlišit,
  - "decision_rule"    1 věta: jak rozhodnout, když entita padá na hranici s tím druhým,
  - "counter_examples" 2-6 výrazů, které vypadají jako TENHLE label, ale patří tomu
                       druhému (čerpej z dodaných zaměněných výrazů).

Pravidla: NEVYMÝŠLEJ nové labely, NEMĚŇ kódy, drž se jazyka vstupu. Vrať POUZE
validní JSON {"labels": [ {...}, {...} ]}, žádný text okolo."""


def _build_sharpen_messages(
    label_a: TaxonomyLabel, label_b: TaxonomyLabel, evidence: list[str]
) -> list[dict]:
    def fmt(lb: TaxonomyLabel) -> str:
        s = f"  {lb.code}: {lb.description}"
        if lb.examples:
            s += f"\n     příklady: {', '.join(lb.examples[:6])}"
        return s

    ev = ", ".join(evidence[:12]) if evidence else "(žádné konkrétní)"
    user = (
        f"ZAMĚŇOVANÉ LABELY:\n{fmt(label_a)}\n{fmt(label_b)}\n\n"
        f"REÁLNĚ ZAMĚNĚNÉ VÝRAZY (model u nich váhal mezi {label_a.code} "
        f"a {label_b.code}):\n  {ev}\n\n"
        f"Zostři hranici obou labelů. Vrať JSON "
        '{"labels": [{"code","description","decision_rule","counter_examples"}, ...]}.'
    )
    return [
        {"role": "system", "content": _SHARPEN_SYSTEM},
        {"role": "user", "content": user},
    ]


def _apply_sharpen_result(tax: Taxonomy, refined: list) -> int:
    """
    Zapracuje výstup zostření zpět do taxonomie. Aktualizuje POUZE description,
    decision_rule a counter_examples u labelů, jejichž kód sedí. NEMĚNÍ kódy,
    examples, domains ani aliases. Pure/deterministické (testovatelné bez LLM).
    Vrací počet reálně upravených labelů.
    """
    by_code = {lb.code.upper(): lb for lb in tax.labels}
    changed = 0
    for r in refined:
        if not isinstance(r, dict):
            continue
        lb = by_code.get(str(r.get("code") or "").strip().upper())
        if lb is None:
            continue
        touched = False
        desc = str(r.get("description") or "").strip()
        if len(desc) >= 10 and desc != lb.description:
            lb.description = desc
            touched = True
        rule = str(r.get("decision_rule") or "").strip()
        if rule and rule != lb.decision_rule:
            lb.decision_rule = rule
            touched = True
        for ce in (r.get("counter_examples") or []):
            ce = str(ce).strip()
            if ce and ce not in lb.counter_examples:
                lb.counter_examples.append(ce)
                touched = True
        if touched:
            changed += 1
    return changed


async def _sharpen_pair(
    client, tax: Taxonomy, pair: ConfusablePair, evidence: list[str],
    *, max_tokens: int = 4096,
) -> int:
    """Zostří jeden zaměňovaný pár. Best-effort — chyba volání = žádná změna."""
    by_code = {lb.code.upper(): lb for lb in tax.labels}
    la, lb = by_code.get(pair.a.upper()), by_code.get(pair.b.upper())
    if la is None or lb is None:
        return 0
    messages = _build_sharpen_messages(la, lb, evidence)
    try:
        resp = await client.chat(
            messages, max_tokens=max_tokens,
            reasoning=CONFIG.taxonomy_reasoning,
            # Strop i pro thinking trace — jinak llm_client s reasoning=True
            # tlačí minimum 16384 a při těsném kontextu vrátí server HTTP 400.
            reasoning_max_tokens=max(1024, min(max_tokens, CONFIG.taxonomy_max_tokens)),
        )
    except (LLMFatalError, LLMTransientError) as e:
        logger.warning(f"audit: zostření páru {pair.a}/{pair.b} selhalo ({e})")
        return 0
    refined = _extract_json_list(resp.content, resp.parsed, "labels", "Labels")
    changed = _apply_sharpen_result(tax, refined)
    logger.info(f"audit: zostřen pár {pair.a}/{pair.b} → upraveno {changed} labelů")
    return changed


# ---- Strukturální úprava (za flagem) ----------------------------------------

def _apply_merge(tax: Taxonomy, a: str, b: str) -> bool:
    """
    Sloučí label b do a (a přežije, b zmizí; jeho examples/domains/aliases/
    counter_examples se přenesou, b.code se přidá do a.aliases). Pure.
    Vrací True když sloučeno. Chrání podlahu 2 labelů (taxonomy musí mít >= 2).
    """
    if len(tax.labels) <= 2:
        return False
    by_code = {lb.code.upper(): lb for lb in tax.labels}
    la, lbm = by_code.get(a.upper()), by_code.get(b.upper())
    if la is None or lbm is None or la is lbm:
        return False
    for ex in lbm.examples:
        if ex not in la.examples:
            la.examples.append(ex)
    for d in lbm.domains:
        if d not in la.domains:
            la.domains.append(d)
    for al in lbm.aliases:
        if al not in la.aliases:
            la.aliases.append(al)
    for ce in lbm.counter_examples:
        if ce not in la.counter_examples:
            la.counter_examples.append(ce)
    if lbm.code not in la.aliases:
        la.aliases.append(lbm.code)
    tax.labels = [lb for lb in tax.labels if lb.code.upper() != b.upper()]
    return True


def _maybe_structural_edit(
    tax: Taxonomy, pair: ConfusablePair, report: ConfusionReport
) -> Optional[str]:
    """
    Strukturální úprava zaměňovaného páru (jen když structural_edits=True).
    Konzervativní: když pár zaměňuje TAK VELKOU část spanů, že ho v praxi nelze
    rozlišit, radši ho SLUČ — dva nerozlišitelné labely škodí víc než jeden
    poctivý. Práh je poměrový (podíl na změřených spanech), ne hard-coded slovo.

    Split se automaticky NEDĚLÁ — potřebuje re-derivaci příkladů; v této verzi
    místo něj zůstává zostření hranice. Vrací "merged"/None, mutuje tax in-place.
    """
    if report.n_spans <= 0:
        return None
    share = pair.support / report.n_spans
    if share >= 0.30 and pair.support >= 3:
        if _apply_merge(tax, pair.a, pair.b):
            logger.info(
                f"audit: pár {pair.a}/{pair.b} zaměňován v {share:.0%} spanů → "
                f"SLUČUJI (structural edit)"
            )
            return "merged"
    return None


# ---- Smyčka ------------------------------------------------------------------

async def audit_and_refine_taxonomy(
    *,
    client,
    tax: Taxonomy,
    samples_per_dataset: int = 10,
    seed: int = 42,
    max_rounds: int = 3,
    target_score: float = 0.15,
    max_pairs: int = 2,
    min_support: int = 2,
    structural_edits: bool = False,
    ruler: Optional[str] = None,
) -> Taxonomy:
    """
    Pilíř A / krok 2 — iterativně měří konfuzi taxonomie a zostřuje zaměňované
    páry. Vrací NEJLEPŠÍ (nejméně zaměňovanou) viděnou variantu; nikdy nevrátí
    horší taxonomii než na vstupu.

    `ruler` volí METRIKU konfuze (default z TAXONOMY_AUDIT_RULER):
      combined  — DEFAULT, OBOJÍ zároveň: embedding podobnost profilů tříd
                  + LLM top-2 probe nad held-out spany; skóre i páry se spojí
                  (TAXONOMY_AUDIT_EMBEDDING_WEIGHT). Když jedno není dostupné
                  (chybí embeddingy / prázdný korpus), použije se druhé.
      auto      — JEDNO z nich: embeddingy, když jsou dostupné, jinak LLM
                  (původní chování)
      embedding — jen embeddingy (bez nich fallback na LLM)
      llm       — jen LLM top-2 self-consistency probe nad held-out spany

    Best-effort: chyby měření/zostření se logují a přeskakují, běh se nezastaví.
    """
    ruler_name = (ruler or CONFIG.taxonomy_audit_ruler).strip().lower()
    if ruler_name not in ("auto", "combined", "llm", "embedding"):
        logger.warning(
            f"audit: neznámý ruler {ruler_name!r} — používám 'combined'"
        )
        ruler_name = "combined"
    combined = ruler_name == "combined"

    # Embedding ruler potřebuje volitelnou závislost → rozhodnout PŘED smyčkou
    # (a model načíst jen jednou).
    embedding_ruler = None
    if combined or ruler_name in ("auto", "embedding"):
        embedding_ruler = create_embedding_ruler()
        if embedding_ruler is None and ruler_name == "embedding":
            logger.warning(
                "audit: ruler=embedding vyžádán, ale embeddingy nejsou "
                "dostupné — spadám na LLM ruler."
            )
    use_embedding = embedding_ruler is not None
    # combined = obojí; auto/embedding = embeddingy, a LLM jen když nejsou.
    use_llm = combined or ruler_name == "llm" or not use_embedding

    # LLM probe sampluje held-out spany z korpusu; embedding pracuje jen
    # s profily tříd, takže data nepotřebuje.
    items: list = []
    if use_llm:
        items = list(scan_data_dir(CONFIG.data_dir))
        if not items:
            if not use_embedding:
                logger.info(
                    "audit: prázdný DATA_DIR — přeskakuji, vracím taxonomii beze změny"
                )
                return tax
            logger.warning(
                "audit: prázdný DATA_DIR — LLM probe se přeskočí, měřím jen embeddingy"
            )
            use_llm = False

    ruler_used = (
        "embedding+llm" if (use_embedding and use_llm)
        else ("embedding" if use_embedding else "llm")
    )
    logger.info(
        f"audit: ruler={ruler_used} (konfigurace: {ruler_name}"
        + (f", váha embeddingů {CONFIG.taxonomy_audit_embedding_weight}"
           if use_embedding and use_llm else "")
        + ")"
    )

    best = deepcopy(tax)
    best_score = float("inf")
    current = deepcopy(tax)

    for rnd in range(1, max_rounds + 1):
        emb_report = (
            measure_confusion_embedding(embedding_ruler, current)
            if use_embedding else None
        )
        llm_report = None
        if use_llm:
            snippets = _heldout_snippets(
                items, samples_per_dataset=samples_per_dataset,
                seed=seed + 101 * rnd,
            )
            llm_report = await _measure_confusion(client, current, snippets)
        report = _combine_confusion_reports(
            emb_report, llm_report, CONFIG.taxonomy_audit_embedding_weight,
        )
        if report.n_spans == 0:
            logger.info(f"audit: kolo {rnd} nic nezměřilo — končím")
            break

        improved = report.score < best_score
        if improved:
            best = deepcopy(current)
            best_score = report.score

        detail = ""
        if report.ruler == "combined":
            detail = (
                f" [embedding={emb_report.score:.3f}, llm={llm_report.score:.3f}, "
                f"páry both/emb/llm="
                f"{sum(1 for p in report.pairs if p.sources == 'both')}/"
                f"{sum(1 for p in report.pairs if p.sources == 'emb')}/"
                f"{sum(1 for p in report.pairs if p.sources == 'llm')}]"
            )
        logger.info(
            f"audit: kolo {rnd}/{max_rounds} konfuze={report.score:.3f}{detail} "
            f"(nejlepší={best_score:.3f}, {'zlepšení' if improved else 'bez zlepšení'})"
        )

        if report.score <= target_score:
            logger.info(f"audit: cíl {target_score} splněn — hotovo")
            break
        if not improved and rnd > 1:
            logger.info("audit: bez zlepšení oproti nejlepší — končím, vracím nejlepší")
            break

        pairs = _select_confusable_pairs(report, max_pairs, min_support)
        if not pairs:
            logger.info("audit: žádný pár nad prahem podpory — končím")
            break

        # ③ Oprava: zostři hranice zaměňovaných párů (+ volitelně strukturální edit).
        for p in pairs:
            ev = report.evidence.get((p.a, p.b), [])
            await _sharpen_pair(client, current, p, ev)
            if structural_edits:
                _maybe_structural_edit(current, p, report)

        # Pojistka: kdyby úprava rozbila validitu, revert na nejlepší.
        if current.validate():
            logger.warning("audit: úprava rozbila validitu taxonomie — revert na nejlepší")
            current = deepcopy(best)

    prefix = (best.notes + " | ") if best.notes else ""
    if best_score == float("inf"):
        best.notes = prefix + (
            f"disjointness audit ({ruler_used}): nezměřeno — beze změny."
        )
    else:
        best.notes = prefix + (
            f"disjointness audit ({ruler_used}): konfuze={best_score:.3f} "
            f"(cíl {target_score}, max {max_rounds} kol)."
        )
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_taxonomy_discovery(
    *,
    mode: TaxonomyMode,
    user_taxonomy_path: Optional[Path] = None,
    samples_per_dataset: int = 10,
    seed: int = 42,
) -> Optional[Path]:
    """
    Vyřeší fázi 0 podle režimu a uloží entity_taxonomy.json (kromě OFF).

    Returns: cesta k uloženému souboru, nebo None pro OFF.
    """
    if mode == TaxonomyMode.OFF:
        logger.info("Taxonomy mode = OFF — žádná discovery, použije se default set.")
        return None

    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if mode == TaxonomyMode.USER:
        if user_taxonomy_path is None:
            raise ValueError("USER mode vyžaduje --user-taxonomy <path>")
        tax = load_user_taxonomy(user_taxonomy_path)
        path = save_taxonomy(tax)
        _run_embedding_analysis_best_effort(path)
        return path

    # AUTO mode — potřebuje LLM.
    #
    # BEST-EFFORT: fáze 0 NIKDY nezhodí celý běh. Ať selže cokoliv —
    #   * extractor nedostupný / wait_ready timeout (20 min) → LLMFatalError,
    #   * prázdný DATA_DIR → FileNotFoundError,
    #   * přetečení kontextu (velký prompt = všechny domény × N vzorků)
    #     → HTTP 4xx → LLMFatalError,
    #   * nekvalitní/neparsovatelný výstup i po retry → TaxonomyQualityError,
    #   * cokoliv jiného neočekávaného,
    # spadneme na defaultních 16 kategorií (entity_taxonomy.json se NEuloží,
    # chová se jako mode=off) a caller (extract / pipeline / smoke) POKRAČUJE
    # dál na extract → critic → revise → finalize. CancelledError a
    # KeyboardInterrupt (BaseException) NEodchytáváme — Ctrl-C / task-cancel
    # musí projít.
    if mode == TaxonomyMode.AUTO:
        try:
            async with LLMClient(
                base_url=CONFIG.extractor_base_url,
                model=CONFIG.extractor_model,
                max_concurrency=2,  # taxonomy je 1 request, ne hromadný
                request_timeout=CONFIG.request_timeout_s,
                max_retries=CONFIG.max_retries,
                context_len=CONFIG.extractor_context_len,
            ) as client:
                await client.wait_ready(max_wait_s=1200)
                tax = await discover_taxonomy_auto(
                    extractor_client=client,
                    samples_per_dataset=samples_per_dataset,
                    seed=seed,
                )
                # Pilíř A / krok 2 — disjointness audit (best-effort nad validní
                # taxonomií). Selhání auditu NEshazuje běh ani nezahazuje návrh:
                # ponechá se taxonomie z discover_taxonomy_auto.
                if CONFIG.taxonomy_audit_enabled:
                    try:
                        tax = await audit_and_refine_taxonomy(
                            client=client,
                            tax=tax,
                            samples_per_dataset=samples_per_dataset,
                            seed=seed,
                            max_rounds=CONFIG.taxonomy_audit_max_rounds,
                            target_score=CONFIG.taxonomy_audit_target_score,
                            max_pairs=CONFIG.taxonomy_audit_max_pairs,
                            min_support=CONFIG.taxonomy_audit_min_support,
                            structural_edits=CONFIG.taxonomy_audit_structural_edits,
                            ruler=CONFIG.taxonomy_audit_ruler,
                        )
                    except Exception as e:  # noqa: BLE001 — audit je best-effort
                        logger.warning(
                            f"disjointness audit selhal ({type(e).__name__}: {e}); "
                            f"ponechávám taxonomii z návrhu (krok 1).",
                            exc_info=True,
                        )
                # Fáze 0b — embeddingy vyberou zaměnitelné páry, LLM jim
                # zostří hranice; výsledek JDE do entity_taxonomy.json.
                if CONFIG.taxonomy_embedding_enabled:
                    tax = await refine_taxonomy_with_embeddings(
                        client, tax,
                        max_pairs=CONFIG.taxonomy_embedding_refine_max_pairs,
                    )
        except TaxonomyQualityError as e:
            # LLM nedodal kvalitní taxonomii ani po reklamaci + retry.
            logger.error(
                f"Taxonomy auto-discovery selhala na kvalitě: {e}\n"
                f"FALLBACK: pipeline použije defaultní 16 kategorií "
                f"(entity_taxonomy.json se neukládá; chová se jako mode=off)."
            )
            return None
        except Exception as e:  # noqa: BLE001 — best-effort fáze 0, viz výše
            logger.error(
                f"Taxonomy auto-discovery TVRDĚ selhala "
                f"({type(e).__name__}: {e}).\n"
                f"FALLBACK: pipeline pokračuje s defaultními 16 kategoriemi "
                f"(entity_taxonomy.json se neukládá; chová se jako mode=off). "
                f"Fáze 0 je best-effort — běh se NEzastavuje.",
                exc_info=True,
            )
            return None
        path = save_taxonomy(tax)
        _run_embedding_analysis_best_effort(path)
        return path

    raise ValueError(f"Neznámý taxonomy mode: {mode!r}")


def _run_embedding_analysis_best_effort(tax_path: Path) -> None:
    """
    Fáze 0b — embedding analýza právě uložené taxonomie (TAXONOMY_EMBEDDING_
    ENABLED, default ON). Běží AUTOMATICKY všude, kde vzniká taxonomie
    (příkazy taxonomy / extract / pipeline i smoke-test --phase taxonomy),
    protože všechny procházejí run_taxonomy_discovery. Reporty se zapíšou
    vedle taxonomie a výsledky (zaměnitelné / disjunktní třídy, clustery) se
    ZAPÍŠOU DO entity_taxonomy.json. BEST-EFFORT: selhání zaloguje warning,
    taxonomie platí.
    """
    if not CONFIG.taxonomy_embedding_enabled:
        return
    try:
        from .taxonomy_embeddings import (
            TaxonomyEmbeddingConfig, run_taxonomy_embedding_analysis,
        )
        report = run_taxonomy_embedding_analysis(
            tax_path, tax_path.parent, TaxonomyEmbeddingConfig.from_env(),
        )
        tax = Taxonomy.from_dict(json.loads(tax_path.read_text(encoding="utf-8")))
        annotate_taxonomy_with_embeddings(tax, report)
        atomic_write_json(tax_path, tax.to_dict())
        logger.info(
            f"Fáze 0b: embedding analýza hotová, vztahy tříd zapsány do {tax_path}"
        )
    except Exception as e:  # noqa: BLE001 — best-effort, taxonomie už je uložená
        logger.warning(
            f"Fáze 0b: embedding analýza taxonomie selhala "
            f"({type(e).__name__}: {e}) — taxonomie zůstává platná."
        )


_EMB_CONFUSABLE = ("possible_duplicate", "possible_overlap")
_EMB_DISJOINT = "disjoint_candidate"


def annotate_taxonomy_with_embeddings(tax: Taxonomy, report) -> None:
    """
    Zapíše výsledky embedding analýzy do taxonomie (pure, in-place):
      label.confusable_with      partneři s possible_duplicate/possible_overlap,
                                 seřazení od nejpodobnějšího
      label.disjoint_candidates  partneři s disjoint_candidate (abecedně)
      label.embedding_cluster    id clusteru
      tax.embedding_analysis     souhrn (model, distribuce, počty párů)
    `report` je TaxonomyEmbeddingReport (nebo objekt se stejnými atributy).
    """
    by_code = {lb.code: lb for lb in tax.labels}
    confusable: dict[str, list[tuple[float, str]]] = {c: [] for c in by_code}
    disjoint: dict[str, list[str]] = {c: [] for c in by_code}
    for pr in report.pairs:
        a, b = pr.class_a, pr.class_b
        if a not in by_code or b not in by_code:
            continue
        if pr.classification in _EMB_CONFUSABLE:
            confusable[a].append((pr.cosine_similarity, b))
            confusable[b].append((pr.cosine_similarity, a))
        elif pr.classification == _EMB_DISJOINT:
            disjoint[a].append(b)
            disjoint[b].append(a)
    cluster_of = {
        code: c.cluster_id for c in report.clusters for code in c.labels
    }
    for code, lb in by_code.items():
        ranked = sorted(confusable[code], key=lambda t: (-t[0], t[1]))
        lb.confusable_with = [other for _sim, other in ranked]
        lb.disjoint_candidates = sorted(disjoint[code])
        lb.embedding_cluster = int(cluster_of.get(code, -1))
    dist = getattr(report, "similarity_distribution", None)
    summary = getattr(report, "summary", None)
    tax.embedding_analysis = {
        "model": report.model,
        "method": getattr(report, "method", ""),
        "centered": getattr(report, "centered", None),
        "threshold_mode": getattr(report, "threshold_mode", ""),
        "similarity_distribution": dist.model_dump() if dist is not None else None,
        "summary": summary.model_dump() if summary is not None else None,
        "formal_disjointness_proof": False,
    }


def _merge_refinement(
    tax: Taxonomy, refined: list, pair: tuple[str, str],
    rules: dict[str, list[str]], described: set[str],
) -> int:
    """
    Zapracuje zostření jednoho páru tak, aby se výsledky VÍCE párů u téže třídy
    NEPŘEPISOVALY: popis bere první (nejpodobnější) pár, hraniční pravidla se
    sčítají, counter_examples se sjednocují. Kódy mimo `pair` ignoruje.
    Pure — vrací počet upravených labelů.
    """
    by_code = {lb.code.upper(): lb for lb in tax.labels}
    allowed = {pair[0].upper(), pair[1].upper()}
    changed = 0
    for r in refined:
        if not isinstance(r, dict):
            continue
        code = str(r.get("code") or "").strip().upper()
        lb = by_code.get(code)
        if lb is None or code not in allowed:
            continue
        touched = False
        desc = str(r.get("description") or "").strip()
        if code not in described and len(desc) >= 10 and desc != lb.description:
            lb.description = desc
            described.add(code)
            touched = True
        rule = str(r.get("decision_rule") or "").strip()
        if rule and rule not in rules.setdefault(code, []):
            rules[code].append(rule)
            touched = True
        for ce in (r.get("counter_examples") or []):
            ce = str(ce).strip()
            if ce and ce not in lb.counter_examples:
                lb.counter_examples.append(ce)
                touched = True
        changed += int(touched)
    return changed


async def refine_taxonomy_with_embeddings(
    client, tax: Taxonomy, *, max_pairs: int = 10, max_tokens: int = 4096,
) -> Taxonomy:
    """
    Fáze 0b / zostření řízené embeddingy — embeddingy ovlivní VÝSLEDNOU
    taxonomii:

      1. embedding analýza označí zaměnitelné páry tříd
         (possible_duplicate / possible_overlap, od nejpodobnějšího),
      2. LLM každému páru (max `max_pairs`) zostří description,
         decision_rule a counter_examples — pravidla třídy z více párů se
         SČÍTAJÍ, nepřepisují,
      3. výsledek se přijme BEZ embeddingové brány (profil pro embedding
         pravidla ani counter_examples neobsahuje, brána by je nespravedlivě
         zahodila); pojistkou je validita taxonomie.

    Třídy se NESLUČUJÍ — embeddingy občas označí jako "duplicitu" i zjevně
    různé třídy (např. MISTO × OSOBA), sloučení by bylo destruktivní.
    BEST-EFFORT: bez embeddingů nebo při chybě vrátí vstup beze změny.
    """
    if max_pairs <= 0:
        return tax
    try:
        import tempfile
        from .taxonomy_embeddings import (
            TaxonomyEmbeddingConfig, run_taxonomy_embedding_analysis,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "entity_taxonomy.json"
            atomic_write_json(tmp_path, tax.to_dict())
            report = run_taxonomy_embedding_analysis(
                tmp_path, Path(tmp), TaxonomyEmbeddingConfig.from_env(),
            )
    except Exception as e:  # noqa: BLE001 — embeddingy jsou best-effort
        logger.warning(
            f"Fáze 0b: embedding zostření přeskočeno — analýza selhala "
            f"({type(e).__name__}: {e})"
        )
        return tax

    flagged = sorted(
        (p for p in report.pairs if p.classification in _EMB_CONFUSABLE),
        key=lambda p: (-p.cosine_similarity, p.class_a, p.class_b),
    )[:max_pairs]
    if not flagged:
        logger.info("Fáze 0b: embeddingy nenašly zaměnitelné páry — beze změny")
        return tax

    refined = deepcopy(tax)
    by_code = {lb.code.upper(): lb for lb in refined.labels}
    rules: dict[str, list[str]] = {}
    described: set[str] = set()
    sharpened = 0
    for p in flagged:
        la, lb = by_code.get(p.class_a.upper()), by_code.get(p.class_b.upper())
        if la is None or lb is None:
            continue
        messages = _build_sharpen_messages(la, lb, [])
        try:
            resp = await client.chat(
                messages, max_tokens=max_tokens,
                reasoning=CONFIG.taxonomy_reasoning,
                reasoning_max_tokens=max(
                    1024, min(max_tokens, CONFIG.taxonomy_max_tokens)
                ),
            )
        except (LLMFatalError, LLMTransientError) as e:
            logger.warning(f"Fáze 0b: zostření {p.class_a}/{p.class_b} selhalo ({e})")
            continue
        out = _extract_json_list(resp.content, resp.parsed, "labels", "Labels")
        if _merge_refinement(refined, out, (p.class_a, p.class_b), rules, described):
            sharpened += 1
            logger.info(
                f"Fáze 0b: zostřen pár {p.class_a}/{p.class_b} "
                f"({p.classification}, podobnost {p.cosine_similarity:.2f})"
            )
    for code, rs in rules.items():
        by_code[code].decision_rule = " ".join(rs)

    if refined.validate():
        logger.warning("Fáze 0b: zostření rozbilo validitu taxonomie — ponechávám vstup")
        return tax
    prefix = (refined.notes + " | ") if refined.notes else ""
    refined.notes = prefix + (
        f"embedding zostření: {sharpened}/{len(flagged)} zaměnitelných párů "
        f"(model {report.model})."
    )
    return refined
