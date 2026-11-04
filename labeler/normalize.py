"""
Normalizační a validační vrstva nad LLM výstupem (silver dataset).

Volá se ve finalize.py těsně před zápisem `final_labels.{json,jsonl}`, pokud
uživatel zapne `--normalize-entities`. Cílem je z LLM výstupu (často
nekonzistentního) udělat čistší dataset pro spaCy NER / RE trénink.

Co dělá (řízené options přes NormalizeOpts):

  A) Validace offsetů
     - každá entita musí splňovat text[start:end] == entity.text
     - když ne, pokusíme se ji v textu dohledat (case-sensitive → -insensitive);
       pokud existuje 1 výskyt, opravíme. Pokud žádný, dropneme (s --strict-offsets).

  B) Auto-expand mentions
     - LLM často označí jen 1 výskyt entity (typicky v titulku); doplníme ostatní.
     - Bezpečnost je DATA-DRIVEN (žádné hard-coded seznamy slov ani labelů):
        * korpusové statistiky (labeler.category_vote.CorpusCategoryStats):
          výraz s vysokou document frequency = funkční slovo / generikum →
          unsafe; výraz s konzistentní kategorií napříč korpusem → safe;
        * taxonomy unsafe_texts (data z fáze 0 / uživatele);
        * strukturální fallback bez korpusu: min. délka + jednoslovný
          lowercase výraz je unsafe (pravidlo o tvaru, ne seznam slov).

  C) Label remapping
     - Mapping je ODVOZENÝ Z DAT (Taxonomy.label_mapping()): s taxonomií
       fáze 0 z jejích kind/remap_to polí; bez ní z packaged
       labeler/default_taxonomy.json. Žádná hard-coded tabulka v kódu.
     - Přebít lze constructor parametrem `label_mapping`.

  D) Drop strukturálních identifikátorů (URL / EMAIL / TELEFON)
     - URL_RE, EMAIL_RE, PHONE_RE — pokud entity.text matchne plně tyto
       regexy, DROPne se. URL/email/telefon jsou regex-detekovatelné
       struktury, ne pojmenované entity. NER se na ně netrénuje; pokud
       downstream kód takový identifikátor potřebuje, pustí regex přímo
       na input_text — deterministické, levné, žádný model nepotřebuje.
       Counter: `structural_identifiers_dropped`.

  E) Konflikt label v rámci dokumentu
     - Pokud se stejný surface form objeví s 2+ různými labely, rozhodne se
       DATA-DRIVEN řetězem (žádná hard-coded priorita labelů):
        1. korpusový hlas (nejčastější kategorie tohoto lemmatu v korpusu),
        2. dokumentová většina (nejčastější label v tomto dokumentu),
        3. první výskyt v dokumentu (deterministický, bez sémantiky).
       Konflikt se vždy zaznamená do needs_review.

  F) Drop překryvů
     - longest-span wins, ties → earlier start

  G) Relace
     - zachovat jen pokud subject/object stále existují po normalizaci
     - žádný auto-expand relací (riziko false positives)

  H) Report
     - NormalizationResult obsahuje counters pro `--write-report` →
       `normalization_report.json` ve finalize.

Deterministické: stejný input → stejný output. Loguje rozhodnutí na INFO/DEBUG.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Entity, Relation, SkoGraph
from .taxonomy import default_taxonomy as _default_taxonomy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanty — label set a mapování (ODVOZENÉ Z DAT, žádná hard-coded tabulka)
# ---------------------------------------------------------------------------

# Default mapping i default NER set se odvozují z labeler/default_taxonomy.json
# (packaged DATA soubor — jediné místo, kde je default 16-kategorie set
# definovaný). Mapping je vlastnost taxonomie (Taxonomy.label_mapping():
# label s remap_to → remap_to; kind="keyphrase" → KEYPHRASE_<code>;
# kind="ner" → identita). Pokud existuje taxonomie z fáze 0, použije se
# JEJÍ mapping (viz NormalizeOpts.effective_mapping) — tenhle default platí
# jen pro fallback bez entity_taxonomy.json.
#
# Pozn.: URL / EMAIL / TELEFON v defaultu záměrně nejsou. Pokud je LLM přesto
# vrátí, _step_drop_structural_identifiers entitu zahodí (regex struktura).
DEFAULT_LABEL_MAPPING: dict[str, str] = _default_taxonomy().label_mapping()

# Default NER labely (po normalizaci) — FALLBACK bez entity_taxonomy.json.
# S taxonomy použij `effective_ner_labels(taxonomy)` (níže).
DEFAULT_NER_LABELS: frozenset[str] = _default_taxonomy().ner_label_targets()


def effective_ner_labels(taxonomy: object = None) -> frozenset[str]:
    """
    Vrátí finální množinu NER labelů, do které entity smí spadnout.

    Pokud `taxonomy` (labeler.taxonomy.Taxonomy) je dodáno, použijí se
    JEJÍ labely: kind="keyphrase" a KEYPHRASE_* prefix se vylučují (nepatří
    do hard NER); labely s remap_to se počítají pod svým CÍLOVÝM jménem
    (po normalizaci). Phase 0 → SOURCE OF TRUTH pro NER labely v doméně.

    Bez taxonomy → fallback na `DEFAULT_NER_LABELS` (odvozeno z
    labeler/default_taxonomy.json).
    """
    if taxonomy is None:
        return DEFAULT_NER_LABELS
    labels = getattr(taxonomy, "labels", None) or []
    codes = set()
    for lb in labels:
        code = getattr(lb, "code", "")
        if not code or code.startswith("KEYPHRASE_"):
            continue
        if getattr(lb, "kind", "ner") == "keyphrase":
            continue
        target = getattr(lb, "mapped_target", None)
        target_code = target() if callable(target) else code
        if not target_code.startswith("KEYPHRASE_"):
            codes.add(target_code)
    if not codes:
        # Taxonomy bez NER labelů (defensive) — fallback
        return DEFAULT_NER_LABELS
    return frozenset(codes)

# POZN. (Pilíř A/B principy): dřívější hard-coded seznamy _LABEL_PRIORITY,
# _UNSAFE_TEXT_FOR_AUTO_EXPAND a _SAFE_FOR_AUTO_EXPAND_LABELS byly SMAZÁNY.
# Nahrazuje je korpusová statistika (labeler.category_vote.CorpusCategoryStats):
#   - konflikt labelů    → korpusový hlas → dokumentová většina → první výskyt
#   - unsafe pro expand  → vysoká document frequency (funkční slovo/generikum)
#   - safe pro expand    → konzistentní kategorie napříč korpusem
# Jediné zbylé konstanty jsou ČÍSELNÉ prahy (délka, počet slov) — žádná slova.

# Minimální délka textu pro auto-expand bez korpusového potvrzení.
_MIN_AUTO_EXPAND_LEN: int = 5

# ---------------------------------------------------------------------------
# Sentence-like entity filter — univerzální anti-halucinace
# ---------------------------------------------------------------------------
# Slabší LLM (např. Qwen3-0.6B v testu) občas vyplivne celé věty nebo otázky
# jako entitu (např. "Kolik vám vyplatíme?" jako PARAMETR). To je čistě
# šum — entita nesmí být věta. Tahle heuristika je doménově NEUTRÁLNÍ a
# čistě jazyková:
#
#   1. Končí ?, ! → určitě věta/otázka
#   2. Obsahuje vnitřní hranici věty (tečka/!/? + mezera + velké písmeno)
#   3. Má víc než _MAX_WORDS_PER_ENTITY slov → skoro určitě věta
#
# Datum "1. 1. 2026", inicály "J. K. Rowling", "Apple Inc." apod. heuristikou
# bezpečně projdou (tečka tam není sentence boundary).
_MAX_WORDS_PER_ENTITY: int = 15
# KEYPHRASE_* entity (BENEFIT, POUZITI, ...) jsou přirozeně delší fráze, ne
# tvrdé NER. Pro ně je práh počtu slov benevolentnější, aby se neořezávaly
# legitimní víceslovné benefity / popisy použití. Pravidla pro ?/! a vnitřní
# hranici věty zůstávají STEJNÁ (ta indikují větu bez ohledu na kategorii).
_MAX_WORDS_KEYPHRASE: int = 25


# ---------------------------------------------------------------------------
# Regex pro identifier detekci
# ---------------------------------------------------------------------------

# URL: http(s)://... nebo www.xxx nebo domain s explicitní path/query.
# CO ZÁMĚRNĚ NEMATCHUJE: holé brand-style domény typu "AcmeShop.cz", "DemoZpravy.cz",
# "VzorObchod.cz" — to jsou platné ORGANIZACE/PRODUKT entity, ne URL. Detekce
# bare-domain by je zbytečně dropla z NER datasetu.
#
# Pravidla:
#   - http(s):// nebo ftp:// nebo www.X  → vždy URL
#   - bare domain S explicitní cestou (`/`, `?`, `#`) → URL (např. example.cz/blog)
#   - bare domain BEZ cesty (AcmeShop.cz) → NENÍ URL (necháme jako entitu)
_URL_RE = re.compile(
    r"(?:(?:https?|ftp)://|www\.)[^\s<>()]+|"
    r"\b[a-zA-Z0-9][-a-zA-Z0-9_.]*"
    r"\.(?:cz|sk|com|net|org|eu|io|info|de|at|pl|hu)"
    r"[/?#][^\s<>()]*",
    re.IGNORECASE,
)

# Email — RFC 5322 jednodušší varianta
_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# Telefonní formáty: vyžaduje explicitní telefonní markery — buď country
# prefix `+` (typicky +420 / +421), nebo závorky kolem area kódu, nebo
# explicit pomlčky/lomítka separátorů.
#
# CO ZÁMĚRNĚ NEMATCHUJE: holé číselné HODNOTA jako "100 000 000" Kč nebo
# "999 999 999" počet — bez phone markers by jen drop legitimních hodnot.
# (Pokud LLM dá entity label TELEFON explicit, je to pořád dropped via
# downstream filtering — ale URL/EMAIL/TELEFON už nejsou v EntityCategory).
_PHONE_RE = re.compile(
    # +CC ... (s explicit + prefixem) — celosvětový standard
    r"\+\d{1,3}[\s\-]?(?:\(?\d{2,4}\)?[\s\-]?){1,3}\d{2,4}|"
    # (PSČ) num num num (s explicit závorkami nebo pomlčkami)
    r"\(\d{2,4}\)[\s\-]?\d{3,4}[\s\-]?\d{3,4}|"
    # num-num-num-num (s explicit pomlčkami, ne mezerami)
    r"\d{2,4}-\d{2,4}-\d{2,4}(?:-\d{2,4})?"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizeOpts:
    """
    Konfigurace normalizační vrstvy.

    Atributy:
      enabled: master flag. False = no-op (graf projde beze změny).
      strict_offsets: dropnout entity, jejichž offset nelze opravit.
      auto_expand_mentions: doplnit chybějící výskyty stejné entity v textu.
      drop_uncertain_labels: zahodit entity s labelem mimo allowed_ner_labels.
      drop_structural_identifiers: pokud entity.text plně matchne URL /
                                    EMAIL / PHONE regex, entita se zahodí
                                    (regex-detekovatelná struktura, ne NER).
      label_mapping: starý label → nový label. Default DEFAULT_LABEL_MAPPING.
      allowed_ner_labels: pokud drop_uncertain_labels=True, jen tyto se nechají.
      taxonomy: volitelná labeler.taxonomy.Taxonomy. Pokud zadána, aplikují se
                její `force_text_match` a `unsafe_texts` per label (data-driven,
                žádné hard-coding v Python kódu). Pokud None, krok se přeskočí.
    """
    enabled: bool = False
    strict_offsets: bool = True
    auto_expand_mentions: bool = False
    drop_uncertain_labels: bool = False
    drop_structural_identifiers: bool = True
    label_mapping: Optional[dict[str, str]] = None
    allowed_ner_labels: Optional[frozenset[str]] = None
    taxonomy: object = None  # labeler.taxonomy.Taxonomy (lazy — žádný import zde)

    def effective_mapping(self) -> dict[str, str]:
        """
        Mapping labelů (starý → nový) v pořadí priority:
          1. explicitní label_mapping (uživatel/CLI),
          2. mapping ODVOZENÝ z taxonomie fáze 0 (Taxonomy.label_mapping() —
             data-driven; custom labely se bez remap_to/kind=keyphrase NEmění),
          3. default mapping z labeler/default_taxonomy.json (fallback).

        Dřív se DEFAULT_LABEL_MAPPING aplikoval i s custom taxonomií — custom
        label shodný se jménem default kategorie (např. PARAMETR) se potichu
        přejmenoval, i když to autor taxonomie nechtěl. Teď je zdrojem pravdy
        taxonomie sama.
        """
        if self.label_mapping is not None:
            return self.label_mapping
        if self.taxonomy is not None and getattr(self.taxonomy, "labels", None):
            derive = getattr(self.taxonomy, "label_mapping", None)
            if callable(derive):
                return derive()
        return DEFAULT_LABEL_MAPPING

    def effective_allowed(self) -> frozenset[str]:
        """
        Vrátí povolené NER labely. Pokud uživatel explicit nezadal
        `allowed_ner_labels`, derivujeme z taxonomy (Phase 0). Bez
        taxonomy: DEFAULT_NER_LABELS (universal + e-shop-like fallback).
        """
        if self.allowed_ner_labels is not None:
            return self.allowed_ner_labels
        return effective_ner_labels(self.taxonomy)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """
    Výsledek normalizace jednoho grafu (per-item).

    `graph` je upravený SkoGraph (in-place modifikace originálního by byla
    nebezpečná — vždy se vrátí nový).
    """
    graph: SkoGraph
    entities_before: int = 0
    entities_after: int = 0
    offsets_fixed: int = 0
    offsets_unrecoverable: int = 0
    mentions_expanded: int = 0
    labels_remapped: int = 0
    label_conflicts_resolved: int = 0
    overlaps_dropped: int = 0
    # Dropnuté regex-detekovatelné struktury (URL / EMAIL / TELEFON) —
    # to nejsou NER entity, zahodíme je. Pokud je downstream potřebuje,
    # pustí regex přímo na input_text.
    structural_identifiers_dropped: int = 0
    relations_before: int = 0
    relations_after: int = 0
    relations_dropped_missing_entity: int = 0
    # Nové countery z relation_normalize
    relations_reversed_fixed: int = 0
    relations_dropped_type_mismatch: int = 0
    relations_dropped_self_loop: int = 0
    relations_deduplicated: int = 0
    relations_dropped_weak: int = 0
    # Sentence-like entity filter (univerzální anti-halucinace)
    entities_dropped_sentence_like: int = 0
    # Insurance / domain heuristiky
    taxonomy_rules_applied: int = 0
    needs_review: list[str] = field(default_factory=list)
    label_distribution_before: dict[str, int] = field(default_factory=dict)
    label_distribution_after: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "NormalizationResult") -> None:
        """Akumuluje counters napříč více grafy (pro celkový report)."""
        for attr in (
            "entities_before", "entities_after",
            "offsets_fixed", "offsets_unrecoverable",
            "mentions_expanded", "labels_remapped",
            "label_conflicts_resolved", "overlaps_dropped",
            "structural_identifiers_dropped",
            "relations_before", "relations_after",
            "relations_dropped_missing_entity",
            "relations_reversed_fixed", "relations_dropped_type_mismatch",
            "relations_dropped_self_loop", "relations_deduplicated",
            "relations_dropped_weak", "taxonomy_rules_applied",
            "entities_dropped_sentence_like",
        ):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        self.needs_review.extend(other.needs_review)
        for label, n in other.label_distribution_before.items():
            self.label_distribution_before[label] = (
                self.label_distribution_before.get(label, 0) + n
            )
        for label, n in other.label_distribution_after.items():
            self.label_distribution_after[label] = (
                self.label_distribution_after.get(label, 0) + n
            )

    def to_dict(self) -> dict:
        return {
            "entities_before": self.entities_before,
            "entities_after": self.entities_after,
            "offsets_fixed": self.offsets_fixed,
            "offsets_unrecoverable": self.offsets_unrecoverable,
            "mentions_expanded": self.mentions_expanded,
            "labels_remapped": self.labels_remapped,
            "entities_dropped_sentence_like": self.entities_dropped_sentence_like,
            "taxonomy_rules_applied": self.taxonomy_rules_applied,
            "label_conflicts_resolved": self.label_conflicts_resolved,
            "overlaps_dropped": self.overlaps_dropped,
            "structural_identifiers_dropped": self.structural_identifiers_dropped,
            "relations_before": self.relations_before,
            "relations_after": self.relations_after,
            "relations_dropped_missing_entity":
                self.relations_dropped_missing_entity,
            "relations_dropped_self_loop": self.relations_dropped_self_loop,
            "relations_dropped_type_mismatch": self.relations_dropped_type_mismatch,
            "relations_dropped_weak": self.relations_dropped_weak,
            "relations_reversed_fixed": self.relations_reversed_fixed,
            "relations_deduplicated": self.relations_deduplicated,
            "needs_review_count": len(self.needs_review),
            "needs_review_sample": self.needs_review[:50],
            "label_distribution_before": dict(
                sorted(self.label_distribution_before.items(),
                       key=lambda x: -x[1])
            ),
            "label_distribution_after": dict(
                sorted(self.label_distribution_after.items(),
                       key=lambda x: -x[1])
            ),
        }


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------


def _is_word_boundary(text: str, idx: int) -> bool:
    """Stejný čistič jako v positions.py."""
    if idx <= 0 or idx >= len(text):
        return True
    return not (text[idx - 1].isalnum() and text[idx].isalnum())


def _find_all_occurrences(
    text: str, needle: str, *, case_sensitive: bool = True
) -> list[tuple[int, int]]:
    """Najde všechny non-overlapping výskyty needle v text. Vrací (start, end)."""
    if not needle:
        return []
    haystack = text if case_sensitive else text.lower()
    sub = needle if case_sensitive else needle.lower()
    out: list[tuple[int, int]] = []
    pos = 0
    while True:
        idx = haystack.find(sub, pos)
        if idx == -1:
            break
        end = idx + len(sub)
        # Word boundary check — vyhne se "auto" v "automobil"
        if _is_word_boundary(text, idx) and _is_word_boundary(text, end):
            out.append((idx, end))
        pos = idx + 1
    return out


def _is_unsafe_for_auto_expand(
    entity_text: str, label: str, taxonomy: object = None,
    corpus_stats: object = None,
) -> bool:
    """
    Vrátí True, pokud auto-expand této entity je riskantní (FP magnet).

    Plně DATA-DRIVEN — žádné hard-coded seznamy slov ani labelů. Pořadí:
      1. Taxonomy unsafe_texts — text je v unsafe_texts pro daný label
         (data z fáze 0 / uživatele) → unsafe.
      2. Korpus: vysoká document frequency = funkční slovo nebo doménové
         generikum ("se", "pojištění" v pojišťovnickém korpusu) → unsafe.
         Nahrazuje dřívější hard-coded stopword seznam.
      3. Korpus: výraz je napříč korpusem konzistentně TENTO label
         (support + dominance + není generikum) → safe (i krátký text).
         Nahrazuje dřívější hard-coded safe-label seznam.
      4. Taxonomy NER label (label je v effective_ner_labels(taxonomy)) a text
         dost dlouhý → safe. Doménová znalost (Phase 0) superseduje
         strukturální fallback. Bez taxonomy se přeskočí.
      5. Strukturální fallback (bez korpusu; pravidla o TVARU, ne o slovech):
         délka < MIN_AUTO_EXPAND_LEN → unsafe; jednoslovný výraz psaný celý
         lowercase (v původním tvaru) → unsafe (pravděpodobné obecné slovo).
    """
    txt = entity_text.strip().lower()
    if not txt:
        return True
    # 1) Taxonomy-driven unsafe per label (data-driven, žádné hard-coding)
    if taxonomy is not None:
        for lb in getattr(taxonomy, "labels", []) or []:
            if lb.code != label:
                continue
            for ut in lb.unsafe_texts:
                ut_norm = ut.strip().lower() if not lb.case_sensitive else ut.strip()
                cmp_txt = txt if not lb.case_sensitive else entity_text.strip()
                if ut_norm == cmp_txt:
                    return True
    # 2) + 3) Korpusové statistiky (pokud jsou k dispozici — finalize je předá)
    if corpus_stats is not None:
        if corpus_stats.is_generic_surface(entity_text):
            return True
        from .category_vote import norm_key
        if corpus_stats.is_safe_expand(norm_key(entity_text), label, entity_text):
            return False
    # 4) Taxonomy-defined NER label (data-driven): autor taxonomie label označil
    #    jako reálnou entitní třídu a text NENÍ v unsafe_texts (krok 1) → safe.
    #    KEYPHRASE_* sem nikdy nespadne (effective_ner_labels je vylučuje).
    if (
        taxonomy is not None
        and len(txt) >= _MIN_AUTO_EXPAND_LEN
        and label in effective_ner_labels(taxonomy)
    ):
        return False
    # 5) Strukturální fallback — pravidla o tvaru textu, žádný seznam slov
    if len(txt) < _MIN_AUTO_EXPAND_LEN:
        return True
    stripped = entity_text.strip()
    if " " not in stripped and stripped.isalpha() and stripped.islower():
        return True
    return False


def _is_sentence_like(text: str, max_words: int = _MAX_WORDS_PER_ENTITY) -> bool:
    """
    Univerzální (doménově neutrální) heuristika: vrátí True, pokud `text`
    vypadá jako věta / otázka, ne jako entita.

    `max_words`: práh počtu slov pro pravidlo 4 (default 15; pro KEYPHRASE_*
    se předává vyšší práh, viz `_MAX_WORDS_KEYPHRASE`).

    Pravidla:
      1. Končí silným sentence terminátorem (`?` nebo `!`) → True.
      2. Obsahuje vnitřní `?` nebo `!` (s následujícím obsahem) → True.
      3. Obsahuje "real" sentence boundary: `[.?!]` + mezera + Uxx (velké +
         lowercase pokračování), kde slovo končící těsně před `.` má >= 3
         znaky. Tím se vyhneme:
           - "J. K. Rowling" (slovo před `.` je 1 znak — inicála)
           - "Mr. Brown" / "Dr. House" (slovo před `.` je 2 znaky — zkratka)
           - "1. 1. 2026" (po tečce následuje číslice, ne velké písmeno)
      4. Počet slov > _MAX_WORDS_PER_ENTITY → True.

    Tečka úplně na konci sama o sobě nestačí ("Apple Inc.").
    """
    txt = (text or "").strip()
    if not txt:
        return False
    n = len(txt)

    # 1) Silný terminátor na konci
    if txt[-1] in ("?", "!"):
        return True

    # 2) Vnitřní `?` / `!`
    for ch in ("?", "!"):
        idx = txt.find(ch)
        if 0 < idx < n - 1:
            return True

    # 3) Real sentence boundary
    def _word_len_before(idx: int) -> int:
        """Délka contiguous slova končícího na idx-1 (alfanumerické znaky)."""
        j = idx - 1
        count = 0
        while j >= 0 and (txt[j].isalpha() or txt[j].isdigit()):
            count += 1
            j -= 1
        return count

    for i in range(n - 3):
        if txt[i] not in (".", "?", "!"):
            continue
        if txt[i + 1] != " ":
            continue
        if not txt[i + 2].isupper():
            continue
        # Pokračování slova musí být lowercase písmeno (vyřadí "M. K. R.")
        if not txt[i + 3].islower():
            continue
        # Slovo končící na tečce musí mít >= 3 znaky (vyřadí inicály a
        # 2-znakové zkratky typu "Mr."/"Dr.")
        if _word_len_before(i) < 3:
            continue
        return True

    # 4) Příliš mnoho slov
    if len(txt.split()) > max_words:
        return True

    return False


def _resolve_overlap_priority(
    a: Entity, b: Entity, corpus_stats: object = None,
) -> Entity:
    """
    Z překrývajících se entit vybere "vítěze".

    Pravidla v pořadí (žádná hard-coded priorita labelů):
      1. delší span vyhrává
      2. tie → vyšší korpusová podpora (klíč, kategorie) — jen se stats
      3. tie → dřívější start
    """
    a_len = (a.end_char or 0) - (a.start_char or 0)
    b_len = (b.end_char or 0) - (b.start_char or 0)
    if a_len != b_len:
        return a if a_len > b_len else b
    if corpus_stats is not None:
        from .category_vote import norm_key
        a_w = corpus_stats.mention_weight(norm_key(a.lemma or a.text), a.category)
        b_w = corpus_stats.mention_weight(norm_key(b.lemma or b.text), b.category)
        if a_w != b_w:
            return a if a_w > b_w else b
    return a if (a.start_char or 0) <= (b.start_char or 0) else b


# ---------------------------------------------------------------------------
# Jednotlivé kroky normalizace
# ---------------------------------------------------------------------------


def _step_apply_taxonomy_text_rules(
    graph: SkoGraph, opts: NormalizeOpts, result: NormalizationResult
) -> None:
    """
    Aplikuje data-driven heuristiky z taxonomy (žádné hard-coding v kódu).

    Pro každý TaxonomyLabel se vyhodnotí:

    1. `force_text_match`:
       Pokud entity.text matchne (case sensitivity dle label.case_sensitive),
       entity dostane tento label, i když LLM dal jiný. Příklad v taxonomy.json:
         { "code": "PRODUKT",
           "force_text_match": ["ALFA", "BETA", "GAMA"],
           "case_sensitive": true }

    2. `unsafe_texts`:
       Pokud entity.text matchne A entity AKTUÁLNĚ má tento label, downgrade
       na KEYPHRASE_SUBJECT (= generic fallback, který není v žádné relation
       target množině — chrání core NER i RE export). Příklad:
         { "code": "PRODUKT",
           "unsafe_texts": ["pojištění", "auto", "smlouva"],
           "case_sensitive": false }

    Pokud `opts.taxonomy` je None, krok se přeskočí — žádný hard-coding.
    """
    if not opts.enabled or opts.taxonomy is None:
        return

    # Build lookup tabulky: text → label_code (force), text → set[label_code] (unsafe)
    force_map: dict[str, str] = {}
    force_map_lower: dict[str, str] = {}
    unsafe_map: dict[str, set[str]] = {}
    unsafe_map_lower: dict[str, set[str]] = {}

    for lb in opts.taxonomy.labels:
        for t in lb.force_text_match:
            t_stripped = t.strip()
            if not t_stripped:
                continue
            if lb.case_sensitive:
                force_map[t_stripped] = lb.code
            else:
                force_map_lower[t_stripped.lower()] = lb.code
        for t in lb.unsafe_texts:
            t_stripped = t.strip()
            if not t_stripped:
                continue
            if lb.case_sensitive:
                unsafe_map.setdefault(t_stripped, set()).add(lb.code)
            else:
                unsafe_map_lower.setdefault(t_stripped.lower(), set()).add(lb.code)

    for e in graph.entities:
        text = e.text.strip()
        text_lower = text.lower()

        # 1) Force match — vyšší priorita než LLM label
        new_label: Optional[str] = None
        if text in force_map:
            new_label = force_map[text]
        elif text_lower in force_map_lower:
            new_label = force_map_lower[text_lower]
        if new_label is not None and e.category != new_label:
            e.category = new_label
            result.taxonomy_rules_applied += 1
            continue

        # 2) Unsafe match — downgrade na KEYPHRASE_SUBJECT
        unsafe_for_labels: set[str] = set()
        if text in unsafe_map:
            unsafe_for_labels |= unsafe_map[text]
        if text_lower in unsafe_map_lower:
            unsafe_for_labels |= unsafe_map_lower[text_lower]
        if e.category in unsafe_for_labels:
            e.category = "KEYPHRASE_SUBJECT"
            result.taxonomy_rules_applied += 1


def _step_remap_labels(
    graph: SkoGraph, opts: NormalizeOpts, result: NormalizationResult
) -> None:
    """A) přemapuj label podle opts.effective_mapping()."""
    mapping = opts.effective_mapping()
    for e in graph.entities:
        old = e.category
        new = mapping.get(old, old)
        if new != old:
            # Entity.category je str s pattern validátorem — přiřaď přímo
            e.category = new
            result.labels_remapped += 1


def _step_drop_structural_identifiers(
    graph: SkoGraph, text: str, opts: NormalizeOpts,
    result: NormalizationResult,
) -> None:
    """
    D) Drop entit, jejichž surface form je regex-detekovatelná struktura
    (URL / EMAIL / TELEFON). To nejsou pojmenované entity — netrénujeme
    na ně NER.

    Pokud LLM extrahovala `https://www.example.com` (často jako MISTO)
    nebo `info@example.com`, entitu zahodíme. Downstream kód, který tyto
    identifikátory potřebuje, pustí URL_RE / EMAIL_RE / PHONE_RE přímo
    na input_text — to je deterministické a model k tomu netřeba.

    Counter: `result.structural_identifiers_dropped`.

    Pokud opts.drop_structural_identifiers=False, krok se přeskočí (lze
    vypnout pro debugging; default je True).
    """
    if not opts.drop_structural_identifiers:
        return

    kept: list[Entity] = []
    for e in graph.entities:
        surface = (e.text or "").strip()
        if not surface:
            kept.append(e)
            continue
        is_structural = (
            bool(_URL_RE.fullmatch(surface))
            or bool(_EMAIL_RE.fullmatch(surface))
            or bool(_PHONE_RE.fullmatch(surface))
        )
        if is_structural:
            result.structural_identifiers_dropped += 1
            logger.debug(
                f"DROP structural identifier {e.id} ({surface!r}) "
                f"[orig label: {e.category}]"
            )
            continue
        kept.append(e)
    graph.entities = kept


def _step_fix_offsets(
    graph: SkoGraph, text: str, opts: NormalizeOpts,
    result: NormalizationResult,
) -> list[Entity]:
    """A) ověř a oprav offsety. Vrací filtered list bez nedohledatelných."""
    kept: list[Entity] = []
    for e in graph.entities:
        s, end = e.start_char, e.end_char
        # Validní pozice + sedí substring?
        if (
            s is not None and end is not None and 0 <= s < end <= len(text)
            and text[s:end] == e.text
        ):
            kept.append(e)
            continue

        # Pokus 1: case-sensitive search, ne overlapping
        occurrences = _find_all_occurrences(text, e.text, case_sensitive=True)
        if len(occurrences) == 1:
            ns, ne = occurrences[0]
            e.start_char, e.end_char = ns, ne
            result.offsets_fixed += 1
            kept.append(e)
            continue

        if not occurrences:
            # Pokus 2: case-insensitive
            ci = _find_all_occurrences(text, e.text, case_sensitive=False)
            if len(ci) == 1:
                ns, ne = ci[0]
                e.start_char, e.end_char = ns, ne
                # Přepíšeme entity.text na přesný substring
                e.text = text[ns:ne]
                result.offsets_fixed += 1
                kept.append(e)
                continue

        # Více výskytů nebo nic
        if not occurrences:
            result.offsets_unrecoverable += 1
            if opts.strict_offsets:
                logger.debug(
                    f"DROP entity {e.id} ({e.text!r}): nedohledatelná v textu"
                )
                continue
            # Bez strict: ponecháme, ale označíme needs_review
            result.needs_review.append(
                f"entity {e.id} ({e.text!r}): offset nedohledatelný"
            )
            kept.append(e)
        else:
            # Více výskytů — necháme ten s původním offsetem, pokud je rozumný,
            # jinak vezmeme první výskyt
            if s is not None and 0 <= s < len(text):
                # zkus najít nejbližší
                closest = min(occurrences, key=lambda x: abs(x[0] - s))
                e.start_char, e.end_char = closest
                result.offsets_fixed += 1
            else:
                ns, ne = occurrences[0]
                e.start_char, e.end_char = ns, ne
                result.offsets_fixed += 1
            kept.append(e)
    return kept


def _step_auto_expand(
    entities: list[Entity], text: str, opts: NormalizeOpts,
    result: NormalizationResult, corpus_stats: object = None,
) -> list[Entity]:
    """
    B) Pro entity, které jsou SAFE pro auto-expand, doplň ostatní výskyty
    stejného surface form jako nové entity se stejným labelem.
    """
    if not opts.auto_expand_mentions:
        return entities

    # Najdi nejvyšší existující ID pro nové entity
    max_id = 0
    for e in entities:
        try:
            max_id = max(max_id, int(e.id[1:]))
        except (ValueError, IndexError):
            pass

    existing_spans = {
        (e.start_char, e.end_char): e for e in entities
        if e.start_char is not None and e.end_char is not None
    }
    new_entities: list[Entity] = list(entities)

    # Pro každý unikátní (text, label) zkoušíme expand
    seen: set[tuple[str, str]] = set()
    for e in entities:
        key = (e.text, e.category)
        if key in seen:
            continue
        seen.add(key)
        if _is_unsafe_for_auto_expand(
            e.text, e.category, opts.taxonomy, corpus_stats,
        ):
            continue
        # Najdi všechny výskyty
        occurrences = _find_all_occurrences(text, e.text, case_sensitive=True)
        for s, end in occurrences:
            if (s, end) in existing_spans:
                continue
            max_id += 1
            new_e = Entity(
                id=f"e{max_id}",
                text=text[s:end],
                lemma=e.lemma,
                category=e.category,
                start_char=s,
                end_char=end,
            )
            new_entities.append(new_e)
            existing_spans[(s, end)] = new_e
            result.mentions_expanded += 1
    return new_entities


def _step_resolve_label_conflicts(
    entities: list[Entity], result: NormalizationResult,
    corpus_stats: object = None,
) -> list[Entity]:
    """
    E) Stejný surface form v jednom dokumentu má více labelů.

    Rozhodovací řetěz — plně data-driven, ŽÁDNÁ hard-coded priorita labelů:
      1. korpusový hlas: nejčastější kategorie tohoto lemmatu napříč korpusem
         (jen s corpus_stats; remíza/neznámý klíč = žádný signál),
      2. dokumentová většina: nejčastější label v tomto dokumentu
         (remíza = žádný signál),
      3. první výskyt: label entity s nejmenším start_char (deterministické
         pravidlo bez sémantického obsahu — poslední instance).
    """
    # Skupina podle case-insensitive textu — sjednoceno s deduplikačním klíčem
    # v finalize._norm_entity_text (ten taky lowercasuje), aby se variantám
    # lišícím se jen velikostí písmen ("Procesor" vs "procesor") přiřadil
    # konzistentní label. Případy lišící se jen interpunkcí jsou vzácné a
    # řeší je merge ve finalize; tady stačí casefold (hlavní zdroj neshody).
    by_text: dict[str, list[Entity]] = {}
    for e in entities:
        by_text.setdefault(e.text.casefold(), []).append(e)

    kept: list[Entity] = []
    for _key, ents in by_text.items():
        labels = {e.category for e in ents}
        if len(labels) <= 1:
            kept.extend(ents)
            continue

        best_label: Optional[str] = None
        # 1) Korpusový hlas (kandidáti = labely v konfliktu)
        if corpus_stats is not None:
            from .category_vote import norm_key
            key = norm_key(ents[0].lemma or ents[0].text)
            best_label = corpus_stats.preferred_category(key, sorted(labels))
        # 2) Dokumentová většina (remíza = žádný signál)
        if best_label is None:
            doc_counts: dict[str, int] = {}
            for e in ents:
                doc_counts[e.category] = doc_counts.get(e.category, 0) + 1
            ranked = sorted(doc_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                best_label = ranked[0][0]
        # 3) První výskyt v dokumentu (deterministická poslední instance)
        if best_label is None:
            first = min(ents, key=lambda e: (e.start_char or 0, e.id))
            best_label = first.category

        result.label_conflicts_resolved += 1
        result.needs_review.append(
            f"text {ents[0].text!r}: konflikt labelů {sorted(labels)} → {best_label}"
        )
        # Sjednoť: všechny entity s tímto textem dostanou best_label
        for e in ents:
            if e.category != best_label:
                e.category = best_label
        kept.extend(ents)
    # Zachovej původní pořadí podle start_char pro determinismus
    kept.sort(key=lambda x: (x.start_char or 0, x.id))
    return kept


def _step_drop_overlaps(
    entities: list[Entity], result: NormalizationResult,
    corpus_stats: object = None,
) -> list[Entity]:
    """F) překryvy: longest > korpusová podpora > earlier."""
    valid = [
        e for e in entities
        if e.start_char is not None and e.end_char is not None
    ]
    # Sort by start, then longer first
    valid.sort(key=lambda e: (e.start_char, -(e.end_char - e.start_char)))
    accepted: list[Entity] = []
    for e in valid:
        overlap = None
        for a in accepted:
            if not (e.end_char <= a.start_char or e.start_char >= a.end_char):
                overlap = a
                break
        if overlap is None:
            accepted.append(e)
        else:
            winner = _resolve_overlap_priority(overlap, e, corpus_stats)
            if winner is e:
                accepted.remove(overlap)
                accepted.append(e)
                result.overlaps_dropped += 1
                logger.debug(
                    f"OVERLAP DROP {overlap.id} ({overlap.text!r}) "
                    f"loser of {e.id} ({e.text!r})"
                )
            else:
                result.overlaps_dropped += 1
                logger.debug(
                    f"OVERLAP DROP {e.id} ({e.text!r}) "
                    f"loser of {overlap.id} ({overlap.text!r})"
                )
    accepted.sort(key=lambda x: x.start_char)
    return accepted


def _step_drop_sentence_like(
    entities: list[Entity], result: NormalizationResult,
) -> list[Entity]:
    """
    Drop entit, jejichž surface form vypadá jako celá věta nebo otázka
    (LLM halucinace běžná u slabších modelů).

    Univerzální, doménově neutrální heuristika — viz `_is_sentence_like`.
    """
    kept: list[Entity] = []
    for e in entities:
        # KEYPHRASE_* fráze smí mít víc slov; ?/! a vnitřní hranice věty
        # se ale posuzují stejně (ty indikují větu bez ohledu na kategorii).
        max_words = (
            _MAX_WORDS_KEYPHRASE
            if str(e.category).startswith("KEYPHRASE_")
            else _MAX_WORDS_PER_ENTITY
        )
        if _is_sentence_like(e.text, max_words=max_words):
            result.entities_dropped_sentence_like += 1
            logger.debug(
                f"DROP sentence-like entity {e.id} ({e.text!r}) [{e.category}]"
            )
            continue
        kept.append(e)
    return kept


def _step_drop_uncertain(
    entities: list[Entity], opts: NormalizeOpts,
    result: NormalizationResult,
) -> list[Entity]:
    """Pokud --drop-uncertain-labels, zahoď entity mimo allowed_ner_labels."""
    if not opts.drop_uncertain_labels:
        return entities
    allowed = opts.effective_allowed()
    kept: list[Entity] = []
    for e in entities:
        if e.category in allowed:
            kept.append(e)
        else:
            logger.debug(
                f"DROP uncertain label {e.id} ({e.text!r}) [{e.category}]"
            )
    return kept


def _step_rebuild_relations(
    relations: list[Relation], kept_entity_ids: set[str],
    result: NormalizationResult,
) -> list[Relation]:
    """G) zachovej jen relace, jejichž subject+object stále existují."""
    kept: list[Relation] = []
    for r in relations:
        if r.subjectId in kept_entity_ids and r.objectId in kept_entity_ids:
            kept.append(r)
        else:
            result.relations_dropped_missing_entity += 1
    return kept


def _step_reindex_entities(entities: list[Entity]) -> dict[str, str]:
    """
    Přečísluj entity na e1, e2, ... (po dropnutí mohou být díry).
    Vrátí mapu starý_id → nové_id pro update relations.
    """
    id_map: dict[str, str] = {}
    entities.sort(key=lambda x: (x.start_char or 0, x.id))
    for i, e in enumerate(entities, 1):
        new_id = f"e{i}"
        if e.id != new_id:
            id_map[e.id] = new_id
            e.id = new_id
        else:
            id_map[e.id] = e.id
    return id_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_graph(
    graph: SkoGraph, text: str, opts: NormalizeOpts = NormalizeOpts(),
    corpus_stats: object = None,
) -> NormalizationResult:
    """
    Aplikuj normalizační vrstvu na graf. Vrátí NormalizationResult s upraveným
    grafem a statistikami.

    Pokud `opts.enabled=False`, vrátí beze změny + prázdné statistiky.

    corpus_stats: volitelně labeler.category_vote.CorpusCategoryStats —
    korpusové statistiky z finalize. Řídí data-driven rozhodování v krocích
    auto-expand (5), label konflikty (6) a překryvy (7). Bez nich se použijí
    strukturální fallbacky (délka, tvar, dokumentová většina, první výskyt) —
    nikdy hard-coded seznamy.
    """
    # Deep-copy graph (přes Pydantic model_copy) — nemodifikujeme vstup
    new_graph = graph.model_copy(deep=True)
    result = NormalizationResult(graph=new_graph)
    result.entities_before = len(new_graph.entities)
    result.relations_before = len(new_graph.relations)
    for e in new_graph.entities:
        lb = e.category
        result.label_distribution_before[lb] = (
            result.label_distribution_before.get(lb, 0) + 1
        )

    if not opts.enabled:
        result.entities_after = result.entities_before
        result.relations_after = result.relations_before
        result.label_distribution_after = dict(result.label_distribution_before)
        return result

    # 1) Drop strukturálních identifikátorů (URL / EMAIL / TELEFON) —
    #    regex-detekovatelné struktury, ne NER entity. Pokud LLM takovou
    #    extrahovala (často jako MISTO), zahodíme.
    _step_drop_structural_identifiers(new_graph, text, opts, result)

    # 2) Taxonomy text rules (data-driven force_text_match / unsafe_texts
    #    per label). Pokud opts.taxonomy je None, krok se přeskočí — žádné
    #    hard-coding. Příklady použití viz examples/entity_taxonomy.example.json.
    _step_apply_taxonomy_text_rules(new_graph, opts, result)

    # 3) Remap labelů (16 → konsolidovaný set)
    _step_remap_labels(new_graph, opts, result)

    # 4) Fix offsets (case-sensitive → -insensitive → drop)
    new_graph.entities = _step_fix_offsets(new_graph, text, opts, result)

    # 4b) Drop sentence-like entities (univerzální anti-halucinace —
    #     entity končící ?, !, s vnitřní sentence boundary nebo > 15 slov).
    #     Doménově neutrální, nezávisí na taxonomy.
    new_graph.entities = _step_drop_sentence_like(new_graph.entities, result)

    # 5) Auto-expand mentions (jen safe entity — korpus/taxonomy/tvar)
    new_graph.entities = _step_auto_expand(
        new_graph.entities, text, opts, result, corpus_stats,
    )

    # 6) Resolve label conflicts (korpus → dokumentová většina → první výskyt)
    new_graph.entities = _step_resolve_label_conflicts(
        new_graph.entities, result, corpus_stats,
    )

    # 7) Drop overlaps (longest > korpusová podpora > earlier)
    new_graph.entities = _step_drop_overlaps(new_graph.entities, result, corpus_stats)

    # 8) Drop uncertain labels (pokud zapnuto)
    new_graph.entities = _step_drop_uncertain(new_graph.entities, opts, result)

    # 9) Reindex entity IDs + update relations
    id_map = _step_reindex_entities(new_graph.entities)
    for r in new_graph.relations:
        r.subjectId = id_map.get(r.subjectId, r.subjectId)
        r.objectId = id_map.get(r.objectId, r.objectId)

    # 10) Filtruj relace (subject/object musí existovat)
    kept_ids = {e.id for e in new_graph.entities}
    new_graph.relations = _step_rebuild_relations(
        new_graph.relations, kept_ids, result
    )

    # 11) Relation type normalization (reverse fix + drop type-mismatch +
    #     drop located_in s URL + drop covers "pojištění" + dedup)
    from .relation_normalize import normalize_relations
    normalize_relations(new_graph, result=result)

    # 12) Rebuild triples (deterministicky z entities + relations — invariant)
    new_graph.triples = []
    new_graph.build_triples()

    result.entities_after = len(new_graph.entities)
    result.relations_after = len(new_graph.relations)
    for e in new_graph.entities:
        lb = e.category
        result.label_distribution_after[lb] = (
            result.label_distribution_after.get(lb, 0) + 1
        )

    return result
