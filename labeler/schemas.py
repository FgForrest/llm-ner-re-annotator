"""
Pydantic schémata pro Mudrc SKG formát.

Výstupní struktura odpovídá formátu popsanému v Mudrc README:
{
  "entities":  [ { id, text, lemma, category } ],
  "relations": [ { subjectId, relation, relationCs, relationEn, objectId } ],
  "triples":   [ { subject, relation, object } ]
}
"""

from __future__ import annotations
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EntityCategory(str, Enum):
    """
    Default NER kategorie — **FALLBACK** sada pro případ, že uživatel
    nemá `entity_taxonomy.json` (Phase 0).

    Pokud taxonomie existuje (`python -m labeler.main taxonomy --mode auto|user`
    nebo manuálně vytvořený soubor v checkpoints/entity_taxonomy.json),
    LLM dostane v promptu tu — tahle enum třída se nepoužije.

    Bez taxonomie tato sada slouží jako rozumný default pro většinu textů
    (e-shop, pojišťovnictví, obecné dokumenty). Display strings
    (entity.text, entity.lemma) zůstávají v originálním jazyce.

    Pozn.: URL / EMAIL / TELEFON tu ZÁMĚRNĚ NEJSOU. Jsou to regex-
    detekovatelné struktury, ne pojmenované entity — netrénujeme na ně
    NER. Pokud LLM takovou strukturu extrahuje, normalize ji DROPne
    (viz `_step_drop_structural_identifiers` v normalize.py).

    Pro nové domény, kde tahle fallback sada nepasuje (např. právo,
    medicína), použij `--taxonomy-mode auto|user`.

    Entity.category je str s pattern validátorem (`^[A-Z][A-Z0-9_]{0,49}$`),
    takže libovolný UPPERCASE_SNAKE label projde — taxonomy můžeš přidávat
    domain-specifické labely (INSURANCE_RISK, MEDICAL_CONDITION, …) bez
    úprav této enum třídy.
    """

    # === LLM input set (default 16 kategorií pro fallback prompt) ===
    PRODUKT = "PRODUKT"
    KATEGORIE = "KATEGORIE"
    KOMPONENTA = "KOMPONENTA"
    PARAMETR = "PARAMETR"
    HODNOTA = "HODNOTA"
    MATERIAL = "MATERIAL"
    VLASTNOST = "VLASTNOST"
    BENEFIT = "BENEFIT"
    POUZITI = "POUZITI"
    TECHNOLOGIE = "TECHNOLOGIE"
    SPECIFIKACE = "SPECIFIKACE"

    ORGANIZACE = "ORGANIZACE"
    OSOBA = "OSOBA"
    MISTO = "MISTO"
    DATUM = "DATUM"
    SUBJEKT = "SUBJEKT"

    # === Normalized labels (po průchodu labeler.normalize) ===
    # Hard NER kandidáti
    PRODUKT_TYP = "PRODUKT_TYP"        # z KATEGORIE (zúžené)
    OSOBA_ROLE = "OSOBA_ROLE"          # rename z OSOBA (pojistník, makléř, lékárník...)
    # Keyphrase kandidáti (mimo "tvrdé" NER — extrahuje se ale neukládá do core NER datasetu)
    KEYPHRASE_BENEFIT = "KEYPHRASE_BENEFIT"
    KEYPHRASE_PARAMETR = "KEYPHRASE_PARAMETR"
    KEYPHRASE_KOMPONENTA = "KEYPHRASE_KOMPONENTA"
    KEYPHRASE_SPEC = "KEYPHRASE_SPEC"
    KEYPHRASE_USECASE = "KEYPHRASE_USECASE"
    KEYPHRASE_TECH = "KEYPHRASE_TECH"
    KEYPHRASE_PROPERTY = "KEYPHRASE_PROPERTY"
    KEYPHRASE_SUBJECT = "KEYPHRASE_SUBJECT"


import re

RELATION_TYPES: dict[str, str] = {
    "is_a": "je",
    "part_of": "je součástí",
    "belongs_to": "patří do",
    "related_to": "souvisí s",
    "has_component": "má komponentu",
    "has_parameter": "má parametr",
    "has_value": "má hodnotu",
    "has_property": "má vlastnost",
    "has_design": "má design",
    "has_feature": "má funkci",
    "has_style": "má styl",
    "has_dimension": "má rozměr",
    "has_color": "má barvu",
    "has_material": "má materiál",
    "has_technology": "má technologii",
    "has_connectivity": "má konektivitu",
    "has_battery_life": "má výdrž baterie",
    "has_abbreviation": "má zkratku",
    "made_of": "je vyroben z",
    "covered_with": "je potažen",
    "manufactured_by": "vyrábí",
    "is_brand": "je značka",
    "supports_charging": "podporuje nabíjení",
    "supports_function": "podporuje funkci",
    "provides_benefit": "poskytuje přínos",
    "provides_audio_quality": "poskytuje kvalitu zvuku",
    "allows": "umožňuje",
    "protects_against": "chrání před",
    "suitable_for": "vhodný pro",
    "covers": "kryje",
    "excludes": "vylučuje",
    "requires": "vyžaduje",
    "valid_for": "platí pro",
    "applies_to": "vztahuje se na",
    "issued_by": "vydává",
    "located_in": "nachází se v",
    "available_in": "dostupný v",
}

_RELATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,49}$")

# Pattern pro entity labely — UPPERCASE_SNAKE (1-50 znaků).
# Default set je `EntityCategory` výše; ale Entity.category přijme libovolný
# label který odpovídá tomuto patternu, aby fungovala vlastní taxonomie
# z labeler/taxonomy.py (auto/user mode).
_LABEL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,49}$")


class Entity(BaseModel):
    """Jedna pojmenovaná entita extrahovaná z textu.

    POZN.: start_char / end_char ANI lemma se NEpřebírají od LLM. Pozice
    počítá deterministicky labeler.positions.compute_entity_positions;
    lemma doplní validator (kopie textu) a přepíše deterministicky Stanza
    ve finalize (labeler/lemma.py). LLM vrací jen surface form ("text")
    a kategorii — zbytek dopočítá aplikace. Viz docstringy positions.py
    a lemma.py.
    """

    id: str = Field(..., description="Unikátní ID v rámci grafu, formát 'e1', 'e2', …")
    text: str = Field(
        ..., min_length=1, description="Původní text z dokumentu (surface form)"
    )
    lemma: str = Field(
        ...,
        min_length=1,
        description="Kanonická forma — NEPŘEBÍRÁ se od LLM: validator ji doplní "
        "z textu a finalize ji přepíše deterministickou Stanza lemmatizací "
        "(labeler/lemma.py)",
    )
    # `category` je str (ne strict Enum), aby fungovala vlastní taxonomie
    # z labeler/taxonomy.py. Default set (16 kategorií) je v EntityCategory výše;
    # custom labely musí splňovat UPPERCASE_SNAKE pattern.
    category: str = Field(
        ...,
        description="Label kategorie entity (UPPERCASE_SNAKE). Default set viz "
        "EntityCategory; custom labely povolené z labeler/taxonomy.py.",
    )
    start_char: Optional[int] = Field(
        None,
        description="Pozice začátku entity v původním textu (počítá aplikace, "
        "ne LLM — pro spaCy/Stanza)",
    )
    end_char: Optional[int] = Field(
        None,
        description="Pozice konce entity v původním textu (počítá aplikace, "
        "ne LLM — pro spaCy/Stanza)",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_entity_defaults(cls, data):
        """Tolerantní doplnění: když LLM (zejm. critic) zapomene `lemma`,
        odvodíme ho z `text` (lemma je povinné, ale rozumně se odvodí). `id`
        ZÁMĚRNĚ negenerujeme — to by rozbilo reference subjectId/objectId
        z relations (relace by ukázala na neexistující entitu)."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not d.get("lemma") and d.get("text"):
            d["lemma"] = d["text"]
        return d

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not v or not v[0] == "e" or not v[1:].isdigit():
            raise ValueError(f"ID musí být ve formátu 'e<N>', dostal jsem: {v!r}")
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v) -> str:
        """Akceptuje EntityCategory enum nebo libovolný UPPERCASE_SNAKE string."""
        if isinstance(v, EntityCategory):
            return v.value
        if isinstance(v, str):
            if not _LABEL_NAME_RE.match(v):
                raise ValueError(
                    f"Label '{v}' musí být UPPERCASE_SNAKE pattern "
                    f"(^[A-Z][A-Z0-9_]{{0,49}}$)"
                )
            return v
        raise ValueError(f"Label musí být str nebo EntityCategory, dostal jsem: {type(v).__name__}")


class Relation(BaseModel):
    """Relace mezi dvěma entitami."""

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data):
        """Tolerantní příjem alternativních názvů polí, které LLM (zejména
        critic) občas použije místo kanonických: source/target → subjectId/
        objectId, RDF styl subject/predicate/object → subjectId/relation/objectId.
        Mapujeme PŘED validací, aby se corrected graf nezahodil jen kvůli jinému
        pojmenování (jinak 'Field required' → degrading na ok)."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "subjectId" not in d:
            if d.get("source") is not None:
                d["subjectId"] = d["source"]
            elif d.get("subject") is not None:
                d["subjectId"] = d["subject"]
        if "objectId" not in d:
            if d.get("target") is not None:
                d["objectId"] = d["target"]
            elif d.get("object") is not None:
                d["objectId"] = d["object"]
        if not d.get("relation") and d.get("predicate"):
            d["relation"] = d["predicate"]
        return d

    subjectId: str = Field(..., description="ID subjektu (musí existovat v entities)")
    objectId: str = Field(..., description="ID objektu (musí existovat v entities)")
    relation: str = Field(
        ..., description="Kanonický typ relace (anglicky, snake_case)"
    )
    # relationCs/relationEn jsou ODVOZENÉ z `relation` (viz _normalize_relation
    # níže), proto NEpovinné s defaultem "". Critic (Gemma) běžně vrací jen
    # `relation`; kdyby byly povinné, pydantic by jeho corrected graf odmítl
    # ("Field required") DŘÍV, než je _normalize_relation stihne dopočítat →
    # oprava by se zahodila (to byly ty "corrected graph nevaliadní → degrading
    # na ok" warningy v critic fázi).
    relationCs: str = Field("", description="Česká verze relace (dopočítá se z relation)")
    relationEn: str = Field(
        "", description="Anglická verze relace (dopočítá se z relation)"
    )

    @field_validator("relation")
    @classmethod
    def _validate_relation_format(cls, v: str) -> str:
        """Relation name musí být snake_case anglicky — povoluje neznámé relace,
        ale formálně omezuje (žádné mezery, žádná velká písmena, žádné speciální znaky).
        """
        if not _RELATION_NAME_RE.match(v):
            raise ValueError(
                f"Relation '{v}' není validní snake_case formát "
                f"(očekáván [a-z][a-z0-9_]{{1,49}}, např. 'has_component')"
            )
        return v

    @model_validator(mode="after")
    def _normalize_relation(self) -> "Relation":
        """Doplní/normalizuje relationCs a relationEn na základě relation."""
        if self.relation in RELATION_TYPES:
            canonical_cs = RELATION_TYPES[self.relation]
            if self.relationCs != canonical_cs:
                object.__setattr__(self, "relationCs", canonical_cs)
        elif not self.relationCs.strip():
            object.__setattr__(self, "relationCs", self.relation.replace("_", " "))

        if self.relationEn != self.relation:
            object.__setattr__(self, "relationEn", self.relation)

        if self.subjectId == self.objectId:
            raise ValueError(
                f"subjectId a objectId nemohou být stejné: {self.subjectId}"
            )
        return self


class Triple(BaseModel):
    """Lidsky čitelná trojice (subject text, relation, object text) — generuje se z entities+relations."""

    subject: str
    relation: str
    object: str


class SkoGraph(BaseModel):
    """Kompletní SKG graf pro jeden item."""

    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_refs(self) -> "SkoGraph":
        """Všechny subjectId/objectId v relations musí existovat v entities."""
        ids = {e.id for e in self.entities}
        dup = [e.id for e in self.entities]
        if len(dup) != len(ids):
            raise ValueError(
                f"Duplicitní ID entit: {[x for x in dup if dup.count(x) > 1]}"
            )
        for r in self.relations:
            if r.subjectId not in ids:
                raise ValueError(
                    f"Relation odkazuje na neexistující subjectId: {r.subjectId}"
                )
            if r.objectId not in ids:
                raise ValueError(
                    f"Relation odkazuje na neexistující objectId: {r.objectId}"
                )
        return self

    @model_validator(mode="after")
    def _regenerate_triples_invariant(self) -> "SkoGraph":
        """
        BEZPEČNOSTNÍ INVARIANT (hardening): triples NIKDY nepocházejí přímo
        z LLM výstupu, vždy se regenerují deterministicky z entities + relations
        po validaci. I když LLM nebo nepřátelský JSON soubor pošle v poli
        `triples` halucinaci se subject/object MIMO entities, tento validator
        ji okamžitě přepíše.

        Důsledky:
          - LLM-provided triples se zahodí.
          - Po jakékoli modifikaci entities/relations se triples NEAKTUALIZUJÍ
            automaticky (Pydantic v2 model_validator běží jen při instantiaci).
            Pro re-build použij explicitně `graph.build_triples()`.
        """
        emap = {e.id: e for e in self.entities}
        regenerated = [
            Triple(
                subject=emap[r.subjectId].lemma,
                relation=r.relation,
                object=emap[r.objectId].lemma,
            )
            for r in self.relations
            if r.subjectId in emap and r.objectId in emap
        ]
        object.__setattr__(self, "triples", regenerated)
        return self

    def build_triples(self) -> None:
        """
        Re-sestaví triples z aktuálních entities + relations.
        Volat po každé modifikaci (auto-expand, drop, normalize). Garantuje,
        že triples reflektují aktuální stav grafu.
        """
        emap = {e.id: e for e in self.entities}
        self.triples = [
            Triple(
                subject=emap[r.subjectId].lemma,
                relation=r.relation,
                object=emap[r.objectId].lemma,
            )
            for r in self.relations
            if r.subjectId in emap and r.objectId in emap
        ]


class ExtractorResult(BaseModel):
    """Výsledek extraktoru pro jeden item."""

    item_id: str
    source_dataset: str
    input_hash: str
    input_text: str
    is_synthetic: bool = False
    graph: SkoGraph
    model: str
    latency_ms: int
    retries: int = 0
    # Strukturovaná metadata, která měl extractor k dispozici. Persistujeme je,
    # aby je fáze 2 (critic) mohla dostat taky — jinak critic validuje "naslepo"
    # a může zbytečně revidovat entity odvozené z kontextu. Optional kvůli
    # zpětné kompatibilitě se staršími extractions.jsonl (default None).
    context: Optional[str] = None


CriticStatus = Literal["ok", "revised", "reject"]


class CriticResult(BaseModel):
    """Výsledek critica — buď schválí, opraví, nebo zamítne."""

    item_id: str
    status: CriticStatus
    reasoning: str = Field(..., description="Zdůvodnění rozhodnutí")
    corrected: Optional[SkoGraph] = Field(
        None, description="Opravený graf navrhnutý criticem (jen pokud status='revised')"
    )
    extractor_revised: Optional[SkoGraph] = Field(
        None,
        description="Graf po re-evaluaci extraktorem na základě zpětné vazby critica",
    )
    extractor_revised_reasoning: Optional[str] = Field(
        None, description="Zdůvodnění extraktoru při re-evaluaci"
    )
    model: str
    latency_ms: int

    @model_validator(mode="after")
    def _check_corrected(self) -> "CriticResult":
        if self.status == "revised" and self.corrected is None and self.extractor_revised is None:
            raise ValueError("status='revised' vyžaduje vyplněný 'corrected' nebo 'extractor_revised' graf")
        return self


class FinalLabel(BaseModel):
    """Finální olabelovaný záznam po merge extractor + critic."""

    item_id: str
    source_dataset: str
    input_text: str
    input_hash: str
    is_synthetic: bool = False
    graph: SkoGraph
    critic_status: CriticStatus
    critic_reasoning: str
    # Reasoning revisora (fáze 2b) ODDĚLENĚ od critic_reasoning — ať je jasné,
    # kdo co tvrdí (critic navrhuje opravu, revisor rozhoduje, zda ji vezme).
    revisor_reasoning: Optional[str] = None
    # Z jakého grafu finální záznam reálně pochází: "extractor_revised"
    # (revisor), "critic_corrected" (critic), "extractor_original" (beze
    # změny), "mixed" (po merge chunků).
    final_graph_source: str = "extractor_original"
    # DETERMINISTICKÝ diff extractor-original → finální graf. Ground truth —
    # NIKDY neodporuje grafu (na rozdíl od LLM reasoningu). Klíče:
    # entities_added / entities_removed / relations_added / relations_removed.
    changes_applied: dict = Field(default_factory=dict)
    extractor_model: str
    critic_model: str
    was_revised: bool
