"""
Prompty pro extractor (Qwen3.5-122B-A10B) a critic (Gemma 4 31B).

KLÍČOVÝ PRINCIP:
  Jeden univerzální systém kategorií pro jakoukoli doménu — zboží v e-shopu,
  pojistné produkty, nemovitosti, automobily, restaurace, software, průmyslové
  stroje, zdravotnické přístroje, finanční produkty, atd.
  LLM mapuje doménově specifické pojmy na obecné kategorie, nikdy nevytváří
  vlastní doménové kategorie.
"""

from __future__ import annotations
import json
from typing import Optional

from .schemas import EntityCategory, RELATION_TYPES

# Default fallback kategorie pro LLM prompt, když uživatel neposkytl
# vlastní taxonomii (Phase 0). Pokud `entity_taxonomy.json` existuje,
# `build_extractor_messages` použije `_taxonomy_block(taxonomy)` místo
# tohoto fallbacku.
ALL_CATEGORIES = [
    EntityCategory.PRODUKT,
    EntityCategory.KATEGORIE,
    EntityCategory.KOMPONENTA,
    EntityCategory.PARAMETR,
    EntityCategory.HODNOTA,
    EntityCategory.MATERIAL,
    EntityCategory.VLASTNOST,
    EntityCategory.BENEFIT,
    EntityCategory.POUZITI,
    EntityCategory.TECHNOLOGIE,
    EntityCategory.SPECIFIKACE,
    EntityCategory.ORGANIZACE,
    EntityCategory.OSOBA,
    EntityCategory.MISTO,
    EntityCategory.DATUM,
    EntityCategory.SUBJEKT,
]

# ---------------------------------------------------------------------------
# Ilustrační příklady výstupu (ukazují aplikaci kategorií na různé domény)
# ---------------------------------------------------------------------------

# Příklad A: technologický produkt (e-shop / elektronika)
EXAMPLE_OUTPUT_TECH = {
    "entities": [
        {"id": "e1", "text": "ProBook 450 G10", "category": "PRODUKT"},
        {"id": "e2", "text": "firemní notebook", "category": "KATEGORIE"},
        {"id": "e3", "text": "procesor", "category": "KOMPONENTA"},
        {"id": "e4", "text": "Intel Core i7-1355U", "category": "HODNOTA"},
        {"id": "e5", "text": "kapacita RAM", "category": "PARAMETR"},
        {"id": "e6", "text": "16 GB", "category": "HODNOTA"},
        {"id": "e7", "text": "výdrž baterie", "category": "PARAMETR"},
        {"id": "e8", "text": "až 10 hodin", "category": "HODNOTA"},
        {"id": "e9", "text": "celodenní práci", "category": "BENEFIT"},
        {"id": "e10", "text": "kanceláři", "category": "POUZITI"},
        {"id": "e11", "text": "Windows 11 Pro", "category": "SPECIFIKACE"},
    ],
    "relations": [
        {"subjectId": "e1", "objectId": "e2", "relation": "is_a", "relationCs": "je", "relationEn": "is_a"},
        {"subjectId": "e1", "objectId": "e3", "relation": "has_component", "relationCs": "má komponentu", "relationEn": "has_component"},
        {"subjectId": "e3", "objectId": "e4", "relation": "has_value", "relationCs": "má hodnotu", "relationEn": "has_value"},
        {"subjectId": "e1", "objectId": "e5", "relation": "has_parameter", "relationCs": "má parametr", "relationEn": "has_parameter"},
        {"subjectId": "e5", "objectId": "e6", "relation": "has_value", "relationCs": "má hodnotu", "relationEn": "has_value"},
        {"subjectId": "e1", "objectId": "e7", "relation": "has_parameter", "relationCs": "má parametr", "relationEn": "has_parameter"},
        {"subjectId": "e7", "objectId": "e8", "relation": "has_value", "relationCs": "má hodnotu", "relationEn": "has_value"},
        {"subjectId": "e7", "objectId": "e9", "relation": "provides_benefit", "relationCs": "poskytuje přínos", "relationEn": "provides_benefit"},
        {"subjectId": "e1", "objectId": "e10", "relation": "suitable_for", "relationCs": "vhodný pro", "relationEn": "suitable_for"},
        {"subjectId": "e1", "objectId": "e11", "relation": "has_feature", "relationCs": "má funkci", "relationEn": "has_feature"},
    ],
}

# Příklad B: realitní inzerát (zcela jiná doména — stejné kategorie)
EXAMPLE_OUTPUT_REALTY = {
    "entities": [
        {"id": "e1", "text": "Byt 3+kk Praha Žižkov", "category": "PRODUKT"},
        {"id": "e2", "text": "byt k pronájmu", "category": "KATEGORIE"},
        {"id": "e3", "text": "podlahová plocha", "category": "PARAMETR"},
        {"id": "e4", "text": "78 m²", "category": "HODNOTA"},
        {"id": "e5", "text": "nájemné", "category": "PARAMETR"},
        {"id": "e6", "text": "22 000 Kč/měsíc", "category": "HODNOTA"},
        {"id": "e7", "text": "balkón", "category": "KOMPONENTA"},
        {"id": "e8", "text": "cihlová zástavba", "category": "MATERIAL"},
        {"id": "e9", "text": "klidné bydlení v centru", "category": "BENEFIT"},
        {"id": "e10", "text": "Žižkov", "category": "MISTO"},
        {"id": "e11", "text": "rodiny s dětmi", "category": "POUZITI"},
    ],
    "relations": [
        {"subjectId": "e1", "objectId": "e2", "relation": "is_a", "relationCs": "je", "relationEn": "is_a"},
        {"subjectId": "e1", "objectId": "e3", "relation": "has_parameter", "relationCs": "má parametr", "relationEn": "has_parameter"},
        {"subjectId": "e3", "objectId": "e4", "relation": "has_value", "relationCs": "má hodnotu", "relationEn": "has_value"},
        {"subjectId": "e1", "objectId": "e5", "relation": "has_parameter", "relationCs": "má parametr", "relationEn": "has_parameter"},
        {"subjectId": "e5", "objectId": "e6", "relation": "has_value", "relationCs": "má hodnotu", "relationEn": "has_value"},
        {"subjectId": "e1", "objectId": "e7", "relation": "has_component", "relationCs": "má komponentu", "relationEn": "has_component"},
        {"subjectId": "e1", "objectId": "e8", "relation": "made_of", "relationCs": "je vyroben z", "relationEn": "made_of"},
        {"subjectId": "e1", "objectId": "e9", "relation": "provides_benefit", "relationCs": "poskytuje přínos", "relationEn": "provides_benefit"},
        {"subjectId": "e1", "objectId": "e10", "relation": "located_in", "relationCs": "nachází se v", "relationEn": "located_in"},
        {"subjectId": "e1", "objectId": "e11", "relation": "suitable_for", "relationCs": "vhodný pro", "relationEn": "suitable_for"},
    ],
}

EXAMPLE_BAD_OUTPUT = {
    "entities": [
        {"id": "e1", "text": "12345678", "category": "PARAMETR"},
        {"id": "e2", "text": "12345678", "category": "HODNOTA"},
        {"id": "e3", "text": "12345678", "category": "ORGANIZACE"},
        {"id": "e4", "text": "Klientská zóna", "category": "OSOBA"},
        {"id": "e5", "text": "Více", "category": "OSOBA"},
    ],
    "relations": [
        {"subjectId": "e1", "objectId": "e2", "relation": "related_to",
         "relationCs": "souvisí s", "relationEn": "related_to"},
    ],
}

# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def _relations_table() -> str:
    lines = [f"  - {en:<20} ({cs})" for en, cs in RELATION_TYPES.items()]
    return "\n".join(lines)


def _strip_lemma_entities(entities: list) -> list:
    """
    Odstraní pole "lemma" z entity dictů PŘED vložením do LLM promptu.

    Lemma se od LLM nepřebírá (dopočítá ho deterministicky Stanza ve finalize
    — viz labeler/lemma.py); interně je to jen kopie textu z validatoru.
    Posílat ho modelům = zbytečné input tokeny + svádí model lemma vracet.
    """
    out = []
    for e in entities or []:
        if isinstance(e, dict):
            out.append({k: v for k, v in e.items() if k != "lemma"})
        else:
            out.append(e)
    return out


def _strip_lemma_graph(graph: dict) -> dict:
    """Totéž pro celý graf ({"entities": [...], "relations": [...]})."""
    if not isinstance(graph, dict):
        return graph
    g = dict(graph)
    if isinstance(g.get("entities"), list):
        g["entities"] = _strip_lemma_entities(g["entities"])
    return g


def _categories_block(cats: list[EntityCategory]) -> str:
    return ", ".join(c.value for c in cats)


def _taxonomy_block(taxonomy) -> str:
    """
    Naformátuje vlastní taxonomii (z labeler.taxonomy.Taxonomy) jako čitelný
    blok pro LLM prompt:

      Povolené labely:
        - PRODUKT: Konkrétní pojmenovaný produkt nebo služba.
            Příklady: ACME Basic, iPhone 15 Pro Max
            Domény: e-shop, pojišťovnictví
        ...
    """
    lines = ["Povolené labely (POUZE tyto, ŽÁDNÉ JINÉ nevracej):"]
    for lb in taxonomy.labels:
        lines.append(f"  - {lb.code}: {lb.description}")
        if lb.examples:
            ex = ", ".join(lb.examples[:6])
            lines.append(f"      Příklady: {ex}")
        if lb.domains:
            lines.append(f"      Domény: {', '.join(lb.domains)}")
        # Auto-generovaná hraniční vodítka z disjointness auditu (Pilíř A).
        # Renderujeme je jen když existují — u čerstvé taxonomie bez auditu
        # jsou prázdná a blok vypadá jako dřív.
        decision_rule = getattr(lb, "decision_rule", "")
        if decision_rule:
            lines.append(f"      Rozhodni na hranici: {decision_rule}")
        counter_examples = getattr(lb, "counter_examples", None)
        if counter_examples:
            ce = ", ".join(counter_examples[:6])
            lines.append(f"      NE sem (patří jinam): {ce}")
        # Embeddingy (fáze 0b): třídy, se kterými se tahle nejsnáz splete.
        confusable_with = getattr(lb, "confusable_with", None)
        if confusable_with:
            lines.append(f"      Pozor na záměnu s: {', '.join(confusable_with[:4])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extractor — system prompt
# ---------------------------------------------------------------------------

EXTRACTOR_SYSTEM = """Jsi expertní systém pro extrakci pojmenovaných entit (NER) a relací (RE) z textu.
Tvůj jediný úkol je vracet přesně strukturovaný JSON podle zadaného schématu.

═══════════════════════════════════════════════════════════════
KLÍČOVÝ PRINCIP: JEDEN SYSTÉM KATEGORIÍ PRO JAKOUKOLI DOMÉNU
═══════════════════════════════════════════════════════════════
Existuje JEDEN obecný systém kategorií — PRODUKT, KATEGORIE, KOMPONENTA, PARAMETR,
HODNOTA, MATERIAL, VLASTNOST, BENEFIT, POUZITI, TECHNOLOGIE, SPECIFIKACE, ORGANIZACE,
OSOBA, MISTO, DATUM, SUBJEKT.

Tyto kategorie platí universálně. Doménově specifické pojmy vždy mapuješ na tento
systém — NIKDY nevytváříš vlastní kategorie jako "INSURANCE_PRODUCT", "PERIL",
"DEDUCTIBLE", "LISTING", "SPEC", "MAKE" apod.

Ilustrace mapování (stejná kategorie, různé domény):

  PRODUKT:     "ProBook 450 G10" / "Pojištění auta OMEGA" / "Byt 3+kk Žižkov"
               "VW Passat Combi 2.0 TDI" / "Věrnostní karta Premium" / "ACME Basic"
               → vždy PRODUKT, nikdy "LAPTOP", "POLICY", "LISTING", "SUBSCRIPTION"

  KATEGORIE:   "firemní notebook" / "povinné ručení" / "byt k pronájmu"
               "rodinný automobil" / "prémiová kreditní karta" / "účetní software"
               → vždy KATEGORIE, nikdy "DEVICE_TYPE", "INSURANCE_TYPE", "PROPERTY_TYPE"

  PARAMETR:    "kapacita RAM" / "limit pojistného plnění" / "podlahová plocha"
               "výkon motoru" / "kreditní limit" / "počet uživatelů"
               → vždy PARAMETR, nikdy "SPEC", "LIMIT", "AREA", "POWER"

  HODNOTA:     "16 GB" / "100 000 Kč" / "78 m²" / "150 kW" / "do 50 uživatelů"
               → vždy HODNOTA, nikdy "AMOUNT", "SIZE", "CAPACITY"

  BENEFIT:     "celodenní práce bez nabíjení" / "ochrana před finanční ztrátou"
               "klidné bydlení v centru" / "nižší spotřeba" / "automatické zálohy"
               → vždy BENEFIT

  POUZITI:     "kancelář, cestování" / "havárie, živelná pohroma" / "rodiny s dětmi"
               "terén a silnice" / "malé firmy do 10 zaměstnanců"
               → vždy POUZITI; rizika, situace, cílové skupiny — vše sem

  KOMPONENTA:  "procesor, baterie, displej" / "hlavní pojistná smlouva, klauzule"
               "balkón, kuchyň" / "motor, převodovka" / "dashboard, API modul"
               → část nebo složka produktu

  MATERIAL:    "hliník, sklo, titan" / "cihla, beton" / "kůže, textil"
               → surový materiál nebo povrchová úprava

  VLASTNOST:   "voděodolný, kompaktní" / "celosvětová platnost" / "rekuperace energie"
               → kvalitativní popis bez přiřaditelné číselné hodnoty

  TECHNOLOGIE: "Intel Evo platforma" / "asistenční služba 24/7" / "tepelné čerpadlo"
               "GORE-TEX" / "Face ID" / "aktivní potlačení hluku"
               → technologie, metoda, inovace

  SPECIFIKACE: "Windows 11 Pro" / "USB-C Thunderbolt 4" / "IP68" / "Bluetooth 5.3"
               "ISO 9001" / "GDPR compliant" / "EN 13402"
               → norma, standard, certifikace, protokol

  ORGANIZACE:  výrobce, poskytovatel, instituce, vydavatel
  OSOBA:       jmenovaná fyzická osoba I obecná role/funkce osoby
               (např. Jan Novák, pojistník, klient, lékárník, řidič, výrobce-osoba)
  MISTO:       geografické místo, adresa, region, stát
  DATUM:       konkrétní datum nebo časový interval
  SUBJEKT:     generický fallback pro vše, co nesedí jinam

════════════════════════════
OBECNÁ PRAVIDLA (vždy platí)
════════════════════════════
1. Vracej POUZE validní JSON — žádný úvod, žádné markdown bloky, žádné komentáře.
2. Každá entita má ID ve formátu "e1", "e2", "e3", … (pořadí v seznamu).
3. Pole "text" = PŘESNÝ substring z původního textu (surface form, včetně pádu).
   ZAKÁZÁNO vymýšlet text, který v originálu není.
4. Pole "lemma" NEVRACEJ — kanonickou/slovníkovou formu dopočítá aplikace
   deterministicky (lemmatizace mimo LLM). Vracíš jen "id", "text", "category".
5. Entity přidávej pouze tehdy, jsou-li EXPLICITNĚ zmíněné v textu.
6. Relace vytvárej POUZE pro vztahy PŘÍMO podporované textem. Nevymýšlej si.
7. Každá relace musí mít subjectId a objectId existující v entities.
8. relationEn = shodné s relation (snake_case, anglicky).
9. Pokud text neobsahuje žádné věcné entity, vrať {"entities": [], "relations": []}.
10. NIKDY nevkládej reasoning, chain-of-thought ani jiný text mimo JSON.
11. NIKDY nepřidávej numerické pozice (start_char, end_char, offset, span ap.).
    Indexy v textu si dopočítá aplikace sama. Stačí ti vrátit "text" (přesný
    substring) — pozice i lemma doplní deterministický kód.
12. URL adresy, domény, e-maily a telefonní čísla NIKDY neextrahuj jako entity.
    Jsou to regex-detekovatelné struktury (https://..., www.example.cz,
    info@example.com, +420 123 456 789) — aplikace si je dotáhne sama
    regexem nad input textem, NER se na ně netrénuje.
    MISTO je VYHRAZENO pro skutečné geografické místo (Praha, ČR, Žižkov).
13. KONZISTENCE LABELU v rámci dokumentu: pokud se stejný surface form (např.
    "Životní pojištění" nebo "ALFA") vyskytuje v textu vícekrát, MUSÍ mít
    VŽDY stejný label. Nesměšuj KATEGORIE/PRODUKT pro stejný výraz.
14. KAŽDÝ DŮLEŽITÝ SURFACE FORM ALESPOŇ JEDNOU — ne nutně všechny výskyty.
    Stačí vrátit každou unikátní entitu jednou (typicky první výskyt v textu).
    Auto-expand aplikace deterministicky doplní ostatní výskyty stejného textu.
    POKUD ale výslovně vracíš více výskytů, MUSÍ mít stejný label.
15. NIKDY NEVRACEJ pole "triples". Aplikace si triples sestaví deterministicky
    z entities + relations. Tvůj JSON OBSAHUJE POUZE "entities" a "relations".

═══════════════════════════
SMĚROVÁ PRAVIDLA PRO RELACE
═══════════════════════════

issued_by:
  subject = PRODUKT / KATEGORIE (= typ produktu), object = ORGANIZACE
  SPRÁVNĚ:  "Životní pojištění" issued_by "ACME Pojišťovna"
  ŠPATNĚ:   "ACME Pojišťovna" issued_by "Životní pojištění"

part_of:
  subject = část, object = celek
  SPRÁVNĚ:  "připojištění odpovědnosti" part_of "pojištění profesní odpovědnosti"

has_component:
  subject = produkt/celek, object = komponenta
  SPRÁVNĚ:  "OMEGA" has_component "investiční složka"

has_parameter:
  subject = produkt, object = parametr
  SPRÁVNĚ:  "ALFA" has_parameter "limit plnění"

has_value:
  subject = parametr, object = hodnota
  SPRÁVNĚ:  "limit plnění" has_value "100 000 Kč"

covers:
  subject = pojištění/produkt, object = konkrétní riziko/událost/škoda
  SPRÁVNĚ:  "ALFA" covers "zkrat"
  ŠPATNĚ:   "pojištění" covers "auto"  (obě strany příliš obecné)

suitable_for:
  subject = produkt, object = cílová skupina / role / použití
  SPRÁVNĚ:  "flotilové pojištění" suitable_for "firma"
  ŠPATNĚ:   "osobní údaje" suitable_for "pojistník"  (nesmysl typů)

located_in:
  POUZE pro skutečné geografické místo, NIKDY pro URL.
  ŠPATNĚ:   "Ministerstvo vnitra" located_in "https://smlouvy.gov.cz/"
            (URL není geografické místo — žádná relace)

Pokud si NEJSI JIST směrem relace, RADŠI ji nevracej. Postprocessing umí pár
typů otočit, ale jen jednoznačné případy.

══════════════════════
ANTI-HALUCINACE PRAVIDLA
══════════════════════
11. NIKDY neopakuj stejný "text" napříč různými entitami s různou kategorií.
12. Číselné identifikátory (katalogová čísla, EAN, ID, čísla smluv) NEJSOU entity.
13. Materiál (kov, plast, sklo, cihla) = MATERIAL nebo KOMPONENTA, nikdy ORGANIZACE.
14. KONTEXT slouží pouze pro porozumění textu — není zdrojem entit.

═══════════════════════
ZÁKAZ NAVIGAČNÍHO ŠUMU
═══════════════════════
15. Navigační prvky webu nejsou entity: "Klientská zóna", "Více", "Zpět", "Domů",
    "Kontakt", "Přihlášení", breadcrumby, názvy sekcí webu, tlačítka, štítky polí.
16. Pokud je celý vstupní text navigace nebo UI prvky, vrať {"entities": [], "relations": []}.
17. Zaměř se VÝHRADNĚ na věcný obsah textu.

═══════════════════════
ZÁKAZ ZBYTEČNÝCH RELACÍ
═══════════════════════
18. "related_to" je poslední možnost. Použij konkrétní relaci (has_component,
    has_value, provides_benefit, covers, suitable_for, …). Pokud žádná nesedí,
    relaci RADŠI nevracej.
19. Nespojuj entity v lineární řetěz jen kvůli pořadí v textu.
"""


# ---------------------------------------------------------------------------
# Extractor — user prompt
# ---------------------------------------------------------------------------

EXTRACTOR_USER_TEMPLATE = """DOMÉNA TEXTU: {domain}

POVOLENÉ KATEGORIE ENTIT (identické pro všechny domény, UPPERCASE):
{categories}

POVOLENÉ TYPY RELACÍ (do "relation"/"relationEn" patří anglická část, do "relationCs" česká):
{relations}

POZNÁMKA: Kategorie jsou univerzální — platí pro e-shop, pojišťovnu, realitní
portál, automotive, restauraci, SaaS, průmyslový katalog i cokoliv jiného.
Nikdy nevytvářej nové kategorie specifické pro aktuální doménu.

──────────────────────────────
ILUSTRACE — technologický produkt:
{example_tech}

──────────────────────────────
ILUSTRACE — realitní inzerát (stejné kategorie, zcela jiná doména):
{example_realty}

──────────────────────────────
PŘÍKLAD ŠPATNÉHO VÝSTUPU (TAKHLE NIKDY!):
{bad_example}

Chyby v špatném výstupu:
- "12345678" je katalogové číslo, NIKDY nesmí být entita.
- Stejný text opakovaný ve více kategoriích = halucinace.
- "Klientská zóna", "Více" jsou navigace webu, nejsou entity.
- "related_to" bez sémantického důvodu = zbytečná relace.

──────────────────────────────
KONTEXT (strukturovaná metadata — pouze pro porozumění, NEPOUŽÍVEJ je jako zdroj entit):
{context}

TEXT K ANALÝZE:
\"\"\"
{text}
\"\"\"

Vrať JSON s klíči "entities" a "relations"."""


# Slim varianta pro custom taxonomy (z Phase 0). Default examples
# (EXAMPLE_OUTPUT_TECH s PRODUKT/KATEGORIE atd.) by konfliktovaly s
# uživatelovou taxonomií typu INSURANCE_RISK / MEDICAL_CONDITION —
# LLM by viděl "use only these labels" + "tady jsou příklady jiných
# labelů". Proto pro custom taxonomy zahodíme hardcoded examples a
# necháme LLM pracovat čistě podle taxonomie.
EXTRACTOR_USER_TEMPLATE_CUSTOM_TAXONOMY = """DOMÉNA TEXTU: {domain}

{categories}

POVOLENÉ TYPY RELACÍ (do "relation"/"relationEn" patří anglická část, do "relationCs" česká):
{relations}

KRITICKÉ PRAVIDLO: Používej VÝHRADNĚ labely z výše uvedené taxonomie. NIC
jiného. Pokud entitu nelze přiřadit žádnému uvedenému labelu, nezahrnuj ji.

──────────────────────────────
PŘÍKLAD ŠPATNÉHO VÝSTUPU (struktura — labely se budou lišit dle taxonomie):
{bad_example}

Chyby v špatném výstupu:
- Katalogová čísla a kódy NIKDY nesmí být entity.
- Stejný text opakovaný ve více kategoriích = halucinace.
- Navigační prvky webu ("Klientská zóna", "Více") nejsou entity.
- "related_to" bez sémantického důvodu = zbytečná relace.

──────────────────────────────
KONTEXT (strukturovaná metadata — pouze pro porozumění, NEPOUŽÍVEJ je jako zdroj entit):
{context}

TEXT K ANALÝZE:
\"\"\"
{text}
\"\"\"

Vrať JSON s klíči "entities" a "relations" — labely VÝHRADNĚ z poskytnuté taxonomie."""


def build_extractor_messages(
    text: str,
    domain: str,
    context: Optional[str] = None,
    taxonomy=None,
) -> list[dict]:
    """
    Sestaví messages pro OpenAI-compatible chat completion API pro extractor.

    Args:
        text:     Text k analýze (HTML již stripnutý, max ~3000 tokenů).
        domain:   Popis domény — libovolný volný text.
        context:  Volitelná strukturovaná metadata.
        taxonomy: Volitelně labeler.taxonomy.Taxonomy — pokud zadáno, nahradí
                  defaultní 16 kategorií vlastní taxonomií (auto/user mode
                  z fáze 0). Kategorie a popis se vloží do promptu místo
                  defaultního seznamu.
    """
    has_custom_taxonomy = (
        taxonomy is not None and getattr(taxonomy, "labels", None)
    )

    if has_custom_taxonomy:
        # CUSTOM TAXONOMY: vlastní labely (např. INSURANCE_RISK,
        # MEDICAL_CONDITION). Default examples (PRODUKT/KATEGORIE) by
        # konfliktovaly s instrukcí "use only these labels".
        # Použijeme slim template bez hardcoded example outputů.
        user = EXTRACTOR_USER_TEMPLATE_CUSTOM_TAXONOMY.format(
            domain=domain,
            categories=_taxonomy_block(taxonomy),
            relations=_relations_table(),
            bad_example=json.dumps(
                EXAMPLE_BAD_OUTPUT, ensure_ascii=False, indent=2
            ),
            context=context or "(žádná)",
            text=text,
        )
    else:
        # DEFAULT 16-category fallback (žádná taxonomy z Phase 0).
        user = EXTRACTOR_USER_TEMPLATE.format(
            domain=domain,
            categories=_categories_block(ALL_CATEGORIES),
            relations=_relations_table(),
            example_tech=json.dumps(
                EXAMPLE_OUTPUT_TECH, ensure_ascii=False, indent=2
            ),
            example_realty=json.dumps(
                EXAMPLE_OUTPUT_REALTY, ensure_ascii=False, indent=2
            ),
            bad_example=json.dumps(
                EXAMPLE_BAD_OUTPUT, ensure_ascii=False, indent=2
            ),
            context=context or "(žádná)",
            text=text,
        )
    return [
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Extractor revision — prompt pro feedback smyčku
# ---------------------------------------------------------------------------

EXTRACTOR_REVISION_SYSTEM = """Jsi expertní systém pro extrakci pojmenovaných entit (NER) a relací (RE).
Dostal jsi zpětnou vazbu od nezávislého validátora k tvému předchozímu výstupu.

Přezkoumej původní výstup ve světle zpětné vazby a rozhodni:
  a) Pokud kritik identifikoval reálné chyby (halucinace, špatné kategorie,
     vymyšlené entity, navigační šum) → vrať OPRAVENÝ výstup.
  b) Pokud s kritikem nesouhlasíš nebo je výhrada nepodložená → vrať původní
     výstup beze změn.

{categories_rule}

Vrať POUZE validní JSON s klíči "entities" a "relations". Žádný text navíc."""


EXTRACTOR_REVISION_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT:
{context}

TVŮJ PŮVODNÍ VÝSTUP:
{original_json}

ZPĚTNÁ VAZBA VALIDÁTORA (status: {critic_status}):
{critic_reasoning}
{corrected_section}
Vrať JSON s klíči "entities" a "relations" — opravený nebo původní:"""


def build_extractor_revision_messages(
    text: str,
    original_graph: dict,
    critic_status: str,
    critic_reasoning: str,
    corrected_graph: Optional[dict] = None,
    context: Optional[str] = None,
    taxonomy=None,
) -> list[dict]:
    """
    Sestaví messages pro re-evaluaci extraktoru po zpětné vazbě critica.
    """
    if corrected_graph:
        corrected_section = (
            "\nKRITIKEM NAVRHOVANÝ OPRAVENÝ VÝSTUP (pouze pro referenci):\n"
            + json.dumps(
                _strip_lemma_graph(corrected_graph), ensure_ascii=False, indent=2
            )
            + "\n"
        )
    else:
        corrected_section = ""

    user = EXTRACTOR_REVISION_USER_TEMPLATE.format(
        text=text,
        context=context or "(žádná)",
        original_json=json.dumps(
            _strip_lemma_graph(original_graph), ensure_ascii=False, indent=2
        ),
        critic_status=critic_status,
        critic_reasoning=critic_reasoning,
        corrected_section=corrected_section,
    )
    return [
        {"role": "system", "content": EXTRACTOR_REVISION_SYSTEM.replace(
            "{categories_rule}",
            _critic_categories_rule(taxonomy, _REVISION_DEFAULT_CATEGORIES_RULE),
        )},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Critic — system prompt a user template
# ---------------------------------------------------------------------------

def _indent_block(block: str, prefix: str = "   ") -> str:
    """Odsadí víceřádkový blok, ať sedí pod odrážkou pravidla v promptu."""
    return "\n".join(prefix + ln if ln else ln for ln in block.split("\n"))


# Default 16 kategorií — platí JEN když fáze 0 nedodala vlastní taxonomii.
_CRITIC_DEFAULT_CATEGORIES_RULE = (
    "B) KATEGORIE jsou univerzální a platí pro jakoukoli doménu:\n"
    "   PRODUKT, KATEGORIE, KOMPONENTA, PARAMETR, HODNOTA, MATERIAL, VLASTNOST,\n"
    "   BENEFIT, POUZITI, TECHNOLOGIE, SPECIFIKACE, ORGANIZACE, OSOBA, MISTO, DATUM, SUBJEKT.\n"
    "   Pokud model vrátil neexistující kategorii (INSURANCE_PRODUCT, DEDUCTIBLE, PERIL,\n"
    "   LISTING, MAKE, MODEL, SPEC, AMOUNT apod.) → oprav v 'revised'."
)

_CRITIC_NER_DEFAULT_CATEGORIES_RULE = (
    "B) Kategorie jsou univerzální: PRODUKT, KATEGORIE, KOMPONENTA, PARAMETR, HODNOTA,\n"
    "   MATERIAL, VLASTNOST, BENEFIT, POUZITI, TECHNOLOGIE, SPECIFIKACE, ORGANIZACE,\n"
    "   OSOBA, MISTO, DATUM, SUBJEKT. Neexistující kategorii (INSURANCE_PRODUCT,\n"
    "   DEDUCTIBLE, MAKE…) → oprav v 'revised'."
)

_REVISION_DEFAULT_CATEGORIES_RULE = (
    "POVOLENÉ KATEGORIE: PRODUKT, KATEGORIE, KOMPONENTA, PARAMETR, HODNOTA,\n"
    "MATERIAL, VLASTNOST, BENEFIT, POUZITI, TECHNOLOGIE, SPECIFIKACE, ORGANIZACE,\n"
    "OSOBA, MISTO, DATUM, SUBJEKT. Jinou kategorii nevracej."
)


def _critic_categories_rule(taxonomy, default_rule: str) -> str:
    """
    Pravidlo o povolených kategoriích pro critic / revisora.

    S vlastní taxonomií z fáze 0 (entity_taxonomy.json) je povolený slovník
    PŘESNĚ ta taxonomie — critic i revisor vidí tentýž seznam labelů (včetně
    decision_rule a counter_examples), jaký dostal extractor. Bez taxonomie
    zůstává default 16 kategorií.

    Tohle je podstatné: dřív měl critic v promptu natvrdo 16 defaultních
    kategorií, takže u auto-discovered taxonomie označil platné labely za
    neexistující a "opravil" je — a revisor, který povolený slovník neznal
    vůbec, tu opravu potvrdil.
    """
    if taxonomy is None or not getattr(taxonomy, "labels", None):
        return default_rule
    return (
        "B) KATEGORIE — povolený slovník je PŘESNĚ taxonomie tohoto projektu\n"
        "   (fáze 0, entity_taxonomy.json), nic jiného a nic méně:\n"
        + _indent_block(_taxonomy_block(taxonomy)) + "\n"
        "   Label z tohohle seznamu je VŽDY platný — NIKDY ho neměň jen proto,\n"
        "   že ti není známý nebo že bys čekal obecnější kategorii.\n"
        "   Teprve kategorii, která v seznamu NENÍ, oprav na nejbližší z něj."
    )


CRITIC_SYSTEM = """Jsi nezávislý validátor výstupů NER/RE systému. Dostáváš původní text
a JSON s entitami a relacemi, který vytvořil jiný model. Tvůj úkol je posoudit správnost.

TVÉ MOŽNÉ ODPOVĚDI (vrať POUZE JSON, nic jiného):

1. Pokud je výstup správný:
{"status": "ok", "reasoning": "<stručné zdůvodnění, max 2 věty>"}

2. Pokud obsahuje opravitelné chyby:
{"status": "revised",
 "reasoning": "<co jsi změnil a proč>",
 "corrected": {"entities": [...], "relations": [...]}}

3. Pokud je výstup zásadně špatný (vymyšlené entity, halucinace, text vůbec nepochopen):
{"status": "reject", "reasoning": "<proč>"}

VALIDAČNÍ PRAVIDLA:

A) ENTITA musí být explicitně zmíněna v textu — "text" musí odpovídat substringu.
   Pole "text" stačí mít přesné — pozice (start_char, end_char) NEKONTROLUJ a
   NEPOSÍLEJ ve výstupu, dopočítá je deterministický kód mimo LLM.

{categories_rule}

C) RELAČNÍ PRAVIDLA — PREFERUJ přesné, ale OBECNÉ jsou taky OK:
   Tahle PŘESNÁ mapování jsou PREFEROVANÁ (použij je, když sedí):
   - "has_value" spojuje PARAMETR → HODNOTA (ne dvě HODNOTY)
   - "has_component" spojuje PRODUKT/KOMPONENTA → KOMPONENTA
   - "has_material" / "made_of" spojuje PRODUKT/KOMPONENTA → MATERIAL
   - "provides_benefit" spojuje PARAMETR/VLASTNOST/TECHNOLOGIE → BENEFIT
   - "covers" spojuje PRODUKT → POUZITI (pojišťovací nebo jiný kontext)
   DŮLEŽITÉ: tohle je jen výběr nejčastějších — povolený slovník relací je
   ŠIRŠÍ. Když ŽÁDNÁ přesná relace nesedí, ale vztah mezi entitami EXISTUJE,
   použij OBECNĚJŠÍ strukturální relaci (is_a, part_of, belongs_to) — radši
   obecná relace než žádná. NIKDY nemaž is_a/part_of/belongs_to jen proto, že
   nejsou v tomhle krátkém výčtu; jsou platné.

D) HALUCINACE IDENTIFIKÁTORŮ:
   Stejné číslo/text ve více entitách s různou kategorií = halucinace. Smaž.
   Katalogová čísla, EAN, ID záznamu, čísla smluv NEJSOU entity.

E) NAVIGAČNÍ ŠUM:
   "Klientská zóna", "Více", "Domů", "Kontakt", "Přihlášení", breadcrumby,
   názvy menu položek, tlačítka — NEJSOU entity. Odstraň v 'revised'.
   Pokud takto označena většina entit → 'reject'.

F) MATERIÁL JAKO ORGANIZACE:
   Kov, plast, sklo, cihla, dřevo = MATERIAL nebo KOMPONENTA, nikdy ORGANIZACE.

G) "related_to" — jen jako POSLEDNÍ možnost:
   Preferuj konkrétnější relaci (i obecnou is_a/part_of) před "related_to".
   "related_to" smaž JEN když je to lineární řetěz BEZ jakékoli sémantiky
   (čistý šum). Když vztah reálně existuje a nic přesnějšího nesedí, NECH ho —
   radši slabá relace než žádná (lepší pro RE dataset).

H) KONZISTENCE LABELU:
   Pokud se stejný surface form objevuje v dokumentu vícekrát, MUSÍ mít stejný
   label. "Životní pojištění" jednou jako KATEGORIE a podruhé jako PRODUKT =
   nekonzistence → oprav v 'revised'.

I) URL / EMAIL / TELEFON NIKDY JAKO ENTITY:
   https://..., www.example.cz, info@example.com, +420 123 456 789 NEJSOU
   pojmenované entity — jsou to regex-detekovatelné struktury a NER se
   na ně netrénuje. Pokud extractor takovou entitu vrátil, je to chyba
   → flag jako 'revised' a smaž entitu. Postprocessing tyto struktury
   stejně automaticky dropne, ale lepší je je vůbec nevracet.

J) CHYBĚJÍCÍ VÝSKYTY:
   Pokud LLM označil entitu jen v titulku, ale stejný surface form se vyskytuje
   ještě 3× v dalším textu a chybí v entities, je to nedostatek (postprocessing
   to ale dokáže doplnit auto-expandem, takže pro 'revised' nestačí — flag jen
   pokud LLM označil 1 výskyt jako PRODUKT a jiný jako KATEGORIE).

ROZHODOVACÍ PRAVIDLO:
- 1–2 menší chyby → 'revised'
- 50 %+ entit jsou nesmysly → 'reject'
- Výstup věcně správný, jen drobné mezery → 'ok'

NIKDY nevracej žádný text mimo JSON."""


CRITIC_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT (strukturovaná metadata, pokud k dispozici):
{context}

NAVRŽENÁ EXTRAKCE:
{proposed_json}

Posuď správnost extrakce. Vrať JSON s klíči "status", "reasoning" a volitelně "corrected":"""


def build_critic_messages(
    text: str,
    proposed: dict,
    context: Optional[str] = None,
    taxonomy=None,
) -> list[dict]:
    """Sestaví messages pro critic.

    `taxonomy` (fáze 0) určuje povolený slovník kategorií — MUSÍ to být ta
    samá taxonomie, podle které extrahoval extractor.
    """
    # .replace, ne .format — prompt obsahuje doslovné JSON závorky
    # ({"status": "ok", ...}), které by .format() interpretoval jako pole.
    system = CRITIC_SYSTEM.replace(
        "{categories_rule}",
        _critic_categories_rule(taxonomy, _CRITIC_DEFAULT_CATEGORIES_RULE),
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": CRITIC_USER_TEMPLATE.format(
                text=text,
                context=context or "(žádná)",
                proposed_json=json.dumps(
                    _strip_lemma_graph(proposed), ensure_ascii=False, indent=2
                ),
            ),
        },
    ]


# ===========================================================================
# SPLIT NER/RE — fokusované varianty (CONFIG.split_ner_re)
# ---------------------------------------------------------------------------
# Každá fáze se rozdělí na dvě LLM volání: jedno jen na entity (NER), druhé
# jen na relace (RE). RE volání vždy dostane už hotové entity jako kontext —
# relace na ně odkazují přes subjectId/objectId.
#
# VŠECHNY fáze (extract / critic / revise) mají pro NER i RE DEDIKOVANÉ čisté
# prompty (viz níže) — každý jen s pravidly své úlohy. ŽÁDNÁ recyklace plného
# combined promptu + přílepku "dělej jen NER/RE": ten dřív vláčel v NER fázi
# i všechna relační pravidla/příklady, což model mátlo a vedlo k runaway.
# Combined buildery (build_extractor_messages / build_critic_messages /
# build_extractor_revision_messages) zůstávají pro ne-split cestu (SPLIT_NER_RE=false).
# ===========================================================================


def _entities_block(entities: list) -> str:
    return json.dumps(
        {"entities": _strip_lemma_entities(entities)},
        ensure_ascii=False, indent=2,
    )


# ---------------------------------------------------------------------------
# NER-only — DEDIKOVANÝ čistý prompt (NE recyklace combined promptu).
# ---------------------------------------------------------------------------
# Combined prompt je nabitý relačními pravidly a příklady S relacemi — v "jen
# entity" úloze to model mate a vede k přeextrahování / runaway. Tady je čistý
# entitní prompt: kategorie + kvalita + DŮRAZ NA KONEČNOST. Combined cesta
# (build_extractor_messages) zůstává beze změny.

# Entity-only varianty příkladů (bez relací — ať neprosakují do NER fáze).
EXAMPLE_NER_OUTPUT = {"entities": EXAMPLE_OUTPUT_TECH["entities"]}
EXAMPLE_NER_BAD = {"entities": EXAMPLE_BAD_OUTPUT["entities"]}


EXTRACTOR_NER_SYSTEM = """Jsi expertní systém pro extrakci pojmenovaných entit (NER).
V TÉTO fázi řešíš VÝHRADNĚ entity — relace neřešíš. Vracíš jen strukturovaný JSON.

═══════════════════════════════════════════════════════════════
JEDEN SYSTÉM KATEGORIÍ PRO JAKOUKOLI DOMÉNU
═══════════════════════════════════════════════════════════════
Doménově specifické pojmy VŽDY mapuješ na obecné kategorie — NIKDY nevytváříš
vlastní (žádné "INSURANCE_PRODUCT", "LISTING", "SPEC", "MAKE"…).

Ilustrace mapování (stejná kategorie, různé domény):
  PRODUKT:     "ProBook 450 G10" / "Pojištění auta OMEGA" / "Byt 3+kk Žižkov"
  KATEGORIE:   "firemní notebook" / "povinné ručení" / "byt k pronájmu"
  KOMPONENTA:  "procesor" / "balkón" / "investiční složka" (část / složka celku)
  PARAMETR:    "kapacita RAM" / "limit pojistného plnění" / "podlahová plocha"
  HODNOTA:     "16 GB" / "100 000 Kč" / "78 m²" / "až 10 hodin"
  MATERIAL:    "hliník, sklo, titan" / "cihla, beton" / "kůže, textil"
  VLASTNOST:   "voděodolný, kompaktní" (kvalita bez přiřaditelné číselné hodnoty)
  BENEFIT:     "celodenní práce bez nabíjení" / "ochrana před finanční ztrátou"
  POUZITI:     "kancelář, cestování" / "havárie" / "rodiny s dětmi" (i rizika, cílovky)
  TECHNOLOGIE: "Intel Evo" / "tepelné čerpadlo" / "Face ID"
  SPECIFIKACE: "Windows 11 Pro" / "USB-C" / "IP68" / "GDPR" (norma / standard / certifikace)
  ORGANIZACE:  výrobce, poskytovatel, instituce, značka
  OSOBA:       jmenovaná osoba I obecná role (klient, řidič, lékárník)
  MISTO:       skutečné geografické místo, adresa, region
  DATUM:       konkrétní datum nebo časový interval
  SUBJEKT:     generický fallback pro to, co nesedí jinam

═══════════════════════════
PRAVIDLA EXTRAKCE ENTIT
═══════════════════════════
1. Vracej POUZE validní JSON — žádný úvod, markdown ani komentáře.
2. Každá entita má: "id" ("e1","e2",…), "text" (PŘESNÝ substring z originálu
   včetně pádu), "category" (POUZE z povolených). NIC jiného — pole "lemma"
   NEVRACEJ (kanonickou formu dopočítá aplikace deterministicky mimo LLM).
3. Entitu přidávej JEN když je v textu EXPLICITNĚ. Nikdy nevymýšlej text, který
   v originálu není.

4. ⚠️ KONEČNOST (NEJDŮLEŽITĚJŠÍ): vrať každou UNIKÁTNÍ věcnou entitu PRÁVĚ
   JEDNOU (typicky první výskyt). NEopakuj stejný text, NElabeluj každé slovo,
   NEnaplňuj výstup balastem. Typický odstavec má JEDNOTKY až NÍZKÉ DESÍTKY
   unikátních entit — výstup je proto KRÁTKÝ a KONEČNÝ. Jakmile začneš sypat
   desítky skoro stejných položek, děláš chybu: vrať jen smysluplné entity a SKONČI.

5. Stejný surface form → VŽDY stejná kategorie (konzistence v dokumentu).
6. NIKDY stejný "text" ve více entitách s různou kategorií = halucinace.
7. Číselné identifikátory (katalogová čísla, EAN, ID, čísla smluv) NEJSOU entity.
8. Materiál (kov, plast, sklo, cihla, dřevo) = MATERIAL nebo KOMPONENTA, nikdy ORGANIZACE.
9. NAVIGAČNÍ / UI ŠUM nejsou entity: "Klientská zóna", "Více", "Zpět", "Domů",
   "Kontakt", "Přihlášení", breadcrumby, tlačítka, názvy menu. Je-li CELÝ vstup
   navigace, vrať {"entities": []}.
10. URL, e-maily, telefony NIKDY jako entity (regex-detekovatelné struktury —
    aplikace si je dotáhne sama). MISTO je vyhrazeno pro skutečné geografické místo.
11. NEVRACEJ pozice (start_char/end_char) ani "lemma" — obojí dopočítá
    aplikace deterministicky z "text".
12. Vracíš POUZE klíč "entities". Žádné "relations", žádné "triples".

KONTEXT slouží jen pro porozumění textu — NENÍ zdrojem entit."""


EXTRACTOR_NER_USER_TEMPLATE = """DOMÉNA TEXTU: {domain}

POVOLENÉ KATEGORIE ENTIT (UPPERCASE, identické pro všechny domény):
{categories}

──────────────────────────────
PŘÍKLAD VÝSTUPU (jiná doména — jen ilustrace struktury a kategorií):
{example_entities}

──────────────────────────────
PŘÍKLAD ŠPATNÉHO VÝSTUPU (TAKHLE NIKDY):
{bad_example}
Chyby: "12345678" je katalogové číslo (není entita); stejný text ve více
kategoriích = halucinace; "Klientská zóna" / "Více" jsou navigace, ne entity.

──────────────────────────────
KONTEXT (strukturovaná metadata — jen pro porozumění, NE zdroj entit):
{context}

TEXT K ANALÝZE:
\"\"\"
{text}
\"\"\"

Vrať JSON {{"entities": [...]}} — POUZE ENTITY z textu (relace teď neřeš), každou
unikátní věcnou entitu jednou. Výstup je krátký a konečný."""


EXTRACTOR_NER_USER_TEMPLATE_CUSTOM = """DOMÉNA TEXTU: {domain}

{categories}

KRITICKÉ: používej VÝHRADNĚ labely z výše uvedené taxonomie. Když entitu nelze
přiřadit žádnému uvedenému labelu, NEzahrnuj ji.

──────────────────────────────
PŘÍKLAD ŠPATNÉHO VÝSTUPU (struktura; labely se liší dle taxonomie):
{bad_example}
Chyby: katalogová čísla a kódy nejsou entity; stejný text ve více kategoriích =
halucinace; navigace ("Klientská zóna", "Více") není entita.

──────────────────────────────
KONTEXT (strukturovaná metadata — jen pro porozumění, NE zdroj entit):
{context}

TEXT K ANALÝZE:
\"\"\"
{text}
\"\"\"

Vrať JSON {{"entities": [...]}} — labely VÝHRADNĚ z taxonomie, POUZE ENTITY z textu,
každou unikátní jednou. Výstup je krátký a konečný."""


def build_extractor_ner_messages(
    text: str, domain: str, context: Optional[str] = None, taxonomy=None
) -> list[dict]:
    """Fáze 1a — extrakce POUZE entit. Dedikovaný čistý NER prompt (bez
    relačního balastu z combined promptu) s důrazem na konečnost výstupu."""
    has_custom_taxonomy = taxonomy is not None and getattr(taxonomy, "labels", None)
    if has_custom_taxonomy:
        user = EXTRACTOR_NER_USER_TEMPLATE_CUSTOM.format(
            domain=domain,
            categories=_taxonomy_block(taxonomy),
            bad_example=json.dumps(EXAMPLE_NER_BAD, ensure_ascii=False, indent=2),
            context=context or "(žádná)",
            text=text,
        )
    else:
        user = EXTRACTOR_NER_USER_TEMPLATE.format(
            domain=domain,
            categories=_categories_block(ALL_CATEGORIES),
            example_entities=json.dumps(EXAMPLE_NER_OUTPUT, ensure_ascii=False, indent=2),
            bad_example=json.dumps(EXAMPLE_NER_BAD, ensure_ascii=False, indent=2),
            context=context or "(žádná)",
            text=text,
        )
    return [
        {"role": "system", "content": EXTRACTOR_NER_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# RE-only — DEDIKOVANÝ čistý prompt (entity jsou DANÉ, žádný entitní balast).
# ---------------------------------------------------------------------------

EXTRACTOR_RE_SYSTEM = """Jsi expertní systém pro extrakci RELACÍ (RE) mezi už hotovými
entitami. Entity dostáváš jako DANÉ — needituj je ani nepřidávej nové. Hledáš jen
smysluplné vztahy mezi nimi, podložené textem. Vracíš jen strukturovaný JSON.

═══════════════════════════
PRAVIDLA EXTRAKCE RELACÍ
═══════════════════════════
1. Vracej POUZE validní JSON — žádný úvod, markdown ani komentáře.
2. Každá relace má: "subjectId" a "objectId" (MUSÍ být "id" z DODANÝCH entit) a
   "relation" (snake_case, anglicky, z povoleného slovníku).
3. ⚠️ KONEČNOST: relaci vytvoř JEN pro vztah PŘÍMO podložený textem. Nevymýšlej
   a NEspojuj entity do řetězu jen kvůli jejich pořadí v textu. Když mezi dvěma
   entitami věcný vztah není, žádnou relaci nevracej. Počet relací je malý a
   konečný — radši méně přesných relací než hodně vymyšlených.
4. self-loop (subjectId == objectId) je zakázaný.

═══════════════════════════
SMĚROVÁ PRAVIDLA (preferuj přesné; obecné jsou taky OK)
═══════════════════════════
  issued_by:        PRODUKT / KATEGORIE → ORGANIZACE   (ne naopak)
  part_of:          část → celek
  has_component:    produkt / celek → komponenta
  has_parameter:    produkt → parametr
  has_value:        parametr → hodnota
  covers:           pojištění / produkt → riziko / událost / použití
  suitable_for:     produkt → cílová skupina / použití
  made_of:          produkt / komponenta → materiál
  provides_benefit: parametr / vlastnost / technologie → benefit
  located_in:       jen skutečné geografické místo (NIKDY URL)

Když ŽÁDNÁ přesná relace nesedí, ale vztah EXISTUJE, použij obecnější strukturální
relaci (is_a, part_of, belongs_to) — radši obecná než žádná. "related_to" je
POSLEDNÍ možnost; čistě lineární řetěz bez sémantiky RADŠI nevracej. Pokud si
nejsi jist směrem relace, RADŠI ji nevracej.

5. Vracíš POUZE klíč "relations". Žádné "entities", žádné "triples"."""


EXTRACTOR_RE_USER_TEMPLATE = """DOMÉNA TEXTU: {domain}

POVOLENÉ TYPY RELACÍ (do "relation" patří anglická snake_case část):
{relations}

UŽ EXTRAHOVANÉ ENTITY — subjectId/objectId v relacích MUSÍ odkazovat na jejich "id":
{entities}

──────────────────────────────
KONTEXT (strukturovaná metadata — jen pro porozumění):
{context}

TEXT K ANALÝZE:
\"\"\"
{text}
\"\"\"

Vrať JSON {{"relations": [...]}} — jen smysluplné vztahy mezi DODANÝMI entitami,
podložené textem. Žádné nové entity."""


def build_extractor_re_messages(
    text: str,
    domain: str,
    entities: list,
    context: Optional[str] = None,
    taxonomy=None,  # nepoužito (relace mají fixní slovník) — ponecháno pro signaturu
) -> list[dict]:
    """Fáze 1b — extrakce POUZE relací nad DODANÝMI entitami. Dedikovaný RE
    prompt; entity jsou dané, žádný entitní balast z combined promptu."""
    user = EXTRACTOR_RE_USER_TEMPLATE.format(
        domain=domain,
        relations=_relations_table(),
        entities=_entities_block(entities),
        context=context or "(žádná)",
        text=text,
    )
    return [
        {"role": "system", "content": EXTRACTOR_RE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# CRITIC NER-only — DEDIKOVANÝ čistý prompt (jen validace entit).
# ---------------------------------------------------------------------------
CRITIC_NER_SYSTEM = """Jsi nezávislý validátor ENTITY výstupu NER systému. Dostáváš
původní text a navržené entity (JSON). Posuzuješ POUZE entity — relace neřešíš.

TVÉ MOŽNÉ ODPOVĚDI (vrať POUZE JSON, nic jiného):
1. Správné:     {"status": "ok", "reasoning": "<max 2 věty>"}
2. Opravitelné: {"status": "revised", "reasoning": "<co a proč>", "corrected": {"entities": [...]}}
3. Zásadně špatné (vymyšlené entity, text nepochopen): {"status": "reject", "reasoning": "<proč>"}

VALIDAČNÍ PRAVIDLA (entity):
A) Entita musí být EXPLICITNĚ v textu — "text" = přesný substring. Pozice
   (start_char/end_char) NEKONTROLUJ ani NEVRACEJ (dopočítá je aplikace).
{categories_rule}
C) HALUCINACE: stejný text ve více entitách s různou kategorií → smaž. Katalogová
   čísla, EAN, ID, čísla smluv NEJSOU entity.
D) MATERIÁL (kov, plast, sklo, cihla, dřevo) = MATERIAL/KOMPONENTA, nikdy ORGANIZACE.
E) NAVIGAČNÍ ŠUM ("Klientská zóna", "Více", "Domů", breadcrumby, tlačítka) nejsou
   entity → odstraň. Když je takto označena většina → 'reject'.
F) KONZISTENCE: stejný surface form = vždy stejná kategorie.
G) URL / e-mail / telefon NIKDY jako entity → smaž (regex struktury, ne NER).

ROZHODOVÁNÍ: 1-2 menší chyby → 'revised'; 50 %+ nesmyslů → 'reject'; jinak 'ok'.
V "corrected" vracej POUZE {"entities": [...]}. NIKDY žádný text mimo JSON."""


CRITIC_NER_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT (strukturovaná metadata, pokud k dispozici):
{context}

NAVRŽENÉ ENTITY:
{proposed_entities}

Posuď POUZE ENTITY. Vrať JSON se "status", "reasoning" a volitelně "corrected"
(jen {{"entities": [...]}}):"""


def build_critic_ner_messages(
    text: str, entities: list, context: Optional[str] = None, taxonomy=None
) -> list[dict]:
    """Fáze 2a — validace POUZE entit (dedikovaný critic NER prompt).

    `taxonomy` (fáze 0) = povolený slovník kategorií, stejný jako u extractoru.
    """
    user = CRITIC_NER_USER_TEMPLATE.format(
        text=text,
        context=context or "(žádná)",
        proposed_entities=json.dumps(
            {"entities": _strip_lemma_entities(entities)},
            ensure_ascii=False, indent=2,
        ),
    )
    system = CRITIC_NER_SYSTEM.replace(
        "{categories_rule}",
        _critic_categories_rule(taxonomy, _CRITIC_NER_DEFAULT_CATEGORIES_RULE),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# CRITIC RE-only — DEDIKOVANÝ čistý prompt (entity dané; jen validace relací).
# ---------------------------------------------------------------------------
CRITIC_RE_SYSTEM = """Jsi nezávislý validátor RELACÍ výstupu RE systému. Dostáváš
původní text, entity jako DANÝ kontext (NEEDITUJ je) a navržené relace. Posuzuješ
POUZE relace.

TVÉ MOŽNÉ ODPOVĚDI (vrať POUZE JSON):
1. {"status": "ok", "reasoning": "<max 2 věty>"}
2. {"status": "revised", "reasoning": "<co a proč>", "corrected": {"relations": [...]}}
3. {"status": "reject", "reasoning": "<proč>"}   (relace jsou vesměs nesmysl)

VALIDAČNÍ PRAVIDLA (relace):
A) subjectId i objectId MUSÍ odkazovat na "id" z dodaných entit. self-loop zakázán.
B) SMĚR — preferuj přesné, ale OBECNÉ jsou taky OK:
   - has_value: PARAMETR → HODNOTA
   - has_component: PRODUKT/KOMPONENTA → KOMPONENTA
   - made_of / has_material: PRODUKT/KOMPONENTA → MATERIAL
   - provides_benefit: PARAMETR/VLASTNOST/TECHNOLOGIE → BENEFIT
   - issued_by: PRODUKT/KATEGORIE → ORGANIZACE
   - covers: PRODUKT → riziko / použití
   Když ŽÁDNÁ přesná nesedí, ale vztah EXISTUJE, použij obecnou (is_a, part_of,
   belongs_to) — NIKDY je nemaž jen kvůli obecnosti; radši slabá relace než žádná.
C) "related_to" smaž JEN když je to lineární řetěz BEZ jakékoli sémantiky (šum).
   Když vztah reálně existuje a nic přesnějšího nesedí, NECH ho.
D) located_in jen pro skutečné geografické místo (NIKDY URL).

ROZHODOVÁNÍ: 1-2 menší chyby → 'revised'; většina relací nesmysl → 'reject'; jinak 'ok'.
V "corrected" vracej POUZE {"relations": [...]}. Entity NEEDITUJ. NIKDY text mimo JSON."""


CRITIC_RE_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT (strukturovaná metadata, pokud k dispozici):
{context}

DANÉ ENTITY (kontext — NEEDITUJ; relace na ně odkazují přes "id"):
{entities}

NAVRŽENÉ RELACE:
{proposed_relations}

Posuď správnost RELACÍ. Vrať JSON se "status", "reasoning" a volitelně "corrected"
(jen {{"relations": [...]}}):"""


def build_critic_re_messages(
    text: str, relations: list, entities: list, context: Optional[str] = None
) -> list[dict]:
    """Fáze 2b — validace POUZE relací (entity jako daný kontext; dedikovaný)."""
    user = CRITIC_RE_USER_TEMPLATE.format(
        text=text,
        context=context or "(žádná)",
        entities=_entities_block(entities),
        proposed_relations=json.dumps(
            {"relations": relations}, ensure_ascii=False, indent=2
        ),
    )
    return [
        {"role": "system", "content": CRITIC_RE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# REVISE NER-only — DEDIKOVANÝ prompt (re-eval entit po kritice).
# ---------------------------------------------------------------------------
REVISION_NER_SYSTEM = """Jsi NER systém. Dostal jsi zpětnou vazbu od nezávislého
validátora ke svým ENTITÁM. Přezkoumej je ve světle vazby a rozhodni:
  a) Kritik našel reálné chyby (halucinace, špatné kategorie, vymyšlené entity,
     navigační šum) → vrať OPRAVENÉ entity.
  b) Výhrada je nepodložená nebo s ní nesouhlasíš → vrať PŮVODNÍ entity beze změny.

{categories_rule}

Řeš POUZE ENTITY (relace neřešíš). Vrať POUZE validní JSON {"entities": [...]} —
žádný text navíc."""


REVISION_NER_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT:
{context}

TVÉ PŮVODNÍ ENTITY:
{original_entities}

ZPĚTNÁ VAZBA VALIDÁTORA (status: {critic_status}):
{critic_reasoning}
{corrected_section}
Vrať JSON {{"entities": [...]}} — opravené nebo původní (POUZE ENTITY):"""


def build_revision_ner_messages(
    text: str,
    original_entities: list,
    critic_status: str,
    critic_reasoning: str,
    corrected_entities: Optional[list] = None,
    context: Optional[str] = None,
    taxonomy=None,
) -> list[dict]:
    """Fáze 2b-revise (NER) — re-eval kritiky entit (dedikovaný prompt)."""
    if corrected_entities:
        corrected_section = (
            "\nKRITIKEM NAVRŽENÉ OPRAVENÉ ENTITY (jen pro referenci):\n"
            + json.dumps(
                {"entities": _strip_lemma_entities(corrected_entities)},
                ensure_ascii=False, indent=2,
            )
            + "\n"
        )
    else:
        corrected_section = ""
    user = REVISION_NER_USER_TEMPLATE.format(
        text=text,
        context=context or "(žádná)",
        original_entities=json.dumps(
            {"entities": _strip_lemma_entities(original_entities)},
            ensure_ascii=False, indent=2,
        ),
        critic_status=critic_status,
        critic_reasoning=critic_reasoning,
        corrected_section=corrected_section,
    )
    return [
        {"role": "system", "content": REVISION_NER_SYSTEM.replace(
            "{categories_rule}",
            _critic_categories_rule(taxonomy, _REVISION_DEFAULT_CATEGORIES_RULE),
        )},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# REVISE RE-only — DEDIKOVANÝ prompt (re-eval relací po kritice; entity dané).
# ---------------------------------------------------------------------------
REVISION_RE_SYSTEM = """Jsi RE systém. Dostal jsi zpětnou vazbu od nezávislého
validátora k RELACÍM. Entity jsou DANÉ (NEEDITUJ je). Přezkoumej relace a rozhodni:
  a) Kritik našel reálné chyby (špatný směr, vymyšlené vztahy, zbytečné
     related_to) → vrať OPRAVENÉ relace.
  b) Výhrada je nepodložená → vrať PŮVODNÍ relace beze změny.

subjectId/objectId musí odkazovat na "id" daných entit. Řeš POUZE RELACE. Vrať
POUZE validní JSON {"relations": [...]} — žádný text navíc."""


REVISION_RE_USER_TEMPLATE = """PŮVODNÍ TEXT:
\"\"\"
{text}
\"\"\"

KONTEXT:
{context}

DANÉ ENTITY (NEEDITUJ; relace na ně odkazují přes "id"):
{entities}

TVÉ PŮVODNÍ RELACE:
{original_relations}

ZPĚTNÁ VAZBA VALIDÁTORA (status: {critic_status}):
{critic_reasoning}
{corrected_section}
Vrať JSON {{"relations": [...]}} — opravené nebo původní (POUZE RELACE):"""


def build_revision_re_messages(
    text: str,
    original_relations: list,
    entities: list,
    critic_status: str,
    critic_reasoning: str,
    corrected_relations: Optional[list] = None,
    context: Optional[str] = None,
) -> list[dict]:
    """Fáze 2b-revise (RE) — re-eval kritiky relací (entity jako kontext;
    dedikovaný prompt)."""
    if corrected_relations:
        corrected_section = (
            "\nKRITIKEM NAVRŽENÉ OPRAVENÉ RELACE (jen pro referenci):\n"
            + json.dumps({"relations": corrected_relations}, ensure_ascii=False, indent=2)
            + "\n"
        )
    else:
        corrected_section = ""
    user = REVISION_RE_USER_TEMPLATE.format(
        text=text,
        context=context or "(žádná)",
        entities=_entities_block(entities),
        original_relations=json.dumps(
            {"relations": original_relations}, ensure_ascii=False, indent=2
        ),
        critic_status=critic_status,
        critic_reasoning=critic_reasoning,
        corrected_section=corrected_section,
    )
    return [
        {"role": "system", "content": REVISION_RE_SYSTEM},
        {"role": "user", "content": user},
    ]
