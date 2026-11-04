"""
Dataset readery pro vstupní JSON soubory (Stage export + generický JSON).

Každý reader je generátor LabelItem objektů. LabelItem obsahuje:
  - item_id       (stabilní ID)
  - source        (dataset name)
  - text          (primární text k labelování, HTML stripnutý, chunkovaný)
  - context       (strukturované metadata, volitelně)
  - domain        (volný český popis domény pro LLM prompt)

Dlouhé texty (> CHUNK_MAX_CHARS) se rozdělí na chunky. Každý chunk má unikátní
item_id = "{base_id}#chunk{N}". V finalize fázi se pak spojí zpět.
"""

from __future__ import annotations
import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from .config import CONFIG

logger = logging.getLogger(__name__)

import os as _os

CHUNK_MAX_CHARS = int(_os.environ.get("CHUNK_MAX_CHARS", "12000"))
CHUNK_OVERLAP_CHARS = 200

_HTML_RE = re.compile(r"<[a-zA-Z/!][^>]{0,500}>")
_WS_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    s = _HTML_RE.sub(" ", s)
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def split_into_chunks(
    text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """
    Rozdělí text na chunky po větách (greedy). Snaží se řezat na tečkách / nových řádcích.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                chunks.append(current)
            while len(s) > max_chars:
                chunks.append(s[:max_chars])
                s = s[max_chars - overlap :]
            current = s
    if current:
        chunks.append(current)
    return chunks


@dataclass
class LabelItem:
    """Jedna jednotka k olabelování."""

    item_id: str
    source: str
    domain: str
    text: str
    context: Optional[str] = None
    is_synthetic: bool = False
    original_id: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    extra: dict = field(default_factory=dict)


def _looks_like_navigation(text: str) -> bool:
    """
    Heuristika: text je pravděpodobně navigační šum (menu, breadcrumb, footer).

    Indikátory:
      - vysoký podíl krátkých frází (< 5 slov) oddělených mezerami
      - obsahuje typická menu slova ("Klientská zóna", "Více", "Domů", "Kontakt")
      - žádné věty (žádné tečky uvnitř, jen krátké úryvky)

    Nedělá to perfektní filtraci — jen vyhodí to nejhorší. Zbytek dolaďuje LLM
    + critic přes prompty.
    """
    if not text:
        return False

    sentence_density = text.count(".") + text.count("!") + text.count("?")
    if len(text) < 500 and sentence_density < 2:
        return True

    nav_keywords = [
        "Klientská zóna",
        "Přihlášení",
        "Registrace",
        "Domů",
        "Kontakt",
        "O nás",
        "Nápověda",
        "Více informací",
        "Zpět",
        "Další",
        "Předchozí",
        "Cookies",
        "Newsletter",
    ]
    matches = sum(1 for kw in nav_keywords if kw in text)
    if matches >= 3 and len(text) < 800:
        return True

    return False


# ---------------------------------------------------------------------------
# Univerzální Stage export reader
# (export-*-latest.json — dataset/schemas/records/fields struktura)
# ---------------------------------------------------------------------------

STAGE_MIN_TEXT_LEN = 80
STAGE_MIN_PRODUCT_TEXT_LEN = 40

# Defaultní doména pro Stage exporty (LLM dostane český popisek domény pro
# prompt). Per soubor ji přepíšeš v labeler_config.json ("domain") nebo přes
# default_domain / --domain — žádné dataset názvy nejsou v kódu hard-coded.
DEFAULT_STAGE_DOMAIN = "produktový / obsahový katalog"

# Schemas, které neobsahují věcný text k labelování — nikdy se neposílá do LLM.
# Matchuje se substringem v schema.code/name (case-insensitive).
_STAGE_SKIP_SCHEMA_SUBSTRINGS = (
    "-files",   # cs-files, sk-files atd. — jen názvy souborů, URL, žádný popis
    "_files",
)

# Prototypy stránek, které jsou navigační/systémové — nelabelovat (banner, alert,
# system include, contact branch list, atd.). Matchuje se substringem v hodnotě
# `prototype` (case-insensitive).
_STAGE_SKIP_PROTOTYPE_SUBSTRINGS = (
    "alertstrip",
    "banners",
    "system/includes",
    "contactbranches",
    "headernav",
    "footer",
    "header",
    "navigation",
    "menu",
    "search",
    "newsletter",
    "cookies",
    "login",
    "register",
)

# Typy objektů bez věcného textu — v2 exporty míchají stránky a soubory v jednom
# schématu (objectType edeePage / edeeFile). Substring v `objectType`
# (case-insensitive), stejná politika jako _STAGE_SKIP_SCHEMA_SUBSTRINGS.
_STAGE_SKIP_OBJECT_TYPE_SUBSTRINGS = ("file",)


# ---------------------------------------------------------------------------
# Výběr jazyků (locale)
# ---------------------------------------------------------------------------

LocalesArg = Union[str, Iterable[str], None]


def _norm_locale(loc: str) -> str:
    return loc.strip().lower().replace("_", "-")


_ALL_LOCALES = {"*", "all"}


def parse_locales(value: LocalesArg) -> Optional[frozenset[str]]:
    """
    "cs, sk" / ["cs", "sk"] → frozenset({"cs", "sk"}) (normalizované).
    "*" / "all" / prázdná hodnota / None → None = bez filtru (všechny jazyky).
    """
    if value is None:
        return None
    parts = re.split(r"[,;\s]+", value) if isinstance(value, str) else value
    out = frozenset(_norm_locale(str(p)) for p in parts if p and str(p).strip())
    if not out or out & _ALL_LOCALES:
        return None
    return out


def _locale_selected(loc: str, selected: Optional[frozenset[str]]) -> bool:
    """True, pokud locale patří do výběru. Výběr "cs" matchuje i "cs-CZ" / "cs_CZ"."""
    if not selected:
        return True
    n = _norm_locale(loc)
    return n in selected or n.split("-", 1)[0] in selected


def _stage_field_value(fld: dict) -> Optional[str]:
    """Hodnota Stage fieldu jako string. REFERENCE pole (list ID) → "id1, id2"."""
    val = fld.get("value")
    if val is None:
        val = fld.get("values")
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v is not None)
    return str(val)


def _stage_record_locale_views(
    rec: dict, schema_locales: list[str]
) -> list[tuple[Optional[str], dict[str, str]]]:
    """
    Rozloží Stage záznam {id, fields:[{code, value, locale}]} na jazykové verze
    [(locale, {code: value})].

    Pole s `locale: null` (nelokalizovaná) jsou v každé verzi, lokalizovaná pole
    jen ve verzi svého jazyka. Záznam bez lokalizovaných polí (i v1 exporty bez
    `locale`) → jediná verze s locale = jediný allowedLocale schématu, jinak None.
    """
    shared: dict[str, str] = {}
    per_locale: dict[str, dict[str, str]] = {}
    for fld in rec.get("fields", []) or []:
        code = fld.get("code") or fld.get("name")
        if not code:
            continue
        val = _stage_field_value(fld)
        if val is None:
            continue
        loc = fld.get("locale")
        if loc:
            per_locale.setdefault(loc, {})[code] = val
        else:
            shared[code] = val

    if not per_locale:
        return [(schema_locales[0] if len(schema_locales) == 1 else None, shared)]
    return [(loc, {**shared, **fields}) for loc, fields in per_locale.items()]


def _stage_is_skipped_prototype(prototype: str) -> bool:
    if not prototype:
        return False
    p = prototype.lower()
    return any(sub in p for sub in _STAGE_SKIP_PROTOTYPE_SUBSTRINGS)


def _stage_is_skipped_record(fields: dict[str, str]) -> bool:
    """Navigační/systémový prototyp nebo objekt bez textu (soubor) → nelabelovat."""
    if _stage_is_skipped_prototype(fields.get("prototype", "")):
        return True
    object_type = fields.get("objectType", "").lower()
    return any(sub in object_type for sub in _STAGE_SKIP_OBJECT_TYPE_SUBSTRINGS)


def _stage_compose_product_text(fields: dict[str, str]) -> tuple[str, str]:
    """
    Pro produktový/kategorie/brand záznam sestaví hlavní text k labelování
    a zvlášť kontext pro LLM (metadata).

    Vrátí (primary_text, context). Pokud nejde dohromady ničemu rozumnému, vrátí
    ("", "").
    """
    # Hlavní text: description > descriptionShort > content
    desc = strip_html(fields.get("description", "")).strip()
    short = strip_html(fields.get("descriptionShort", "")).strip()
    if len(desc) >= STAGE_MIN_PRODUCT_TEXT_LEN:
        primary = desc
    elif len(short) >= STAGE_MIN_PRODUCT_TEXT_LEN:
        primary = short
    else:
        # Pokud máme aspoň name + parameters, poskládáme synthetic text
        name = fields.get("titleOrName") or fields.get("name") or ""
        params = fields.get("parameters", "")
        if name and params:
            params_pretty = ", ".join(
                p.strip() for p in params.split(";")[:25] if p.strip()
            )
            primary = f"{name.strip()}. {params_pretty}".strip()
        else:
            primary = ""

    if len(primary) < STAGE_MIN_PRODUCT_TEXT_LEN:
        return "", ""

    # Context: pouze pole, která LLM pomohou (značka, název, kategorie přes URL)
    ctx_parts = []
    if fields.get("brandName"):
        ctx_parts.append(f"Značka: {fields['brandName']}")
    if fields.get("titleOrName") or fields.get("name"):
        title = fields.get("titleOrName") or fields.get("name")
        if title.strip() and title.strip() not in primary:
            ctx_parts.append(f"Název: {title}")
    if fields.get("parameters") and "parameters" not in (
        # neopakovat parameters, pokud už jsou v primary
        primary,
    ):
        params_clean = strip_html(fields["parameters"])[:300]
        if params_clean and params_clean not in primary:
            ctx_parts.append(f"Parametry: {params_clean}")
    return primary, "\n".join(ctx_parts) if ctx_parts else ""


# Pole, která obsahují hlavní text článku — v prioritě (v1 Stage: 'content',
# zpravodajské exporty: 'body'). První nenalezené pole se ignoruje.
_PAGE_TEXT_FIELDS = ("content", "body")


def _stage_compose_page_text(fields: dict[str, str]) -> tuple[str, str]:
    """
    Pro page/news/blog/faq sestaví hlavní text + kontext.

    Hlavní text hledá v polích `_PAGE_TEXT_FIELDS` (content, body) —
    podpoří tak i zpravodajské exporty, kde je text v `body`.
    Kontext: title, perex (zpravodajské exporty), kategorie, autoři.
    """
    content = ""
    for fld in _PAGE_TEXT_FIELDS:
        val = fields.get(fld, "")
        if val:
            content = strip_html(val).strip()
            if content:
                break
    if len(content) < STAGE_MIN_TEXT_LEN:
        return "", ""

    # Skip navigace
    if _looks_like_navigation(content):
        return "", ""

    ctx_parts = []
    title = strip_html(fields.get("title", "")).strip()
    if title and title not in content:
        ctx_parts.append(f"Titulek: {title}")
    perex = strip_html(fields.get("perex", "")).strip()
    if perex and perex not in content:
        ctx_parts.append(f"Perex: {perex}")
    # kategorie — zpravodajské exporty: list[str], ostatní: string
    cats_raw = fields.get("category_names") or fields.get("categories") or ""
    if isinstance(cats_raw, list):
        cats = ", ".join(str(c) for c in cats_raw if c)
    else:
        cats = str(cats_raw).strip()
    if cats:
        ctx_parts.append(f"Kategorie: {cats}")
    # autoři
    authors_raw = fields.get("authors") or ""
    if isinstance(authors_raw, list):
        authors = ", ".join(str(a) for a in authors_raw if a)
    else:
        authors = str(authors_raw).strip()
    if authors:
        ctx_parts.append(f"Autoři: {authors}")

    return content, "\n".join(ctx_parts) if ctx_parts else ""


def iter_stage_export(
    path: Path,
    *,
    domain_override: Optional[str] = None,
    locales: LocalesArg = None,
    limit: Optional[int] = None,
) -> Iterator[LabelItem]:
    """
    Univerzální reader pro Stage export soubory typu `export-*-latest.json`.

    Struktura (v1 i v2):
      root = {
        version: 1|2, mode: "EMBEDDED" (v2), exportedAt: "...",
        dataset: {name, path, description},
        schemas: [
          {
            name, code, allowedLocales: [...],
            fields:    [{code, name, type, localized, ...}],   # definice fieldů
            relations: [{fieldCode, targetSchemaCode, ...}],   # v2
            records: [{
              id,
              fields:  [{code, value, locale}],   # value: str | list[str] (REFERENCE)
              related: [...]                      # v2, embedded vazby — ignoruje se
            }]
          }
        ]
      }

    Jazyky: `locales` (např. "cs,sk" nebo ["cs"]; None / "*" = všechny). Schéma,
    jehož allowedLocales nemá s výběrem průnik, se přeskočí celé. Lokalizovaná
    pole (`locale` != null) jiných jazyků se ignorují; záznam dostupný jen
    v jiných jazycích se přeskočí. Záznam s více vybranými jazyky dá jednu
    verzi per jazyk — item_id pak nese suffix "@{locale}" (jen u vícejazyčných
    schémat/záznamů, jednojazyčná zůstávají beze změny).

    Pro každý record vybere POUZE věcný obsah pro NER/RE/triples:
      - produkty/kategorie/brands → description (fallback descriptionShort, pak
        synthetic z name+parameters)
      - stránky/news/blog → content (s odfiltrováním navigace a krátkých
        nicneříkajících záznamů)
      - soubory (cs-files, sk-files atd., objectType edeeFile) → SKIP (jen
        názvy a URL, žádný obsah)
      - system/banner/header/footer prototypy → SKIP

    Doména je defaultně DEFAULT_STAGE_DOMAIN; per soubor ji přepíšeš přes
    `domain_override` (labeler_config.json "domain", případně --domain).
    """
    logger.info(f"Načítám Stage export: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    dataset_meta = data.get("dataset", {}) or {}
    dataset_name = dataset_meta.get("name", "") or path.stem
    schemas = data.get("schemas", []) or []
    if not schemas:
        logger.warning(f"{path.name}: chybí 'schemas', přeskakuji.")
        return

    domain = domain_override or DEFAULT_STAGE_DOMAIN
    selected = parse_locales(locales)
    logger.info(
        f"Stage export {dataset_name!r}: {len(schemas)} schémat, doména='{domain}', "
        f"jazyky={sorted(selected) if selected else 'všechny'}"
    )

    produced = 0
    skipped_empty = 0
    skipped_nav = 0
    skipped_short = 0
    skipped_locale = 0
    skipped_schemas: list[str] = []
    skipped_locale_schemas: list[str] = []

    for schema in schemas:
        if limit is not None and produced >= limit:
            break
        schema_code = (schema.get("code") or schema.get("name") or "").strip()
        if not schema_code:
            continue
        schema_lower = schema_code.lower()

        # SKIP whole-schema bordel: jen názvy souborů, žádný věcný popis.
        if any(sub in schema_lower for sub in _STAGE_SKIP_SCHEMA_SUBSTRINGS):
            skipped_schemas.append(schema_code)
            continue

        schema_locales = [
            loc for loc in (schema.get("allowedLocales") or [])
            if isinstance(loc, str) and loc.strip()
        ]
        if selected and schema_locales and not any(
            _locale_selected(loc, selected) for loc in schema_locales
        ):
            skipped_locale_schemas.append(schema_code)
            continue

        records = schema.get("records", []) or []
        if not records:
            continue

        # Typ schématu (produkty: 'description'/'descriptionShort', stránky:
        # 'content') se určí z prvního nepřeskočeného záznamu — v2 schémata
        # míchají stránky se soubory, records[0] proto nestačí.
        is_product_like: Optional[bool] = None
        is_page_like = False

        for rec in records:
            if limit is not None and produced >= limit:
                break

            all_views = _stage_record_locale_views(rec, schema_locales)
            views = [
                (loc, fields) for loc, fields in all_views
                if loc is None or _locale_selected(loc, selected)
            ]
            if not views:
                skipped_locale += 1
                continue

            # prototype / objectType jsou nelokalizované → stejné ve všech verzích
            if _stage_is_skipped_record(views[0][1]):
                skipped_nav += 1
                continue

            # Typ schématu se určí z PRVNÍHO záznamu s věcným textem — některé
            # schémata začínají záznamy bez textu (např. zpravodajský záznam bez 'body'),
            # records[0] proto nestačí.
            if is_product_like is None:
                sample_fields = views[0][1].keys()
                has_product_fields = (
                    "description" in sample_fields or "descriptionShort" in sample_fields
                )
                has_page_fields = any(f in sample_fields for f in _PAGE_TEXT_FIELDS)
                if has_product_fields or has_page_fields:
                    is_product_like = has_product_fields
                    is_page_like = has_page_fields
                # jinak: typ neznámý, prozkoumáme další záznam

            # Suffix @locale jen u vícejazyčných schémat/záznamů; nezávisí na
            # výběru jazyků → stabilní item_id napříč běhy s různým DATA_LOCALES.
            tag_locale = len(schema_locales) > 1 or len(all_views) > 1
            seen_texts: set[str] = set()

            for loc, fields in views:
                if limit is not None and produced >= limit:
                    break

                # Vyber správný skladač podle typu schématu
                if is_product_like:
                    primary, context = _stage_compose_product_text(fields)
                elif is_page_like:
                    primary, context = _stage_compose_page_text(fields)
                else:
                    primary, context = "", ""

                if not primary:
                    has_text = any(
                        len(strip_html(fields.get(f, ""))) > 0 for f in _PAGE_TEXT_FIELDS
                    )
                    if is_page_like and has_text:
                        skipped_short += 1
                    else:
                        skipped_empty += 1
                    continue
                # Jazykové verze sdílející jen nelokalizovaný text → labelovat jednou
                if primary in seen_texts:
                    continue
                seen_texts.add(primary)

                record_id = (
                    fields.get("externalId")
                    or fields.get("id")
                    or rec.get("id")
                    or f"row{produced}"
                )
                base_id = f"{dataset_name}:{schema_code}:{record_id}"
                if loc and tag_locale:
                    base_id = f"{base_id}@{loc}"
                prototype = fields.get("prototype", "")

                chunks = split_into_chunks(primary)
                for i, chunk in enumerate(chunks):
                    chunk_id = base_id if len(chunks) == 1 else f"{base_id}#chunk{i}"
                    yield LabelItem(
                        item_id=chunk_id,
                        source=dataset_name,
                        domain=domain,
                        text=chunk,
                        context=context or None,
                        is_synthetic=False,
                        original_id=str(record_id),
                        chunk_index=i if len(chunks) > 1 else None,
                        total_chunks=len(chunks) if len(chunks) > 1 else None,
                        extra={
                            "schema": schema_code,
                            "prototype": prototype or None,
                            "locale": loc,
                            "source_file": path.name,
                        },
                    )
                    produced += 1
                    if limit is not None and produced >= limit:
                        break

    if skipped_schemas:
        logger.info(
            f"Stage {dataset_name}: skipped {len(skipped_schemas)} schémat "
            f"(jen názvy souborů / bez věcného obsahu): {skipped_schemas[:6]}"
            f"{'...' if len(skipped_schemas) > 6 else ''}"
        )
    if skipped_locale_schemas:
        logger.info(
            f"Stage {dataset_name}: skipped {len(skipped_locale_schemas)} schémat "
            f"(jiný jazyk): {skipped_locale_schemas[:6]}"
            f"{'...' if len(skipped_locale_schemas) > 6 else ''}"
        )
    logger.info(
        f"Stage {dataset_name}: vyprodukováno {produced} label items, "
        f"přeskočeno {skipped_empty} (bez obsahu) + {skipped_nav} (navigace/system) "
        f"+ {skipped_short} (krátký text) + {skipped_locale} (jiný jazyk)"
    )


def _is_stage_export(path: Path) -> bool:
    """Zjistí, zda JSON soubor je Stage export (klíče version + schemas + dataset)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            # Úvod souboru — schemas list je vždy až za dataset metadaty, ale
            # dataset.description může mít stovky znaků (v2 exporty), 512 nestačí.
            head = f.read(65536)
        # Rychlý sniff bez parsování celého souboru
        return (
            '"version"' in head
            and '"schemas"' in head
            and '"dataset"' in head
        )
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Auto-scan DATA_DIR
# ---------------------------------------------------------------------------

_CONFIG_FILENAME = "labeler_config.json"


def load_data_config(data_dir: Path) -> dict[str, dict]:
    """
    Načte volitelný konfigurační soubor labeler_config.json ze složky data.

    Formát souboru:
      {
        "produkty.json": {
          "domain": "e-shop s elektronikou",
          "text_field": "popis",
          "id_field": "catalogId",
          "context_fields": ["nazev", "kategorie", "cena"]
        },
        "inzeraty.json": {
          "domain": "realitní inzeráty"
        },
        "export-nabytek-demo-latest.json": {
          "locales": ["sk"]            # jen Stage exporty; přepíše DATA_LOCALES
        }
      }

    Pokud soubor neexistuje, vrátí prázdný dict (vše se auto-detekuje).
    """
    cfg_path = data_dir / _CONFIG_FILENAME
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"{cfg_path}: Očekával jsem JSON objekt, ignoruji")
            return {}
        logger.info(f"Načten labeler_config.json: {list(data.keys())}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Nelze číst {cfg_path}: {e}")
        return {}


def scan_data_dir(
    data_dir: Path,
    *,
    default_domain: Optional[str] = None,
    locales: LocalesArg = None,
    limit: Optional[int] = None,
) -> Iterator[LabelItem]:
    """
    Automaticky projde všechny *.json soubory ve složce data_dir a načte je.

    Logika výběru readeru:
      - Stage export soubor (root má 'version' + 'schemas' + 'dataset')
            → iter_stage_export (univerzální Stage reader; filtruje navigaci,
              system stránky a soubory bez věcného obsahu — do LLM jde jen
              produktové popisy, blog/news content, FAQ atd.)
      - cokoli jiného → iter_generic_json (univerzální dict/list reader)

    Konfigurace (volitelná) v souboru labeler_config.json:
      Umožňuje nastavit domain, text_field, id_field, context_fields, locales
      per soubor. Pokud soubor chybí, vše se auto-detekuje nebo použije název
      souboru jako doménu.

    Parametry:
      data_dir:       Složka se vstupními JSON soubory (CONFIG.data_dir).
      default_domain: Pokud je zadáno, přepíše doménu pro všechny generické soubory.
      locales:        Jazyky načítané ze Stage exportů ("cs,sk" / ["cs"]).
                      None → CONFIG.data_locales (env DATA_LOCALES, default
                      "cs"); "*" = všechny. Klíč "locales" v labeler_config.json
                      má přednost.
      limit:          Max. počet LabelItem per soubor (pro dry run).
    """
    if not data_dir.exists():
        logger.warning(f"DATA_DIR {data_dir} neexistuje")
        return

    file_config = load_data_config(data_dir)
    global_locales = locales if locales is not None else CONFIG.data_locales

    json_files = sorted(data_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != _CONFIG_FILENAME and not f.name.startswith(".")]

    if not json_files:
        logger.warning(f"V {data_dir} nenalezeny žádné *.json soubory")
        return

    logger.info(f"Nalezeno {len(json_files)} JSON souborů v {data_dir}: "
                f"{[f.name for f in json_files]}")

    for json_path in json_files:
        stem = json_path.stem
        per_file = file_config.get(json_path.name, {})

        # 1. Stage export (priorita — má jednoznačně detekovatelný formát)
        if _is_stage_export(json_path):
            logger.info(f"Načítám {json_path.name} jako Stage export (universal reader)")
            domain_override = default_domain or per_file.get("domain")
            yield from iter_stage_export(
                json_path,
                domain_override=domain_override,
                locales=per_file.get("locales", global_locales),
                limit=limit,
            )
            continue

        # 2. Generický reader (libovolný jiný JSON)
        domain = (
            default_domain
            or per_file.get("domain")
            or stem  # název souboru bez přípony jako fallback
        )
        context_fields_raw = per_file.get("context_fields")
        context_fields = (
            context_fields_raw if isinstance(context_fields_raw, list)
            else [s.strip() for s in context_fields_raw.split(",") if s.strip()]
            if isinstance(context_fields_raw, str)
            else None
        )
        logger.info(f"Načítám {json_path.name} jako generický dataset (doména: '{domain}')")
        yield from iter_generic_json(
            json_path,
            domain=domain,
            text_field=per_file.get("text_field"),
            id_field=per_file.get("id_field"),
            context_fields=context_fields,
            limit=limit,
        )


# ---------------------------------------------------------------------------
# Univerzální JSON reader
# ---------------------------------------------------------------------------

GENERIC_MIN_TEXT_LEN = 40
_AUTO_DETECT_SAMPLE = 20  # kolik prvních itemů prohledat při auto-detekci


def _collect_records(data: object, max_depth: int = 3) -> list[dict]:
    """
    Z libovolné JSON struktury vytáhne list dict objektů.

    Podporuje:
      - list[dict]          → přímo vrátí
      - dict s hodnotami list[dict] (prohledá do hloubky max_depth)
      - dict přímo jako jediný record
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        # Zkusíme najít array jako hodnotu klíče (do hloubky max_depth)
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]
        if max_depth > 1:
            for value in data.values():
                if isinstance(value, dict):
                    records = _collect_records(value, max_depth - 1)
                    if records:
                        return records
        # Samotný dict jako jediný record
        return [data]

    return []


def _detect_text_field(records: list[dict]) -> Optional[str]:
    """
    Automaticky najde pole s nejdelším průměrným textem.
    Kandidáti: string hodnoty delší než GENERIC_MIN_TEXT_LEN.
    """
    sample = records[:_AUTO_DETECT_SAMPLE]
    field_lengths: dict[str, list[int]] = {}

    for rec in sample:
        for key, val in rec.items():
            if isinstance(val, str) and len(val) >= GENERIC_MIN_TEXT_LEN:
                field_lengths.setdefault(key, []).append(len(val))

    if not field_lengths:
        return None

    best = max(field_lengths, key=lambda k: sum(field_lengths[k]) / len(field_lengths[k]))
    avg = sum(field_lengths[best]) / len(field_lengths[best])
    logger.info(f"Auto-detekce text_field: '{best}' (průměrná délka {avg:.0f} znaků)")
    return best


def _detect_id_field(records: list[dict]) -> Optional[str]:
    """Heuristika pro nalezení ID pole — preferujeme 'id', 'ID', 'uuid', 'externalId', …"""
    id_candidates = ["id", "ID", "Id", "uuid", "UUID", "externalId", "key", "slug", "url"]
    sample = records[:_AUTO_DETECT_SAMPLE]
    for candidate in id_candidates:
        if all(candidate in rec for rec in sample if rec):
            return candidate
    return None


def iter_generic_json(
    path: Path,
    *,
    domain: str = "obecný",
    text_field: Optional[str] = None,
    id_field: Optional[str] = None,
    context_fields: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> Iterator[LabelItem]:
    """
    Univerzální iterátor přes libovolný JSON soubor.

    Parametry:
      path:           Cesta k JSON souboru.
      domain:         Popis domény pro LLM (volný text, např. "e-shop", "realitní inzeráty").
      text_field:     Název pole obsahujícího hlavní text. Pokud None, auto-detekuje.
      id_field:       Název pole s ID záznamu. Pokud None, auto-detekuje nebo použije index.
      context_fields: Seznam polí, která se přidají do kontextu (metadata pro LLM).
      limit:          Maximální počet vygenerovaných LabelItem.

    Podporované struktury JSON:
      - Pole objektů:            [{"id": 1, "text": "..."}, ...]
      - Objekt s polem objektů:  {"items": [{"id": 1, ...}], ...}
      - Vnořené struktury:       {"data": {"records": [...]}} (do hloubky 3)
      - Jediný objekt:           {"id": 1, "text": "..."}
    """
    logger.info(f"Načítám generický dataset: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    records = _collect_records(raw)
    if not records:
        logger.warning(f"Generický dataset {path}: nenalezeny žádné záznamy")
        return

    logger.info(f"Generický dataset {path}: nalezeno {len(records)} záznamů")

    # Auto-detekce polí
    if text_field is None:
        text_field = _detect_text_field(records)
        if text_field is None:
            logger.error(
                f"Nelze auto-detekovat textové pole v {path}. "
                f"Zadej --json-text-field explicitně."
            )
            return

    if id_field is None:
        id_field = _detect_id_field(records)

    dataset_name = path.stem  # název souboru bez přípony jako jméno datasetu

    produced = 0
    skipped = 0

    for idx, rec in enumerate(records):
        if limit is not None and produced >= limit:
            break

        # Primární text
        raw_text = rec.get(text_field, "") if text_field else ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text) if raw_text is not None else ""
        text = strip_html(raw_text)

        if len(text) < GENERIC_MIN_TEXT_LEN:
            skipped += 1
            continue

        # ID záznamu
        if id_field and id_field in rec:
            record_id = str(rec[id_field])
        else:
            record_id = str(idx)

        # Kontext z dalších polí
        ctx_parts = []
        if context_fields:
            for cf in context_fields:
                val = rec.get(cf)
                if val and isinstance(val, str):
                    cleaned = strip_html(val)
                    if cleaned:
                        ctx_parts.append(f"{cf}: {cleaned}")
        context = "\n".join(ctx_parts) if ctx_parts else None

        chunks = split_into_chunks(text)
        for i, chunk in enumerate(chunks):
            base_id = f"{dataset_name}:{record_id}"
            chunk_id = base_id if len(chunks) == 1 else f"{base_id}#chunk{i}"
            yield LabelItem(
                item_id=chunk_id,
                source=dataset_name,
                domain=domain,
                text=chunk,
                context=context,
                is_synthetic=False,
                original_id=record_id,
                chunk_index=i if len(chunks) > 1 else None,
                total_chunks=len(chunks) if len(chunks) > 1 else None,
                extra={"source_file": str(path), "text_field": text_field},
            )
            produced += 1
            if limit is not None and produced >= limit:
                break

    logger.info(
        f"Generický dataset {dataset_name}: vyprodukováno {produced} label items, "
        f"přeskočeno {skipped}"
    )
