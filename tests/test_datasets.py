"""
Unit testy pro labeler.datasets — Stage export reader (v1 + v2) a výběr jazyků.

Pokrývá:
  - detekce v2 exportu s dlouhým dataset.description (sniff hlavičky)
  - locale filtr: celá schémata (allowedLocales) i jednotlivá pole (locale)
  - vícejazyčné záznamy: verze per jazyk, stabilní item_id se suffixem @locale
  - REFERENCE pole (list ID), soubory (objectType edeeFile) → skip
  - per-soubor "locales" v labeler_config.json

Spuštění: python tests/test_datasets.py
"""

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import labeler.datasets as datasets_mod
from labeler.datasets import (
    _is_stage_export,
    _locale_selected,
    _stage_field_value,
    iter_stage_export,
    parse_locales,
    scan_data_dir,
)

SK_CONTENT = (
    "Testovací poťah ALFA je smyšlený produkt pre unit testy. Je z bavlnenej "
    "látky, dá sa prať a vyrába sa v piatich farbách."
)
CS_CONTENT = (
    "Testovací potah ALFA je smyšlený produkt pro unit testy. Je z bavlněné "
    "látky, dá se prát a vyrábí se v pěti barvách."
)
PL_CONTENT = (
    "Testowy pokrowiec ALFA to fikcyjny produkt do testów jednostkowych. Jest "
    "z bawełny, można go prać i jest dostępny w pięciu kolorach."
)


def _fld(code, value, locale=None):
    return {"code": code, "value": value, "locale": locale}


def _page(rec_id, fields, object_type="edeePage", prototype="page"):
    return {
        "id": rec_id,
        "fields": [
            _fld("externalId", rec_id),
            _fld("objectType", object_type),
            _fld("prototype", prototype),
            _fld("contentBlocks", ["501", "502"]),
            *fields,
        ],
        "related": [
            {
                "fieldCode": "contentBlocks",
                "targetSchemaCode": "sk",
                "records": [],
                "unresolved": [{"id": "501", "reason": "NOT_FOUND"}],
            }
        ],
    }


def _export(schemas, description="Data source containing demo e-shop content."):
    return {
        "version": 2,
        "mode": "EMBEDDED",
        "exportedAt": "2026-01-01T00:00:00Z",
        "dataset": {
            "name": "nabytek-demo",
            "path": "nabytek-demo",
            "description": description,
        },
        "schemas": schemas,
    }


def _sk_schema():
    return {
        "name": "sk",
        "code": "sk",
        "allowedLocales": ["sk"],
        "fields": [],
        "relations": [{"fieldCode": "contentBlocks", "targetSchemaCode": "sk"}],
        "records": [
            # soubor jako první záznam — nesmí rozhodit detekci typu schématu
            _page(
                "edeeFile_1",
                [_fld("description", "Popis súboru z publish metadat, dostatočne dlhý na produktový text.")],
                object_type="edeeFile",
            ),
            _page(
                "edeePage_100",
                [_fld("title", "Úvodná stránka", "sk"), _fld("content", SK_CONTENT, "sk")],
            ),
        ],
    }


def _multi_schema():
    return {
        "name": "blog",
        "code": "blog",
        "allowedLocales": ["cs", "sk", "pl"],
        "fields": [],
        "records": [
            _page(
                "edeePage_1",
                [
                    _fld("title", "Potahy", "cs"),
                    _fld("content", CS_CONTENT, "cs"),
                    _fld("title", "Poťahy", "sk"),
                    _fld("content", SK_CONTENT, "sk"),
                ],
            ),
            # existuje jen v polštině
            _page("edeePage_2", [_fld("content", PL_CONTENT, "pl")]),
        ],
    }


def _write(tmp: Path, data, name="export-nabytek-demo-latest.json") -> Path:
    p = tmp / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _items(data, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        return list(iter_stage_export(_write(Path(tmp), data), **kw))


# ---------- locale helpery -------------------------------------------------

def test_parse_locales():
    assert parse_locales("CS, sk") == frozenset({"cs", "sk"})
    assert parse_locales(["cs_CZ"]) == frozenset({"cs-cz"})
    assert parse_locales("") is None
    assert parse_locales([]) is None
    assert parse_locales(None) is None
    assert parse_locales("*") is None
    assert parse_locales("all") is None


def test_locale_selected_matches_region_variants():
    sel = parse_locales("cs")
    assert _locale_selected("cs", sel)
    assert _locale_selected("cs-CZ", sel)
    assert _locale_selected("cs_CZ", sel)
    assert not _locale_selected("sk", sel)
    assert _locale_selected("sk", None)


def test_reference_value_joined():
    assert _stage_field_value({"code": "x", "value": ["1", "2"]}) == "1, 2"
    assert _stage_field_value({"code": "x", "value": None}) is None


# ---------- detekce + routing ----------------------------------------------

def test_v2_export_with_long_description_is_detected():
    data = _export([_sk_schema()], description="x " * 400)  # > 512 znaků
    with tempfile.TemporaryDirectory() as tmp:
        assert _is_stage_export(_write(Path(tmp), data))


def test_scan_data_dir_routes_v2_to_stage_reader():
    data = _export([_sk_schema()], description="x " * 400)
    with tempfile.TemporaryDirectory() as tmp:
        _write(Path(tmp), data)
        items = list(scan_data_dir(Path(tmp), locales="*"))
    assert [it.item_id for it in items] == ["nabytek-demo:sk:edeePage_100"]
    assert items[0].text == SK_CONTENT
    assert items[0].extra["locale"] == "sk"


# ---------- filtr jazyků ---------------------------------------------------

def test_single_locale_schema_skipped_when_not_selected():
    assert _items(_export([_sk_schema()]), locales="cs") == []
    assert len(_items(_export([_sk_schema()]), locales="sk")) == 1


def test_file_object_type_skipped():
    items = _items(_export([_sk_schema()]))
    assert all("edeeFile" not in it.item_id for it in items)


def test_multi_locale_record_split_per_language():
    items = _items(_export([_multi_schema()]))
    by_id = {it.item_id: it for it in items}
    assert set(by_id) == {
        "nabytek-demo:blog:edeePage_1@cs",
        "nabytek-demo:blog:edeePage_1@sk",
        "nabytek-demo:blog:edeePage_2@pl",
    }
    assert by_id["nabytek-demo:blog:edeePage_1@cs"].text == CS_CONTENT
    assert by_id["nabytek-demo:blog:edeePage_1@sk"].text == SK_CONTENT


def test_multi_locale_filter_keeps_stable_ids():
    items = _items(_export([_multi_schema()]), locales="cs")
    assert [it.item_id for it in items] == ["nabytek-demo:blog:edeePage_1@cs"]
    assert items[0].text == CS_CONTENT
    assert items[0].context == "Titulek: Potahy"  # žádný slovenský titulek


def test_shared_nonlocalized_text_labeled_once():
    schema = {
        "code": "products",
        "allowedLocales": ["cs", "sk"],
        "records": [{
            "id": "p1",
            "fields": [
                _fld("externalId", "p1"),
                _fld("description", CS_CONTENT),
                _fld("name", "Potah", "cs"),
                _fld("name", "Poťah", "sk"),
            ],
        }],
    }
    items = _items(_export([schema]))
    assert len(items) == 1
    assert items[0].text == CS_CONTENT


def test_scan_data_dir_defaults_to_config_locales():
    # bez `locales` rozhoduje CONFIG.data_locales (env DATA_LOCALES, default "cs")
    data = _export([_sk_schema(), _multi_schema()])
    orig = datasets_mod.CONFIG
    datasets_mod.CONFIG = dataclasses.replace(orig, data_locales="cs")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp), data)
            items = list(scan_data_dir(Path(tmp)))
    finally:
        datasets_mod.CONFIG = orig
    assert [it.item_id for it in items] == ["nabytek-demo:blog:edeePage_1@cs"]


def test_per_file_config_overrides_global_locales():
    data = _export([_sk_schema(), _multi_schema()])
    with tempfile.TemporaryDirectory() as tmp:
        _write(Path(tmp), data)
        (Path(tmp) / "labeler_config.json").write_text(
            json.dumps({"export-nabytek-demo-latest.json": {"locales": ["sk"]}}),
            encoding="utf-8",
        )
        items = list(scan_data_dir(Path(tmp), locales="cs"))
    assert sorted(it.item_id for it in items) == [
        "nabytek-demo:blog:edeePage_1@sk",
        "nabytek-demo:sk:edeePage_100",
    ]


# ---------- v1 export (kosmetika-demo, bez mode/relations/related) --------------------

COSMETICS_CONTENT = (
    "Kosmetika Vzor je smyšlená značka pro unit testy. "
    "Testovací e-shop nabízí pleťové krémy, séra a pleťové vody. "
    "Všechny texty v tomto souboru jsou vymyšlené."
)


def _cosmetics_v1():
    def page(rec_id, title, prototype, url, content):
        return {
            "id": rec_id,
            "fields": [
                _fld("externalId", rec_id),
                _fld("title", title, "cs"),
                _fld("prototype", prototype),
                _fld("url", url),
                _fld("content", content, "cs"),
                _fld("changed", "2026-01-01T00:00:00Z"),
                _fld("objectType", "edeePage"),
            ],
        }

    return {
        "version": 1,
        "exportedAt": "2026-01-02T00:00:00Z",
        "dataset": {
            "name": "kosmetika-demo",
            "path": "kosmetika-demo",
            "description": (
                "Data source containing demo cosmetics e-shop content — products, "
                "categories and groups in Czech and Slovak mutations, plus CMS "
                "pages, blog articles and downloadable files."
            ),
        },
        "schemas": [
            {
                "name": "cs",
                "code": "cs",
                "allowedLocales": ["cs"],
                "fields": [],
                "records": [
                    page("edeePage_201", "Lišta", "alertStrip",
                         "/system/lista", "Lišta"),
                    page("edeePage_202", "Tým", "people",
                         "/system/tym", "Tým"),
                    page("edeePage_203", "Feed", "igFeed",
                         "/system/feed", "Feed"),
                    page("edeePage_204", "Informační lišta", "alertStrip",
                         "/system/lista/info",
                         "Informační lišta Nákup nad 999 Kč - doprava zdarma!"),
                    page("edeePage_205", "O nás", "page", "/o-nas", COSMETICS_CONTENT),
                ],
            },
            {
                "name": "products",
                "code": "kosmetika-products",
                "allowedLocales": ["cs", "sk"],
                "fields": [],
                "records": [{
                    "id": "123",
                    "fields": [
                        _fld("externalId", "123"),
                        _fld("name", "Pleťový krém", "cs"),
                        _fld("description", CS_CONTENT, "cs"),
                        _fld("name", "Pleťový krém", "sk"),
                        _fld("description", SK_CONTENT, "sk"),
                    ],
                }],
            },
        ],
    }


def test_v1_export_pages_and_ids_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), _cosmetics_v1(), name="export-kosmetika-demo-latest.json")
        assert _is_stage_export(path)
        items = list(scan_data_dir(Path(tmp), locales="cs"))
    # alertStrip = navigace, "Tým"/"Feed" = krátké → jen stránka O nás,
    # ID bez @locale suffixu (jednojazyčné schéma) = stejné jako před úpravou
    page_ids = [it.item_id for it in items if ":cs:" in it.item_id]
    assert page_ids == ["kosmetika-demo:cs:edeePage_205"]
    assert items[0].domain == datasets_mod.DEFAULT_STAGE_DOMAIN


def test_v1_multi_locale_products_split_and_filtered():
    all_items = _items(_cosmetics_v1())
    prod_ids = sorted(it.item_id for it in all_items if ":kosmetika-products:" in it.item_id)
    assert prod_ids == ["kosmetika-demo:kosmetika-products:123@cs", "kosmetika-demo:kosmetika-products:123@sk"]

    cs_items = _items(_cosmetics_v1(), locales="cs")
    cs_prod = [it for it in cs_items if ":kosmetika-products:" in it.item_id]
    assert [it.item_id for it in cs_prod] == ["kosmetika-demo:kosmetika-products:123@cs"]
    assert cs_prod[0].text == CS_CONTENT

    assert [it.item_id for it in _items(_cosmetics_v1(), locales="sk")] == [
        "kosmetika-demo:kosmetika-products:123@sk"
    ]


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
