"""
OWL ontology export pro silver dataset — generuje se POUZE v rámci
production smoke testu (`smoke-test --phase finalize`).

Postavené přes `rdflib` (žádné hand-written serializery). Výstup je
formálně OWL 2 ontologie obsahující:

  TBox (schéma):
    - owl:Ontology declaration s metadata
    - owl:Class :Document
    - owl:Class :Entity (+ owl:Class subclasses pro každý label —
      :PRODUKT, :ORGANIZACE, :KEYPHRASE_BENEFIT, …)
    - owl:ObjectProperty per relation type (:issued_by, :covers, :part_of, …)
      s rdfs:domain :Entity a rdfs:range :Entity
    - owl:ObjectProperty :hasEntity (Document → Entity)
    - owl:DatatypeProperty pro :text, :lemma, :category, :startChar,
      :endChar, :sourceDataset, :inputText, :criticStatus, :wasRevised

  ABox (data):
    - každý FinalLabel record = owl:NamedIndividual typu :Document
    - každá entity = owl:NamedIndividual typu konkrétního labelu (subclass :Entity)
    - každá relace = předikát mezi dvěma entity individuals

Výstupní formáty (oba generujeme):
  - `.owl` (RDF/XML — kanonický OWL serialization)
  - `.ttl` (Turtle — čitelnější, stejná data)

Volitelně: pokud je nainstalováno `pylode`, vygeneruje se HTML
dokumentace ontologie (`.html`).

Filtrace anti-halucinace (stejná jako JSONL):
  - Label regex `^[A-Z][A-Z0-9_]{0,49}$`
  - Relation regex `^[a-z][a-z0-9_]{0,49}$`
  - Self-loop drop
  - Dangling subject/object drop

Architektura: VŠECHNO se sestaví v rdflib.Graph a knihovna sama
serializuje. Žádné hand-written escaping ani URI manipulace.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

VOCAB_NS_STR = "http://mudrc.example/vocab#"
DOC_NS_STR = "urn:mudrc:doc:"
ENT_NS_STR = "urn:mudrc:ent:"
ONTOLOGY_IRI = "http://mudrc.example/ontology"

VOCAB = Namespace(VOCAB_NS_STR)
DOC = Namespace(DOC_NS_STR)
ENT = Namespace(ENT_NS_STR)


# ---------------------------------------------------------------------------
# Schema validation (anti-halucinace)
# ---------------------------------------------------------------------------

_VALID_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,49}$")
_VALID_RELATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,49}$")
_INVALID_URI_CHARS = re.compile(r"[^A-Za-z0-9._\-:#]")


def _is_valid_label(label: str) -> bool:
    return bool(label) and bool(_VALID_LABEL_RE.match(label))


def _is_valid_relation(rel: str) -> bool:
    return bool(rel) and bool(_VALID_RELATION_RE.match(rel))


def _safe_uri_component(value: str) -> str:
    """
    Sanitize string pro URI komponent. rdflib akceptuje URI s mezerou v
    `URIRef(...)`, ale lepší je explicit escape — vyhneme se ambiguitě
    při serializaci. Empty / whitespace → "anon".
    """
    stripped = (value or "").strip()
    if not stripped:
        return "anon"
    return _INVALID_URI_CHARS.sub("_", stripped)


def _doc_uri(item_id: str) -> URIRef:
    return URIRef(DOC_NS_STR + _safe_uri_component(item_id))


def _ent_uri(item_id: str, entity_id: str) -> URIRef:
    return URIRef(
        f"{ENT_NS_STR}{_safe_uri_component(item_id)}:{_safe_uri_component(entity_id)}"
    )


# ---------------------------------------------------------------------------
# Sběr labelů + relací z records (pro TBox build)
# ---------------------------------------------------------------------------


def collect_labels_and_relations(records: Iterable[dict]) -> tuple[set[str], set[str]]:
    """Posbírá použité validní labely + relation predikáty z all records."""
    labels: set[str] = set()
    relations: set[str] = set()
    for rec in records:
        graph = rec.get("graph") or {}
        for e in graph.get("entities") or []:
            cat = e.get("category")
            if cat and _is_valid_label(cat):
                labels.add(cat)
        for r in graph.get("relations") or []:
            rel = r.get("relation")
            if rel and _is_valid_relation(rel):
                relations.add(rel)
    return labels, relations


# ---------------------------------------------------------------------------
# TBox: postaví schema do rdflib.Graph
# ---------------------------------------------------------------------------


def _add_tbox(
    g: Graph,
    labels: Iterable[str],
    relations: Iterable[str],
    ontology_iri: str = ONTOLOGY_IRI,
    ontology_label: str = "Mudrc Labeler Silver Dataset Ontology",
    ontology_comment: Optional[str] = None,
) -> URIRef:
    """
    Přidá kompletní TBox (OWL schéma) do grafu `g`.

    Vrátí URIRef ontologie pro pozdější použití (např. version info).
    """
    # Bind prefixy (pro pěknou serializaci do Turtle)
    g.bind("", VOCAB)
    g.bind("doc", DOC)
    g.bind("ent", ENT)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Ontology declaration
    ont = URIRef(ontology_iri)
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal(ontology_label, lang="en")))
    if ontology_comment:
        g.add((ont, RDFS.comment, Literal(ontology_comment, lang="en")))
    else:
        g.add((ont, RDFS.comment, Literal(
            "Auto-generated from Mudrc Labeler smoke test. Contains Document "
            "and Entity classes (subclasses per NER label) + ObjectProperty "
            "per relation type. Conforms to OWL 2 DL.", lang="en"
        )))

    # === Core classes ===========================================================
    g.add((VOCAB.Document, RDF.type, OWL.Class))
    g.add((VOCAB.Document, RDFS.label, Literal("Document", lang="en")))
    g.add((VOCAB.Document, RDFS.comment, Literal(
        "Labelovaný vstupní dokument (text + entity + relace mezi entitami).",
        lang="cs"
    )))

    g.add((VOCAB.Entity, RDF.type, OWL.Class))
    g.add((VOCAB.Entity, RDFS.label, Literal("Entity", lang="en")))
    g.add((VOCAB.Entity, RDFS.comment, Literal(
        "Pojmenovaná entita (NER span). Subclass určuje konkrétní label.",
        lang="cs"
    )))

    # Document a Entity jsou disjointní
    g.add((VOCAB.Document, OWL.disjointWith, VOCAB.Entity))

    # Per-label subclasses of Entity
    for label in sorted(set(lb for lb in labels if _is_valid_label(lb))):
        cls = VOCAB[label]
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.subClassOf, VOCAB.Entity))
        g.add((cls, RDFS.label, Literal(label, lang="cs")))

    # === Object properties ======================================================
    g.add((VOCAB.hasEntity, RDF.type, OWL.ObjectProperty))
    g.add((VOCAB.hasEntity, RDFS.label, Literal("has entity", lang="en")))
    g.add((VOCAB.hasEntity, RDFS.comment, Literal(
        "Document obsahuje Entity (mention).", lang="cs"
    )))
    g.add((VOCAB.hasEntity, RDFS.domain, VOCAB.Document))
    g.add((VOCAB.hasEntity, RDFS.range, VOCAB.Entity))

    # Per-relation predicates
    for rel in sorted(set(r for r in relations if _is_valid_relation(r))):
        prop = VOCAB[rel]
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.label, Literal(rel, lang="en")))
        g.add((prop, RDFS.domain, VOCAB.Entity))
        g.add((prop, RDFS.range, VOCAB.Entity))

    # === Datatype properties ====================================================
    # Document
    for prop_name, label, datatype, domain in [
        ("sourceDataset", "source dataset", XSD.string, VOCAB.Document),
        ("inputText", "input text", XSD.string, VOCAB.Document),
        ("criticStatus", "critic status", XSD.string, VOCAB.Document),
        ("wasRevised", "was revised", XSD.boolean, VOCAB.Document),
        # Entity
        ("text", "entity surface form", XSD.string, VOCAB.Entity),
        ("lemma", "entity lemma", XSD.string, VOCAB.Entity),
        ("category", "entity label/category", XSD.string, VOCAB.Entity),
        ("startChar", "start char offset (inclusive)", XSD.nonNegativeInteger,
         VOCAB.Entity),
        ("endChar", "end char offset (exclusive)", XSD.nonNegativeInteger,
         VOCAB.Entity),
    ]:
        prop = VOCAB[prop_name]
        g.add((prop, RDF.type, OWL.DatatypeProperty))
        g.add((prop, RDFS.label, Literal(label, lang="en")))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, datatype))

    return ont


# ---------------------------------------------------------------------------
# ABox: data instances → rdflib.Graph
# ---------------------------------------------------------------------------


def _add_abox_record(
    g: Graph, record: dict, max_text_len: int = 5000,
) -> None:
    """
    Přidá jeden FinalLabel record do grafu jako OWL individuals.

    - Document individual + datatype properties
    - Entity individuals (validní label only)
    - hasEntity links
    - Per-relation predikáty mezi entitami (subject/object musí být v records,
      žádné self-loops, žádné nevalidní relation názvy)
    """
    item_id = record.get("item_id") or ""
    if not item_id:
        return
    source = record.get("source_dataset") or ""
    text = record.get("input_text") or ""
    if max_text_len > 0 and len(text) > max_text_len:
        text = text[:max_text_len] + "…"
    critic_status = record.get("critic_status") or ""
    was_revised = bool(record.get("was_revised", False))

    graph_data = record.get("graph") or {}
    entities = graph_data.get("entities") or []
    relations = graph_data.get("relations") or []

    doc_iri = _doc_uri(item_id)
    g.add((doc_iri, RDF.type, OWL.NamedIndividual))
    g.add((doc_iri, RDF.type, VOCAB.Document))
    g.add((doc_iri, VOCAB.sourceDataset, Literal(source)))
    if text:
        g.add((doc_iri, VOCAB.inputText, Literal(text)))
    if critic_status:
        g.add((doc_iri, VOCAB.criticStatus, Literal(critic_status)))
    g.add((doc_iri, VOCAB.wasRevised, Literal(was_revised)))

    # Entity individuals
    ent_uri_by_id: dict[str, URIRef] = {}
    for e in entities:
        eid = e.get("id")
        if not eid:
            continue
        category = e.get("category") or ""
        if not _is_valid_label(category):
            # Schema-invalid label — přeskočíme entity celou
            continue
        ent_iri = _ent_uri(item_id, eid)
        ent_uri_by_id[eid] = ent_iri
        g.add((ent_iri, RDF.type, OWL.NamedIndividual))
        g.add((ent_iri, RDF.type, VOCAB[category]))
        g.add((ent_iri, VOCAB.category, Literal(category)))
        if e.get("text"):
            g.add((ent_iri, VOCAB.text, Literal(e["text"])))
        if e.get("lemma"):
            g.add((ent_iri, VOCAB.lemma, Literal(e["lemma"])))
        sc, ec = e.get("start_char"), e.get("end_char")
        if isinstance(sc, int) and sc >= 0:
            g.add((ent_iri, VOCAB.startChar, Literal(sc, datatype=XSD.nonNegativeInteger)))
        if isinstance(ec, int) and ec >= 0:
            g.add((ent_iri, VOCAB.endChar, Literal(ec, datatype=XSD.nonNegativeInteger)))
        # Document → hasEntity → Entity
        g.add((doc_iri, VOCAB.hasEntity, ent_iri))

    # Relations: subject → predicate → object
    for r in relations:
        sid, oid = r.get("subjectId"), r.get("objectId")
        rel_name = r.get("relation") or ""
        if not _is_valid_relation(rel_name):
            continue
        if sid not in ent_uri_by_id or oid not in ent_uri_by_id:
            continue
        if sid == oid:
            continue
        g.add((ent_uri_by_id[sid], VOCAB[rel_name], ent_uri_by_id[oid]))


# ---------------------------------------------------------------------------
# Public API — build graph + serialize
# ---------------------------------------------------------------------------


def build_owl_graph(
    records: list[dict],
    *,
    ontology_iri: str = ONTOLOGY_IRI,
    ontology_label: str = "Mudrc Labeler Silver Dataset Ontology",
    ontology_comment: Optional[str] = None,
    max_text_len: int = 5000,
) -> Graph:
    """
    Vrátí kompletní rdflib.Graph obsahující TBox (schema) + ABox (data).

    Lze pak serializovat do libovolného formátu (Turtle, RDF/XML, JSON-LD).
    """
    g = Graph()
    labels, relations = collect_labels_and_relations(records)
    _add_tbox(
        g, labels, relations,
        ontology_iri=ontology_iri,
        ontology_label=ontology_label,
        ontology_comment=ontology_comment,
    )
    for rec in records:
        _add_abox_record(g, rec, max_text_len=max_text_len)
    return g


def write_owl_files(
    records: list[dict],
    base_path: Path,
    *,
    ontology_iri: str = ONTOLOGY_IRI,
    ontology_label: Optional[str] = None,
    max_text_len: int = 5000,
    write_ttl: bool = True,
    write_html: bool = True,
) -> dict:
    """
    Hlavní entry point: vyrobí OWL ontologii a serializuje do souborů.

    Vyrobí (pokud `write_ttl=True` i Turtle, pokud `write_html=True` a
    pyLODE je nainstalované, i HTML doc):

      <base>.owl   — RDF/XML (kanonický OWL serialization)
      <base>.ttl   — Turtle (čitelnější, stejná data)
      <base>.html  — pyLODE HTML doc (volitelně, jen pokud pylode importuje OK)

    `base_path` typicky `checkpoints/smoke/final_labels.owl` (přípona
    `.owl` se odstraní, doplníme všechny varianty).
    """
    g = build_owl_graph(
        records,
        ontology_iri=ontology_iri,
        ontology_label=ontology_label or "Mudrc Labeler Silver Dataset Ontology",
        max_text_len=max_text_len,
    )

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    stem_dir = base_path.parent / base_path.stem  # path bez .owl
    owl_path = stem_dir.with_suffix(".owl")
    ttl_path = stem_dir.with_suffix(".ttl")
    html_path = stem_dir.with_suffix(".html")

    # OWL (RDF/XML) — kanonický pro .owl extension
    g.serialize(destination=str(owl_path), format="pretty-xml")
    logger.info(f"OWL ontology: {owl_path} ({len(g)} triplů)")

    written = {
        "owl": owl_path.name,
        "triples": len(g),
        "records": len(records),
    }

    if write_ttl:
        g.serialize(destination=str(ttl_path), format="turtle")
        written["ttl"] = ttl_path.name
        logger.info(f"OWL ontology (Turtle): {ttl_path}")

    if write_html:
        html_written = _try_write_html_doc(owl_path, html_path)
        if html_written:
            written["html"] = html_path.name

    return written


def _try_write_html_doc(owl_path: Path, html_path: Path) -> bool:
    """
    Pokus o vygenerování HTML dokumentace ontologie přes pyLODE.

    pyLODE je VOLITELNÁ závislost (není v requirements.txt jako hard
    dep). Pokud nelze:
      - naimportovat (ImportError)
      - inicializovat (e.g. Windows pyproject.toml bug v pyLODE 3.x)
      - použít pro náš graf
    tiše vynecháme HTML doc — `.owl` a `.ttl` výstupy jsou tak jako tak.

    Vrátí True pokud HTML doc vznikl, jinak False.
    """
    OntDoc = None  # type: ignore
    try:
        # Některé pyLODE verze padají už při importu (např. Windows
        # FileNotFoundError na pyproject.toml v 3.2). Catch BaseException
        # opatrně — chceme jen failure detekci, ne KeyboardInterrupt.
        from pylode import OntDoc  # type: ignore  # noqa: F811
    except ImportError:
        logger.info(
            "pylode není nainstalováno (volitelná závislost) — "
            "skipping HTML dokumentaci. Instaluj `pip install pylode` pro HTML doc."
        )
        return False
    except Exception as e:
        logger.info(
            f"pylode import selhal ({type(e).__name__}: {e}); "
            f"skipping HTML dokumentaci. .owl/.ttl výstupy jsou OK."
        )
        return False

    try:
        od = OntDoc(ontology=str(owl_path))
        od.make_html(destination=str(html_path))
        logger.info(f"OWL ontology (HTML doc via pyLODE): {html_path}")
        return True
    except Exception as e:
        logger.warning(
            f"pyLODE selhalo při generování HTML ({type(e).__name__}: {e}); "
            f"HTML doc se nevyrobil. .owl/.ttl výstupy jsou OK."
        )
        return False


def write_per_source_owl(
    records: list[dict], base_path: Path,
    *, max_text_len: int = 5000,
    write_ttl: bool = True, write_html: bool = True,
) -> list[dict]:
    """
    Rozdělí records podle source_dataset; pro každý source vyrobí samostatnou
    OWL ontologii (.owl + volitelně .ttl + .html).

    Soubory pojmenované podle `base_path`: pokud `base_path` je
    `final_labels.owl`, výstup je `final_labels_<source>.owl`,
    `final_labels_<source>.ttl`, `final_labels_<source>.html` ve stejné
    složce.

    Vrátí list dict statistik per source.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        src = rec.get("source_dataset") or "unknown"
        src_safe = re.sub(r"[^A-Za-z0-9._\-]", "_", src) or "unknown"
        by_source[src_safe].append(rec)

    base_stem = base_path.stem
    base_dir = base_path.parent
    out_stats: list[dict] = []
    for src_safe, recs in sorted(by_source.items()):
        per_path = base_dir / f"{base_stem}_{src_safe}.owl"
        stats = write_owl_files(
            recs, per_path,
            ontology_label=f"Mudrc Labeler — {src_safe}",
            max_text_len=max_text_len,
            write_ttl=write_ttl,
            write_html=write_html,
        )
        stats["source"] = src_safe
        out_stats.append(stats)
    return out_stats


def export_owl(
    records: list[dict], base_jsonl_path: Path,
    *,
    per_source: bool = True,
    combined: bool = True,
    max_text_len: int = 5000,
    write_ttl: bool = True,
    write_html: bool = True,
) -> dict:
    """
    Hlavní entry point z finalize.py.

    `base_jsonl_path` — typicky `checkpoints/smoke/final_labels.jsonl`.
    Z toho odvodíme `final_labels.owl` a per-source split.

    Vrátí dict se statistikou (počty souborů, počty triplů, počty records).
    """
    base_owl_path = base_jsonl_path.with_suffix(".owl")
    stats: dict = {
        "records": len(records),
        "combined": None,
        "per_source": [],
    }

    if combined:
        sub = write_owl_files(
            records, base_owl_path,
            max_text_len=max_text_len,
            write_ttl=write_ttl,
            write_html=write_html,
        )
        stats["combined"] = sub

    if per_source:
        stats["per_source"] = write_per_source_owl(
            records, base_owl_path,
            max_text_len=max_text_len,
            write_ttl=write_ttl,
            write_html=write_html,
        )

    return stats


# ---------------------------------------------------------------------------
# Backward-compat aliases (předchozí .ttl-only API)
# ---------------------------------------------------------------------------

# Některé starší volání použily `export_rdf` / `write_combined_turtle`.
# Necháme alias, aby nic v pipeline nepuklo. Vnitřně dispatchuje na OWL.

def export_rdf(records, base_jsonl_path, **kwargs):
    """Deprecated alias pro export_owl. Zachováno pro backward compat."""
    return export_owl(records, base_jsonl_path, **kwargs)


def write_combined_turtle(records, out_path, max_text_len: int = 5000):
    """Deprecated alias — postaví OWL graf a zapíše jako Turtle (.ttl)."""
    g = build_owl_graph(records, max_text_len=max_text_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    return out_path
