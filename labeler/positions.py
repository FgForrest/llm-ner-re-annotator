"""
Kódový výpočet pozic entit (start_char, end_char) v původním textu.

PRINCIP (anti-halucinace):
  LLM nikdy neposílá numerické pozice — vrací pouze surface form (text) a
  lemma. Tato vrstva pozice DOPOČÍTÁ z původního textu pomocí substring
  vyhledání + ověření hranic slov.

  Důvod: LLM počítá indexy nespolehlivě (off-by-one, nesoulad s tokenizerem,
  halucinace u opakovaných slov). Server-side výpočet je deterministický a
  garantuje invarianty potřebné pro spaCy/Stanza dataset.

GARANTOVANÉ INVARIANTY pro spaCy/Stanza:
  a) text[start:end] == entity.text
     (případný case-insensitive fallback přepíše entity.text na přesný
     substring originálu, nikdy neposune indexy)
  b) span začíná i končí na hranici slova
     (žádné rozseknutí tokenu, kompatibilní se spaCy tokenizátorem)
  c) entity se nepřekrývají
     (greedy: vyhrává delší span, ties → dřívější výskyt)

Entity, které poruší a/b nebo prohrají c), jsou ZAHOZENY a kaskádově
se odstraní i relace, které je odkazují (FG162: nikdy nehádej pozice).
"""

from __future__ import annotations
import logging

from .schemas import SkoGraph

logger = logging.getLogger(__name__)


def _is_word_boundary(text: str, idx: int) -> bool:
    """True, pokud idx leží na hranici slova (start textu, konec textu,
    nebo přechod alfanum ↔ ne-alfanum). Aproximuje spaCy tokenizaci
    bez nutnosti pouštět tokenizer."""
    if idx <= 0 or idx >= len(text):
        return True
    return not (text[idx - 1].isalnum() and text[idx].isalnum())


def compute_entity_positions(text: str, graph: SkoGraph) -> None:
    """
    Doplní start_char a end_char pro každou entitu v grafu deterministickým
    výpočtem nad originálním textem.

    Vstup:
      text   — původní (HTML-stripped) text dokumentu
      graph  — SkoGraph s entitami z LLM. Případné start_char/end_char,
               které LLM vrátil, se IGNORUJÍ a přepíšou se vypočítanými
               hodnotami (nebo entita padá do drop_ids, viz níže).

    Postup:
      1. Pro každou entitu nejdřív zresetuje start_char/end_char na None
         (anti-halucinace: nedůvěřujeme LLM-provided pozicím).
      2. Najde první výskyt entity.text v textu (case-sensitive, fallback
         case-insensitive — v takovém případě přepíše entity.text na
         přesný substring originálu).
      3. Ověří, že začátek i konec spanu leží na hranici slova.
      4. Vyřeší překryvy: greedy preferuje delší span, ties → dřívější.
      5. Entity bez validní pozice ZAHODÍ a odstraní relace, které je
         odkazují (kaskádové čištění).

    Po návratu platí pro všechny zachované entity:
      - start_char a end_char jsou nastaveny (nikdy None)
      - text[start_char:end_char] == entity.text
      - span je validní pro spaCy/Stanza training (Doc.char_span by neměl
         vrátit None)

    Funkce mutuje graph in-place (entities, relations).
    """
    text_lower = text.lower()
    drop_ids: set[str] = set()

    for entity in graph.entities:
        # Anti-halucinace: zahodit cokoli, co LLM poslal jako pozice.
        entity.start_char = None
        entity.end_char = None

        if not entity.text:
            drop_ids.add(entity.id)
            continue

        def _find_boundary_match(haystack: str, sub: str) -> int:
            pos = 0
            while True:
                idx = haystack.find(sub, pos)
                if idx == -1:
                    return -1
                end = idx + len(sub)
                if _is_word_boundary(text, idx) and _is_word_boundary(text, end):
                    return idx
                pos = idx + 1

        idx = _find_boundary_match(text, entity.text)
        if idx == -1:
            # Fallback to case-insensitive
            idx = _find_boundary_match(text_lower, entity.text.lower())
            if idx >= 0:
                entity.text = text[idx : idx + len(entity.text)]
        
        if idx < 0:
            drop_ids.add(entity.id)
            continue

        entity.start_char = idx
        entity.end_char = idx + len(entity.text)

    # Vyřešit překryvy: delší span vyhrává, ties → dřívější.
    valid = [
        e for e in graph.entities
        if e.id not in drop_ids and e.start_char is not None and e.end_char is not None
    ]
    valid.sort(key=lambda e: (-(e.end_char - e.start_char), e.start_char))
    accepted: list = []
    for e in valid:
        if any(
            not (e.end_char <= a.start_char or e.start_char >= a.end_char)
            for a in accepted
        ):
            drop_ids.add(e.id)
        else:
            accepted.append(e)

    if drop_ids:
        logger.warning(
            f"Zahazuji {len(drop_ids)} entit (nedohledatelné / nesedí na hranici / "
            f"překryv): {sorted(drop_ids)}"
        )
        graph.entities = [e for e in graph.entities if e.id not in drop_ids]
        graph.relations = [
            r for r in graph.relations
            if r.subjectId not in drop_ids and r.objectId not in drop_ids
        ]
