"""Data enrichment: Glottolog info, morphological derivatives."""
from __future__ import annotations
import json
from pathlib import Path
import pymorphy3

DATA_DIR = Path(__file__).parent.parent.parent / "data"

morph = pymorphy3.MorphAnalyzer()

_glottolog_cache: dict | None = None


def _load_glottolog() -> dict:
    global _glottolog_cache
    if _glottolog_cache is None:
        path = DATA_DIR / "glottolog_info.json"
        _glottolog_cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _glottolog_cache


def get_glottolog_info(donor_language: str) -> dict:
    """Return Glottolog metadata for the given donor language."""
    data = _load_glottolog()
    for key, val in data.items():
        if key.lower() == donor_language.lower():
            return val
    return {}


def get_morphological_derivatives(lemma: str) -> list[str]:
    """
    Attempt to find common derivative forms in Russian.
    Uses simple suffix-based heuristics.
    """
    lemma = lemma.lower().strip()
    derivatives: list[str] = []

    # For nouns ending in -инг → try -инговый (adjective)
    if lemma.endswith("инг"):
        base = lemma[:-3]
        derivatives.append(f"{base}инговый")
        derivatives.append(f"{base}ингом")

    # For nouns ending in -ция → try -ционный
    if lemma.endswith("ция"):
        base = lemma[:-3]
        derivatives.append(f"{base}ционный")
        derivatives.append(f"{base}циям")

    # For nouns ending in -изм → try -ист
    if lemma.endswith("изм"):
        base = lemma[:-3]
        derivatives.append(f"{base}ист")
        derivatives.append(f"{base}истский")

    # For nouns ending in -ер → try -ерский, -ерство
    if lemma.endswith("ер"):
        derivatives.append(f"{lemma}ский")
        derivatives.append(f"{lemma}ство")

    # For nouns ending in -ент → try -ентный
    if lemma.endswith("ент"):
        derivatives.append(f"{lemma}ный")

    # For nouns ending in consonant → try -овый (adjective), -ировать (verb)
    cyrillic_consonants = "бвгджзйклмнпрстфхцчшщ"
    if lemma and lemma[-1] in cyrillic_consonants:
        # verbal derivative
        verb_candidate = f"{lemma}ировать"
        derivatives.append(verb_candidate)

    # Validate: only return forms that pymorphy2 considers valid Russian words
    validated = []
    for d in derivatives:
        parses = morph.parse(d)
        if parses and parses[0].is_known:
            validated.append(d)
        elif d not in validated:
            # include even if not in dict — it might be a real word
            if len(validated) < 4:
                validated.append(d)

    return validated[:4]


def get_word_card(
    word: str,
    lemma: str,
    morph_info: dict,
    prediction: dict,
    glottolog: dict,
    derivatives: list[str],
    explanation: list[str],
) -> dict:
    """Assemble the complete word card response."""
    from .explainer import get_semantic_domain_label, DONOR_LANG_DISPLAY

    donor_lang = prediction.get("donor_language", "Unknown")
    semantic_domain = prediction.get("semantic_domain", "")

    return {
        "word": word,
        "lemma": lemma,
        "pos": morph_info.get("POS", "UNKN"),
        "gender": morph_info.get("gender"),
        "number": morph_info.get("number"),
        "is_declinable": morph_info.get("declinable", True),
        "loanword_probability": prediction.get("loanword_probability", 0.5),
        "is_loanword": prediction.get("is_loanword", False),
        "donor_language": donor_lang,
        "donor_language_ru": DONOR_LANG_DISPLAY.get(donor_lang, donor_lang),
        "donor_family": prediction.get("donor_family", glottolog.get("family", "")),
        "donor_subfamily": glottolog.get("subfamily", ""),
        "source_word": prediction.get("source_word", ""),
        "semantic_domain": semantic_domain,
        "semantic_domain_ru": get_semantic_domain_label(semantic_domain),
        "explanation": explanation,
        "morphological_derivatives": derivatives,
        "glottolog": glottolog,
        "top_donors": prediction.get("top_donors", []),
        "in_seed": prediction.get("in_seed", False),
    }
