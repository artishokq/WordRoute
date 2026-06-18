"""Glottolog lookup and morphological derivative generation."""
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
    data = _load_glottolog()
    for key, val in data.items():
        if key.lower() == donor_language.lower():
            return val
    return {}


def get_morphological_derivatives(lemma: str) -> list[str]:
    """Suffix-based heuristic to suggest common derivative forms."""
    lemma = lemma.lower().strip()
    derivatives: list[str] = []

    if lemma.endswith("инг"):
        base = lemma[:-3]
        derivatives.append(f"{base}инговый")
        derivatives.append(f"{base}ингом")

    if lemma.endswith("ция"):
        base = lemma[:-3]
        derivatives.append(f"{base}ционный")
        derivatives.append(f"{base}циям")

    if lemma.endswith("изм"):
        base = lemma[:-3]
        derivatives.append(f"{base}ист")
        derivatives.append(f"{base}истский")

    if lemma.endswith("ер"):
        derivatives.append(f"{lemma}ский")
        derivatives.append(f"{lemma}ство")

    if lemma.endswith("ент"):
        derivatives.append(f"{lemma}ный")

    cyrillic_consonants = "бвгджзйклмнпрстфхцчшщ"
    if lemma and lemma[-1] in cyrillic_consonants:
        derivatives.append(f"{lemma}ировать")

    validated = []
    for d in derivatives:
        parses = morph.parse(d)
        if parses and parses[0].is_known:
            validated.append(d)
        elif d not in validated:
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
