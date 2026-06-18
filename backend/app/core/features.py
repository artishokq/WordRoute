"""Feature extraction for borrowing detection."""
from __future__ import annotations
import os
from pathlib import Path
from Levenshtein import distance as lev_distance
from .preprocessor import transliterate, count_vowels

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "donor_words"

DONOR_LANGUAGES = ["english", "french", "german", "greek_latin", "turkic", "arabic_persian", "italian"]

_donor_cache: dict[str, list[str]] = {}


def _load_donor_words(lang: str) -> list[str]:
    if lang in _donor_cache:
        return _donor_cache[lang]
    path = DATA_DIR / f"{lang}.txt"
    if not path.exists():
        _donor_cache[lang] = []
        return []
    words = [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _donor_cache[lang] = words
    return words


def _min_lev_normalized(word_translit: str, donor_words: list[str], max_check: int = 150) -> float:
    """Minimum normalized Levenshtein distance to a donor word list."""
    if not donor_words or not word_translit:
        return 1.0
    candidates = donor_words[:max_check]
    min_dist = min(
        lev_distance(word_translit, w) / max(len(word_translit), len(w), 1)
        for w in candidates
    )
    return round(min_dist, 4)


# Borrowing-indicator suffix patterns (Cyrillic)
SUFFIX_PATTERNS: dict[str, list[str]] = {
    "ing":    ["инг"],
    "tsiya":  ["ция"],
    "izm":    ["изм"],
    "ist":    ["ист"],
    "ment":   ["мент"],
    "er_end": ["ер"],
    "azh":    ["аж"],
    "or_end": ["ор"],
    "tor":    ["тор"],
    "siya":   ["сия"],
    "iya":    ["ия"],
    "ika":    ["ика"],
    "atsiya": ["ация"],
    "al":     ["аль"],
}

# Native Slavic suffix patterns — evidence AGAINST borrowing
# These are highly productive native derivational morphemes
NATIVE_SUFFIX_PATTERNS: dict[str, list[str]] = {
    "sfx_ost":    ["ость", "есть"],               # радость, смелость, верность
    "sfx_nost":   ["ность"],                       # верность, сложность (subset of ost)
    "sfx_enie":   ["ение", "ание", "яние"],        # движение, желание, знание
    "sfx_stvo":   ["ство", "ество"],               # богатство, братство
    "sfx_nik":    ["ник", "ница"],                 # работник, ученица
    "sfx_tel":    ["тель"],                        # учитель, писатель
    "sfx_ny_adj": ["ный", "ной", "ная", "ное",
                   "ний", "няя", "нее"],           # белый, добрый, синий
    "sfx_ovy":    ["овый", "евый", "овой", "евой"], # дубовый, весенний
}

# Germanic-language discriminators (шт/шн/шп clusters are >95% German loans)
GERMANIC_CLUSTERS = {
    "has_sht": "шт",   # Sturm→шторм, штраф, Staat→штат
    "has_shn": "шн",   # Schnur→шнур, шницель
    "has_shp": "шп",   # Spion→шпион, шпага
    "has_shv": "шв",   # Schwarz→шварц
}

LETTER_PATTERNS: dict[str, str] = {
    "has_f":  "ф",
    "has_dj": "дж",
    "has_ae": "аэ",
}


def _check_suffix(word: str, suffixes: list[str]) -> int:
    return int(any(word.endswith(s) for s in suffixes))


def extract_features(word: str, lemma: str, morph: dict) -> dict[str, float]:
    """
    Extract a feature vector for borrowing classification.
    Returns a flat dict of feature_name → numeric value.
    """
    w = lemma.lower() if lemma else word.lower()
    translit = transliterate(w)

    feats: dict[str, float] = {}

    # length
    feats["len_word"] = len(w)
    feats["num_vowels"] = count_vowels(w)
    feats["vowel_ratio"] = count_vowels(w) / max(len(w), 1)

    # borrowing suffix patterns
    for feat_name, suffixes in SUFFIX_PATTERNS.items():
        feats[f"sfx_{feat_name}"] = _check_suffix(w, suffixes)

    native_er = {"вечер", "ветер", "тетерев"}
    feats["sfx_er_end"] = int(w.endswith("ер") and w not in native_er)

    # native Slavic suffix patterns (evidence against borrowing)
    for feat_name, suffixes in NATIVE_SUFFIX_PATTERNS.items():
        feats[feat_name] = _check_suffix(w, suffixes)

    # Germanic cluster patterns (шт/шн/шп are almost exclusively German loans)
    for feat_name, cluster in GERMANIC_CLUSTERS.items():
        feats[feat_name] = int(cluster in w)

    # French: vowel + ёр/ер (актёр, суфлёр)
    feats["sfx_eur"] = int(
        (w.endswith("ёр") or w.endswith("ер"))
        and len(w) > 3
        and w[-3] in "аеёиоуыэюя"
    )
    # English: -мен (бизнесмен, джентльмен)
    feats["sfx_men"] = int(w.endswith("мен") and len(w) > 4)
    # French/Italian: -ир (командир, сувенир)
    feats["sfx_ir"] = int(w.endswith("ир") and len(w) > 3)

    # letter patterns
    feats["has_f"] = int("ф" in w)
    feats["has_dj"] = int("дж" in w)
    feats["has_ae"] = int("аэ" in w or "эа" in w)
    feats["has_ex"] = int(w.startswith("экс"))
    feats["has_double_vowel"] = int(
        any(w[i] == w[i + 1] for i in range(len(w) - 1) if w[i] in "аеёиоуы")
    )

    # morphological features from pymorphy3
    pos = morph.get("POS", "UNKN")
    feats["pos_noun"] = int(pos == "NOUN")
    feats["pos_adj"] = int(pos in ("ADJF", "ADJS"))
    feats["pos_verb"] = int(pos in ("VERB", "INFN"))
    feats["pos_other"] = int(pos not in ("NOUN", "ADJF", "ADJS", "VERB", "INFN"))
    feats["is_known"] = int(morph.get("is_known", True))
    feats["is_declinable"] = int(morph.get("declinable", True))
    feats["gender_neut"] = int(morph.get("gender") == "neut")
    feats["gender_masc"] = int(morph.get("gender") == "masc")

    # Levenshtein distance to donor word lists
    for lang in DONOR_LANGUAGES:
        donor_words = _load_donor_words(lang)
        feats[f"lev_{lang}"] = _min_lev_normalized(translit, donor_words)

    lev_scores = {lang: feats[f"lev_{lang}"] for lang in DONOR_LANGUAGES}
    feats["lev_best"] = min(lev_scores.values())
    feats["lev_en_vs_native"] = feats["lev_english"] - 0.5

    return feats


def get_feature_names() -> list[str]:
    """Return ordered list of feature names (matches extract_features output)."""
    dummy_morph = {"POS": "NOUN", "gender": "masc", "is_known": True, "declinable": True}
    sample = extract_features("тест", "тест", dummy_morph)
    return list(sample.keys())


def features_to_vector(feats: dict[str, float]) -> list[float]:
    """Convert feature dict to ordered numeric list."""
    names = get_feature_names()
    return [feats.get(n, 0.0) for n in names]
