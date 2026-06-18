"""Rule-based explanation generator for borrowing predictions."""
from __future__ import annotations

SUFFIX_EXPLANATIONS = {
    "sfx_ing": "содержит суффикс -инг (характерен для английских заимствований)",
    "sfx_tsiya": "содержит суффикс -ция (латинское/французское происхождение)",
    "sfx_atsiya": "содержит суффикс -ация (латинское происхождение, очень продуктивный)",
    "sfx_izm": "содержит суффикс -изм (философский/научный термин из греческого/латыни)",
    "sfx_ist": "содержит суффикс -ист (из греческого через западные языки)",
    "sfx_ment": "содержит суффикс -мент (французское или латинское происхождение)",
    "sfx_er_end": "оканчивается на -ер (характерно для английских и немецких заимствований)",
    "sfx_azh": "содержит суффикс -аж (французское происхождение)",
    "sfx_or_end": "оканчивается на -ор (латинское или итальянское происхождение)",
    "sfx_tor": "содержит суффикс -тор (латинское происхождение)",
    "sfx_siya": "содержит суффикс -сия (латинское происхождение)",
    "sfx_iya": "оканчивается на -ия (греческое/латинское происхождение, научная/религиозная лексика)",
    "sfx_ika": "оканчивается на -ика (греческое происхождение, научная терминология)",
    "sfx_al": "оканчивается на -аль (французское или латинское происхождение)",
}

LETTER_EXPLANATIONS = {
    "has_f": "содержит букву «ф» — редкая для исконно русских слов",
    "has_dj": "содержит сочетание «дж» — нехарактерно для исконной лексики",
    "has_ae": "содержит сочетание «аэ/эа» — признак иноязычного происхождения",
    "has_ex": "начинается с «экс-» (латинский префикс)",
}

LEVENSHTEIN_EXPLANATIONS = {
    "english": "фонетически близко к английскому слову",
    "french": "фонетически близко к французскому слову",
    "german": "фонетически близко к немецкому слову",
    "greek_latin": "фонетически близко к греческому или латинскому слову",
    "turkic": "фонетически близко к тюркскому слову",
    "arabic_persian": "фонетически близко к арабскому или персидскому слову",
    "italian": "фонетически близко к итальянскому слову",
}

MORPHO_EXPLANATIONS = {
    "is_declinable_false": "слово несклоняемо, что характерно для недавних заимствований",
    "gender_neut_true": "средний род у несклоняемого существительного — типичная черта заимствований",
    "is_known_false": "слово отсутствует в морфологическом словаре — возможно, новое заимствование",
}

DONOR_LANG_DISPLAY = {
    "English": "английский",
    "French": "французский",
    "German": "немецкий",
    "Greek/Latin": "греческий/латынь",
    "Arabic/Persian": "арабский/персидский",
    "Turkic": "тюркские языки",
    "Italian": "итальянский",
    "Dutch": "нидерландский",
    "Slavic": "славянское происхождение",
    "Unknown": "неизвестный язык",
}

LEV_THRESHOLD = 0.35  # distances below this = "close"


def generate_explanation(
    word: str,
    lemma: str,
    feats: dict,
    loanword_prob: float,
    donor_language: str,
    source_word: str,
    in_seed: bool,
) -> list[str]:
    """
    Build a human-readable list of reasons for the borrowing prediction.
    """
    reasons: list[str] = []

    if not feats:
        return ["недостаточно данных для объяснения"]

    # ── suffix / pattern evidence ──────────────────────────────────────────────
    for key, explanation in SUFFIX_EXPLANATIONS.items():
        if feats.get(key, 0) == 1:
            reasons.append(explanation)

    # ── letter pattern evidence ────────────────────────────────────────────────
    for key, explanation in LETTER_EXPLANATIONS.items():
        if feats.get(key, 0) == 1:
            reasons.append(explanation)

    # ── Levenshtein proximity ──────────────────────────────────────────────────
    lev_key = f"lev_{donor_language.lower().replace('/', '_').replace(' ', '_')}"
    # Map donor to feature key
    lang_key_map = {
        "English": "lev_english",
        "French": "lev_french",
        "German": "lev_german",
        "Greek/Latin": "lev_greek_latin",
        "Arabic/Persian": "lev_arabic_persian",
        "Turkic": "lev_turkic",
        "Italian": "lev_italian",
    }
    lev_key = lang_key_map.get(donor_language, "")
    if lev_key and feats.get(lev_key, 1.0) < LEV_THRESHOLD:
        lang_short = donor_language.lower().split("/")[0].strip()
        reasons.append(LEVENSHTEIN_EXPLANATIONS.get(lang_short.replace(" ", "_"), f"фонетически близко к {lang_short}"))

    if source_word:
        reasons.append(f"близко к слову в языке-доноре: «{source_word}»")

    # ── morphological evidence ─────────────────────────────────────────────────
    if not feats.get("is_declinable", 1):
        reasons.append(MORPHO_EXPLANATIONS["is_declinable_false"])
    if feats.get("gender_neut", 0) and not feats.get("is_declinable", 1):
        reasons.append(MORPHO_EXPLANATIONS["gender_neut_true"])
    if not feats.get("is_known", 1):
        reasons.append(MORPHO_EXPLANATIONS["is_known_false"])

    # ── fallback for native words ──────────────────────────────────────────────
    if loanword_prob < 0.4 and not reasons:
        reasons.append("слово не содержит характерных маркеров заимствования")
        reasons.append("соответствует типичной фонетической структуре исконной лексики")
        reasons.append("хорошо вписано в русскую морфологию")

    # ── source database note ───────────────────────────────────────────────────
    if in_seed:
        reasons.append("слово найдено в эталонной базе данных (Wiktionary/WOLD)")

    return reasons[:7]  # cap at 7 reasons


def get_semantic_domain_label(domain: str) -> str:
    translations = {
        "technology": "технологии",
        "business": "бизнес / экономика",
        "finance": "финансы",
        "food": "еда / кулинария",
        "clothing": "одежда / мода",
        "military": "военное дело",
        "science": "наука",
        "medicine": "медицина",
        "art": "искусство / культура",
        "music": "музыка",
        "sport": "спорт",
        "media": "медиа / коммуникация",
        "politics": "политика",
        "religion": "религия",
        "transport": "транспорт",
        "commerce": "торговля",
        "social": "социальные сети",
        "fashion": "мода",
        "education": "образование",
        "industry": "промышленность",
        "labor": "ремесло / труд",
        "tourism": "туризм",
        "architecture": "архитектура",
        "navigation": "навигация / мореплавание",
        "chemistry": "химия",
        "general": "общая лексика",
        "household": "бытовая лексика",
        "nature": "природа",
        "animal": "животные",
        "family": "семья",
        "body": "тело",
        "abstract": "абстрактная лексика",
        "action": "действие",
        "time": "время",
        "history": "история",
        "craft": "ремесло",
        "agriculture": "сельское хозяйство",
        "administration": "администрация",
        "law": "право",
        "typography": "типография",
        "furniture": "мебель",
        "psychology": "психология",
        "textile": "текстиль",
    }
    return translations.get(domain, domain)
