"""Russian morphological preprocessor using pymorphy2."""
import re
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

STOP_WORDS = {
    "и", "в", "не", "на", "я", "быть", "с", "он", "а", "то", "что",
    "это", "по", "из", "у", "же", "как", "к", "но", "они", "мы",
    "за", "так", "от", "все", "о", "его", "она", "при", "если",
    "когда", "тот", "мой", "уже", "бы", "до", "или", "вот", "где",
    "ещё", "ни", "чем", "со", "без", "вы", "нет", "тут", "там",
    "ли", "для", "её", "об", "во", "то", "вс", "один", "два",
}

TARGET_POS = {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "ADVB"}

VOWELS = set("аеёиоуыэюя")


def transliterate(word: str) -> str:
    """Transliterate Russian word to approximate Latin form."""
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
        "й": "j", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
    return "".join(table.get(c, c) for c in word.lower())


def count_vowels(word: str) -> int:
    return sum(1 for c in word.lower() if c in VOWELS)


def analyze_word(word: str) -> dict:
    """Return pymorphy2 morphological analysis for a word."""
    parses = morph.parse(word)
    if not parses:
        return {"lemma": word, "POS": "UNKN", "is_known": False, "declinable": True}

    best = parses[0]
    tag = best.tag

    pos = tag.POS or "UNKN"
    gender = str(tag.gender) if tag.gender else None
    number = str(tag.number) if tag.number else None
    case = str(tag.case) if tag.case else None
    animacy = str(tag.animacy) if tag.animacy else None

    # Check declinability: if word has no inflection forms it's likely indeclinable
    forms = {f.word for f in best.lexeme}
    declinable = len(forms) > 1

    return {
        "lemma": best.normal_form,
        "POS": pos,
        "gender": gender,
        "number": number,
        "case": case,
        "animacy": animacy,
        "is_known": best.is_known,
        "declinable": declinable,
        "score": best.score,
    }


def tokenize(text: str) -> list[str]:
    """Extract Russian word tokens from text."""
    tokens = re.findall(r"[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*", text)
    return [t.lower() for t in tokens if len(t) >= 2]


def preprocess_text(text: str, keep_all_pos: bool = False) -> list[dict]:
    """
    Full preprocessing pipeline for Russian text.
    Returns list of token dicts with morphological info.
    """
    tokens = tokenize(text)
    results = []
    seen_lemmas: set[str] = set()

    for token in tokens:
        if token in STOP_WORDS:
            continue

        info = analyze_word(token)
        lemma = info["lemma"]

        if lemma in STOP_WORDS or lemma in seen_lemmas:
            continue

        if not keep_all_pos and info["POS"] not in TARGET_POS:
            continue

        seen_lemmas.add(lemma)
        info["token"] = token
        results.append(info)

    return results


def preprocess_word(word: str) -> dict:
    """Preprocess a single word and return full morphological info."""
    word = word.strip().lower()
    info = analyze_word(word)
    info["token"] = word
    return info
