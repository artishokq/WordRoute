"""
WordRoute — Dataset Builder via Russian Wiktionary API
======================================================
Fetches Russian words grouped by donor language from ru.wiktionary.org
categories, normalises them with pymorphy3, and merges into seed_dataset.csv.

How it works:
  1. For each donor language, tries multiple Wiktionary category names
     (categories vary across Wiktionary versions)
  2. Fetches page titles from each working category (paginated API)
  3. Filters out proper nouns, phrases, abbreviations, etc.
  4. Runs pymorphy3 to confirm each word is a real lemma
  5. Native Slavic words are taken from the Swadesh list (authoritative)
  6. Merges with the existing seed_dataset.csv (no duplicates)
  7. Backs up the original before writing

Usage:
    cd backend
    source venv/bin/activate
    python build_dataset.py              # up to 350 words per donor language
    python build_dataset.py --limit 50   # quick test (fewer words)
    python build_dataset.py --dry-run    # preview stats without saving
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from app.core.preprocessor import analyze_word

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_PATH   = Path(__file__).parent / "data" / "seed_dataset.csv"
BACKUP_PATH = Path(__file__).parent / "data" / "seed_dataset.backup.csv"

# ─── Wiktionary API ───────────────────────────────────────────────────────────

WIKI_API      = "https://en.wiktionary.org/w/api.php"
REQUEST_DELAY = 2.0    # seconds — conservative to avoid 429s
MAX_RETRIES   = 4

# English Wiktionary has well-structured etymology categories for Russian words
CATEGORY_GROUPS: dict[str, list[str]] = {
    "English":        ["Russian terms derived from English"],
    "French":         ["Russian terms derived from French"],
    "German":         ["Russian terms derived from German"],
    "Greek/Latin":    ["Russian terms derived from Ancient Greek",
                       "Russian terms derived from Latin"],
    "Arabic/Persian": ["Russian terms derived from Arabic",
                       "Russian terms derived from Persian"],
    "Turkic":         ["Russian terms derived from Turkish",
                       "Russian terms derived from Proto-Turkic"],
    "Italian":        ["Russian terms derived from Italian"],
    "Dutch":          ["Russian terms derived from Dutch"],
}

# Wiktionary categories for native (inherited) Slavic words
NATIVE_CATEGORIES: list[str] = [
    "Russian terms inherited from Proto-Slavic",
    "Russian terms inherited from Old East Slavic",
    "Russian terms inherited from Proto-Balto-Slavic",
]

DONOR_FAMILY: dict[str, str] = {
    "English":        "Indo-European/Germanic",
    "French":         "Indo-European/Romance",
    "German":         "Indo-European/Germanic",
    "Greek/Latin":    "Indo-European/Classical",
    "Arabic/Persian": "Afro-Asiatic/Iranian",
    "Turkic":         "Turkic",
    "Italian":        "Indo-European/Romance",
    "Dutch":          "Indo-European/Germanic",
    "Slavic":         "Indo-European/Slavic",
}

# ─── Swadesh list — authoritative native Slavic words ─────────────────────────
# Source: standard 207-item Swadesh list for Russian (Swadesh 1952).
# All items are unambiguously Indo-European / native East Slavic vocabulary.

SWADESH_NATIVE: list[str] = [
    "я", "ты", "он", "мы", "вы", "они",
    "этот", "тот", "здесь", "там", "кто", "что", "где", "когда", "как",
    "все", "много", "один", "два", "большой", "длинный", "широкий",
    "толстый", "тяжёлый", "маленький", "короткий", "узкий", "тонкий",
    "женщина", "мужчина", "человек", "ребёнок", "жена", "муж",
    "мать", "отец", "зверь", "рыба", "птица", "собака", "змея",
    "червь", "дерево", "лес", "палка", "плод", "семя", "лист",
    "корень", "кора", "цветок", "трава", "верёвка", "кожа", "мясо",
    "кровь", "кость", "жир", "яйцо", "рог", "хвост", "перо",
    "волос", "голова", "ухо", "глаз", "нос", "рот", "зуб",
    "язык", "ноготь", "нога", "колено", "рука", "живот", "шея",
    "грудь", "сердце", "печень",
    "пить", "есть", "кусать", "видеть", "слышать", "знать",
    "думать", "нюхать", "бояться", "спать", "жить", "умереть",
    "убить", "ударить", "резать", "идти", "лежать", "сидеть",
    "стоять", "говорить", "петь", "давать", "брать", "жечь",
    "чёрный", "белый", "красный", "жёлтый", "зелёный",
    "полный", "новый", "хороший", "круглый", "сухой",
    "вода", "земля", "огонь", "дым", "пепел", "гора",
    "соль", "день", "ночь", "луна", "звезда", "море",
    "берег", "озеро", "река", "снег", "лёд", "ветер",
    "дождь", "камень", "песок", "пыль", "небо",
    "солнце", "облако", "гром", "молния",
    "год", "зима", "лето", "весна", "осень",
    "путь", "дорога", "дом", "поле", "деревня", "город",
    "имя", "слово", "язык",
    "нет", "да", "и", "или", "если", "потому",
    # Extended basic vocabulary
    "окно", "дверь", "стол", "стул", "кровать", "пол",
    "потолок", "стена", "крыша", "лестница",
    "нож", "ложка", "вилка", "чашка", "тарелка",
    "хлеб", "молоко", "масло", "яблоко", "груша",
    "вишня", "слива", "малина", "клубника",
    "морковь", "репа", "капуста", "чеснок", "горох",
    "пшеница", "рожь", "гречка",
    "кот", "лошадь", "корова", "свинья", "овца",
    "коза", "курица", "утка", "гусь",
    "дуб", "сосна", "берёза", "ель", "осина",
    "гриб", "трава", "мох", "болото",
    "нос", "рот", "спина",
    "брат", "сестра", "сын", "дочь", "дед", "баба",
    "друг", "враг", "господин",
    "белый", "серый", "синий",
    "спать", "плыть", "летать", "нести", "везти",
    "открыть", "закрыть", "найти", "потерять",
    "помнить", "забыть", "понять", "помочь", "ждать",
    "богатый", "бедный", "сложный", "простой", "трудный",
    "холодный", "тёплый", "горячий",
    "сладкий", "кислый", "горький", "солёный",
    "мокрый", "сухой", "чистый", "грязный",
    "старый", "молодой", "первый", "последний",
    "правый", "левый", "прямой",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

W = 62

def banner(title: str) -> None:
    print(); print("╔" + "═" * W + "╗")
    print("║" + title.center(W) + "║")
    print("╚" + "═" * W + "╝")

def section(title: str) -> None:
    print(); print("─" * W); print(f"  {title}"); print("─" * W)

def ok(msg: str)   -> None: print(f"  ✓  {msg}")
def info(msg: str) -> None: print(f"     {msg}")
def warn(msg: str) -> None: print(f"  ⚠  {msg}")

# ─── Wiktionary fetcher ───────────────────────────────────────────────────────

def _get_with_backoff(
    session: requests.Session,
    params: dict,
    attempt: int = 0,
) -> dict | None:
    """GET request with exponential backoff on 429 / network errors."""
    wait = REQUEST_DELAY * (2 ** attempt)
    time.sleep(wait)
    try:
        resp = session.get(WIKI_API, params=params, timeout=20)
        if resp.status_code == 429:
            if attempt < MAX_RETRIES:
                warn(f"    Rate-limited (429). Waiting {wait*2:.0f}s before retry {attempt+1}/{MAX_RETRIES}...")
                return _get_with_backoff(session, params, attempt + 1)
            warn("    Rate-limit retries exhausted.")
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        if attempt < MAX_RETRIES:
            return _get_with_backoff(session, params, attempt + 1)
        warn(f"    Request failed after {MAX_RETRIES} retries: {exc}")
        return None


def fetch_category(
    category: str,
    session: requests.Session,
    limit: int = 500,
) -> list[str]:
    """
    Fetch page titles (Russian words) from an English Wiktionary category.
    en.wiktionary.org uses namespace 0 for word entries.
    Returns empty list if the category is empty or not found.
    """
    words: list[str] = []
    params: dict = {
        "action":      "query",
        "list":        "categorymembers",
        "cmtitle":     f"Category:{category}",
        "cmlimit":     min(limit, 500),
        "cmtype":      "page",
        "cmnamespace": 0,
        "format":      "json",
    }

    while True:
        data = _get_with_backoff(session, params)
        if data is None:
            break

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            if m.get("ns") == 0:
                words.append(m["title"])

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont or len(words) >= limit:
            break
        params["cmcontinue"] = cont

    return words[:limit]


def category_exists(category: str, session: requests.Session) -> bool:
    """Quick probe: does this Wiktionary category have any members?"""
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 1,
        "format":  "json",
    }
    data = _get_with_backoff(session, params)
    if data is None:
        return False
    return bool(data.get("query", {}).get("categorymembers"))

# ─── Word validation ──────────────────────────────────────────────────────────

_CYRILLIC_ONLY = re.compile(r"^[а-яёА-ЯЁ\-]+$")

def is_valid_word(word: str) -> bool:
    """Accept only single lowercase Cyrillic words of reasonable length."""
    if not word or not _CYRILLIC_ONLY.match(word):
        return False
    if word[0].isupper():          # proper noun
        return False
    if " " in word:                # phrase
        return False
    if "-" in word and len(word.split("-")) > 2:   # complex hyphenated
        return False
    if not (2 <= len(word) <= 30):
        return False
    return True


def morph_is_lemma(word: str) -> bool:
    """
    Return True if pymorphy3 agrees this is a normal form (lemma) of itself,
    OR if pymorphy3 doesn't know the word at all (likely a new borrowing).
    """
    morph = analyze_word(word)
    if not morph.get("is_known", True):
        return True   # unknown word — keep it (probably a new loanword)
    return morph.get("lemma", word).lower() == word.lower()

# ─── Row builders ─────────────────────────────────────────────────────────────

def build_loanword_rows(
    words: list[str],
    donor_lang: str,
    existing_words: set[str],
) -> list[dict]:
    rows = []
    for word in words:
        w = word.lower().strip()
        if not is_valid_word(w) or w in existing_words:
            continue
        if not morph_is_lemma(w):
            continue
        morph = analyze_word(w)
        rows.append({
            "word":           w,
            "lemma":          morph.get("lemma", w),
            "is_loanword":    1,
            "donor_language": donor_lang,
            "donor_family":   DONOR_FAMILY.get(donor_lang, ""),
            "source_word":    "",
            "semantic_domain": "",
            "confidence":     0.88,
        })
        existing_words.add(w)
    return rows


def build_native_rows(existing_words: set[str]) -> list[dict]:
    rows = []
    for word in SWADESH_NATIVE:
        w = word.lower().strip()
        if not is_valid_word(w) or w in existing_words:
            continue
        morph = analyze_word(w)
        rows.append({
            "word":           w,
            "lemma":          morph.get("lemma", w),
            "is_loanword":    0,
            "donor_language": "Slavic",
            "donor_family":   "Indo-European/Slavic",
            "source_word":    "",
            "semantic_domain": "",
            "confidence":     0.04,
        })
        existing_words.add(w)
    return rows

# ─── Main ─────────────────────────────────────────────────────────────────────

def main(limit: int, dry_run: bool) -> None:
    banner("WordRoute — Wiktionary Dataset Builder")

    # ── Load existing dataset ──────────────────────────────────────────────────
    section("EXISTING DATASET")
    df_existing = pd.read_csv(DATA_PATH)
    existing_words: set[str] = set(df_existing["word"].str.lower())
    info(f"Loaded {len(df_existing)} existing words (will skip duplicates)")

    # ── Fetch from Wiktionary ──────────────────────────────────────────────────
    session = requests.Session()
    session.headers["User-Agent"] = "WordRoute/1.0 (NLP educational project; contact via GitHub)"

    new_rows: list[dict] = []
    fetch_stats: dict[str, int] = {}

    section(f"FETCHING FROM WIKTIONARY  (up to {limit} words per donor language)")

    for donor_lang, categories in CATEGORY_GROUPS.items():
        collected: list[str] = []
        tried: list[str] = []

        for cat in categories:
            if len(collected) >= limit:
                break

            time.sleep(REQUEST_DELAY)
            remaining = limit - len(collected)

            if not category_exists(cat, session):
                continue

            tried.append(cat)
            time.sleep(REQUEST_DELAY)
            words = fetch_category(cat, session, limit=remaining + 50)
            collected.extend(words)
            info(f"  [{donor_lang}] '{cat}': {len(words)} words")

        rows = build_loanword_rows(collected, donor_lang, existing_words)
        new_rows.extend(rows)
        fetch_stats[donor_lang] = len(rows)

        if rows:
            ok(f"{donor_lang:<20} +{len(rows)} new words added")
        else:
            warn(f"{donor_lang:<20} no new words found (all may already be in dataset)")

    # ── Native words: Wiktionary inherited categories + Swadesh ───────────────
    section("NATIVE WORDS  (Wiktionary inherited + Swadesh list)")

    native_wikt: list[str] = []
    for cat in NATIVE_CATEGORIES:
        time.sleep(REQUEST_DELAY)
        if not category_exists(cat, session):
            continue
        time.sleep(REQUEST_DELAY)
        words = fetch_category(cat, session, limit=limit)
        native_wikt.extend(words)
        info(f"  [Slavic] '{cat}': {len(words)} words")

    native_wikt_rows = build_loanword_rows.__wrapped__ if hasattr(build_loanword_rows, '__wrapped__') else None
    # Re-use build_native_rows for the Wiktionary inherited words
    wikt_native_rows: list[dict] = []
    for word in native_wikt:
        w = word.lower().strip()
        if not is_valid_word(w) or w in existing_words:
            continue
        morph = analyze_word(w)
        wikt_native_rows.append({
            "word":           w,
            "lemma":          morph.get("lemma", w),
            "is_loanword":    0,
            "donor_language": "Slavic",
            "donor_family":   "Indo-European/Slavic",
            "source_word":    "",
            "semantic_domain": "",
            "confidence":     0.06,
        })
        existing_words.add(w)

    new_rows.extend(wikt_native_rows)
    fetch_stats["Slavic (Wiktionary)"] = len(wikt_native_rows)

    # Swadesh as fallback / supplement
    swadesh_rows = build_native_rows(existing_words)
    new_rows.extend(swadesh_rows)
    fetch_stats["Slavic (Swadesh)"] = len(swadesh_rows)

    ok(f"{'Slavic total':<20} +{len(wikt_native_rows) + len(swadesh_rows)} new native words")

    # ── Stats preview ─────────────────────────────────────────────────────────
    section("RESULTS")
    total_new = len(new_rows)
    total_merged = len(df_existing) + total_new

    print(f"\n  Words in original dataset   : {len(df_existing)}")
    print(f"  New words fetched           : {total_new}")
    print(f"  Total after merge           : {total_merged}")
    print()
    print(f"  {'Donor language':<24} {'New words':>10}")
    print(f"  {'─'*24}  {'─'*10}")
    for lang, count in sorted(fetch_stats.items(), key=lambda x: -x[1]):
        print(f"  {lang:<24} {count:>10}")

    if dry_run:
        print()
        warn("DRY RUN — no files written. Remove --dry-run to save.")
        return

    if total_new == 0:
        warn("Nothing new to add. Dataset unchanged.")
        return

    # ── Backup original ───────────────────────────────────────────────────────
    import shutil
    shutil.copy(DATA_PATH, BACKUP_PATH)
    ok(f"Backup saved to {BACKUP_PATH.name}")

    # ── Merge and save ────────────────────────────────────────────────────────
    df_new = pd.DataFrame(new_rows, columns=[
        "word", "lemma", "is_loanword", "donor_language",
        "donor_family", "source_word", "semantic_domain", "confidence",
    ])
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset="word", keep="first")
    df_merged.to_csv(DATA_PATH, index=False)

    ok(f"Dataset saved: {len(df_merged)} words → {DATA_PATH.name}")

    # ── Donor distribution after merge ────────────────────────────────────────
    section("FINAL DONOR DISTRIBUTION")
    donor_counts = df_merged["donor_language"].value_counts()
    for lang, cnt in donor_counts.items():
        bar = "▓" * int(cnt / donor_counts.max() * 20)
        print(f"  {lang:<24} {cnt:>4}  {bar}")

    print()
    ok("Done. Run 'python train.py' to retrain models on the expanded dataset.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WordRoute dataset from Wiktionary")
    parser.add_argument(
        "--limit", type=int, default=350,
        help="Max new words per donor language category (default: 350)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview results without writing files",
    )
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
