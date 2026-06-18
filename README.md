# WordRoute — анализ лексических заимствований в русском языке

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)

Учебный NLP-проект по компьютерной лингвистике. Принимает русское слово или текст и:

- определяет, является ли слово заимствованием;
- предсказывает язык-донор (английский, французский, немецкий, греко-латинский, тюркский, арабо-персидский, итальянский);
- объясняет предсказание через лингвистические признаки;
- анализирует морфологическую адаптацию слова в русском языке;
- проводит **probing-эксперимент**: кодируют ли предобученные multilingual embeddings информацию об истории слова.

---

## Содержание

- [Быстрый старт](#быстрый-старт)
- [NLP-пайплайн](#nlp-пайплайн)
- [Датасет](#датасет)
- [Обучение и метрики](#обучение-и-метрики)
- [Probing-эксперимент](#probing-эксперимент)
- [API](#api)
- [Технологии](#технологии)
- [Структура проекта](#структура-проекта)

---

## Быстрый старт

**Требования:** Python 3.11+, Node.js 18+

```bash
git clone https://github.com/ВАШ_НИК/WordRoute.git
cd WordRoute
```

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Обучение моделей — обязательно перед первым запуском
python train.py     # ~20 сек, выводит полный отчёт с метриками

# API-сервер
uvicorn app.main:app --reload --port 8000
```

```bash
# Frontend (новый терминал)
cd frontend
npm install
npm run dev         # → http://localhost:3000
```

---

## NLP-пайплайн

```
Входное слово / текст
        │
        ▼
  [Препроцессор]  ── pymorphy3
  │  лемматизация: маркетинговый → маркетинг
  │  POS-тег: NOUN / ADJF / VERB / ...
  │  грамматика: род, число, одушевлённость
  │  is_known: есть ли слово в словаре pymorphy3
  │  транслитерация: маркетинг → marketing
        │
        ▼
  [Feature Engineering]  ── 39 признаков
  │
  │  Графо-фонетические
  │    • наличие букв ф, дж, кс, э — редки в исконной лексике
  │    • суффиксы: -инг, -ция, -ия, -ер, -аж, -тор, -изм ...
  │    • длина слова, доля гласных, символьные н-граммы
  │
  │  Морфологические
  │    • POS one-hot, род, склоняемость, declinable/indeclinable
  │
  │  Фонетическое сходство  ── python-Levenshtein
  │    • нормированное расстояние до 7 донорских словарей
  │      (English / French / German / Greek / Turkic / Arabic / Italian)
        │
        ▼
  [L1 — бинарный классификатор]   заимствование / исконное
  │  Baseline: LogisticRegression + StandardScaler (sklearn Pipeline)
  │  Main:     CatBoost (auto_class_weights=SqrtBalanced)
  │  F1-weighted ≈ 0.91 | 5-fold CV F1 ≈ 0.90
        │
        ▼  (только для заимствований)
  [L2 — многоклассовый классификатор]   язык-донор
  │  Baseline: RandomForest (n_estimators=200)
  │  Main:     CatBoost MultiClass
  │  Top-1 ≈ 0.75 | Top-3 ≈ 0.93
        │
        ▼
  [Explainer]  правило-ориентированные объяснения на русском языке
  [Enricher]   данные Glottolog: языковая семья, регион
        │
        ▼
    FastAPI  →  Next.js UI
```

---

## Датасет

| Файл | Описание |
|------|----------|
| `backend/data/seed_dataset.csv` | **3660 слов** с метками: word, lemma, is_loanword, donor_language, donor_family, source_word, semantic_domain, confidence |
| `backend/data/donor_words/*.txt` | Словари донорских языков (используются для Левенштейна) |
| `backend/data/glottolog_info.json` | Метаданные Glottolog: языковая семья, регион |

**Распределение по классам:**

| Класс | Слов | Источник |
|-------|------|----------|
| Native (Slavic) | 1048 | Wiktionary inherited + Swadesh |
| English | 520 | Wiktionary + ручная разметка |
| Greek/Latin | 474 | Wiktionary + ручная разметка |
| French | 421 | Wiktionary + ручная разметка |
| German | 408 | Wiktionary + ручная разметка |
| Arabic/Persian | 297 | Wiktionary + ручная разметка |
| Italian | 241 | Wiktionary + ручная разметка |
| Dutch | 161 | Wiktionary |
| Turkic | 90 | Wiktionary + ручная разметка |

Датасет собирался в два этапа: 627 слов вручную (с `source_word` и `semantic_domain`), затем ~3000 автоматически через `build_dataset.py` из English Wiktionary API по категориям этимологии.

---

## Обучение и метрики

Запустить из `backend/`:

```bash
python train.py
```

Скрипт выводит:
- распределение классов и визуализацию доноров
- **L1**: classification report (precision / recall / F1) для LogReg и CatBoost, 5-fold CV, confusion matrix, топ-15 признаков по важности
- **L2 flat**: per-donor classification report, Top-1 / Top-3 accuracy, confusion matrix
- **L2 hierarchical**: двухуровневая классификация (семья → конкретный язык)
- анализ ошибок: ложные положительные и ложные отрицательные
- сохранение артефактов в `models_cache/`

**Метрики на 3660 словах (Wiktionary + ручная разметка):**

| Задача | Модель | CV F1 | Accuracy |
|--------|--------|-------|----------|
| L1: Borrowed vs Native | CatBoost | **0.848** | 0.839 |
| L2: Donor language (flat) | CatBoost | 0.442 | 0.426 |
| L2: Family (Stage A) | CatBoost | **0.547** | 0.564 |
| L2: Language within family (Stage B) | CatBoost | — | **0.750** |
| L2: Top-3 accuracy | — | — | **0.732** |

> Flat L2 (0.44 F1) уступает иерархическому подходу потому, что
> Germanic-языки (En/De/Nl) и Romance (Fr/It) используют схожие
> паттерны адаптации. Семейная классификация первым уровнем
> устраняет это смешение: Stage B в рамках семьи достигает 75%.

**Сохранённые артефакты:**

```
models_cache/
├── l1_model.joblib        # L1-классификатор (CatBoost)
├── l2_model.joblib        # L2-классификатор (CatBoost)
├── label_encoder.joblib   # кодировщик языков-доноров
├── feature_names.joblib   # имена 39 признаков
└── seed_lookup.joblib     # O(1)-поиск по 3660 словам
```

При старте API-сервер загружает готовые модели из `models_cache/` — повторное обучение не нужно.

---

## Probing-эксперимент

**Исследовательский вопрос:** кодируют ли multilingual embeddings сигнал об этимологии русских слов?

```bash
python probe.py           # первый запуск скачивает модель ~120 MB
python probe.py --no-plot # без PCA-графиков
```

**Метод:**
1. Загружается `paraphrase-multilingual-MiniLM-L12-v2` (SBERT, dim=384)
2. Все 627 слов кодируются в векторы (кэшируются в `models_cache/`)
3. На embeddings обучается линейный зонд (LogisticRegression)
4. Результат сравнивается с зондом на ручных признаках

**Результаты (5-fold CV F1):**

| Метод | L1 (binary) | L2 (donor) |
|-------|:-----------:|:----------:|
| Embeddings только | 0.871 | 0.541 |
| Ручные признаки | 0.893 | **0.730** |
| Embeddings + Ручные | **0.908** | 0.684 |

**Вывод:** Multilingual embeddings частично кодируют информацию о происхождении слова (L1 ≈ паритет), однако для определения конкретного языка-донора (L2) ручные морфо-фонетические признаки значительно эффективнее. Комбинация даёт наилучший результат по L1, что подтверждает гипотезу о взаимодополняющей природе двух типов признаков.

PCA-визуализации сохраняются в `models_cache/probe_pca_binary.png` и `probe_pca_donor.png`.

---

## API

Базовый URL: `http://localhost:8000`

### `POST /api/analyze`

```json
{
  "text": "маркетинг и менеджмент",
  "detailed": true
}
```

Ответ:
```json
{
  "results": [{
    "word": "маркетинг",
    "lemma": "маркетинг",
    "is_loanword": true,
    "loanword_probability": 0.97,
    "donor_language": "English",
    "donor_family": "Indo-European/Germanic",
    "source_word": "marketing",
    "top_donors": [
      {"language": "English", "probability": 0.89},
      {"language": "German",  "probability": 0.06}
    ],
    "features": { "has_ing": 1, "len_word": 9, "lev_english": 0.12 },
    "explanations": ["Суффикс -инг — характерен для английских заимствований"]
  }]
}
```

### `GET /api/word/{word}`

Анализ одного слова.

### `GET /health`

```json
{ "status": "ok", "model_source": "cache" }
```

Документация Swagger: `http://localhost:8000/docs`

---

## Технологии

| Слой | Инструменты |
|------|-------------|
| Морфология | pymorphy3 |
| Фонетика | python-Levenshtein |
| ML | scikit-learn, CatBoost |
| Probing | sentence-transformers, matplotlib |
| Данные | pandas, numpy |
| API | FastAPI, Pydantic, uvicorn |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Сериализация | joblib |

---

## Структура проекта

```
WordRoute/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py         # FastAPI эндпоинты
│   │   ├── core/
│   │   │   ├── preprocessor.py   # pymorphy3, транслитерация
│   │   │   ├── features.py       # 39 лингвистических признаков
│   │   │   ├── classifier.py     # L1/L2, загрузка из кэша
│   │   │   ├── explainer.py      # объяснения предсказаний
│   │   │   └── enricher.py       # Glottolog, морфологические производные
│   │   └── main.py               # точка входа FastAPI
│   ├── data/
│   │   ├── seed_dataset.csv      # 627 размеченных слов
│   │   ├── donor_words/          # словари для Левенштейна
│   │   └── glottolog_info.json   # метаданные Glottolog
│   ├── models/
│   │   └── schemas.py            # Pydantic-схемы запросов/ответов
│   ├── models_cache/             # сохранённые модели (в .gitignore)
│   ├── build_dataset.py          # автосбор данных из English Wiktionary API
│   ├── train.py                  # обучение + полный отчёт с метриками
│   ├── probe.py                  # probing-эксперимент с SBERT embeddings
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── components/
    │   │   ├── InputForm.tsx
    │   │   ├── WordCard.tsx
    │   │   ├── ResultsTable.tsx
    │   │   └── DonorChart.tsx
    │   ├── layout.tsx
    │   └── page.tsx
    └── lib/
        ├── types.ts
        └── api.ts
```
