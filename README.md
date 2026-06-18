# WordRoute — анализ лексических заимствований в русском языке

Python
FastAPI
Next.js
scikit-learn
CatBoost

NLP-проект по теоретичсекой лингвистике в контексте NLP. Принимает русское слово или текст и:

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
  [Feature Engineering]  ── 54 признака
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
  │  CV F1 ≈ 0.845 | Accuracy ≈ 0.846
        │
        ▼  (только для заимствований)
  [L2 — многоклассовый классификатор]   язык-донор
  │  Baseline: RandomForest (n_estimators=200)
  │  Main:     CatBoost MultiClass + иерархический (семья → язык)
  │  Top-1 ≈ 0.56 (flat) | Top-3 ≈ 0.775 | Stage B ≈ 0.798
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


| Файл                               | Описание                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `backend/data/seed_dataset.csv`    | **5063 слова** с метками: word, lemma, is_loanword, donor_language, donor_family, source_word, semantic_domain, confidence |
| `backend/data/donor_words/*.txt`   | Словари донорских языков (используются для Левенштейна)                                                                    |
| `backend/data/glottolog_info.json` | Метаданные Glottolog: языковая семья, регион                                                                               |


**Распределение по классам:**


| Класс            | Слов | Источник                       |
| ---------------- | ---- | ------------------------------ |
| Native (Slavic)  | 1459 | Wiktionary inherited + Swadesh |
| German (+ Dutch) | 809  | Wiktionary + ручная разметка   |
| English          | 780  | Wiktionary + ручная разметка   |
| Greek/Latin      | 739  | Wiktionary + ручная разметка   |
| French           | 648  | Wiktionary + ручная разметка   |
| Arabic/Persian   | 297  | Wiktionary + ручная разметка   |
| Italian          | 241  | Wiktionary + ручная разметка   |
| Turkic           | 90   | Wiktionary + ручная разметка   |


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

**Метрики (5063 слова, 54 признака):**


| Задача                               | Модель   | CV F1     | Accuracy  |
| ------------------------------------ | -------- | --------- | --------- |
| L1: Borrowed vs Native               | CatBoost | **0.845** | 0.846     |
| L2: Donor language (flat)            | CatBoost | 0.454     | 0.444     |
| L2: Family — Stage A                 | CatBoost | **0.542** | 0.562     |
| L2: Language within family — Stage B | CatBoost | —         | **0.798** |
| L2: Top-3 accuracy                   | —        | —         | **0.775** |


> Flat L2 (~0.45 F1) уступает иерархическому подходу потому, что
> Germanic-языки (English/German/Dutch) и Romance (French/Italian)
> используют схожие паттерны адаптации в русском.
> Stage B в рамках семьи достигает 80% — это результат объединения
> Dutch→German и добавления германских кластерных признаков (шт/шн/шп).

**Сохранённые артефакты:**

```
models_cache/
├── l1_model.joblib        # L1-классификатор (CatBoost)
├── l2_model.joblib        # L2-классификатор (CatBoost)
├── label_encoder.joblib   # кодировщик языков-доноров
├── feature_names.joblib   # имена 54 признаков
└── seed_lookup.joblib     # O(1)-поиск по 5063 словам
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
2. Слова из датасета кодируются в векторы (кэшируются в `models_cache/`)
3. На embeddings обучается линейный зонд (LogisticRegression)
4. Результат сравнивается с зондом на ручных признаках

**Результаты (5-fold CV F1):**


| Метод               | L1 (binary) | L2 (donor) |
| ------------------- | ----------- | ---------- |
| Embeddings только   | 0.871       | 0.541      |
| Ручные признаки     | 0.893       | **0.730**  |
| Embeddings + Ручные | **0.908**   | 0.684      |


**Вывод:** Multilingual embeddings частично кодируют информацию о происхождении слова (L1 ≈ паритет), однако для определения конкретного языка-донора (L2) ручные морфо-фонетические признаки значительно эффективнее. Комбинация даёт наилучший результат по L1, что подтверждает гипотезу о взаимодополняющей природе двух типов признаков.

PCA-визуализации сохраняются в `models_cache/probe_pca_binary.png` и `probe_pca_donor.png`.

---

## API

Базовый URL: `http://localhost:8000`

### `POST /api/analyze`

```json
{
  "input": "маркетинг и менеджмент",
  "mode": "text"
}
```

`mode`: `"text"` (полный текст с препроцессингом) или `"word"` (одно слово / список через запятую).

Ответ:

```json
{
  "input_text": "маркетинг и менеджмент",
  "mode": "text",
  "words": [{
    "word": "маркетинг",
    "lemma": "маркетинг",
    "pos": "NOUN",
    "is_loanword": true,
    "loanword_probability": 0.97,
    "donor_language": "English",
    "donor_language_ru": "английский",
    "donor_family": "Indo-European/Germanic",
    "source_word": "marketing",
    "top_donors": [
      {"language": "English", "probability": 0.89},
      {"language": "German",  "probability": 0.06}
    ],
    "explanation": ["содержит суффикс -инг (характерен для английских заимствований)"]
  }],
  "stats": {
    "total_words": 2,
    "borrowings_found": 2,
    "borrowing_rate": 1.0,
    "top_donor": "English"
  }
}
```

### `GET /api/word/{word}`

Анализ одного слова.

### `GET /api/health`

```json
{ "status": "ok", "trained": true }
```

Документация Swagger: `http://localhost:8000/docs`

---

## Технологии


| Слой         | Инструменты                                    |
| ------------ | ---------------------------------------------- |
| Морфология   | pymorphy3                                      |
| Фонетика     | python-Levenshtein                             |
| ML           | scikit-learn, CatBoost                         |
| Probing      | sentence-transformers, matplotlib              |
| Данные       | pandas, numpy                                  |
| API          | FastAPI, Pydantic, uvicorn                     |
| Frontend     | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Сериализация | joblib                                         |


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
│   │   │   ├── features.py       # 54 лингвистических признака
│   │   │   ├── classifier.py     # L1/L2, загрузка из кэша
│   │   │   ├── explainer.py      # объяснения предсказаний
│   │   │   └── enricher.py       # Glottolog, морфологические производные
│   │   └── main.py               # точка входа FastAPI
│   ├── data/
│   │   ├── seed_dataset.csv      # 5063 размеченных слова
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

