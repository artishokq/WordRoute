# WordRoute — Лингвистический анализ заимствований в русском языке

Учебный NLP-проект, который принимает русское слово или текст и:
- определяет, является ли слово заимствованием;
- предсказывает язык-донор (английский, французский, немецкий, греко-латинский, тюркский, арабо-персидский, итальянский);
- объясняет предсказание через лингвистические признаки;
- анализирует морфологическую адаптацию;
- проводит probing-эксперимент: несут ли предобученные embeddings информацию об истории слова.

---

## Быстрый старт

```bash
# 1. Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Обучение моделей (обязательно!)
python train.py          # ~20 сек — полный отчёт с метриками

# 3. API-сервер
uvicorn app.main:app --reload --port 8000

# 4. Frontend (в новом терминале)
cd ../frontend
npm install
npm run dev              # → http://localhost:3000
```

---

## Probing-эксперимент

```bash
cd backend && source venv/bin/activate
python probe.py          # первый запуск скачивает модель ~120 MB
                         # последующие — мгновенные (кэш embeddings)
```

Что делает `probe.py`:
1. Загружает `paraphrase-multilingual-MiniLM-L12-v2` (SBERT, 384 измерения)
2. Кодирует все 627 слов из датасета
3. Обучает линейный зонд (logistic regression) на embeddings
4. Сравнивает с зондом на ручных признаках и комбинированном векторе
5. Строит PCA-визуализацию (сохраняется в `models_cache/`)

---

## Архитектура NLP-пайплайна

```
Слово
  │
  ▼
[Препроцессор]  pymorphy3
  ├── лемматизация
  ├── POS-тег (NOUN / VERB / ADJF / ...)
  ├── грамматические категории (род, число)
  └── транслитерация (Кириллица → латиница)
  │
  ▼
[Feature Engineering]  39 признаков
  ├── Графо-фонетические
  │     ├── н-граммы символов
  │     ├── наличие «ф», «дж», «кс», «э»
  │     ├── суффиксы (-инг, -ция, -ия, -ер, -аж, ...)
  │     └── длина / доля гласных
  ├── Морфологические
  │     ├── POS, род, склоняемость
  │     └── флаги known / declinable
  └── Фонетическое сходство (Левенштейн)
        └── нормированное расстояние до словарей
            English / French / German / Greek / Turkic / Arabic / Italian
  │
  ▼
[L1 — Бинарный классификатор]
  ├── Baseline:  Logistic Regression (sklearn Pipeline + StandardScaler)
  └── Main:      CatBoost (auto_class_weights=SqrtBalanced)
  Метрики:  F1-weighted ~0.91 | CV F1 ~0.90 | confusion matrix
  │
  ▼  (только для заимствований)
[L2 — Многоклассовый классификатор]
  ├── Baseline:  RandomForest (n_estimators=200)
  └── Main:      CatBoost MultiClass
  Метрики:  Top-1 acc ~0.75 | Top-3 acc ~0.93 | classification_report per donor
  │
  ▼
[Explainer]  — правило-ориентированные объяснения
[Enricher]   — Glottolog + морфологические производные
  │
  ▼
FastAPI  →  Next.js UI
```

---

## Датасет

| Файл | Описание |
|------|----------|
| `backend/data/seed_dataset.csv` | 627 слов с метками: word, lemma, is_loanword, donor_language, donor_family, source_word, semantic_domain, confidence |
| `backend/data/donor_words/*.txt` | Словари донорских языков для Левенштейна |
| `backend/data/glottolog_info.json` | Метаданные Glottolog (семья, регион) |

Распределение: English 129, Greek/Latin 100, French 56, Native 231, German 32, Turkic 27, Italian 26, Arabic/Persian 26.

---

## Скрипты обучения

### `train.py` — полное обучение с метриками

Запускать из директории `backend/`:

```bash
python train.py
```

Выводит:
- Распределение классов в датасете
- **L1** (binary): precision / recall / F1 для LogReg и CatBoost, 5-fold CV, confusion matrix
- **L2** (multiclass): per-donor classification_report, Top-1 / Top-3 accuracy, confusion matrix
- Топ-15 признаков по коэффициентам LogReg
- Анализ ошибок: ложные положительные и ложные отрицательные
- Сохранение всех артефактов в `models_cache/`

Артефакты:
```
models_cache/
├── l1_model.joblib        # L1-классификатор
├── l2_model.joblib        # L2-классификатор (RandomForest)
├── label_encoder.joblib   # LabelEncoder для языков-доноров
├── feature_names.joblib   # названия 39 признаков
└── seed_lookup.joblib     # точный поиск по датасету (dict)
```

### `probe.py` — probing-эксперимент с embeddings

```bash
python probe.py           # с PCA-графиками
python probe.py --no-plot # без matplotlib
```

Исследовательский вопрос: несут ли multilingual embeddings (SBERT) сигнал об истории происхождения слова?

Результаты (5-fold CV F1):

| Метод | L1 (binary) | L2 (donor) |
|-------|-------------|------------|
| Embeddings только | 0.871 | 0.541 |
| Ручные признаки | **0.893** | **0.730** |
| Embeddings + Ручные | **0.908** | 0.684 |

**Вывод**: embeddings частично кодируют происхождение слова (L1 ≈ паритет), но для детального определения языка-донора (L2) ручные морфо-фонетические признаки значительно эффективнее. Комбинация даёт лучший результат для L1, что подтверждает гипотезу о взаимодополняющей информации.

---

## API

Сервер: `http://localhost:8000`

### `POST /api/analyze`

```json
{
  "text": "маркетинг",
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
    "top_donors": [{"language": "English", "probability": 0.89}],
    "features": { "has_ing": 1, "len_word": 9, "lev_english": 0.12, ... },
    "explanations": ["Суффикс -инг — типичен для английских заимствований"],
    "morphology": { "POS": "NOUN", "gender": "masc", ... }
  }]
}
```

### `GET /api/word/{word}`

Анализ одного слова.

### `GET /health`

```json
{ "status": "ok", "model_source": "cache" }
```

---

## Технологии

| Слой | Стек |
|------|------|
| Морфология | pymorphy3 |
| Признаки | python-Levenshtein, numpy, pandas |
| ML | scikit-learn, CatBoost |
| Probing | sentence-transformers, matplotlib |
| API | FastAPI, Pydantic |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Сериализация | joblib |

---

## Структура проекта

```
WordRoute/
├── backend/
│   ├── app/
│   │   ├── api/routes.py         # FastAPI эндпоинты
│   │   ├── core/
│   │   │   ├── preprocessor.py   # pymorphy3, транслитерация
│   │   │   ├── features.py       # 39 признаков
│   │   │   ├── classifier.py     # L1/L2, load from cache
│   │   │   ├── explainer.py      # объяснения предсказаний
│   │   │   └── enricher.py       # Glottolog, производные
│   │   └── main.py               # FastAPI app
│   ├── data/
│   │   ├── seed_dataset.csv      # 627 размеченных слов
│   │   ├── donor_words/          # словари для Левенштейна
│   │   └── glottolog_info.json
│   ├── models/schemas.py         # Pydantic-схемы
│   ├── models_cache/             # сохранённые модели (git ignore)
│   ├── train.py                  # 🔑 полное обучение + метрики
│   ├── probe.py                  # 🔑 probing-эксперимент
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── components/           # InputForm, WordCard, DonorChart, ...
    │   ├── layout.tsx
    │   └── page.tsx
    └── lib/
        ├── types.ts
        └── api.ts
```
