"""
Two-level borrowing classifier.

Startup behaviour:
  1. If trained artefacts exist in models_cache/ → load them instantly.
  2. Otherwise → train on seed dataset and save for next run.

To force retrain:  python train.py   (from backend/)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .features import extract_features, features_to_vector, get_feature_names
from .preprocessor import analyze_word

DATA_DIR  = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent.parent / "models_cache"

DONOR_LABEL_MAP = {
    "English":        "English",
    "French":         "French",
    "German":         "German",
    "Greek/Latin":    "Greek/Latin",
    "Arabic/Persian": "Arabic/Persian",
    "Turkic":         "Turkic",
    "Italian":        "Italian",
    "Dutch":          "Dutch",
    "Slavic":         "Slavic",
}

_ARTEFACT_NAMES = ["l1_model", "l2_model", "label_encoder",
                   "feature_names", "seed_lookup"]


class BorrowingClassifier:
    def __init__(self) -> None:
        self.l1_model    = None
        self.l2_model    = None
        self.le          = LabelEncoder()
        self.feature_names: list[str] = []
        self._seed_lookup: dict[str, dict] = {}
        self.trained     = False
        self._source     = "none"   # "cache" | "trained"

    # ── Loading ───────────────────────────────────────────────────────────────

    def _all_artefacts_exist(self) -> bool:
        return all((MODEL_DIR / f"{n}.joblib").exists() for n in _ARTEFACT_NAMES)

    def _load_from_cache(self) -> bool:
        """Try to load pre-trained artefacts. Returns True on success."""
        if not self._all_artefacts_exist():
            return False
        try:
            self.l1_model       = joblib.load(MODEL_DIR / "l1_model.joblib")
            self.l2_model       = joblib.load(MODEL_DIR / "l2_model.joblib")
            self.le             = joblib.load(MODEL_DIR / "label_encoder.joblib")
            self.feature_names  = joblib.load(MODEL_DIR / "feature_names.joblib")
            self._seed_lookup   = joblib.load(MODEL_DIR / "seed_lookup.joblib")
            self.trained        = True
            self._source        = "cache"
            print("[classifier] Loaded from cache (models_cache/).")
            return True
        except Exception as exc:
            print(f"[classifier] Cache load failed: {exc}. Falling back to training.")
            return False

    # ── Fallback training (used when cache is missing) ────────────────────────

    def _load_seed_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "seed_dataset.csv")
        df = df.dropna(subset=["word", "lemma", "is_loanword"])
        df["is_loanword"] = df["is_loanword"].astype(int)
        return df

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            morph = analyze_word(str(row["lemma"]))
            feats = extract_features(str(row["word"]), str(row["lemma"]), morph)
            rows.append(features_to_vector(feats))
        return np.array(rows, dtype=float)

    def _build_seed_lookup(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            entry = {
                "is_loanword":     int(row["is_loanword"]),
                "donor_language":  str(row.get("donor_language", "Unknown")),
                "donor_family":    str(row.get("donor_family", "")),
                "source_word":     str(row.get("source_word", "")),
                "semantic_domain": str(row.get("semantic_domain", "")),
                "confidence":      float(row.get("confidence", 0.5)),
            }
            self._seed_lookup[str(row["word"]).lower()]  = entry
            self._seed_lookup[str(row["lemma"]).lower()] = entry

    def _train_fallback(self) -> None:
        """Minimal training used when no saved artefacts are available."""
        df = self._load_seed_dataset()
        self._build_seed_lookup(df)
        self.feature_names = get_feature_names()

        X = self._build_feature_matrix(df)
        y_l1 = df["is_loanword"].values

        self.l1_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])
        self.l1_model.fit(X, y_l1)

        df_borrow = df[df["is_loanword"] == 1].copy()
        df_borrow["donor_mapped"] = df_borrow["donor_language"].map(
            lambda x: DONOR_LABEL_MAP.get(str(x), str(x))
        )
        X_l2 = self._build_feature_matrix(df_borrow)
        y_l2 = self.le.fit_transform(df_borrow["donor_mapped"].values)

        self.l2_model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.l2_model.fit(X_l2, y_l2)

        self.trained = True
        self._source = "trained"
        print(f"[classifier] Fallback training done on {len(df)} words. "
              "Run 'python train.py' for full training with CatBoost.")

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Load from cache or fall back to in-process training."""
        if not self._load_from_cache():
            self._train_fallback()

    def predict(self, word: str) -> dict:
        if not self.trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        word_lower = word.lower().strip()
        morph      = analyze_word(word_lower)
        lemma      = morph.get("lemma", word_lower)

        # Ground-truth lookup from seed dataset
        seed_entry = (self._seed_lookup.get(word_lower)
                      or self._seed_lookup.get(lemma))

        feats = extract_features(word_lower, lemma, morph)
        vec   = features_to_vector(feats)
        X     = np.array([vec])

        # L1
        l1_proba      = self.l1_model.predict_proba(X)[0]
        loanword_prob = float(l1_proba[1])

        # L2
        l2_proba  = self.l2_model.predict_proba(X)[0]
        classes   = self.le.classes_
        top_donors = sorted(
            [{"language": cls, "probability": float(p)}
             for cls, p in zip(classes, l2_proba)],
            key=lambda d: d["probability"],
            reverse=True,
        )[:3]
        top_donor = top_donors[0]["language"] if top_donors else "Unknown"

        result: dict = {
            "word":            word,
            "lemma":           lemma,
            "pos":             morph.get("POS", "UNKN"),
            "gender":          morph.get("gender"),
            "number":          morph.get("number"),
            "is_known_to_morph": morph.get("is_known", True),
            "is_declinable":   morph.get("declinable", True),
            "loanword_probability": round(loanword_prob, 3),
            "is_loanword":     loanword_prob >= 0.5,
            "donor_language":  top_donor,
            "top_donors":      top_donors,
            "features":        feats,
            "donor_family":    "",
            "source_word":     "",
            "semantic_domain": "",
            "in_seed":         False,
        }

        if seed_entry:
            result["loanword_probability"] = seed_entry["confidence"]
            result["is_loanword"]          = bool(seed_entry["is_loanword"])
            result["donor_language"]       = (seed_entry["donor_language"]
                                               if seed_entry["is_loanword"] else "Slavic")
            result["donor_family"]         = seed_entry["donor_family"]
            result["source_word"]          = seed_entry["source_word"]
            result["semantic_domain"]      = seed_entry["semantic_domain"]
            result["in_seed"]              = True

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────

_classifier: BorrowingClassifier | None = None


def get_classifier() -> BorrowingClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BorrowingClassifier()
        _classifier.train()
    return _classifier
