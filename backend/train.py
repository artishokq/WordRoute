"""
WordRoute — Training Script
===========================
Trains and evaluates the two-level borrowing classifier:

  L1 — Binary:     borrowed vs. native
  L2 — Multiclass: donor language (English / French / German / ...)

Models compared per level:
  Baseline   — LogisticRegression (L1), RandomForest (L2)
  Main       — CatBoost (both)

Outputs:
  • Full classification report (precision / recall / F1)
  • Confusion matrix (ASCII)
  • 5-fold cross-validation F1
  • Top feature importances (L1)
  • Top-1 and Top-3 accuracy (L2)
  • Saved model artefacts in models_cache/

Usage:
  cd backend
  source venv/bin/activate
  python train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
from app.core.preprocessor import analyze_word
from app.core.features import extract_features, features_to_vector, get_feature_names

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "seed_dataset.csv"
MODEL_DIR = Path(__file__).parent / "models_cache"
MODEL_DIR.mkdir(exist_ok=True)

DONOR_LABEL_MAP = {
    "English": "English",
    "French": "French",
    "German": "German",
    "Greek/Latin": "Greek/Latin",
    "Arabic/Persian": "Arabic/Persian",
    "Turkic": "Turkic",
    "Italian": "Italian",
    "Dutch": "English",      # merge Dutch into Germanic/English group (small class)
    "Slavic": "Slavic",
}

# ─── Display helpers ──────────────────────────────────────────────────────────

W = 62  # box width


def banner(title: str) -> None:
    print()
    print("╔" + "═" * W + "╗")
    print("║" + title.center(W) + "║")
    print("╚" + "═" * W + "╝")


def section(title: str) -> None:
    print()
    print("─" * W)
    print(f"  {title}")
    print("─" * W)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


def ascii_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Pretty-print a confusion matrix with row/column headers."""
    col_w = max(max(len(l) for l in labels), 5) + 2
    header = " " * (col_w + 2) + "".join(l[:col_w].center(col_w) for l in labels)
    print(header)
    print(" " * (col_w + 2) + "─" * (col_w * len(labels)))
    for i, row_label in enumerate(labels):
        row_str = row_label[:col_w].rjust(col_w) + " │"
        for j, val in enumerate(cm[i]):
            cell = str(val).center(col_w)
            if i == j:
                cell = f"\033[1m{cell}\033[0m"   # bold diagonal (true positives)
            row_str += cell
        print(row_str)
    print()


# ─── Data loading & feature extraction ────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["word", "lemma", "is_loanword"])
    df["is_loanword"] = df["is_loanword"].astype(int)
    df["donor_mapped"] = df["donor_language"].fillna("Unknown").map(
        lambda x: DONOR_LABEL_MAP.get(str(x), str(x))
    )
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract feature vectors for every row in the dataset."""
    rows = []
    for _, row in df.iterrows():
        morph = analyze_word(str(row["lemma"]))
        feats = extract_features(str(row["word"]), str(row["lemma"]), morph)
        rows.append(features_to_vector(feats))
    return np.array(rows, dtype=float)


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def cv_f1(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> tuple[float, float]:
    """sklearn cross_val_score — works for Pipeline / LogReg / RF."""
    scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1_weighted",
    )
    return float(scores.mean()), float(scores.std())


def cv_f1_factory(model_factory, X: np.ndarray, y: np.ndarray,
                   cv: int = 5) -> tuple[float, float]:
    """
    Manual k-fold CV using a factory function.
    Use for estimators that sklearn cannot clone (e.g. CatBoost with class_weights).
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        m = model_factory()
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx]).ravel()
        scores.append(f1_score(y[val_idx], y_pred, average="weighted", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Top-k accuracy: correct if true label is in top-k predictions."""
    top_k = np.argsort(proba, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


# ─── L1: Borrowing detection ──────────────────────────────────────────────────

def train_l1(X_train, X_test, y_train, y_test) -> tuple:
    section("L1 — BORROWING DETECTION  (binary classification)")
    results = {}

    # ── Baseline: LogisticRegression ───────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    t0 = time.perf_counter()
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    lr_time = time.perf_counter() - t0

    lr_f1_cv, lr_f1_std = cv_f1(lr_pipe, np.vstack([X_train, X_test]),
                                  np.hstack([y_train, y_test]))

    print("\n  Baseline — Logistic Regression")
    print(classification_report(y_test, y_pred_lr,
                                 target_names=["Native", "Borrowed"],
                                 digits=3))
    info(f"5-fold CV F1 (weighted): {lr_f1_cv:.3f} ± {lr_f1_std:.3f}")
    info(f"Train time: {lr_time*1000:.1f} ms")

    results["lr"] = {
        "model": lr_pipe,
        "f1": f1_score(y_test, y_pred_lr, average="weighted"),
        "cv_f1": lr_f1_cv,
        "y_pred": y_pred_lr,
    }

    # ── CatBoost ───────────────────────────────────────────────────────────────
    CB_L1_PARAMS = dict(
        iterations=400, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="F1",
        auto_class_weights="SqrtBalanced",
        random_seed=42, verbose=0,
    )
    t0 = time.perf_counter()
    cb = CatBoostClassifier(**CB_L1_PARAMS)
    cb.fit(X_train, y_train)
    y_pred_cb = cb.predict(X_test)
    cb_time = time.perf_counter() - t0

    X_full, y_full = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])
    cb_f1_cv, cb_f1_std = cv_f1_factory(
        lambda: CatBoostClassifier(**CB_L1_PARAMS), X_full, y_full
    )

    print("\n  CatBoost")
    print(classification_report(y_test, y_pred_cb,
                                 target_names=["Native", "Borrowed"],
                                 digits=3))
    info(f"5-fold CV F1 (weighted): {cb_f1_cv:.3f} ± {cb_f1_std:.3f}")
    info(f"Train time: {cb_time*1000:.1f} ms")

    results["catboost"] = {
        "model": cb,
        "f1": f1_score(y_test, y_pred_cb, average="weighted"),
        "cv_f1": cb_f1_cv,
        "y_pred": y_pred_cb,
    }

    # ── Confusion matrix (best model) ─────────────────────────────────────────
    best_key = max(results, key=lambda k: results[k]["cv_f1"])
    best = results[best_key]
    print(f"\n  Confusion matrix ({best_key}):")
    cm = confusion_matrix(y_test, best["y_pred"])
    ascii_confusion_matrix(cm, ["Native", "Borrowed"])

    # ── Winner summary ─────────────────────────────────────────────────────────
    print(f"  {'Model':<22} {'Test F1':>8}  {'CV F1':>8}  {'± Std':>6}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*6}")
    for k, v in results.items():
        marker = " ← best" if k == best_key else ""
        print(f"  {k:<22} {v['f1']:>8.3f}  {v['cv_f1']:>8.3f}  {v['f1']-v['cv_f1']:>+6.3f}{marker}")

    return results, best_key


def show_feature_importance(model, feature_names: list[str], n: int = 15) -> None:
    section(f"TOP {n} FEATURES  (L1 — Logistic Regression coefficients)")
    if not hasattr(model, "named_steps"):
        return
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return

    coef = clf.coef_[0]
    pairs = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)[:n]
    for name, val in pairs:
        bar_len = int(abs(val) * 12)
        bar = ("█" * bar_len)[:20]
        direction = "+" if val > 0 else "-"
        print(f"  {name:<28} {direction}{abs(val):>6.3f}  {bar}")
    print()
    info("Positive coefficient → evidence FOR borrowing")
    info("Negative coefficient → evidence AGAINST borrowing (native word)")


# ─── L2: Donor language ───────────────────────────────────────────────────────

def train_l2(
    X_train, X_test, y_train, y_test, le: LabelEncoder
) -> tuple:
    section("L2 — DONOR LANGUAGE DETECTION  (multiclass)")
    labels_str = list(le.classes_)
    results = {}

    # ── Baseline: RandomForest ─────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)
    rf_time = time.perf_counter() - t0

    rf_f1_cv, rf_f1_std = cv_f1(rf, np.vstack([X_train, X_test]),
                                  np.hstack([y_train, y_test]))

    print("\n  Baseline — RandomForest")
    print(classification_report(y_test, y_pred_rf, target_names=labels_str, digits=3,
                                 zero_division=0))
    info(f"5-fold CV F1 (weighted): {rf_f1_cv:.3f} ± {rf_f1_std:.3f}")
    info(f"Top-1 accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    info(f"Top-3 accuracy: {top_k_accuracy(proba_rf, y_test, k=3):.3f}")
    info(f"Train time: {rf_time*1000:.1f} ms")

    results["rf"] = {
        "model": rf,
        "f1": f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        "cv_f1": rf_f1_cv,
        "y_pred": y_pred_rf,
        "proba": proba_rf,
    }

    # ── CatBoost ───────────────────────────────────────────────────────────────
    CB_L2_PARAMS = dict(
        iterations=500, learning_rate=0.04, depth=6,
        loss_function="MultiClass", eval_metric="Accuracy",
        auto_class_weights="SqrtBalanced",
        random_seed=42, verbose=0,
    )
    t0 = time.perf_counter()
    cb = CatBoostClassifier(**CB_L2_PARAMS)
    cb.fit(X_train, y_train)
    y_pred_cb = cb.predict(X_test).ravel()
    proba_cb = cb.predict_proba(X_test)
    cb_time = time.perf_counter() - t0

    X_full2, y_full2 = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])
    cb_f1_cv, cb_f1_std = cv_f1_factory(
        lambda: CatBoostClassifier(**CB_L2_PARAMS), X_full2, y_full2
    )

    print("\n  CatBoost")
    print(classification_report(y_test, y_pred_cb, target_names=labels_str, digits=3,
                                 zero_division=0))
    info(f"5-fold CV F1 (weighted): {cb_f1_cv:.3f} ± {cb_f1_std:.3f}")
    info(f"Top-1 accuracy: {accuracy_score(y_test, y_pred_cb):.3f}")
    info(f"Top-3 accuracy: {top_k_accuracy(proba_cb, y_test, k=3):.3f}")
    info(f"Train time: {cb_time*1000:.1f} ms")

    results["catboost"] = {
        "model": cb,
        "f1": f1_score(y_test, y_pred_cb, average="weighted", zero_division=0),
        "cv_f1": cb_f1_cv,
        "y_pred": y_pred_cb,
        "proba": proba_cb,
    }

    # ── Confusion matrix (best model) ─────────────────────────────────────────
    best_key = max(results, key=lambda k: results[k]["cv_f1"])
    best = results[best_key]
    print(f"\n  Confusion matrix ({best_key}):")
    cm = confusion_matrix(y_test, best["y_pred"])
    ascii_confusion_matrix(cm, labels_str)

    # ── Winner summary ─────────────────────────────────────────────────────────
    print(f"  {'Model':<22} {'Test F1':>8}  {'CV F1':>8}  {'Top-1':>6}  {'Top-3':>6}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for k, v in results.items():
        marker = " ← best" if k == best_key else ""
        top1 = accuracy_score(y_test, v["y_pred"])
        top3 = top_k_accuracy(v["proba"], y_test, k=3)
        print(f"  {k:<22} {v['f1']:>8.3f}  {v['cv_f1']:>8.3f}  {top1:>6.3f}  {top3:>6.3f}{marker}")

    return results, best_key


# ─── Error analysis ───────────────────────────────────────────────────────────

def error_analysis(
    df_test: pd.DataFrame,
    y_true_l1: np.ndarray,
    y_pred_l1: np.ndarray,
    n: int = 10,
) -> None:
    section("ERROR ANALYSIS  (L1 — false positives & false negatives)")
    errors = []
    for i, (true, pred) in enumerate(zip(y_true_l1, y_pred_l1)):
        if true != pred:
            row = df_test.iloc[i]
            kind = "FP (predicted borrowed)" if pred == 1 else "FN (predicted native)"
            errors.append({
                "word": row["word"],
                "donor": row.get("donor_language", ""),
                "error": kind,
            })

    if not errors:
        ok("No misclassifications on test set!")
        return

    print(f"\n  {len(errors)} errors out of {len(y_true_l1)} test samples:\n")
    print(f"  {'Word':<20} {'True donor':<18} {'Error type'}")
    print(f"  {'─'*20}  {'─'*18}  {'─'*30}")
    for e in errors[:n]:
        print(f"  {e['word']:<20} {e['donor']:<18} {e['error']}")
    if len(errors) > n:
        print(f"  ... and {len(errors) - n} more")


# ─── Save artefacts ───────────────────────────────────────────────────────────

def save_models(
    l1_model,
    l2_model,
    le: LabelEncoder,
    feature_names: list[str],
    seed_lookup: dict,
) -> None:
    section("SAVING MODEL ARTEFACTS")
    joblib.dump(l1_model,      MODEL_DIR / "l1_model.joblib")
    joblib.dump(l2_model,      MODEL_DIR / "l2_model.joblib")
    joblib.dump(le,            MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    joblib.dump(seed_lookup,   MODEL_DIR / "seed_lookup.joblib")

    for name in ["l1_model", "l2_model", "label_encoder", "feature_names", "seed_lookup"]:
        path = MODEL_DIR / f"{name}.joblib"
        size_kb = path.stat().st_size / 1024
        ok(f"{name}.joblib  ({size_kb:.1f} KB)")

    print()
    info(f"Artefacts saved to: {MODEL_DIR}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    banner("WordRoute — Training & Evaluation Report")

    # ── Dataset ────────────────────────────────────────────────────────────────
    section("DATASET")
    t_start = time.perf_counter()
    df = load_dataset()
    n_total = len(df)
    n_borrowed = df["is_loanword"].sum()
    n_native = n_total - n_borrowed
    donor_counts = df[df["is_loanword"] == 1]["donor_mapped"].value_counts()

    print(f"\n  Total words       : {n_total}")
    print(f"  Borrowings        : {n_borrowed}  ({n_borrowed/n_total*100:.1f}%)")
    print(f"  Native Slavic     : {n_native}  ({n_native/n_total*100:.1f}%)")
    print(f"\n  Donor distribution:")
    for lang, cnt in donor_counts.items():
        bar = "▓" * int(cnt / donor_counts.max() * 20)
        print(f"    {lang:<20} {cnt:>4}  {bar}")

    # ── Feature extraction ─────────────────────────────────────────────────────
    section("FEATURE EXTRACTION")
    info(f"Extracting features for {n_total} words...")
    t0 = time.perf_counter()
    X_all = build_feature_matrix(df)
    feature_names = get_feature_names()
    elapsed = time.perf_counter() - t0
    ok(f"Done. {len(feature_names)} features per word  ({elapsed:.1f}s)")

    # ── L1 split ───────────────────────────────────────────────────────────────
    y_l1 = df["is_loanword"].values
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_all, y_l1, np.arange(len(df)),
        test_size=0.20, stratify=y_l1, random_state=42,
    )
    info(f"Train / Test split: {len(X_tr)} / {len(X_te)}  (80 / 20, stratified)")

    # ── Train L1 ───────────────────────────────────────────────────────────────
    l1_results, l1_best_key = train_l1(X_tr, X_te, y_tr, y_te)
    show_feature_importance(l1_results["lr"]["model"], feature_names)

    # ── Error analysis L1 ─────────────────────────────────────────────────────
    error_analysis(
        df.iloc[idx_te].reset_index(drop=True),
        y_te,
        l1_results[l1_best_key]["y_pred"],
    )

    # ── L2 split (borrowings only) ────────────────────────────────────────────
    df_borrow = df[df["is_loanword"] == 1].reset_index(drop=True)
    X_borrow = build_feature_matrix(df_borrow)
    le = LabelEncoder()
    y_l2 = le.fit_transform(df_borrow["donor_mapped"].values)

    X_tr2, X_te2, y_tr2, y_te2, idx_tr2, idx_te2 = train_test_split(
        X_borrow, y_l2, np.arange(len(df_borrow)),
        test_size=0.20, stratify=y_l2, random_state=42,
    )

    l2_results, l2_best_key = train_l2(X_tr2, X_te2, y_tr2, y_te2, le)

    # ── Build seed lookup from full dataset ───────────────────────────────────
    seed_lookup: dict = {}
    for _, row in df.iterrows():
        entry = {
            "is_loanword": int(row["is_loanword"]),
            "donor_language": str(row.get("donor_language", "Unknown")),
            "donor_family": str(row.get("donor_family", "")),
            "source_word": str(row.get("source_word", "")),
            "semantic_domain": str(row.get("semantic_domain", "")),
            "confidence": float(row.get("confidence", 0.5)),
        }
        seed_lookup[str(row["word"]).lower()] = entry
        seed_lookup[str(row["lemma"]).lower()] = entry

    # ── Save best models ───────────────────────────────────────────────────────
    save_models(
        l1_model=l1_results[l1_best_key]["model"],
        l2_model=l2_results[l2_best_key]["model"],
        le=le,
        feature_names=feature_names,
        seed_lookup=seed_lookup,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    section("SUMMARY")
    total_time = time.perf_counter() - t_start
    print(f"\n  Dataset          : {n_total} words")
    print(f"  Feature dim      : {len(feature_names)}")
    print(f"  L1 best model    : {l1_best_key}  (CV F1 = {l1_results[l1_best_key]['cv_f1']:.3f})")
    print(f"  L2 best model    : {l2_best_key}  (CV F1 = {l2_results[l2_best_key]['cv_f1']:.3f})")
    print(f"  Total time       : {total_time:.1f}s")
    print()
    ok("Training complete. Run the API server to use the saved models.")
    print()


if __name__ == "__main__":
    main()
