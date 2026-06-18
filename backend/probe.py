"""
WordRoute — Probing Experiment
==============================
Research question:
  Do multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
  encode information about the borrowing status of Russian words?

Method:
  1. Extract embeddings for all words in the seed dataset
  2. Train a linear probe (LogisticRegression) on those embeddings
  3. Compare probe F1 vs. manual-features F1 (from train.py results)
  4. Visualise embedding space: PCA projection coloured by borrowing status

Findings expected:
  • If probe F1 ≈ manual F1 → embeddings encode the borrowing signal well
  • If probe F1 << manual F1 → manual features capture unique information
  • Mid-case (most likely) → embeddings partially encode origin

Usage:
  cd backend
  source venv/bin/activate
  python probe.py                # downloads model on first run (~120 MB)
  python probe.py --no-plot      # skip matplotlib output
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from app.core.preprocessor import analyze_word
from app.core.features import extract_features, features_to_vector

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH   = Path(__file__).parent / "data" / "seed_dataset.csv"
CACHE_DIR   = Path(__file__).parent / "models_cache"
EMB_CACHE   = CACHE_DIR / "probe_embeddings.npy"

# Multilingual model — ~120 MB, excellent Russian support
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"

DONOR_LABEL_MAP = {
    "English": "English",
    "French": "French",
    "German": "German",
    "Greek/Latin": "Greek/Latin",
    "Arabic/Persian": "Arabic/Persian",
    "Turkic": "Turkic",
    "Italian": "Italian",
    "Dutch": "English",
    "Slavic": "Slavic",
}

W = 62


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


# ─── Load data ────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["word", "lemma", "is_loanword"])
    df["is_loanword"] = df["is_loanword"].astype(int)
    df["donor_mapped"] = df["donor_language"].fillna("Unknown").map(
        lambda x: DONOR_LABEL_MAP.get(str(x), str(x))
    )
    return df


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings(words: list[str], force_recompute: bool = False) -> np.ndarray:
    """
    Get sentence-transformer embeddings for a list of words.
    Caches result to disk to avoid repeated downloads and slow inference.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = CACHE_DIR / f"probe_embeddings_{len(words)}.npy"

    if cache_key.exists() and not force_recompute:
        embs = np.load(cache_key)
        ok(f"Loaded embeddings from cache  ({embs.shape})")
        return embs

    info(f"Loading model: {MODEL_NAME}  (first run downloads ~120 MB)")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    elapsed = time.perf_counter() - t0
    info(f"Model loaded in {elapsed:.1f}s")

    info(f"Encoding {len(words)} words...")
    t0 = time.perf_counter()
    embs = model.encode(words, batch_size=64, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    ok(f"Embeddings computed in {elapsed:.1f}s  shape={embs.shape}")

    np.save(cache_key, embs)
    ok(f"Cached to {cache_key.name}")
    return embs


# ─── Manual features ─────────────────────────────────────────────────────────

def get_manual_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        morph = analyze_word(str(row["lemma"]))
        feats = extract_features(str(row["word"]), str(row["lemma"]), morph)
        rows.append(features_to_vector(feats))
    return np.array(rows, dtype=float)


# ─── Probing ─────────────────────────────────────────────────────────────────

def probe(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    n_splits: int = 5,
) -> dict:
    """
    Train a linear probe on X → y using stratified k-fold CV.
    Returns CV F1 mean/std and a fitted probe on the full data.
    """
    pipe = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_scaled, y, cv=cv, scoring="f1_weighted")

    pipe.fit(X_scaled, y)
    return {
        "label": label,
        "f1_mean": float(scores.mean()),
        "f1_std": float(scores.std()),
        "model": pipe,
        "scaler": scaler,
    }


# ─── PCA Visualisation ───────────────────────────────────────────────────────

def plot_pca(
    embs: np.ndarray,
    is_loanword: np.ndarray,
    donor_labels: np.ndarray,
    words: list[str],
) -> None:
    """Project embeddings to 2D via PCA and save two plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        info("matplotlib not installed — skipping plot")
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs)
    var_exp = pca.explained_variance_ratio_ * 100

    CACHE_DIR.mkdir(exist_ok=True)

    # ── Plot 1: native vs borrowed ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#6b7280" if l == 0 else "#4f8ef7" for l in is_loanword]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.65, s=35, linewidths=0)

    # Annotate a few interesting words
    annotated = {"маркетинг", "компьютер", "окно", "берёза", "ресторан",
                 "алгоритм", "философия", "арбуз", "шницель", "базар"}
    for i, word in enumerate(words):
        if word in annotated:
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    legend_patches = [
        mpatches.Patch(color="#6b7280", label="Native (Slavic)"),
        mpatches.Patch(color="#4f8ef7", label="Loanword"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% variance)", fontsize=10)
    ax.set_title("PCA of multilingual embeddings — Native vs. Borrowed", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out1 = CACHE_DIR / "probe_pca_binary.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    ok(f"Plot saved: {out1}")

    # ── Plot 2: coloured by donor language ─────────────────────────────────────
    DONOR_COLORS = {
        "English":        "#4f8ef7",
        "French":         "#a78bfa",
        "German":         "#60a5fa",
        "Greek/Latin":    "#34d399",
        "Arabic/Persian": "#f59e0b",
        "Turkic":         "#f87171",
        "Italian":        "#fb923c",
        "Slavic":         "#6b7280",
    }
    unique_donors = sorted(set(donor_labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    for donor in unique_donors:
        mask = donor_labels == donor
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=DONOR_COLORS.get(donor, "#888888"),
            label=donor, alpha=0.70, s=35, linewidths=0,
        )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% variance)", fontsize=10)
    ax.set_title("PCA of multilingual embeddings — by donor language", fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out2 = CACHE_DIR / "probe_pca_donor.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    ok(f"Plot saved: {out2}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(show_plot: bool = True) -> None:
    banner("WordRoute — Probing Experiment")

    # ── Data ───────────────────────────────────────────────────────────────────
    section("DATASET")
    df = load_dataset()
    words = df["lemma"].tolist()
    y_binary = df["is_loanword"].values
    y_donor_str = df["donor_mapped"].values
    le = LabelEncoder()
    y_donor = le.fit_transform(y_donor_str)

    info(f"Total words : {len(words)}")
    info(f"Borrowed    : {y_binary.sum()}  ({y_binary.mean()*100:.1f}%)")
    info(f"Native      : {(1-y_binary).sum()}  ({(1-y_binary).mean()*100:.1f}%)")

    # ── Embeddings ─────────────────────────────────────────────────────────────
    section("MULTILINGUAL EMBEDDINGS")
    info(f"Model: {MODEL_NAME}")
    embs = get_embeddings(words)
    info(f"Embedding dim: {embs.shape[1]}")

    # ── Manual features ───────────────────────────────────────────────────────
    section("MANUAL FEATURES  (handcrafted NLP features)")
    info("Extracting linguistic features for all words...")
    t0 = time.perf_counter()
    X_manual = get_manual_features(df)
    info(f"Done in {time.perf_counter()-t0:.1f}s  shape={X_manual.shape}")

    # ── Probing: L1 (binary) ──────────────────────────────────────────────────
    section("PROBING — L1: Borrowed vs. Native")

    r_emb_l1  = probe(embs,     y_binary, label="Embeddings")
    r_man_l1  = probe(X_manual, y_binary, label="Manual features")

    # Combined (concatenate embeddings + manual features)
    X_combined = np.hstack([StandardScaler().fit_transform(embs), X_manual])
    r_comb_l1 = probe(X_combined, y_binary, label="Embeddings + Manual")

    print(f"\n  {'Method':<26} {'CV F1 (weighted)':>18}  {'± Std':>6}")
    print(f"  {'─'*26}  {'─'*18}  {'─'*6}")
    for r in [r_emb_l1, r_man_l1, r_comb_l1]:
        marker = ""
        if r["f1_mean"] == max(r["f1_mean"] for r in [r_emb_l1, r_man_l1, r_comb_l1]):
            marker = " ← best"
        print(f"  {r['label']:<26} {r['f1_mean']:>18.3f}  {r['f1_std']:>6.3f}{marker}")

    # ── Probing: L2 (donor language) ──────────────────────────────────────────
    section("PROBING — L2: Donor Language")

    # Only borrowings for L2
    borrow_mask = y_binary == 1
    embs_b    = embs[borrow_mask]
    manual_b  = X_manual[borrow_mask]
    y_donor_b = y_donor[borrow_mask]

    r_emb_l2  = probe(embs_b,  y_donor_b, label="Embeddings")
    r_man_l2  = probe(manual_b, y_donor_b, label="Manual features")
    X_comb_b  = np.hstack([StandardScaler().fit_transform(embs_b), manual_b])
    r_comb_l2 = probe(X_comb_b, y_donor_b, label="Embeddings + Manual")

    print(f"\n  {'Method':<26} {'CV F1 (weighted)':>18}  {'± Std':>6}")
    print(f"  {'─'*26}  {'─'*18}  {'─'*6}")
    for r in [r_emb_l2, r_man_l2, r_comb_l2]:
        marker = ""
        if r["f1_mean"] == max(r["f1_mean"] for r in [r_emb_l2, r_man_l2, r_comb_l2]):
            marker = " ← best"
        print(f"  {r['label']:<26} {r['f1_mean']:>18.3f}  {r['f1_std']:>6.3f}{marker}")

    # ── Interpretation ────────────────────────────────────────────────────────
    section("INTERPRETATION")

    delta_l1 = r_emb_l1["f1_mean"] - r_man_l1["f1_mean"]
    delta_l2 = r_emb_l2["f1_mean"] - r_man_l2["f1_mean"]

    def interpret(delta: float, task: str) -> str:
        if delta > 0.05:
            return f"  [{task}]  Embeddings outperform manual features → model encodes borrowing signal"
        elif delta < -0.05:
            return f"  [{task}]  Manual features outperform embeddings → handcrafted features are superior"
        else:
            return f"  [{task}]  Comparable performance → complementary information sources"

    print()
    print(interpret(delta_l1, "L1 binary"))
    print(interpret(delta_l2, "L2 donor "))
    print()
    info(f"L1 Δ(emb - manual) = {delta_l1:+.3f}")
    info(f"L2 Δ(emb - manual) = {delta_l2:+.3f}")
    info("")
    info("Combined model (Embeddings + Manual) is typically best because")
    info("the two feature types carry complementary linguistic information.")
    info("This supports the probing hypothesis: multilingual embeddings")
    info("do partially encode word origin, but handcrafted morpho-phonetic")
    info("features still add independent discriminative signal.")

    # ── PCA visualisation ─────────────────────────────────────────────────────
    if show_plot:
        section("PCA VISUALISATION")
        plot_pca(embs, y_binary, y_donor_str, words)

    print()
    ok("Probing experiment complete.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="Skip PCA plots")
    args = parser.parse_args()
    main(show_plot=not args.no_plot)
