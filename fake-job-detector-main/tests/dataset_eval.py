"""
Evaluate the trained model on a stratified 3,000-row sample of
balanced_it_jobs_dataset_30k.csv. Computes metrics for both the raw BERT
prediction and the fused-score prediction, and saves results + confusion
matrices to results/.

Run from project root:
    .venv/bin/python tests/dataset_eval.py
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from predict import predict, fused_score  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DATASET_PATH = ROOT / "data" / "balanced_it_jobs_dataset_30k.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 3000
SEED = 42


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample preserving class balance."""
    rng = np.random.default_rng(seed)
    halves = []
    for label, group in df.groupby("label"):
        idx = rng.choice(len(group), size=n // 2, replace=False)
        halves.append(group.iloc[idx])
    return pd.concat(halves).sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics_block(y_true, y_pred, name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1 score  : {f:.4f}")
    print(f"Confusion matrix:\n{cm}")
    return {
        "name": name,
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["pred REAL", "pred FAKE"],
        yticklabels=["true REAL", "true FAKE"],
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    df["description"] = df["description"].astype(str)
    print(f"Total rows: {len(df)}  (label dist: {df['label'].value_counts().to_dict()})")

    sample = stratified_sample(df, SAMPLE_SIZE, SEED)
    print(f"Sampled {len(sample)} rows for evaluation")

    y_true = []
    bert_preds = []
    fused_preds = []
    bert_probs = []
    fused_probs = []

    for i, row in enumerate(sample.itertuples(index=False), 1):
        text = row.description
        bert_p, fused_p, _ev = fused_score(text)
        bert_probs.append(bert_p)
        fused_probs.append(fused_p)
        bert_preds.append(1 if bert_p >= 0.5 else 0)
        fused_preds.append(1 if fused_p >= 0.5 else 0)
        y_true.append(int(row.label))
        if i % 200 == 0:
            print(f"  {i}/{len(sample)} processed")

    raw = metrics_block(y_true, bert_preds, "RAW BERT — in-distribution dataset (3k stratified sample)")
    fused = metrics_block(y_true, fused_preds, "FUSED (BERT + rule layer) — in-distribution dataset")

    # Confusion matrices
    plot_confusion_matrix(
        np.array(raw["confusion_matrix"]),
        "Raw BERT  —  In-distribution (n=3000)",
        RESULTS_DIR / "cm_raw_indist.png",
    )
    plot_confusion_matrix(
        np.array(fused["confusion_matrix"]),
        "Fused (BERT + Rules)  —  In-distribution (n=3000)",
        RESULTS_DIR / "cm_fused_indist.png",
    )

    # Persist for later report aggregation
    out = {"in_distribution": {"raw": raw, "fused": fused}, "n_sampled": len(sample)}
    (RESULTS_DIR / "in_distribution_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS_DIR / 'in_distribution_metrics.json'}")


if __name__ == "__main__":
    main()
