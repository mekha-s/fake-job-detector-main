"""
Final evaluation pipeline for the production system:
NEW model (EMSCAD-trained) + v3 fusion (now in predict.py).

Produces:
  results/final_metrics.json
  results/final_cm_*.png  (confusion matrices)
  results/final_comparison_bar.png
  results/FINAL_RESULTS.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

from predict import predict, fused_score  # noqa: E402
from adversarial_test import REAL_JOBS, FAKE_JOBS  # noqa: E402


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
    }


def evaluate_set(name, texts, labels):
    print(f"\n--- {name} (n={len(texts)}) ---")
    raw, fused = [], []
    for t in texts:
        bert_p, fused_p, _ev = fused_score(t)
        raw.append(1 if bert_p >= 0.5 else 0)
        fused.append(1 if fused_p >= 0.5 else 0)
    raw_m = metrics(labels, raw)
    fused_m = metrics(labels, fused)
    print(f"  raw   acc={raw_m['accuracy']:.4f} P={raw_m['precision']:.4f} R={raw_m['recall']:.4f} F1={raw_m['f1']:.4f}")
    print(f"  fused acc={fused_m['accuracy']:.4f} P={fused_m['precision']:.4f} R={fused_m['recall']:.4f} F1={fused_m['f1']:.4f}")
    return {"raw": raw_m, "fused": fused_m}


def save_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred REAL", "pred FAKE"],
                yticklabels=["true REAL", "true FAKE"], ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = {}

    # --- EMSCAD held-out (the headline number) ---
    em = pd.read_csv(ROOT / "data" / "emscad_balanced.csv")
    _, X_val, _, y_val = train_test_split(
        em["description"].astype(str).tolist(),
        em["label"].astype(int).tolist(),
        test_size=0.2, stratify=em["label"], random_state=42,
    )
    out["emscad"] = evaluate_set("EMSCAD held-out (in-distribution)", X_val, y_val)

    # --- Adversarial 40-row ---
    adv_texts = list(REAL_JOBS) + list(FAKE_JOBS)
    adv_labels = [0] * len(REAL_JOBS) + [1] * len(FAKE_JOBS)
    out["adversarial"] = evaluate_set("Adversarial 40-row", adv_texts, adv_labels)

    # --- Persist ---
    (RESULTS / "final_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS / 'final_metrics.json'}")

    # --- Confusion matrices ---
    save_cm(out["emscad"]["raw"]["confusion_matrix"],
            f"NEW model · Raw BERT  —  EMSCAD held-out (n={out['emscad']['raw']['n']})",
            RESULTS / "final_cm_emscad_raw.png")
    save_cm(out["emscad"]["fused"]["confusion_matrix"],
            f"NEW model · Fused (v3)  —  EMSCAD held-out (n={out['emscad']['fused']['n']})",
            RESULTS / "final_cm_emscad_fused.png")
    save_cm(out["adversarial"]["raw"]["confusion_matrix"],
            "NEW model · Raw BERT  —  Adversarial (n=40)",
            RESULTS / "final_cm_adv_raw.png")
    save_cm(out["adversarial"]["fused"]["confusion_matrix"],
            "NEW model · Fused (v3)  —  Adversarial (n=40)",
            RESULTS / "final_cm_adv_fused.png")
    print("Saved confusion-matrix PNGs")

    # --- Comparison bar chart ---
    cells = {
        ("Raw BERT", "EMSCAD held-out"): out["emscad"]["raw"],
        ("Fused (v3)", "EMSCAD held-out"): out["emscad"]["fused"],
        ("Raw BERT", "Adversarial 40"): out["adversarial"]["raw"],
        ("Fused (v3)", "Adversarial 40"): out["adversarial"]["fused"],
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [f"{m}\n{s}" for (m, s) in cells.keys()]
    x = np.arange(len(labels))
    width = 0.2
    palette = sns.color_palette("Set2", 4)
    for i, m in enumerate(["accuracy", "precision", "recall", "f1"]):
        vals = [cells[k][m] for k in cells]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=m.capitalize(), color=palette[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Final System (NEW model + v3 fusion) — Performance by test set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", ncol=4)
    plt.tight_layout()
    fig.savefig(RESULTS / "final_comparison_bar.png", dpi=150)
    plt.close(fig)
    print("Saved final_comparison_bar.png")

    # --- Markdown summary ---
    def fmt_cm(cm):
        cm = np.array(cm)
        return (
            "|              | pred REAL | pred FAKE |\n"
            "|--------------|-----------|-----------|\n"
            f"| **true REAL** | {cm[0,0]:>9} | {cm[0,1]:>9} |\n"
            f"| **true FAKE** | {cm[1,0]:>9} | {cm[1,1]:>9} |"
        )

    md = f"""# Final Evaluation — Production System

**Model:** DistilBERT-base + 18 hand-engineered features, retrained on
EMSCAD (Employment Scam Aegean Dataset, 707 real-labeled fakes paired
1:1 with reals from the same dataset).

**Fusion (v3):** `fused = max(bert_prob, evidence_floor)`
- evidence_floor ∈ {{0, 0.30, 0.55, 0.70, 0.85}} for ev=0..≥4
- ev=0 + BERT in (0.4, 0.6) → down-weight to bert × 0.6

## Headline numbers

| Test set                | Raw BERT  acc / F1 | Fused (v3)  acc / F1 |
|-------------------------|--------------------|----------------------|
| **EMSCAD held-out (n={out['emscad']['raw']['n']})** | **{out['emscad']['raw']['accuracy']*100:.2f}%** / {out['emscad']['raw']['f1']:.3f} | **{out['emscad']['fused']['accuracy']*100:.2f}%** / {out['emscad']['fused']['f1']:.3f} |
| Adversarial 40          | {out['adversarial']['raw']['accuracy']*100:.2f}% / {out['adversarial']['raw']['f1']:.3f} | {out['adversarial']['fused']['accuracy']*100:.2f}% / {out['adversarial']['fused']['f1']:.3f} |

## EMSCAD held-out (the academic headline)

### Raw BERT
- Accuracy: **{out['emscad']['raw']['accuracy']:.4f}**
- Precision: {out['emscad']['raw']['precision']:.4f}
- Recall: {out['emscad']['raw']['recall']:.4f}
- F1: {out['emscad']['raw']['f1']:.4f}

{fmt_cm(out['emscad']['raw']['confusion_matrix'])}

### Fused (v3)
- Accuracy: **{out['emscad']['fused']['accuracy']:.4f}**
- Precision: {out['emscad']['fused']['precision']:.4f}
- Recall: {out['emscad']['fused']['recall']:.4f}
- F1: {out['emscad']['fused']['f1']:.4f}

{fmt_cm(out['emscad']['fused']['confusion_matrix'])}

## Adversarial set (robustness check on novel scam types)

### Raw BERT
- Accuracy: **{out['adversarial']['raw']['accuracy']:.4f}**
- Precision: {out['adversarial']['raw']['precision']:.4f}
- Recall: {out['adversarial']['raw']['recall']:.4f}
- F1: {out['adversarial']['raw']['f1']:.4f}

{fmt_cm(out['adversarial']['raw']['confusion_matrix'])}

### Fused (v3)
- Accuracy: **{out['adversarial']['fused']['accuracy']:.4f}**
- Precision: {out['adversarial']['fused']['precision']:.4f}
- Recall: {out['adversarial']['fused']['recall']:.4f}
- F1: {out['adversarial']['fused']['f1']:.4f}

{fmt_cm(out['adversarial']['fused']['confusion_matrix'])}

## Files in this folder

- `final_cm_emscad_raw.png`   confusion matrix — Raw BERT, EMSCAD held-out
- `final_cm_emscad_fused.png` confusion matrix — Fused, EMSCAD held-out
- `final_cm_adv_raw.png`      confusion matrix — Raw BERT, Adversarial
- `final_cm_adv_fused.png`    confusion matrix — Fused, Adversarial
- `final_comparison_bar.png`  4-cell comparison bar chart
- `emscad_training_curve.png` training history (5 epochs, EMSCAD retrain)
- `final_metrics.json`        raw numbers
"""
    (RESULTS / "FINAL_RESULTS.md").write_text(md)
    print("Saved FINAL_RESULTS.md")


if __name__ == "__main__":
    main()
