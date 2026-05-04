"""
Side-by-side comparison of:
  OLD model — trained on the synthetic-template dataset (models/)
  NEW model — retrained on EMSCAD real labelled fakes (models_emscad/)

Evaluated on three sets:
  EMSCAD held-out (200 from emscad_balanced.csv — 80/20 split)
  Adversarial 40-row hand-curated set
  Synthetic dataset (1000 stratified rows from balanced_it_jobs_dataset_30k.csv)

For each (model, dataset), report raw-BERT and fused metrics.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

from transformers import DistilBertTokenizerFast  # noqa: E402
from predict import (  # noqa: E402
    DigitalFootprintClassifier,
    DEVICE,
    MODEL_NAME,
    MAX_LEN,
    normalize_text,
    extract_features,
    evidence_signal_count,
)
from adversarial_test import REAL_JOBS, FAKE_JOBS  # noqa: E402

OLD_DIR = ROOT / "models"
NEW_DIR = ROOT / "models_emscad"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_model(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DigitalFootprintClassifier().to(DEVICE)
    state = torch.load(model_dir / "model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def predict_with(model, tokenizer, text):
    if not text or len(text.strip()) < 10:
        return 0.0
    normalized = normalize_text(text)
    feats = extract_features(normalized)
    feats_t = torch.tensor(np.array([feats]), dtype=torch.float32).to(DEVICE)
    enc = tokenizer(normalized, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], feats_t)
        return torch.softmax(logits, dim=1)[0, 1].item()


def fused_with(model, tokenizer, text):
    bert_p = predict_with(model, tokenizer, text)
    ev = evidence_signal_count(text)
    if ev >= 5: f = 0.92
    elif ev == 4: f = 0.85
    elif ev == 3: f = 0.72
    elif ev == 2: f = 0.58
    elif ev == 1: f = max(0.30, 0.55 * bert_p)
    else: f = bert_p * 0.35
    return bert_p, f


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


def evaluate_set(name, texts, labels, models):
    """models = {'old': (m, t), 'new': (m, t)}"""
    print(f"\n--- {name}  (n={len(texts)}) ---")
    results = {}
    for tag, (m, tok) in models.items():
        raw_pred, fused_pred = [], []
        for txt in texts:
            bert_p, fused_p = fused_with(m, tok, txt)
            raw_pred.append(1 if bert_p >= 0.5 else 0)
            fused_pred.append(1 if fused_p >= 0.5 else 0)
        m_raw = metrics(labels, raw_pred)
        m_fused = metrics(labels, fused_pred)
        print(f"  [{tag}] raw   acc={m_raw['accuracy']:.3f} f1={m_raw['f1']:.3f}")
        print(f"  [{tag}] fused acc={m_fused['accuracy']:.3f} f1={m_fused['f1']:.3f}")
        results[tag] = {"raw": m_raw, "fused": m_fused}
    return results


def main():
    print("Loading OLD model (synthetic-template trained) ...")
    old_model, old_tok = load_model(OLD_DIR)
    print("Loading NEW model (EMSCAD trained) ...")
    new_model, new_tok = load_model(NEW_DIR)

    models = {"old": (old_model, old_tok), "new": (new_model, new_tok)}
    out = {}

    # --- Adversarial ---
    adv_texts = list(REAL_JOBS) + list(FAKE_JOBS)
    adv_labels = [0] * len(REAL_JOBS) + [1] * len(FAKE_JOBS)
    out["adversarial"] = evaluate_set("Adversarial 40", adv_texts, adv_labels, models)

    # --- EMSCAD held-out (rebuild same 80/20 split as training) ---
    em_df = pd.read_csv(ROOT / "data" / "emscad_balanced.csv")
    em_texts = em_df["description"].astype(str).tolist()
    em_labels = em_df["label"].astype(int).tolist()
    _, X_val, _, y_val = train_test_split(
        em_texts, em_labels, test_size=0.2, stratify=em_labels, random_state=42
    )
    out["emscad_heldout"] = evaluate_set("EMSCAD held-out (n=283)", X_val, y_val, models)

    # --- Synthetic 30k (1000-row stratified subsample) ---
    syn_df = pd.read_csv(ROOT / "data" / "balanced_it_jobs_dataset_30k.csv")
    syn_df["description"] = syn_df["description"].astype(str)
    rng = np.random.default_rng(42)
    halves = []
    for label in (0, 1):
        sub = syn_df[syn_df["label"] == label]
        halves.append(sub.iloc[rng.choice(len(sub), 500, replace=False)])
    syn_sample = pd.concat(halves).sample(frac=1, random_state=7).reset_index(drop=True)
    syn_texts = syn_sample["description"].tolist()
    syn_labels = syn_sample["label"].astype(int).tolist()
    out["synthetic_dataset"] = evaluate_set("Synthetic dataset (n=1000)", syn_texts, syn_labels, models)

    # Persist
    (RESULTS / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS / 'model_comparison.json'}")

    # ----------------------------------------------------------------
    # Comparison bar chart — accuracy across all 6 cells × old/new × raw/fused
    # ----------------------------------------------------------------
    sets = ["emscad_heldout", "synthetic_dataset", "adversarial"]
    set_labels = ["EMSCAD held-out", "Synthetic 30k (sample)", "Adversarial 40"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.18
    x = np.arange(len(sets))
    series = [
        ("Old · Raw",   "old",   "raw",   "#9aa0a6"),
        ("Old · Fused", "old",   "fused", "#1a73e8"),
        ("New · Raw",   "new",   "raw",   "#fbbc04"),
        ("New · Fused", "new",   "fused", "#34a853"),
    ]
    for i, (label, tag, mode, color) in enumerate(series):
        vals = [out[s][tag][mode]["accuracy"] for s in sets]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(set_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Old vs New model · Raw BERT vs Fused — Accuracy by test set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", ncol=4)
    plt.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS / 'model_comparison.png'}")

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Test set':<28}{'OLD raw':>11}{'OLD fused':>13}{'NEW raw':>11}{'NEW fused':>13}")
    print("=" * 90)
    for s, label in zip(sets, set_labels):
        print(f"{label:<28}"
              f"{out[s]['old']['raw']['accuracy']:>10.3f} "
              f"{out[s]['old']['fused']['accuracy']:>12.3f} "
              f"{out[s]['new']['raw']['accuracy']:>10.3f} "
              f"{out[s]['new']['fused']['accuracy']:>12.3f}")


if __name__ == "__main__":
    main()
