"""
Compare two fusion rules on the NEW (EMSCAD-trained) model:

  v1 (current production rule):
       ev=0 -> bert*0.35
       ev=1 -> max(0.30, 0.55*bert)
       ev=2 -> 0.58
       ev=3 -> 0.72
       ev=4 -> 0.85
       ev>=5 -> 0.92

  v2 (OR-style, recommended for NEW model):
       fused = max(bert, evidence_floor)
       evidence_floor: ev=0 -> 0.0
                       ev=1 -> 0.30
                       ev=2 -> 0.55
                       ev=3 -> 0.70
                       ev>=4 -> 0.85

  v3 (compromise — slight floor when BERT is uncertain):
       fused = max(bert, evidence_floor) * confidence_modifier
       Same floors as v2, but if bert is in (0.4, 0.6) and ev=0, drop to bert*0.6
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

from transformers import DistilBertTokenizerFast  # noqa: E402
from predict import (  # noqa: E402
    DigitalFootprintClassifier,
    DEVICE,
    MAX_LEN,
    normalize_text,
    extract_features,
    evidence_signal_count,
)
from adversarial_test import REAL_JOBS, FAKE_JOBS  # noqa: E402

NEW_DIR = ROOT / "models_emscad"
RESULTS = ROOT / "results"


def load_new_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(NEW_DIR))
    model = DigitalFootprintClassifier().to(DEVICE)
    state = torch.load(NEW_DIR / "model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def bert_prob(model, tok, text):
    if not text or len(text.strip()) < 10:
        return 0.0
    n = normalize_text(text)
    feats = torch.tensor(np.array([extract_features(n)]), dtype=torch.float32).to(DEVICE)
    enc = tok(n, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], feats)
        return torch.softmax(logits, 1)[0, 1].item()


def fuse_v1(b, ev):
    if ev >= 5: return 0.92
    if ev == 4: return 0.85
    if ev == 3: return 0.72
    if ev == 2: return 0.58
    if ev == 1: return max(0.30, 0.55 * b)
    return b * 0.35


def fuse_v2(b, ev):
    if ev >= 4: floor = 0.85
    elif ev == 3: floor = 0.70
    elif ev == 2: floor = 0.55
    elif ev == 1: floor = 0.30
    else: floor = 0.0
    return max(b, floor)


def fuse_v3(b, ev):
    if ev >= 4: floor = 0.85
    elif ev == 3: floor = 0.70
    elif ev == 2: floor = 0.55
    elif ev == 1: floor = 0.30
    else:
        # No rule evidence — slightly down-weight only when BERT is mid-range
        if 0.4 < b < 0.6:
            return b * 0.6
        return b
    return max(b, floor)


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate(name, texts, labels, model, tok):
    print(f"\n--- {name} (n={len(texts)}) ---")
    bs, evs = [], []
    for t in texts:
        b = bert_prob(model, tok, t)
        ev = evidence_signal_count(t)
        bs.append(b)
        evs.append(ev)
    rs = {}
    for tag, fn in [("v1", fuse_v1), ("v2_OR", fuse_v2), ("v3_compromise", fuse_v3)]:
        preds = [1 if fn(b, ev) >= 0.5 else 0 for b, ev in zip(bs, evs)]
        m = metrics(labels, preds)
        print(f"  {tag:<14} acc={m['accuracy']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        rs[tag] = m
    raw_preds = [1 if b >= 0.5 else 0 for b in bs]
    rs["raw"] = metrics(labels, raw_preds)
    print(f"  {'raw':<14} acc={rs['raw']['accuracy']:.3f}  F1={rs['raw']['f1']:.3f}")
    return rs


def main():
    model, tok = load_new_model()

    out = {}

    # Adversarial
    adv_texts = list(REAL_JOBS) + list(FAKE_JOBS)
    adv_labels = [0] * len(REAL_JOBS) + [1] * len(FAKE_JOBS)
    out["adversarial"] = evaluate("Adversarial 40", adv_texts, adv_labels, model, tok)

    # EMSCAD held-out
    em_df = pd.read_csv(ROOT / "data" / "emscad_balanced.csv")
    em_texts = em_df["description"].astype(str).tolist()
    em_labels = em_df["label"].astype(int).tolist()
    _, X_val, _, y_val = train_test_split(em_texts, em_labels, test_size=0.2, stratify=em_labels, random_state=42)
    out["emscad_heldout"] = evaluate("EMSCAD held-out", X_val, y_val, model, tok)

    # Synthetic dataset (smaller sample for speed)
    syn_df = pd.read_csv(ROOT / "data" / "balanced_it_jobs_dataset_30k.csv")
    syn_df["description"] = syn_df["description"].astype(str)
    rng = np.random.default_rng(42)
    halves = []
    for label in (0, 1):
        sub = syn_df[syn_df["label"] == label]
        halves.append(sub.iloc[rng.choice(len(sub), 250, replace=False)])
    syn_sample = pd.concat(halves).sample(frac=1, random_state=7).reset_index(drop=True)
    out["synthetic"] = evaluate("Synthetic dataset", syn_sample["description"].tolist(), syn_sample["label"].astype(int).tolist(), model, tok)

    (RESULTS / "fusion_comparison.json").write_text(json.dumps(out, indent=2))

    # Print summary
    print("\n" + "=" * 80)
    print(f"NEW model (EMSCAD-trained) — fusion variant comparison (accuracy)")
    print("=" * 80)
    print(f"{'Test set':<22}{'raw':>8}{'v1 (current)':>15}{'v2 (OR)':>10}{'v3':>8}")
    for name, m in [("adversarial", out["adversarial"]),
                    ("emscad_heldout", out["emscad_heldout"]),
                    ("synthetic", out["synthetic"])]:
        print(f"{name:<22}{m['raw']['accuracy']:>8.3f}{m['v1']['accuracy']:>15.3f}"
              f"{m['v2_OR']['accuracy']:>10.3f}{m['v3_compromise']['accuracy']:>8.3f}")


if __name__ == "__main__":
    main()
