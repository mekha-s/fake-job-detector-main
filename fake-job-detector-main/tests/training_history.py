"""
Retrain the DistilBERT classifier on a 4k/1k stratified subset for 5 epochs
and capture per-epoch train/val accuracy + loss. Produces a clean
epoch-vs-accuracy + loss curve PNG for the report.

Does NOT overwrite the existing trained model — saves to results/history_model/.

Run from project root:
    .venv/bin/python tests/training_history.py
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# Reuse the verbatim feature extractor + normalizer from train.py
sys.path.insert(0, str(ROOT / "backend"))
from predict import (  # noqa: E402
    normalize_text,
    extract_features,
    DigitalFootprintClassifier,
    MAX_LEN,
    MODEL_NAME,
    DEVICE,
)

DATASET_PATH = ROOT / "data" / "balanced_it_jobs_dataset_30k.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUBSET_TRAIN = 800
SUBSET_VAL = 200
EPOCHS = 5
BATCH_SIZE = 16
TRAIN_MAX_LEN = 128  # shorter sequences for faster CPU training
LR = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class JobDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = normalize_text(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=TRAIN_MAX_LEN,
            return_tensors="pt",
        )
        features = extract_features(text)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(features),
            "label": torch.tensor(self.labels[idx]),
        }


def evaluate(model, loader, criterion):
    model.eval()
    preds, trues = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            feat = batch["features"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            logits = model(input_ids, attn, feat)
            loss = criterion(logits, label)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            trues.extend(label.cpu().tolist())
    acc = accuracy_score(trues, preds)
    p, r, f, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }


def main():
    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    df["description"] = df["description"].astype(str)

    texts = df["description"].tolist()
    labels = df["label"].astype(int).tolist()

    # Stratified split — 4k train, 1k val
    X_pool, _, y_pool, _ = train_test_split(
        texts, labels,
        train_size=SUBSET_TRAIN + SUBSET_VAL,
        stratify=labels,
        random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=SUBSET_VAL,
        stratify=y_pool,
        random_state=SEED,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Device: {DEVICE}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds = JobDataset(X_train, y_train, tokenizer)
    val_ds = JobDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = DigitalFootprintClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        epoch_losses = []
        epoch_preds = []
        epoch_trues = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            feat = batch["features"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            logits = model(input_ids, attn, feat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_preds.extend(torch.argmax(logits, 1).detach().cpu().tolist())
            epoch_trues.extend(label.cpu().tolist())

        train_acc = accuracy_score(epoch_trues, epoch_preds)
        train_loss = float(np.mean(epoch_losses))
        val = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0
        history["train"].append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc})
        history["val"].append({"epoch": epoch, **val})
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
            f"val_f1={val['f1']:.4f}  ({elapsed:.0f}s)"
        )

    # Save history JSON
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved {RESULTS_DIR / 'training_history.json'}")

    # Plot
    epochs = [r["epoch"] for r in history["train"]]
    train_acc = [r["accuracy"] for r in history["train"]]
    val_acc = [r["accuracy"] for r in history["val"]]
    train_loss = [r["loss"] for r in history["train"]]
    val_loss = [r["loss"] for r in history["val"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, train_acc, "o-", label="Train accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, "s-", label="Validation accuracy", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Training & Validation Accuracy ({SUBSET_TRAIN} train / {SUBSET_VAL} val)")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(epochs)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_loss, "o-", label="Train loss", linewidth=2)
    ax2.plot(epochs, val_loss, "s-", label="Validation loss", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xticks(epochs)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out = RESULTS_DIR / "training_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
