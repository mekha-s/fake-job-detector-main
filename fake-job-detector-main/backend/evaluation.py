# =====================================================
# FAKE JOB DETECTOR MODEL EVALUATION (ADVANCED VERSION)
# =====================================================

import os
import re
import torch
import numpy as np
import pandas as pd
import spacy

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "../models"
MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = "../data/balanced_it_jobs_dataset_30k.csv"

MAX_LEN = 256
BATCH_SIZE = 32
FEATURE_DIM = 18

# =====================================================
# LOAD SPACY
# =====================================================

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","lemmatizer"])

# =====================================================
# CONSTANTS
# =====================================================

SCAM_WORDS = {
    "whatsapp", "telegram", "registration fee", "processing fee",
    "urgent hiring", "instant joining", "no interview", "earn daily",
    "limited seats", "data entry", "online job", "part time"
}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com"
}

# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text):
    text = text.lower()
    text = text.replace("gmaiI", "gmail")
    text = text.replace("gmai1", "gmail")
    text = re.sub(r"[^a-z0-9@. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(text):
    t = text.lower()
    doc = nlp(text)

    orgs = len({e.text for e in doc.ents if e.label_ == "ORG"})
    gpes = len({e.text for e in doc.ents if e.label_ == "GPE"})

    emails = re.findall(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+)", text)

    free_email = sum(e in FREE_EMAIL_DOMAINS for e in emails)
    corporate_email = sum(e not in FREE_EMAIL_DOMAINS for e in emails)

    domain_score = (2 * corporate_email) - free_email

    url_count = len(re.findall(r"http[s]?://", t))
    linkedin_score = 1 if "linkedin.com" in t else 0
    scam_score = sum(w in t for w in SCAM_WORDS)

    phone = 1 if re.search(r"\b\d{10}\b", t) else 0
    salary = 1 if re.search(r"(₹|\$|rs\.|salary)", t) else 0
    urgent = 1 if any(w in t for w in ["urgent", "immediate", "instant"]) else 0
    whatsapp = 1 if "whatsapp" in t else 0
    telegram = 1 if "telegram" in t else 0
    fee = 1 if "fee" in t and ("registration" in t or "processing" in t) else 0
    data_entry = 1 if "data entry" in t else 0
    wfh = 1 if "work from home" in t else 0
    no_interview = 1 if "no interview" in t else 0
    has_contact = 1 if "@" in t or phone == 1 else 0

    return np.array([
        orgs, gpes, free_email, corporate_email, domain_score,
        url_count, linkedin_score, scam_score,
        phone, salary, urgent, whatsapp, telegram,
        fee, data_entry, wfh, no_interview, has_contact
    ], dtype=np.float32)

# =====================================================
# DATASET
# =====================================================

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
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        features = extract_features(text)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(features),
            "label": torch.tensor(self.labels[idx])
        }

# =====================================================
# MODEL
# =====================================================

class DigitalFootprintClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Linear(768 + FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Masked Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        fused = torch.cat([mean_pooled, features], dim=1)
        return self.classifier(fused)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset {DATASET_PATH} not found.")
        exit(1)
        
    df = pd.read_csv(DATASET_PATH)
    df["description"] = df["description"].astype(str)
    
    texts = df["description"].tolist()
    labels = df["label"].tolist()

    print("Loading tokenizer...")
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory {MODEL_DIR} not found. Please train first.")
        exit(1)
        
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    
    dataset = JobDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print("Loading trained model...")
    model = DigitalFootprintClassifier().to(DEVICE)
    
    model_path = os.path.join(MODEL_DIR, "model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        exit(1)
        
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print("Running evaluation...")
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["features"].to(DEVICE)
            )
            p = torch.argmax(logits, 1).cpu().numpy()
            preds.extend(p)
            trues.extend(batch["label"].numpy())

    # Metrics
    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average="binary")

    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS (STRENGTHENED)")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(trues, preds))
    print("\nClassification Report:")
    print(classification_report(trues, preds))