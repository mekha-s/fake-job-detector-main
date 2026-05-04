import os
import re
import numpy as np
import pandas as pd
import torch
import spacy

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

print("--- DEBUG: IMPORTING DONE ---")


# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization for NVIDIA GPUs
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"RUNNING ON GPU: {torch.cuda.get_device_name(0)}")
else:
    print("RUNNING ON CPU (No GPU detected)")

MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = "../data/balanced_it_jobs_dataset_30k.csv"
SAVE_DIR = "../models"

MAX_LEN = 256
BATCH_SIZE = 32  # Increased for GPU
EPOCHS = 5
LR = 2e-5


# =====================================================
# LOAD SPACY
# =====================================================

nlp = spacy.load(
    "en_core_web_sm",
    disable=["tagger","parser","lemmatizer"]
)


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


FEATURE_DIM = 18


# =====================================================
# DATASET CLASS
# =====================================================

class JobDataset(Dataset):

    def __init__(self,texts,labels,tokenizer):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
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
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}")
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df["description"] = df["description"].astype(str)
    texts = df["description"].tolist()
    labels = df["label"].tolist()
    print("Dataset size:",len(texts))

    # SPLIT
    X_train,X_val,y_train,y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # TOKENIZER
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds = JobDataset(X_train,y_train,tokenizer)
    val_ds = JobDataset(X_val,y_val,tokenizer)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=(DEVICE.type == "cuda"), 
        num_workers=0  # Stability for Windows
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        pin_memory=(DEVICE.type == "cuda"), 
        num_workers=0
    )

    # MODEL
    model = DigitalFootprintClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
    criterion = nn.CrossEntropyLoss()

    # TRAIN LOOP
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader)
        for batch in loop:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["features"].to(DEVICE)
            )
            loss = criterion(logits, batch["label"].to(DEVICE))
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        # VALIDATION
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                    batch["features"].to(DEVICE)
                )
                preds.extend(torch.argmax(logits,1).cpu().tolist())
                trues.extend(batch["label"].tolist())

        acc = accuracy_score(trues,preds)
        p,r,f,_ = precision_recall_fscore_support(trues,preds,average="binary")
        print(f"\nValidation → Acc:{acc:.4f} P:{p:.4f} R:{r:.4f} F1:{f:.4f}")

    # SAVE
    os.makedirs(SAVE_DIR,exist_ok=True)
    torch.save(model.state_dict(),f"{SAVE_DIR}/model.pt")
    tokenizer.save_pretrained(SAVE_DIR)
    print("\nModel saved.")