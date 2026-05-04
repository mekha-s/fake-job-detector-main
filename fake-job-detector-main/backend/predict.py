"""
Fake Job Posting Detector — inference module.

Reconstructed from train.py (verbatim model + features) and the original
predict.cpython-310.pyc disassembly (verbatim branching logic for the
forensic / scam-pattern / hiring-intention reports).
"""

import os
import re

import cv2
import easyocr
import numpy as np
import spacy
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizerFast

print("### FAKE JOB DETECTOR STARTED ###")


# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models_emscad"),
)
MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 256
FEATURE_DIM = 18


# =====================================================
# CONSTANTS
# =====================================================

SCAM_WORDS = {
    "whatsapp", "telegram", "registration fee", "processing fee",
    "urgent hiring", "instant joining", "no interview", "earn daily",
    "limited seats", "data entry", "online job", "part time",
}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
}


# =====================================================
# LOAD NLP / OCR / TOKENIZER
# =====================================================

print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

print("Loading OCR engine...")
ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE.type == "cuda"))


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("gmaiI", "gmail")
    text = text.replace("gmai1", "gmail")
    text = re.sub(r"[^a-z0-9@. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================================================
# FEATURE EXTRACTION (matches train.py — 18 dims)
# =====================================================

def extract_features(text: str, doc=None) -> np.ndarray:
    t = text.lower()
    if doc is None:
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

    return np.array(
        [
            orgs, gpes, free_email, corporate_email, domain_score,
            url_count, linkedin_score, scam_score,
            phone, salary, urgent, whatsapp, telegram,
            fee, data_entry, wfh, no_interview, has_contact,
        ],
        dtype=np.float32,
    )


# =====================================================
# MODEL (architecture verbatim from train.py)
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
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        fused = torch.cat([mean_pooled, features], dim=1)
        return self.classifier(fused)


print("Loading trained model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DigitalFootprintClassifier().to(DEVICE)
model_path = os.path.join(MODEL_DIR, "model.pt")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Model ready.")
else:
    print(f"Warning: Model weights not found at {model_path}. Please train the model first.")


# =====================================================
# OCR — multi-variant preprocessing for robustness
# =====================================================

def _build_image_variants(img):
    """Yield (label, image, scale) variants for OCR."""
    variants = []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Variant 1: raw grayscale at native scale
    variants.append(("gray-1x", gray, 1.0))

    # Variant 2: 2x upscale + CLAHE contrast
    big = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe-2x", clahe.apply(big), 2.0))

    # Variant 3: Otsu binarization at 2x
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    variants.append(("otsu-2x", otsu, 2.0))

    # Variant 4: Gaussian-smoothed at 2x
    blur = cv2.GaussianBlur(big, (3, 3), 0)
    variants.append(("blur-2x", blur, 2.0))

    return variants


def _run_ocr(img):
    return ocr_reader.readtext(img)


def _score_detections(detections):
    """Sum of (text length × confidence) across detections."""
    if not detections:
        return 0.0
    return sum(len(text) * float(conf) for _, text, conf in detections)


def _detections_to_text(detections, line_snap: int) -> str:
    """Group detections into lines by Y-coordinate proximity, then join."""
    if not detections:
        return ""
    items = []
    for bbox, text, _conf in detections:
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        items.append((min(ys), min(xs), text))
    items.sort(key=lambda x: (round(x[0] / max(line_snap, 1)), x[1]))

    lines = []
    cur_y = None
    cur_line = []
    for y, _x, t in items:
        bucket = round(y / max(line_snap, 1))
        if cur_y is None or bucket == cur_y:
            cur_line.append(t)
            cur_y = bucket
        else:
            lines.append(" ".join(cur_line))
            cur_line = [t]
            cur_y = bucket
    if cur_line:
        lines.append(" ".join(cur_line))
    return "\n".join(lines)


def extract_text_from_image(path: str) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""

    variants = _build_image_variants(img)

    best_score = -1
    best_detections = []
    best_scale = 1.0

    for label, variant_img, scale in variants:
        try:
            detections = _run_ocr(variant_img)
            score = _score_detections(detections)
            print(f"  OCR variant '{label}': {len(detections)} detections, score={score:.1f}")
            if score > best_score:
                best_score = score
                best_detections = detections
                best_scale = scale
        except Exception as e:  # noqa: BLE001
            print(f"  OCR variant '{label}' failed: {e}")

    line_snap = int(22 * best_scale)
    raw_text = _detections_to_text(best_detections, line_snap)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    raw_text = re.sub(r"[ \t]{2,}", " ", raw_text)
    return raw_text.strip()


# =====================================================
# PREDICTION
# =====================================================

def predict(text: str, doc=None) -> float:
    """Predict probability that the text is a fake job posting (0..1)."""
    if not text or len(text.strip()) < 10:
        return 0.0

    normalized = normalize_text(text)
    features_vec = extract_features(normalized, doc=doc)
    features_tensor = torch.tensor(np.array([features_vec]), dtype=torch.float32).to(DEVICE)

    enc = tokenizer(
        normalized,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], features_tensor)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    return prob_fake


# =====================================================
# HIRING INTENTION (rules — verbatim from .pyc)
# =====================================================

def detect_hiring_intention(text: str) -> str:
    text = text.lower()

    if "send resume" in text or "submit cv" in text:
        return "Resume Collection"

    if "visit website" in text:
        return "Brand Promotion"

    if "earn daily" in text or "work from home" in text:
        return "Possible Scam Recruitment"

    if "requirements" in text or "experience" in text:
        return "Genuine Hiring"

    return "Unclear Hiring Intention"


# =====================================================
# SCAM PATTERN (rules — verbatim from .pyc)
# =====================================================

def detect_scam_pattern(text: str) -> str:
    t = text.lower()

    if "telegram" in t:
        return "Telegram Recruitment Scam"

    if "whatsapp" in t and "earn" in t:
        return "WhatsApp Work-From-Home Scam"

    if "registration fee" in t or "processing fee" in t:
        return "Advance Fee Job Scam"

    if "send resume" in t and "gmail.com" in t:
        return "Resume Harvesting Scam"

    return "No specific scam pattern detected"


# =====================================================
# FORENSIC ANALYSIS (rules — verbatim from .pyc)
# =====================================================

def forensic_analysis(text: str, prob: float) -> list:
    """Generate a dynamic forensic report based on the 18 signals."""
    report = []
    t = text.lower()

    if prob > 0.85:
        report.append("CRITICAL: Neural engine detected high-confidence fraud signatures")
    elif prob > 0.65:
        report.append("WARNING: Model detected suspicious structural patterns")
    elif prob < 0.35:
        report.append("INFO: Model identifies this as a standard legitimate job layout")

    if re.search(r"\b\d{10}\b", t):
        report.append("EVIDENCE: Direct mobile number contact (common in unofficial recruitment)")

    if "fee" in t and ("registration" in t or "processing" in t):
        report.append("EVIDENCE: Direct request for advance payment (High-Risk Scam)")
    elif re.search(r"(₹|\$|rs\.|salary)", t):
        report.append("NOTICE: Specific salary/currency details mentioned")

    if "whatsapp" in t:
        report.append("RISK: Recruitment redirected to WhatsApp (unencrypted/private channel)")

    if "telegram" in t:
        report.append("RISK: Recruitment redirected to Telegram (common for untraceable scams)")

    if any(w in t for w in ("urgent", "immediate", "instant", "walk-in")):
        report.append("PATTERN: Artificial sense of urgency used to bypass scrutiny")

    emails = re.findall(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+)", t)
    if any(e in FREE_EMAIL_DOMAINS for e in emails):
        report.append("RISK: Non-corporate/free email domain used for official business")

    if "data entry" in t:
        report.append("PATTERN: High-frequency 'Data Entry' scam template detected")
    if "work from home" in t:
        report.append("NOTICE: Remote/WFH offer (requires secondary verification)")

    return report


# =====================================================
# EVIDENCE COUNT (rule-based, independent of BERT prob)
# Used for forensic-fusion scoring in api.py — see fused_score().
# =====================================================

_EXTENDED_FREE_EMAIL = FREE_EMAIL_DOMAINS | {
    "aol.com", "icloud.com", "protonmail.com", "yandex.com", "yandex.ru",
    "mail.ru", "live.com", "msn.com", "rediffmail.com", "zoho.com",
}


# Each tuple: (category, list of phrases). Every phrase that matches the
# normalized text contributes 1 evidence point. Categories are independent.
_EVIDENCE_PHRASES = [
    # Advance-fee / pay-to-apply
    ("fee_words", [
        "registration fee", "processing fee", "activation fee", "starter kit",
        "documentation processing fee", "background check fee", "testing fee",
        "orientation materials", "verification fee", "application processing fee",
        "training kit",
    ]),
    # Comms-channel redirect (each platform counts separately)
    ("comms", [
        "whatsapp", "telegram", "dm me", "tap link in bio", "cashapp",
        "venmo", "zelle me", "@gmail.com", "@yahoo.com", "@aol.com",
    ]),
    # Urgency / pressure (each phrase counts separately)
    ("urgency", [
        "urgent", "immediate", "instant", "walk-in", "limited time",
        "limited positions", "limited seats", "act now", "apply now",
        "today only", "secure your spot", "claim your",
    ]),
    # Suspicious work types
    ("work_type", [
        "data entry", "work from home", "work from anywhere",
        "no interview", "skip the interview", "no experience needed",
        "earn daily", "earn weekly",
    ]),
    # Identity / financial doc harvesting (each item counts).
    # NB: "driver's license" alone is too common in legitimate field-tech /
    # CDL postings — only count the scam-specific phrasings.
    ("identity_harvest", [
        "social security", "ssn", "voided check", "passport copy",
        "passport scan", "copy of your id", "scan of your id",
        "two recent paystubs", "recent paystubs",
        "photo of your driver", "photo of your id",
        "copy of your driver", "scan of your driver",
        "bank info", "online banking login",
        "routing and account number", "full name, home address",
    ]),
    # Money-mule / reshipping / fake check / pay-to-start (each phrase counts)
    ("money_mule", [
        "wire transfer", "wire $", "wire the", "western union",
        "send the balance", "deposit it", "deposit the check",
        "deposit and wire", "purchase gift cards", "buy gift cards",
        "forward using prepaid", "prepaid labels", "repackage",
        "reshipment", "send the remaining", "zelle the remaining",
        "keep the rest", "keep $",
        # Pay-to-start / unlock-the-platform scams
        "deposit to your account", "deposit to your company",
        "unlock the trading", "unlock your tester",
        "company-issued account", "unlock the platform",
        "trade forex", "trade crypto", "trade cryptocurrency",
        "no license required",
    ]),
    # Pyramid / MLM
    ("pyramid", [
        "downline", "recruit two", "recruit your friends", "starter kit",
        "be your own boss", "no cap on earnings", "top earners",
        "top consultants", "build your team", "wellness consultant",
        "brand ambassador", "lifetime access", "join our team of",
        "60% commission", "30% commission", "free product",
    ]),
    # Romance / exploitation framing
    ("romance", [
        "discreet companion", "wealthy gentleman", "mature professional",
        "send a photo", "send recent photo", "send your photo",
        "send a recent photo", "models needed", "easy money",
        "all expenses paid", "set your own hours",
        "attractive females", "premium online platform",
    ]),
]


def evidence_signal_count(text: str) -> int:
    """Count distinct scam-evidence signals (rule-based, BERT-independent).

    Each individual phrase / signal contributes one evidence point. Categories
    are independent and stack — the more signals fire, the higher the score.
    """
    # Lowercase and collapse all whitespace to single spaces so substring
    # matches don't fail on line wraps in pasted postings or OCR output.
    t = re.sub(r"\s+", " ", text.lower())
    count = 0

    # Phrase-table matches
    for _category, phrases in _EVIDENCE_PHRASES:
        for phrase in phrases:
            if phrase in t:
                count += 1

    # 10-digit phone contact
    if re.search(r"\b\d{10}\b", t):
        count += 1

    # Generic "$NN fee/deposit/orientation/etc" pattern (allow ≤4 words gap)
    if re.search(
        r"\$\s?\d{2,4}\s+(?:\w+\s+){0,4}"
        r"(?:fee|deposit|cost|charge|payment|activation|"
        r"orientation|materials|kit|background|verification|documentation|"
        r"voucher|application|training)",
        t,
    ):
        count += 1

    # Refundable / "refunded after" framing
    if re.search(r"\b(?:refundable|refunded\s+after|fully\s+refundable)\b", t) and \
       re.search(r"\$\s?\d", t):
        count += 1

    # Free email domain used for "official" recruitment
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+)", t)
    if any(e in _EXTENDED_FREE_EMAIL for e in emails):
        count += 1

    # Earnings claims (must include cadence — "pay $28/hr" alone is normal).
    # Range with daily/weekly/monthly cadence:
    if re.search(
        r"(?:\$|rs\.|₹)\s?\d[\d,]*[kK]?\s*[-–/]?\s*(?:\$|rs\.|₹)?\s?\d*[\d,]*[kK]?\s*"
        r"(?:per\s+)?(?:daily|weekly|monthly|per\s+day|per\s+week|per\s+month)",
        t,
    ):
        count += 1
    # Earn / make / clear + amount + cadence (must include cadence to suppress legit hourly):
    if re.search(
        r"(?:earn|make|clear|cash|earn\s+up\s+to|net|pocket)\s+(?:up\s+to\s+)?"
        r"(?:\$|rs\.|₹)?\s?[\d,]+[kKmM]?\s*"
        r"(?:[-–]\s*(?:\$|rs\.|₹)?\s?[\d,]+[kKmM]?\s*)?"
        r"(?:daily|weekly|monthly|per\s+(?:day|week|month|assignment|test)|"
        r"in\s+(?:your\s+)?first\s+(?:month|week|payout|dispatch))",
        t,
    ):
        count += 1
    # Per-task earnings ("$200 per assignment", "$400 per test")
    if re.search(r"(?:\$|rs\.|₹)\s?\d[\d,]*\s+per\s+(?:assignment|test|task|gig)", t):
        count += 1
    # Deposit $X to / deposit a $X (pay-to-start scam pattern)
    if re.search(r"\bdeposit(?:\s+(?:the|a|your))?\s+(?:\$|rs\.|₹)\s?\d", t):
        count += 1

    # Generic form-letter salutation
    if re.search(r"^\s*dear\s+(?:candidate|applicant|sir/madam|valued|user)", t):
        count += 1

    # Congratulations / pre-approved framing
    if re.search(r"\b(?:congratulations|your resume has been shortlisted|pre[\s-]?approved|"
                 r"selected for the position)",
                 t):
        count += 1

    return count


def fused_score(text: str, doc=None):
    """Return (bert_prob, fused_prob, evidence_count).

    Compromise (v3) fusion — calibrated for the EMSCAD-trained model:
        fused = max(bert_prob, evidence_floor)
        where evidence_floor: ev=0 -> 0  (trust BERT)
                              ev=1 -> 0.30
                              ev=2 -> 0.55
                              ev=3 -> 0.70
                              ev>=4 -> 0.85

        EXCEPT when ev=0 and BERT is mid-uncertain (0.4 < bert < 0.6),
        down-weight to bert*0.6 to push borderline reals back to REAL.

    Rationale: with the EMSCAD-trained BERT, the model is well-calibrated
    on real-world scam patterns (89.4% raw on EMSCAD held-out). The rule
    layer adds a recall floor for adversarial scam categories EMSCAD
    didn't cover (reshipping, modern crypto MLM, romance pivots, etc.).
    """
    bert_prob = predict(text, doc=doc)
    evidence = evidence_signal_count(text)

    if evidence >= 4:
        floor = 0.85
    elif evidence == 3:
        floor = 0.70
    elif evidence == 2:
        floor = 0.55
    elif evidence == 1:
        floor = 0.30
    else:
        floor = 0.0
        if 0.4 < bert_prob < 0.6:
            return bert_prob, bert_prob * 0.6, evidence

    fused = max(bert_prob, floor)
    return bert_prob, fused, evidence


# =====================================================
# HIGHLIGHT SCAM WORDS (verbatim from .pyc)
# =====================================================

def highlight_scam_words(text: str) -> list:
    """Return list of suspicious tokens (keywords + phone numbers + currency markers)."""
    found = set()
    t = text.lower()

    for w in SCAM_WORDS:
        if w in t:
            found.add(w)

    for p in re.findall(r"\b\d{10}\b", t):
        found.add(p)

    if "fee" in t:
        found.add("fee")

    for c in re.findall(r"(₹|\$|rs\.)", t):
        found.add(c)

    return list(found)


# =====================================================
# DIGITAL FOOTPRINT (DDGS — gated by perform_search)
# =====================================================

# Words that spaCy tags as ORG but are clearly noise in job postings.
_ORG_BLACKLIST = frozenset({
    "age", "experience", "hiring", "office", "location", "knowledge", "urgent",
    "education", "requirements", "contact", "vacancy", "interview", "company",
    "management", "hours", "academic", "sunday", "immediate", "shift",
    "saturday", "technical", "resume", "full time", "skills", "details",
    "interviews", "cv", "apply", "walk-in", "candidates", "gender", "job",
    "joining", "working", "time", "per month", "monday", "required",
    "salary", "qualifications", "friday", "night", "walk", "address", "day",
})


def check_digital_footprint(text: str, doc=None, perform_search: bool = False) -> dict:
    """Identify likely company name(s) from spaCy ORGs; optionally validate via DDGS."""
    if doc is None:
        doc = nlp(text)

    raw_orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
    valid_orgs = []

    for org in raw_orgs:
        org_clean = org.strip().lower()
        if org_clean in _ORG_BLACKLIST:
            continue
        if len(org_clean) < 2 or len(org_clean) > 35:
            continue
        if len(org.split()) > 5:
            continue
        if not re.search(r"[a-zA-Z]", org):
            continue
        valid_orgs.append(org.strip())

    seen = set()
    unique_orgs = [x for x in valid_orgs if not (x in seen or seen.add(x))]

    result = {
        "has_footprint": False,
        "company_name": None,
        "search_results": [],
    }

    if not unique_orgs:
        return result

    company_name = unique_orgs[0]
    result["company_name"] = company_name

    if not perform_search:
        result["has_footprint"] = True
        result["search_results"] = []
        return result

    try:
        from duckduckgo_search import DDGS

        query = f'"{company_name}" company official website'
        search_results = DDGS().text(query, max_results=3)
        if search_results:
            result["has_footprint"] = True
            for r in search_results:
                result["search_results"].append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:  # noqa: BLE001
        print(f"Web search failed: {e}")

    return result
