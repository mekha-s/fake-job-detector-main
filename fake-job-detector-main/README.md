# Fake Job Posting Detector

Multimodal (text + image-OCR) detector for counterfeit job advertisements,
built around a fine-tuned **DistilBERT** classifier fused with 18
hand-engineered scam-signal features and a rule-based forensic layer that
explains *why* a posting was flagged.

Three-tier output: **REAL JOB** / **SUSPICIOUS** / **FAKE JOB** plus a
0–100 fraud probability score, the suspected scam pattern (e.g. *Telegram
Recruitment Scam*, *Advance Fee Job Scam*), highlighted suspicious phrases,
and a multi-line forensic report.

## Headline metrics

Final production system (NEW model + v3 fusion):

| Test set | Accuracy | F1 |
|---|---|---|
| EMSCAD held-out (n=283) | **90.46 %** | **0.911** |
| Adversarial 40-row | 70.00 % | 0.769 |

Full numbers, confusion matrices, training curves, and old-vs-new model
comparisons live in [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

## Project Layout

```
.
├── backend/
│   ├── api.py                    FastAPI app exposing POST /analyze
│   ├── predict.py                Inference + fused scoring + forensic rules
│   ├── train.py                  Original train loop (synthetic dataset)
│   ├── train_emscad.py           Retrain on real EMSCAD scams (production)
│   ├── evaluation.py             Dataset-level metrics
│   ├── extract_dataset.py        Builds the original balanced 30k CSV
│   └── build_emscad_dataset.py   Builds the EMSCAD 1:1 balanced subset
├── tests/
│   ├── adversarial_test.py       40 hand-curated novel scam vs real postings
│   ├── dataset_eval.py           In-distribution evaluation (3000-row sample)
│   ├── training_history.py       Captures epoch-vs-accuracy curve
│   ├── compare_models.py         OLD vs NEW × raw vs fused comparison
│   ├── compare_fusion_v2.py      Fusion variant ablation (v1 / v2 / v3)
│   ├── final_evaluation.py       Production-system eval pipeline
│   ├── generate_report.py        Aggregates earlier results to RESULTS.md
│   ├── test_bias.py              Sanity diagnostic on real/fake samples
│   └── test_search.py            DuckDuckGo company-lookup smoke test
├── data/                         (CSVs are gitignored — see Setup)
├── models/                       Original synthetic-trained artifacts (.pt gitignored)
├── models_emscad/                Production EMSCAD-trained artifacts (.pt gitignored)
├── frontend/                     React + Vite UI
├── results/                      All metrics, confusion matrices, plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Backend

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Python 3.10 specifically — torch 2.2.x (the latest with Intel-Mac
> wheels) doesn't ship for 3.13. On Linux / Apple Silicon you can use
> 3.11 or 3.12.

### 2. Datasets (gitignored — fetch separately)

`data/` is gitignored because the CSVs are 50–65 MB. Two sources:

- **EMSCAD** (production training data) — Employment Scam Aegean Dataset.
  Download via the [Erfaniaa mirror](https://github.com/Erfaniaa/fake-job-posting-detection/blob/master/dataset.csv)
  or from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
  Save to `data/emscad.csv`, then run:

  ```bash
  python backend/build_emscad_dataset.py
  # writes data/emscad_balanced.csv (~1414 rows, 1:1 balance)
  ```

- **Original synthetic-template dataset** (legacy) — `balanced_it_jobs_dataset_30k.csv`.
  Reproducible via `extract_dataset.py` if you have the source EMSCAD +
  LinkedIn postings zips.

### 3. Trained model weights (gitignored)

`models_emscad/model.pt` (~265 MB) is gitignored. Three options:

- **Retrain locally** (recommended, ~15-20 min on CPU):
  ```bash
  python backend/train_emscad.py
  ```
- Download the trained weights from the GitHub release artifacts (if published).
- Use Git LFS if you fork.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

## Running locally

```bash
# Terminal 1 — API on :8001
cd backend
uvicorn api:app --reload --port 8001

# Terminal 2 — UI on :5173 (Vite default)
cd frontend
npm run dev
```

The frontend calls `http://127.0.0.1:8001/analyze`
(`frontend/src/App.jsx:76`). Port 8001 is used because :8000 was already
in use by another local service during development; if 8000 is free for
you, change both spots back.

## API contract

`POST /analyze` accepts either `text` (multipart form field) or `image`
(file upload, processed via easyocr). Returns:

```json
{
  "prediction": "FAKE JOB",
  "score": 97,
  "bert_score": 97,
  "evidence_count": 8,
  "intention": "Possible Scam Recruitment",
  "pattern": "WhatsApp Work-From-Home Scam",
  "analysis": ["EVIDENCE: ...", "RISK: ...", ...],
  "words": ["whatsapp", "registration fee", ...],
  "digital_footprint": {"has_footprint": true, "company_name": "...", ...},
  "ocr_text": null
}
```

## Training & Evaluation

```bash
# Production retrain on EMSCAD (recommended — ~15-20 min on CPU)
python backend/train_emscad.py

# Full dataset evaluation
python tests/dataset_eval.py        # in-distribution metrics
python tests/adversarial_test.py    # 40-row adversarial set
python tests/final_evaluation.py    # generates results/FINAL_RESULTS.md
python tests/compare_models.py      # OLD vs NEW × raw vs fused
```

## How the fused score works

```
fused_prob = max(bert_prob, evidence_floor)

evidence_floor: ev=0  -> 0.0   (trust BERT)
                ev=1  -> 0.30
                ev=2  -> 0.55
                ev=3  -> 0.70
                ev>=4 -> 0.85

EXCEPT when ev=0 and BERT is mid-uncertain (0.4 < bert < 0.6),
down-weight to bert*0.6 to push borderline reals back to REAL.
```

The 18 hand-engineered features (in `backend/predict.py:extract_features`)
plus the keyword-driven `evidence_signal_count` capture: phone-number
contact, advance-fee patterns, whatsapp/telegram redirects, urgency,
free-email domains, suspicious work types (data-entry, WFH, no-interview),
earnings claims with cadence, identity / financial-doc harvesting
phrases, money-mule / reshipping patterns, pyramid / MLM language,
romance-pivot framing, and form-letter salutations.

## Known limitations

- **DistilBERT alone is not robust** to scam categories absent from
  training data. The rule-based forensic layer compensates by adding a
  recall floor for adversarial scam types (reshipping, modern crypto MLM,
  romance pivots, etc.).
- **EMSCAD is from 2014–2015** — modern scam vocabulary (NFTs, Telegram
  bots, OTP harvesting) requires periodic retraining.
- **`dangerouslySetInnerHTML` in the frontend highlight rendering**
  (`frontend/src/App.jsx:99–108`) doesn't HTML-escape user input before
  inserting `<span>` tags. Sanitize with DOMPurify in production.
- **Permissive CORS:** `api.py` uses `allow_origins=["*"]` — restrict to
  known origins in production.
- **The 100% adversarial accuracy** reported earlier in the development
  history (with v1 fusion) is on a 40-row hand-curated set and should
  not be cited as a generalisation guarantee.

## References

- [EMSCAD / Real-or-Fake Job Posting (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- [Detection of Fake Job Postings via ML and NLP — Springer](https://link.springer.com/article/10.1007/s11063-021-10727-z)
- [DistilBERT — Sanh et al. 2019](https://arxiv.org/abs/1910.01108)

## License

This project is provided as-is for academic / research use. See LICENSE
if/when added.
