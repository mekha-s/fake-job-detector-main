# Fake Job Detector — Evaluation Results

Generated automatically by `tests/generate_report.py`.

## 1. Headline 4-cell results

|                | **Raw BERT** | **Fused (BERT + rule layer)** |
|----------------|--------------|-------------------------------|
| **In-distribution (n=3000)** | acc 98.53% · F1 0.986 | acc 80.13% · F1 0.759 |
| **Adversarial (n=40)**       | acc 57.50% · F1 0.452 | acc 100.00% · F1 1.000 |

> **Reading**: Raw BERT memorises the training distribution (98 % in-dist) but
> generalises poorly to novel scams (57 % adversarial). The fused score
> deliberately distrusts BERT when no rule-based evidence is found, so it
> trades some in-distribution recall for far stronger adversarial robustness.

## 2. Detailed metrics

### 2a. Raw BERT — In-distribution (n=3000)
- Accuracy : **0.9853**
- Precision: 0.9727
- Recall   : 0.9987
- F1 score : 0.9855

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |      1458 |        42 |
| **true FAKE** |         2 |      1498 |

### 2b. Fused — In-distribution (n=3000)
- Accuracy : **0.8013**
- Precision: 0.9669
- Recall   : 0.6240
- F1 score : 0.7585

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |      1468 |        32 |
| **true FAKE** |       564 |       936 |

### 2c. Raw BERT — Adversarial (n=40)
- Accuracy : **0.5750**
- Precision: 0.6364
- Recall   : 0.3500
- F1 score : 0.4516

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |        16 |         4 |
| **true FAKE** |        13 |         7 |

### 2d. Fused — Adversarial (n=40)
- Accuracy : **1.0000**
- Precision: 1.0000
- Recall   : 1.0000
- F1 score : 1.0000

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |        20 |         0 |
| **true FAKE** |         0 |        20 |

## 3. Training history (small-subset retrain for report)

A separate retrain on a stratified 800-train / 200-val subset, sequence
length 128, batch 16, AdamW lr=2e-5, 5 epochs.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|------:|-----------:|----------:|---------:|--------:|-------:|
| 1 | 0.4584 | 0.8125 | 0.3671 | 0.8600 | 0.8409 |
| 2 | 0.2901 | 0.9050 | 0.3378 | 0.8800 | 0.8667 |
| 3 | 0.2051 | 0.9387 | 0.3254 | 0.8650 | 0.8475 |
| 4 | 0.1340 | 0.9750 | 0.3661 | 0.8550 | 0.8612 |
| 5 | 0.0738 | 0.9912 | 0.3647 | 0.8750 | 0.8663 |

The val accuracy peaks around epoch 2 and starts oscillating thereafter
while train accuracy keeps climbing — a classic overfitting signature
consistent with the synthetic-template hypothesis discussed in the
limitations section. See `training_curve.png`.

## 4. Files in this folder

- `cm_raw_indist.png`        confusion matrix — raw BERT, in-distribution
- `cm_fused_indist.png`      confusion matrix — fused, in-distribution
- `cm_raw_adversarial.png`   confusion matrix — raw BERT, adversarial
- `cm_fused_adversarial.png` confusion matrix — fused, adversarial
- `training_curve.png`       train/val accuracy + loss across 5 epochs
- `comparison_bar.png`       4-cell metric grid as a grouped bar chart
- `in_distribution_metrics.json`  raw numbers behind §2a, §2b
- `training_history.json`    raw numbers behind §3

## 5. Caveats for the report

1. The adversarial set is hand-curated (n=40). A pass at 100 % does not
   generalise to all unseen scams — realistic field accuracy is likely
   85-95 %. State this explicitly in your Limitations section.
2. The training curve is from a small subset retrain (800 train / 200 val,
   seq_len 128) so it converges fast. The shape is representative of the
   full-dataset training pattern; absolute numbers should not be quoted as
   the production model's training history.
3. The original `models/model.pt` was not retrained — these results are
   evaluated against the model already in the repo.
