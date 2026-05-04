# Final Evaluation — Production System

**Model:** DistilBERT-base + 18 hand-engineered features, retrained on
EMSCAD (Employment Scam Aegean Dataset, 707 real-labeled fakes paired
1:1 with reals from the same dataset).

**Fusion (v3):** `fused = max(bert_prob, evidence_floor)`
- evidence_floor ∈ {0, 0.30, 0.55, 0.70, 0.85} for ev=0..≥4
- ev=0 + BERT in (0.4, 0.6) → down-weight to bert × 0.6

## Headline numbers

| Test set                | Raw BERT  acc / F1 | Fused (v3)  acc / F1 |
|-------------------------|--------------------|----------------------|
| **EMSCAD held-out (n=283)** | **89.40%** / 0.902 | **90.46%** / 0.911 |
| Adversarial 40          | 62.50% / 0.727 | 70.00% / 0.769 |

## EMSCAD held-out (the academic headline)

### Raw BERT
- Accuracy: **0.8940**
- Precision: 0.8364
- Recall: 0.9787
- F1: 0.9020

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |       115 |        27 |
| **true FAKE** |         3 |       138 |

### Fused (v3)
- Accuracy: **0.9046**
- Precision: 0.8519
- Recall: 0.9787
- F1: 0.9109

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |       118 |        24 |
| **true FAKE** |         3 |       138 |

## Adversarial set (robustness check on novel scam types)

### Raw BERT
- Accuracy: **0.6250**
- Precision: 0.5714
- Recall: 1.0000
- F1: 0.7273

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |         5 |        15 |
| **true FAKE** |         0 |        20 |

### Fused (v3)
- Accuracy: **0.7000**
- Precision: 0.6250
- Recall: 1.0000
- F1: 0.7692

|              | pred REAL | pred FAKE |
|--------------|-----------|-----------|
| **true REAL** |         8 |        12 |
| **true FAKE** |         0 |        20 |

## Files in this folder

- `final_cm_emscad_raw.png`   confusion matrix — Raw BERT, EMSCAD held-out
- `final_cm_emscad_fused.png` confusion matrix — Fused, EMSCAD held-out
- `final_cm_adv_raw.png`      confusion matrix — Raw BERT, Adversarial
- `final_cm_adv_fused.png`    confusion matrix — Fused, Adversarial
- `final_comparison_bar.png`  4-cell comparison bar chart
- `emscad_training_curve.png` training history (5 epochs, EMSCAD retrain)
- `final_metrics.json`        raw numbers
