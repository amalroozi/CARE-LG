# NHANES classifier metrics

Dataset: `nhanes_pooled` (primary NHANES dataset). Test-set metrics at threshold 0.5, mean (sd) over seeds 0-4.
Models retrained with the pipeline's seeds and split by `scripts/run_nhanes_classifier_metrics.py`.

| Metric | MLP | XGBoost |
|---|---|---|
| Precision | 0.922 (0.009) | 0.913 (0.002) |
| Recall (sensitivity) | 0.864 (0.015) | 0.874 (0.005) |
| Specificity | 0.927 (0.011) | 0.917 (0.002) |
| F1 | 0.892 (0.004) | 0.893 (0.003) |
| Accuracy | 0.896 (0.003) | 0.896 (0.003) |
| Balanced accuracy | 0.896 (0.003) | 0.896 (0.003) |
| MCC | 0.793 (0.005) | 0.792 (0.006) |
| AUROC | 0.964 (0.002) | 0.964 (0.001) |
| AUPRC | 0.969 (0.001) | 0.969 (0.001) |
| Brier | 0.074 (0.002) | 0.074 (0.002) |
| ECE | 0.013 (0.004) | 0.019 (0.001) |

The two models are effectively tied. The MLP is slightly better calibrated and more precise; XGBoost has slightly higher recall.

Per-seed rows (with confusion-matrix counts) are in `nhanes_classifier_metrics.csv`; 95% bootstrap CIs are in `nhanes_classifier_metrics.json`.

Reproduce (from the repo root):

```bash
.venv/bin/python -W ignore -m scripts.run_nhanes_classifier_metrics
```
