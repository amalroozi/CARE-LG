"""
Classification metrics for the NHANES risk models (MLP and XGBoost).

experiment_final/src/model.py only records AUC, Brier and accuracy. This script
retrains both models with the pipeline's own seeds and split (same code path as
run_seed.setup, minus the VAE/graph) and reports, on the held-out test set:
precision, recall (sensitivity), specificity, NPV, F1, accuracy, balanced
accuracy, MCC, AUROC, AUPRC, Brier, ECE and the confusion matrix, as mean +/- sd
over seeds with a 95% bootstrap CI on the pooled-seed mean.

Threshold 0.5 on P(y=1|x) for the label metrics (the pipeline's 0.55/0.45 gate
is a separate abstention rule, reported at the end as gate coverage).

Usage:
  .venv/bin/python -m scripts.run_nhanes_classifier_metrics [dataset] [variant]
  (dataset defaults to nhanes_pooled, the primary NHANES dataset)
Writes experiment_final/results/tables/nhanes_classifier_metrics.{csv,json}
"""
import json
import os
import sys
from pathlib import Path

# torch and XGBoost each bundle an OpenMP runtime; on macOS XGBoost.fit can deadlock after the MLP has trained.
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401,E402  (import before torch)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

from experiment_final import adapters
from experiment_final.src.common import load_config
from experiment_final.src.model import fit_mlp, fit_xgb
from experiment_final.src.prep import make_split
from src.seeding import set_all_seeds

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 1000
OUT_DIR = REPO_ROOT / "experiment_final" / "results" / "tables"


def ece(y, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins)))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "npv": div(tn, tn + fn), "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mcc": float(matthews_corrcoef(y, pred)), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_ci(y, p, rng, keys=("f1", "precision", "recall", "specificity", "auroc")):
    boots = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = metrics(y[i], p[i])
        for k in keys:
            boots[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boots.items()}


def main(dataset="nhanes_pooled", variant=""):
    cfg = load_config()
    hi, lo = cfg["thresholds"]["high"], cfg["thresholds"]["low"]
    rows = []
    for seed in SEEDS:
        set_all_seeds(seed)
        b = adapters.get(dataset).load(cfg)
        sp = make_split(b, seed, cfg["split"]["test_size"])
        Xfit, yfit = sp.Xtr[sp.fit], sp.ytr[sp.fit]
        models = {"mlp": fit_mlp(Xfit, yfit, seed), "xgboost": fit_xgb(Xfit, yfit, seed)}
        rng = np.random.default_rng(seed)
        for name, m in models.items():
            p = m.predict(sp.Xte)
            y = sp.yte.astype(int)
            r = {"dataset": dataset, "model": name, "seed": seed, "n_test": len(y), "prevalence": float(y.mean()),
                 **metrics(y, p), "gate_high_frac": float((p > hi).mean()), "gate_low_frac": float((p < lo).mean()),
                 "gate_abstain_frac": float(((p >= lo) & (p <= hi)).mean())}
            r["ci95"] = bootstrap_ci(y, p, rng)
            rows.append(r)
            print(f"seed {seed} {name:8s} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"Spec={r['specificity']:.3f} AUROC={r['auroc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("dataset", "model", "seed", "ci95")]
    summary = df.groupby("model")[num].agg(["mean", "std"]).round(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="ci95").to_csv(OUT_DIR / "nhanes_classifier_metrics.csv", index=False)
    (OUT_DIR / "nhanes_classifier_metrics.json").write_text(json.dumps(rows, indent=2))
    show = ["precision", "recall", "specificity", "f1", "accuracy", "balanced_accuracy", "mcc", "auroc", "auprc", "brier", "ece"]
    print(f"\n{dataset}: test metrics at threshold 0.5, mean (sd) over seeds {SEEDS}\n")
    print(pd.DataFrame({c: [f"{summary.loc[m, (c, 'mean')]:.3f} ({summary.loc[m, (c, 'std')]:.3f})" for m in summary.index]
                        for c in show}, index=summary.index).T.to_string())
    print(f"\nwrote {OUT_DIR / 'nhanes_classifier_metrics.csv'}")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
