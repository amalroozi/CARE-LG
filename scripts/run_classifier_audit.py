"""
Phase 2 of experiments_v6_audit/DEEP_AUDIT.md's resolution (issue #4): audits
the RiskClassifier itself, which no prior session (v2-v6) ever independently
validated -- every previous claim (0% CVR, certified infeasibility, "this
patient is high-risk") assumed the classifier's own risk score was
trustworthy without checking.

Computes, for both cohorts, 5 seeds, held-out test set:
  - AUC-ROC
  - Brier score
  - A 10-bin calibration curve (predicted probability vs. observed frequency)
  - Subgroup AUC/Brier by sex (the one immutable feature every constraint
    check treats as sacred)
  - The empirical classifier-probability threshold at which observed outcome
    frequency crosses 7.5% (the PCE/ASCVD threshold used to build the data
    label itself), compared against the pipeline's hardcoded 0.45/0.55 gate

Writes experiments_v6/results/classifier_audit.json (raw numbers) and prints
a summary. experiments_v6/CLASSIFIER_AUDIT.md is written by hand from this
script's actual output -- no number in that document is invented.

Usage:
  .venv/bin/python -m experiments_v6.run_classifier_audit
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed

RESULTS_DIR = REPO_ROOT / "results" / "tables"
SEEDS = [0, 1, 2, 3, 4]
PCE_LABEL_THRESHOLD = 0.075  # 7.5% 10-yr ASCVD risk, the threshold used to build cvd_risk_flag / heart_disease_risk


def empirical_threshold_crossing(y_true, y_prob, target_freq, n_bins=20):
    """
    Sorts test patients by predicted probability into n_bins equal-width bins
    and finds the bin whose OBSERVED positive-outcome frequency first exceeds
    target_freq, returning that bin's predicted-probability midpoint as the
    empirical threshold. This directly answers "at what classifier
    probability does the real, observed outcome rate actually reach 7.5%?"
    """
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        obs_freq = y_true[mask].mean()
        if obs_freq >= target_freq:
            return {
                "bin_index": b, "bin_range": [float(edges[b]), float(edges[b+1])],
                "bin_midpoint": float((edges[b] + edges[b+1]) / 2),
                "n_in_bin": int(mask.sum()), "observed_freq_in_bin": float(obs_freq),
            }
    return None  # never crosses target_freq in any bin


def audit_one(dataset, seed):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    with torch.no_grad():
        y_prob = run.classifier(torch.tensor(run.X_test, dtype=torch.float32)).cpu().numpy().squeeze(-1)
    y_true = run.y_test.astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    sex_idx = run.feature_cols.index("sex")
    sex = run.X_test[:, sex_idx]
    subgroup = {}
    for label, mask in [("male_sex1", sex == 1), ("female_sex0", sex == 0)]:
        if mask.sum() >= 5 and len(np.unique(y_true[mask])) > 1:
            subgroup[label] = {
                "n": int(mask.sum()),
                "auc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "brier": float(brier_score_loss(y_true[mask], y_prob[mask])),
            }
        else:
            subgroup[label] = {"n": int(mask.sum()), "auc": None, "brier": None,
                                "note": "too few samples or single class to compute AUC"}

    crossing = empirical_threshold_crossing(y_true, y_prob, PCE_LABEL_THRESHOLD)

    return {
        "dataset": dataset, "seed": seed, "n_test": len(y_true),
        "auc_roc": auc, "brier_score": brier,
        "calibration_curve": {"mean_predicted": mean_pred.tolist(), "frac_positive_observed": frac_pos.tolist()},
        "subgroup_by_sex": subgroup,
        "empirical_75pct_crossing": crossing,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {"uci": [], "nhanes_real": []}
    for dataset in ["uci", "nhanes_real"]:
        for seed in SEEDS:
            r = audit_one(dataset, seed)
            all_results[dataset].append(r)
            cross = r["empirical_75pct_crossing"]
            cross_str = f"{cross['bin_midpoint']:.3f}" if cross else "never crosses"
            print(f"[audit] {dataset} seed={seed}: AUC={r['auc_roc']:.4f} Brier={r['brier_score']:.4f} "
                  f"empirical-7.5%-crossing(classifier prob)={cross_str}", flush=True)

    with open(RESULTS_DIR / "classifier_audit.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[audit] === SUMMARY (mean +/- std over 5 seeds) ===")
    for dataset in ["uci", "nhanes_real"]:
        aucs = [r["auc_roc"] for r in all_results[dataset]]
        briers = [r["brier_score"] for r in all_results[dataset]]
        crossings = [r["empirical_75pct_crossing"]["bin_midpoint"] for r in all_results[dataset] if r["empirical_75pct_crossing"]]
        print(f"{dataset:12s} AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  Brier={np.mean(briers):.4f}+/-{np.std(briers):.4f}  "
              f"empirical-crossing(n={len(crossings)}/5 seeds)={np.mean(crossings):.3f}+/-{np.std(crossings):.3f}" if crossings else
              f"{dataset:12s} AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  Brier={np.mean(briers):.4f}+/-{np.std(briers):.4f}  empirical-crossing: never observed in any seed")


if __name__ == "__main__":
    main()
