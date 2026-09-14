"""
Option 3 diagnostic (per the brief): for every certified-infeasible patient,
quantify whether their risk ceiling is driven by AGE (a non-decreasing,
un-freezable factor) or by DATA SCARCITY (no single real patient combines
the best individually-observed values, even though such a combination
would work if it existed).

Method: build a "best-case, age-frozen" synthetic patient for each
certified-infeasible query --
  - immutable features (sex) and age: held FIXED at the patient's own value
    (not just non-decreasing -- literally unchanged, testing "can risk drop
    below threshold without touching age or sex at all")
  - each directional_reduce feature: set to the MINIMUM value observed
    anywhere in the REAL training population (not an arbitrary/unbounded
    synthetic extreme -- grounded in an actually-recorded value)
  - each directional_increase feature: set to the MAXIMUM value observed
    anywhere in the REAL training population
Then check the classifier's predicted risk on this best-case combination.

  - best-case risk >= LOW_RISK_THRESHOLD  -> AGE-CEILING: even the best
    achievable combination of every other real-observed value, holding age
    fixed, is not enough. A directed synthetic search (Stage 2) is unlikely
    to help this patient either, since it's not a data-coverage problem.
  - best-case risk <  LOW_RISK_THRESHOLD  -> DATA-SCARCITY: a
    risk-reducing combination provably exists (it's built from real,
    individually-observed values) but no single real patient happens to
    combine them. A directed synthetic search (Stage 2) should be able to
    find something in this region.

No graph search, no training beyond the standard pipeline -- classifier
forward passes only, fast.

Usage:
  .venv/bin/python -m scripts.run_age_ceiling_diagnostic
"""
import json
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed, LOW_RISK_THRESHOLD
from src.graph.clinical_constraints import unscale_features

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DATASETS = ["uci", "nhanes_real"]
SEEDS = [0, 1, 2, 3, 4]


def best_case_age_frozen(x0, X_train, feature_metadata, feature_cols):
    """Returns a synthetic scaled-feature vector: age/immutable fixed at x0's
    value, every directional feature set to the best REAL value observed in
    X_train (min for directional_reduce, max for directional_increase)."""
    x_best = x0.copy()
    for f in feature_metadata.get('directional_reduce', []):
        if f in feature_cols:
            idx = feature_cols.index(f)
            x_best[idx] = X_train[:, idx].min()
    for f in feature_metadata.get('directional_increase', []):
        if f in feature_cols:
            idx = feature_cols.index(f)
            x_best[idx] = X_train[:, idx].max()
    # age (non_decreasing) and immutable features: left untouched (x_best already = x0 there)
    return x_best


def main():
    all_rows = []
    for dataset in DATASETS:
        blindspot_path = TABLES_DIR / f"{dataset}_blindspot.csv"
        blindspot = pd.read_csv(blindspot_path)
        cert = blindspot[blindspot.certified_infeasible == True]

        for seed in SEEDS:
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            cert_seed = cert[cert.seed == seed]
            if len(cert_seed) == 0:
                continue
            t0 = time.perf_counter()
            for pidx in cert_seed.patient_idx:
                x0 = run.X_test[int(pidx)]
                x_best = best_case_age_frozen(x0, run.X_train, run.feature_metadata, run.feature_cols)
                with torch.no_grad():
                    risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
                    risk_best = float(run.classifier(torch.tensor(x_best, dtype=torch.float32).unsqueeze(0)).item())
                category = "age_ceiling" if risk_best >= LOW_RISK_THRESHOLD else "data_scarcity"
                all_rows.append({
                    "dataset": dataset, "seed": seed, "patient_idx": int(pidx),
                    "risk_before": risk_before, "risk_best_case_age_frozen": risk_best,
                    "category": category,
                })
            print(f"[age-ceiling] {dataset} seed={seed}: {len(cert_seed)} certified-infeasible patients checked "
                  f"in {time.perf_counter()-t0:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "age_ceiling_diagnostic.csv", index=False)

    print("\n[age-ceiling] === SUMMARY (mean +/- std over seeds) ===")
    summary = {}
    for dataset in DATASETS:
        d = df[df.dataset == dataset]
        per_seed_age = d.groupby("seed").apply(lambda g: (g.category == "age_ceiling").mean() * 100)
        per_seed_scarce = d.groupby("seed").apply(lambda g: (g.category == "data_scarcity").mean() * 100)
        summary[dataset] = {
            "n_certified_infeasible_checked": len(d),
            "age_ceiling_pct_mean": float(per_seed_age.mean()), "age_ceiling_pct_std": float(per_seed_age.std()),
            "data_scarcity_pct_mean": float(per_seed_scarce.mean()), "data_scarcity_pct_std": float(per_seed_scarce.std()),
            "mean_risk_best_case": float(d.risk_best_case_age_frozen.mean()),
            "mean_risk_before": float(d.risk_before.mean()),
        }
        print(f"{dataset:12s} n={len(d):4d}  age_ceiling={per_seed_age.mean():.1f}+/-{per_seed_age.std():.1f}%  "
              f"data_scarcity={per_seed_scarce.mean():.1f}+/-{per_seed_scarce.std():.1f}%  "
              f"mean_risk_before={d.risk_before.mean():.3f}  mean_risk_best_case={d.risk_best_case_age_frozen.mean():.3f}")

    with open(TABLES_DIR / "age_ceiling_diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[age-ceiling] wrote results/tables/age_ceiling_diagnostic.csv and _summary.json")


if __name__ == "__main__":
    main()
