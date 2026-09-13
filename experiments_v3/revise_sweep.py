"""
Phase C diagnostic: REVISE hyperparameter sweep (lambda_prox x lr), 25 UCI
seed-0 high-risk patients. Writes experiments_v3/results/revise_lr_sweep.csv.
See experiments_v3/lib/revise.py's module docstring for the interpretation
and final choice (lambda_prox=0.01, lr=0.05, steps=100).

Usage: .venv/bin/python -m experiments_v3.revise_sweep
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import experiments_v3.lib.dataset_registration  # noqa: F401
from experiments_v3.lib.pipeline_v3 import run_dataset_seed
from experiments_v3.lib.revise import run_revise

RESULTS_DIR = REPO_ROOT / "experiments_v3" / "results"


def main():
    run = run_dataset_seed("uci", 0, device=torch.device("cpu"))
    patients = run.high_risk_test_idx[:25]

    rows = []
    for lam in [0.001, 0.005, 0.01, 0.05]:
        for lr in [0.05, 0.1]:
            successes, dists, lats = [], [], []
            for pidx in patients:
                x0 = run.X_test[pidx]
                succ, x_final, z_final, lat = run_revise(x0, run.vae, run.classifier, run.device,
                                                          lr=lr, steps=150, lambda_prox=lam)
                successes.append(succ)
                lats.append(lat)
                if succ:
                    dists.append(np.linalg.norm(x_final - x0))
            rows.append({"lambda_prox": lam, "lr": lr, "n": len(patients),
                         "success_rate": np.mean(successes) * 100, "mean_latency": np.mean(lats),
                         "mean_feature_dist_success": np.mean(dists) if dists else np.nan})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "revise_lr_sweep.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
