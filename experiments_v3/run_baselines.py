"""
Phase C+5 (v3): baselines including a genuinely-implemented REVISE, run
under the same seeded pipeline (type-aware VAE, real NHANES) and same
high-risk test patients as the main grid.

DiCE / FACE / GrowingSpheres are reused UNCHANGED from
benchmarks/run_benchmarks.py, exactly as in experiments_v2. REVISE
(experiments_v3.lib.revise.run_revise) is new, faithful, gradient-based
latent-space search sharing the SAME trained VAE + classifier as CARE-LG
for each (dataset, seed) -- see experiments_v3/lib/revise.py for the full
method description and hyperparameter justification.

Writes: experiments_v3/results/{dataset}_baselines.csv
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from benchmarks.run_benchmarks import run_dice, run_face, run_growing_spheres
from benchmarks.metrics import compute_kde_density
from src.graph.clinical_constraints import compute_clinical_effort, check_clinical_violations
import experiments_v3.lib.dataset_registration  # noqa: F401
from experiments_v3.lib.pipeline_v3 import run_dataset_seed
from experiments_v3.lib.revise import run_revise
from experiments_v3.lib.seeding import set_all_seeds

METHODS = ["DiCE", "FACE", "GrowingSpheres", "REVISE"]


def run_one(dataset: str, seed: int):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    rows = []
    for pidx in run.high_risk_test_idx:
        x0 = run.X_test[pidx]
        set_all_seeds(seed * 1000 + int(pidx))

        t0 = time.perf_counter()
        succ, rec_x, _ = run_dice(x0, run.classifier, run.device, target_threshold=0.35)
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "DiCE", succ, x0, rec_x, lat, run))

        nearest_train_idx = int(np.argmin(np.linalg.norm(run.X_train - x0, axis=1)))
        t0 = time.perf_counter()
        succ, rec_x, _ = run_face(nearest_train_idx, run.X_train, run.low_risk_mask)
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "FACE", succ, x0, rec_x, lat, run))

        t0 = time.perf_counter()
        succ, rec_x, _ = run_growing_spheres(x0, run.classifier, run.device, target_threshold=0.35)
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "GrowingSpheres", succ, x0, rec_x, lat, run))

        succ, rec_x, z_final, lat = run_revise(x0, run.vae, run.classifier, run.device)
        rows.append(_row(dataset, seed, pidx, "REVISE", succ, x0, rec_x, lat, run))

    return rows


def _row(dataset, seed, pidx, method, succ, x0, rec_x, lat, run):
    row = {"dataset": dataset, "seed": seed, "patient_idx": int(pidx), "method": method,
           "success": bool(succ), "latency_sec": lat, "cvr_hard": None, "cvr_full": None, "effort": None, "kde": None}
    if succ and rec_x is not None:
        from benchmarks.metrics import compute_cvr
        row["cvr_hard"] = compute_cvr(x0, rec_x, run.feature_metadata, run.scaler, run.feature_cols)
        row["cvr_full"] = float(check_clinical_violations(x0, rec_x, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False))
        row["effort"] = float(compute_clinical_effort(x0, rec_x, run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols))
        row["kde"] = float(compute_kde_density(rec_x.reshape(1, -1), run.X_train, bandwidth=0.5))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "experiments_v3" / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"
    with open(out_dir / "run_baselines_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds, "methods": METHODS,
                   "revise_hparams": {"lr": 0.05, "steps": 100, "lambda_prox": 0.01, "target_threshold": 0.45},
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    for dataset in args.datasets:
        all_rows = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            rows = run_one(dataset, seed)
            t1 = time.perf_counter()
            df_s = pd.DataFrame(rows)
            print(f"[baselines] {dataset} seed={seed}: {len(df_s)//len(METHODS)} patients in {t1-t0:.1f}s", flush=True)
            print(df_s.groupby("method")["success"].mean().to_string())
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out_dir / f"{dataset}_baselines.csv", index=False)


if __name__ == "__main__":
    main()
