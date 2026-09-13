"""
Phase 5: baseline comparison hygiene.

Reuses the EXISTING, unmodified baseline implementations from
benchmarks/run_benchmarks.py (run_dice, run_face, run_growing_spheres) --
not reimplemented -- run under the SAME seeded (dataset, seed) pipeline and
the SAME high-risk test patients as the main ablation grid (experiments_v2/
run_grid.py), so Table 1 compares like-for-like.

Two CVR numbers are reported per baseline recourse:
  - cvr_hard: exactly benchmarks/metrics.py::compute_cvr (hard violations
    only: immutables + age-decrease, check_step_horizon=False). This is
    what the original repo's Table 1 CVR% column means.
  - cvr_full: check_clinical_violations (hard + directional), same
    check_step_horizon=False convention, for an apples-to-apples comparison
    against the grid cells' decoded_cvr_endpoint column.
Baselines output feature-space vectors directly (no VAE decode step), so
there is no separate "decoded-space" check for them beyond cvr_full.

R+soft is already computed by run_grid.py across lambda in {0.1,1,10,100};
per the brief we treat R+soft(lambda) as the "emulation" of Pegios-et-al-
style Riemannian soft-constrained recourse (explicitly labeled as an
emulation, not their released code) rather than re-implementing it here.

Writes: experiments_v2/results/{dataset}_baselines.csv
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
from benchmarks.metrics import compute_cvr, compute_kde_density
from src.graph.clinical_constraints import compute_clinical_effort, check_clinical_violations
from experiments_v2.lib.pipeline import run_dataset_seed
from experiments_v2.lib.seeding import set_all_seeds

METHODS = ["DiCE", "FACE", "GrowingSpheres"]


def run_one(dataset: str, seed: int):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    rows = []
    for pidx in run.high_risk_test_idx:
        x0 = run.X_test[pidx]
        set_all_seeds(seed * 1000 + int(pidx))  # deterministic per-patient stochastic baselines

        t0 = time.perf_counter()
        succ, rec_x, _ = run_dice(x0, run.classifier, run.device, target_threshold=0.35)
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "DiCE", succ, x0, rec_x, lat, run))

        nearest_train_idx = int(np.argmin(np.linalg.norm(run.X_train - x0, axis=1)))
        t0 = time.perf_counter()
        succ, rec_x, _ = run_face(nearest_train_idx, run.X_train, run.low_risk_mask)  # default k_neighbors=25, matching original benchmark usage
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "FACE", succ, x0, rec_x, lat, run))

        t0 = time.perf_counter()
        succ, rec_x, _ = run_growing_spheres(x0, run.classifier, run.device, target_threshold=0.35)
        lat = time.perf_counter() - t0
        rows.append(_row(dataset, seed, pidx, "GrowingSpheres", succ, x0, rec_x, lat, run))

    return rows


def _row(dataset, seed, pidx, method, succ, x0, rec_x, lat, run):
    row = {"dataset": dataset, "seed": seed, "patient_idx": int(pidx), "method": method,
           "success": bool(succ), "latency_sec": lat, "cvr_hard": None, "cvr_full": None, "effort": None, "kde": None}
    if succ and rec_x is not None:
        row["cvr_hard"] = compute_cvr(x0, rec_x, run.feature_metadata, run.scaler, run.feature_cols)
        row["cvr_full"] = float(check_clinical_violations(x0, rec_x, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False))
        row["effort"] = float(compute_clinical_effort(x0, rec_x, run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols))
        row["kde"] = float(compute_kde_density(rec_x.reshape(1, -1), run.X_train, bandwidth=0.5))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes"], choices=["uci", "nhanes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "experiments_v2" / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"
    with open(out_dir / "run_baselines_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "methods": METHODS, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    for dataset in args.datasets:
        all_rows = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            rows = run_one(dataset, seed)
            t1 = time.perf_counter()
            df_s = pd.DataFrame(rows)
            print(f"[baselines] {dataset} seed={seed}: {len(df_s)//len(METHODS)} patients in {t1-t0:.1f}s")
            print(df_s.groupby("method")["success"].mean().to_string())
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out_dir / f"{dataset}_baselines.csv", index=False)


if __name__ == "__main__":
    main()
