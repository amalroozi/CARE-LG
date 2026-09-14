"""
Evaluates the three rebuilt baselines (src/baselines/{dice,growing_spheres,face}.py)
under the SAME corrected protocol as every other baseline in this project:
real high-risk test cohorts, same seeds, real-value CVR (not decoded
reconstruction), KDE plausibility, clinical effort, latency.

Also runs the ORIGINAL (legacy, AI-generated pre-investigation, uncited)
`benchmarks/run_benchmarks.py::run_dice` / `run_growing_spheres` on a
matched patient subset for a direct old-vs-new comparison (Step 4 of this
session's brief) -- FACE's original implementation was already replaced by
`scripts/run_face_rerun.py` in a prior session and is not re-compared here
(no new FACE numbers are at stake; its old-vs-new comparison already
happened in that session).

Usage:
  .venv/bin/python -m scripts.run_baselines_rebuilt --datasets uci nhanes_real --seeds 0 1 2 3 4
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

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.constraints_ext import check_decoded_violations
from src.graph.clinical_constraints import compute_clinical_effort
from src.baselines.growing_spheres import run_growing_spheres
from src.baselines.dice import run_dice_diverse
from benchmarks.metrics import compute_kde_density
from benchmarks.run_benchmarks import run_dice as legacy_run_dice, run_growing_spheres as legacy_run_growing_spheres

TARGET_THRESHOLD = 0.45  # this project's own low-risk threshold, used for BOTH old and new
                         # implementations below so the old-vs-new comparison isolates the
                         # algorithmic rebuild, not a threshold change (the legacy functions'
                         # own default, 0.35, is NOT used here -- documented deviation).
DICE_MAX_ITER = 300
N_OLD_VS_NEW = 15  # patients per (dataset, seed) also run through the legacy implementations


def _cvr_and_metrics(x0, x_target, feature_metadata, scaler, feature_cols, effort_weights):
    v, _ = check_decoded_violations(x0, x_target, feature_metadata, scaler, feature_cols, check_step_horizon=False)
    effort = float(compute_clinical_effort(x0, x_target, effort_weights, feature_metadata, scaler, feature_cols))
    return bool(v), effort


def evaluate_new_methods(pidx, x0, run):
    rows = []

    t0 = time.perf_counter()
    succ, rec, lat = run_growing_spheres(x0, run.classifier, run.device, target_threshold=TARGET_THRESHOLD, seed=0)
    row = {"method": "growing_spheres_new", "patient_idx": int(pidx), "success": succ, "latency_sec": lat,
           "real_cvr": None, "kde": None, "effort": None}
    if succ:
        cvr, effort = _cvr_and_metrics(x0, rec, run.feature_metadata, run.scaler, run.feature_cols, run.effort_weights)
        row["real_cvr"] = cvr
        row["effort"] = effort
        row["kde"] = float(compute_kde_density(np.vstack([x0, rec]), run.X_train, bandwidth=0.5))
    rows.append(row)

    succ, cfs, lat = run_dice_diverse(x0, run.classifier, run.device, run.feature_metadata, run.feature_cols,
                                       run.X_train, k=4, target_threshold=TARGET_THRESHOLD, max_iter=DICE_MAX_ITER, seed=0)
    row = {"method": "dice_new", "patient_idx": int(pidx), "success": succ, "latency_sec": lat,
           "real_cvr": None, "kde": None, "effort": None, "n_diverse_cfs": len(cfs) if cfs else 0}
    if succ:
        # report metrics for the BEST (lowest-effort) of the k valid diverse counterfactuals
        best_cvr, best_effort = None, None
        for cf in cfs:
            cvr, effort = _cvr_and_metrics(x0, cf, run.feature_metadata, run.scaler, run.feature_cols, run.effort_weights)
            if best_effort is None or effort < best_effort:
                best_cvr, best_effort = cvr, effort
        row["real_cvr"] = best_cvr
        row["effort"] = best_effort
        row["kde"] = float(np.mean([compute_kde_density(np.vstack([x0, cf]), run.X_train, bandwidth=0.5) for cf in cfs]))
    rows.append(row)

    return rows


def evaluate_legacy_methods(pidx, x0, run):
    rows = []
    succ, rec, lat = legacy_run_growing_spheres(x0, run.classifier, run.device, target_threshold=TARGET_THRESHOLD)
    row = {"method": "growing_spheres_legacy", "patient_idx": int(pidx), "success": succ, "latency_sec": lat,
           "real_cvr": None, "kde": None, "effort": None}
    if succ:
        cvr, effort = _cvr_and_metrics(x0, rec, run.feature_metadata, run.scaler, run.feature_cols, run.effort_weights)
        row["real_cvr"] = cvr
        row["effort"] = effort
        row["kde"] = float(compute_kde_density(np.vstack([x0, rec]), run.X_train, bandwidth=0.5))
    rows.append(row)

    succ, rec, lat = legacy_run_dice(x0, run.classifier, run.device, target_threshold=TARGET_THRESHOLD, steps=100, lr=0.05)
    row = {"method": "dice_legacy", "patient_idx": int(pidx), "success": succ, "latency_sec": lat,
           "real_cvr": None, "kde": None, "effort": None}
    if succ:
        cvr, effort = _cvr_and_metrics(x0, rec, run.feature_metadata, run.scaler, run.feature_cols, run.effort_weights)
        row["real_cvr"] = cvr
        row["effort"] = effort
        row["kde"] = float(compute_kde_density(np.vstack([x0, rec]), run.X_train, bandwidth=0.5))
    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"

    new_rows, legacy_rows = [], []
    for dataset in args.datasets:
        for seed in args.seeds:
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            patients = run.high_risk_test_idx
            t0 = time.perf_counter()
            for i, pidx in enumerate(patients):
                x0 = run.X_test[pidx]
                for r in evaluate_new_methods(pidx, x0, run):
                    r.update({"dataset": dataset, "seed": seed})
                    new_rows.append(r)
                if i < N_OLD_VS_NEW:
                    for r in evaluate_legacy_methods(pidx, x0, run):
                        r.update({"dataset": dataset, "seed": seed})
                        legacy_rows.append(r)
            t1 = time.perf_counter()
            df_seed = pd.DataFrame(new_rows)
            d = df_seed[(df_seed.dataset == dataset) & (df_seed.seed == seed)]
            for m in ["growing_spheres_new", "dice_new"]:
                dm = d[d.method == m]
                succ = dm["success"].mean() * 100
                ok = dm[dm.success == True]
                cvr = ok["real_cvr"].mean() * 100 if len(ok) else float("nan")
                print(f"[baselines] {dataset} seed={seed} {m:20s}: n={len(dm)} success={succ:.1f}% "
                      f"real_cvr={cvr:.1f}% in {t1-t0:.1f}s (cell total)", flush=True)
            # checkpoint after every seed
            pd.DataFrame(new_rows).to_csv(out_dir / "baselines_rebuilt_per_patient.csv", index=False)
            pd.DataFrame(legacy_rows).to_csv(out_dir / "baselines_old_vs_new.csv", index=False)

    with open(out_dir / "baselines_rebuilt_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "target_threshold": TARGET_THRESHOLD, "dice_max_iter": DICE_MAX_ITER,
                    "dice_k": 4, "n_old_vs_new_per_seed": N_OLD_VS_NEW,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print("\n[baselines] === FINAL SUMMARY: new implementations (mean +/- std over seeds) ===")
    df = pd.DataFrame(new_rows)
    for dataset in args.datasets:
        for m in ["growing_spheres_new", "dice_new"]:
            d = df[(df.dataset == dataset) & (df.method == m)]
            per_seed_succ = d.groupby("seed")["success"].mean() * 100
            ok = d[d.success == True]
            per_seed_cvr = ok.groupby("seed")["real_cvr"].mean() * 100 if len(ok) else pd.Series(dtype=float)
            print(f"{dataset:12s} {m:20s} success={per_seed_succ.mean():.1f}+/-{per_seed_succ.std():.1f}%  "
                  f"real_cvr={per_seed_cvr.mean():.1f}+/-{per_seed_cvr.std():.1f}%  "
                  f"kde={ok['kde'].mean():.3f}  effort={ok['effort'].mean():.2f}  latency={ok['latency_sec'].mean():.4f}s")

    print("\n[baselines] === OLD vs NEW comparison (matched patient subset) ===")
    dfl = pd.DataFrame(legacy_rows)
    for dataset in args.datasets:
        for old_m, new_m in [("growing_spheres_legacy", "growing_spheres_new"), ("dice_legacy", "dice_new")]:
            old_d = dfl[(dfl.dataset == dataset) & (dfl.method == old_m)]
            new_d = df[(df.dataset == dataset) & (df.method == new_m) & (df.patient_idx.isin(old_d.patient_idx.unique()))]
            print(f"{dataset:12s} {old_m:24s} success={old_d['success'].mean()*100:.1f}%  "
                  f"real_cvr={(old_d[old_d.success==True]['real_cvr'].mean()*100 if old_d['success'].any() else float('nan')):.1f}%  "
                  f"n={len(old_d)}")
            print(f"{dataset:12s} {new_m:24s} success={new_d['success'].mean()*100:.1f}%  "
                  f"real_cvr={(new_d[new_d.success==True]['real_cvr'].mean()*100 if new_d['success'].any() else float('nan')):.1f}%  "
                  f"n={len(new_d)}")


if __name__ == "__main__":
    main()
