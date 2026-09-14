"""
FACE rerun under the corrected evaluation protocol -- entry-point script.
The algorithm itself now lives in `src/baselines/face.py` (promoted there
this session for consistency with `src/baselines/dice.py` and
`src/baselines/growing_spheres.py`, the other two baselines rebuilt to the
same standard). See that module's docstring for the full citation and
method description.

Extended this session to also report KDE plausibility and clinical effort,
matching the metrics reported for every other baseline
(`scripts/run_baselines_rebuilt.py`).

Usage:
  .venv/bin/python -m scripts.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4
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
from src.baselines.face import build_face_graph, run_face_query, K_NEIGHBORS_FACE
from benchmarks.metrics import compute_kde_density


def evaluate_patient_face(x0, X_train, base_csr, low_risk_mask, feature_metadata, scaler, feature_cols, effort_weights):
    row = {"success": False, "real_cvr_hop": None, "real_cvr_endpoint": None,
           "kde": None, "effort": None, "latency_sec": None}
    success, path, cost, latency = run_face_query(x0, X_train, base_csr, low_risk_mask)
    row["latency_sec"] = latency
    if not success:
        return row
    row["success"] = True
    N = len(X_train)

    real_points = [x0 if n == N else X_train[n] for n in path]
    any_hop = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, feature_metadata, scaler, feature_cols, check_step_horizon=True)
        any_hop = any_hop or v
    row["real_cvr_hop"] = any_hop
    endpoint_v, _ = check_decoded_violations(real_points[0], real_points[-1], feature_metadata, scaler, feature_cols, check_step_horizon=False)
    row["real_cvr_endpoint"] = endpoint_v

    row["effort"] = float(compute_clinical_effort(real_points[0], real_points[-1], effort_weights, feature_metadata, scaler, feature_cols))
    row["kde"] = float(compute_kde_density(np.vstack(real_points), X_train, bandwidth=0.5))
    return row


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

    all_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            base_csr, _ = build_face_graph(run.X_train, K_NEIGHBORS_FACE[dataset])
            t0 = time.perf_counter()
            n_success, n_cvr = 0, 0
            for pidx in run.high_risk_test_idx:
                r = evaluate_patient_face(run.X_test[pidx], run.X_train, base_csr, run.low_risk_mask,
                                           run.feature_metadata, run.scaler, run.feature_cols, run.effort_weights)
                r.update({"dataset": dataset, "seed": seed, "patient_idx": int(pidx)})
                all_rows.append(r)
                n_success += int(r["success"])
                if r["success"]:
                    n_cvr += int(bool(r["real_cvr_endpoint"]) or bool(r["real_cvr_hop"]))
            t1 = time.perf_counter()
            n = len(run.high_risk_test_idx)
            print(f"[face] {dataset} seed={seed}: success={n_success}/{n} ({100.0*n_success/max(1,n):.1f}%) "
                  f"real_cvr={100.0*n_cvr/max(1,n_success):.1f}% in {t1-t0:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "face_rerun_per_patient.csv", index=False)
    with open(out_dir / "face_rerun_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "k_neighbors": K_NEIGHBORS_FACE, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print("\n[face] === SUMMARY (mean +/- std over seeds) ===")
    for dataset in args.datasets:
        d = df[df.dataset == dataset]
        per_seed_succ = d.groupby("seed")["success"].mean() * 100
        ok = d[d.success == True]
        real_viol = ok["real_cvr_endpoint"].astype(bool) | ok["real_cvr_hop"].astype(bool)
        per_seed_cvr = real_viol.groupby(ok["seed"]).mean() * 100 if len(ok) else pd.Series(dtype=float)
        print(f"{dataset:12s} success={per_seed_succ.mean():.1f}+/-{per_seed_succ.std():.1f}%  "
              f"real_cvr={per_seed_cvr.mean():.1f}+/-{per_seed_cvr.std():.1f}%  "
              f"kde={ok['kde'].mean():.3f}  effort={ok['effort'].mean():.2f}  latency={ok['latency_sec'].mean():.4f}s")


if __name__ == "__main__":
    main()
