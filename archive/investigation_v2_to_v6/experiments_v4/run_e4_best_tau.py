"""
E4 follow-up: the main tau-sweep in run_dsr.py only swept `dsr_full`
(Riemannian). Table 1 and the sweep show dsr_full has ~0 successes at the
q=0.95 operating point on BOTH datasets, so a Riemannian-vs-Euclidean paired
test there is a real, reportable 0-pairs result (nothing to compare). To also
give E4 a POWERED test wherever one is possible, this script runs
`dsr_full_euclidean` at the SAME tau quantile where dsr_full (Riemannian)
achieves its best success rate (from tau_sweep_{dataset}.csv), so the two
metrics can be compared on equal footing at an operating point where the
method actually returns paths.

On UCI, dsr_full has 0% success at every tau level 0.05-0.99 (see
tau_sweep_uci.csv) -- there is no tau at which a powered E4 test is possible,
and this script reports that explicitly rather than picking an arbitrary
level and implying a comparison exists where it does not.

Usage:
  .venv/bin/python -m experiments_v4.run_e4_best_tau --datasets nhanes_real --seeds 0 1 2 3 4
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

import experiments_v4.lib.dataset_registration  # noqa: F401
from experiments_v4.lib.calibration import calibrate
from experiments_v4.lib.dsr_graph import build_dsr_edge_table
from experiments_v4.lib.pipeline_v4 import run_dataset_seed_v4
from experiments_v4.run_dsr import RESULTS_DIR, run_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    for dataset in args.datasets:
        sweep_path = RESULTS_DIR / f"tau_sweep_{dataset}.csv"
        if not sweep_path.exists():
            print(f"[e4] {dataset}: run aggregate.py first to produce {sweep_path}")
            continue
        sweep = pd.read_csv(sweep_path)
        nonzero = sweep[sweep["success_rate_mean"] > 0]
        if len(nonzero) == 0:
            print(f"[e4] {dataset}: dsr_full has 0% success at every tau level tested -- "
                  f"no operating point exists for a powered E4 test. Reporting this as a "
                  f"real limitation, not picking an arbitrary tau.")
            continue
        best_q = float(nonzero.sort_values("success_rate_mean", ascending=False).iloc[0]["tau_quantile"])
        print(f"[e4] {dataset}: running dsr_full_euclidean at tau_q={best_q} (matches dsr_full's best operating point)")

        rows = []
        for seed in args.seeds:
            run = run_dataset_seed_v4(dataset, seed, decoder="v3", device=torch.device("cpu"))
            table = build_dsr_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata, run.scaler,
                                         run.feature_cols, run.k_neighbors, device=run.device)
            cal = calibrate(run.vae, run.X_calib, run.feature_metadata, run.scaler, run.feature_cols,
                            quantile=best_q, device=run.device)
            t0 = time.perf_counter()
            cell_rows = run_cell(run, table, "dsr_full_euclidean",
                                 dict(prune_mode="decoded_naive", verify_mode="decoded_banded",
                                      repair=True, metric="euclidean"),
                                 cal.taus, best_q, run.high_risk_test_idx)
            rows.extend(cell_rows)
            ok = sum(r["success"] for r in cell_rows)
            print(f"[e4]   {dataset} seed={seed}: {ok}/{len(cell_rows)} succeeded in {time.perf_counter()-t0:.1f}s")

        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{dataset}_e4_euclidean_best_tau.csv", index=False)
        print(f"[e4] wrote {len(rows)} rows to {dataset}_e4_euclidean_best_tau.csv")


if __name__ == "__main__":
    main()
