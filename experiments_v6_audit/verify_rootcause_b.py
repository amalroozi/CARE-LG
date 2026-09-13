"""
Phase 3 empirical verification of Root Cause B (unnecessary decoding of real graph
nodes). Reuses experiments_v3's EXACT pipeline (type-aware decoder, same seeds, same
edge table, same search protocol, same R+hard cell = CARE-LG's headline cell) with
ONE change: when computing "decoded-space CVR," a graph node that is a REAL training
patient (which is every non-query node in every path this script evaluates -- R+hard
never routes through synthetic nodes) uses `run.X_train[node]` directly instead of
`decode_node(run.vae, run.Z_train[node])`. The query's own state is `run.X_test[idx]`,
already exact, never decoded either -- reproducing exactly what
`src/recourse/search.py::decode_recourse_trajectory` already does (see
PROJECT_TIMELINE.md / RECOMMENDATION.md for the citation), which the v2-v5 Phase-3
measurement code never did.

Everything else (graph construction, search, success/failure) is BYTE-FOR-BYTE
IDENTICAL to experiments_v3/run_grid.py's R+hard cell -- only the decoded-CVR
MEASUREMENT step differs, isolating exactly the effect of Root Cause B.

Usage:
  .venv/bin/python -m experiments_v6_audit.verify_rootcause_b --datasets uci nhanes_real --seeds 0 1 2 3 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import experiments_v3.lib.dataset_registration  # noqa: F401
from experiments_v3.lib.pipeline_v3 import run_dataset_seed
from experiments_v3.lib.graph_build import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from experiments_v3.lib.search_v2 import find_path_augmented
from experiments_v3.lib.constraints_ext import decode_node, check_decoded_violations

RESULTS_DIR = REPO_ROOT / "experiments_v6_audit" / "results"


def evaluate_patient_fixed(run, edge_table, base_csr, patient_idx):
    """IDENTICAL to experiments_v3.run_grid.evaluate_patient for metric='riemannian',
    constraints='hard', EXCEPT the Phase-3 decoded-space block uses real recorded
    values for real graph nodes and the real query value, instead of decode_node()."""
    x0 = run.X_test[patient_idx]
    with torch.no_grad():
        mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))
    z0 = mu.squeeze(0).cpu().numpy()

    N = edge_table.N
    qedges = build_query_edges(
        run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols,
        k=run.k_neighbors, nbrs=edge_table.nbrs,
    )
    qrow = make_query_row(qedges, "riemannian", "hard", N, lam=1.0)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    row = {"patient_idx": int(patient_idx), "success": False,
          "decoded_cvr_hop_ORIGINAL": None, "decoded_cvr_endpoint_ORIGINAL": None,
          "decoded_cvr_hop_FIXED": None, "decoded_cvr_endpoint_FIXED": None}
    try:
        path, cost = find_path_augmented(aug, N, target_mask)
        success = True
    except ValueError:
        success = False
    row["success"] = success
    if not success:
        return row

    # --- ORIGINAL method (v2-v5 pattern): decode every node, including real ones ---
    decoded_points = []
    for node in path:
        z = z0 if node == N else run.Z_train[node]
        decoded_points.append(decode_node(run.vae, z, device=run.device))
    v_hop = False
    for a, b in zip(decoded_points[:-1], decoded_points[1:]):
        v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        v_hop = v_hop or v
    v_end, _ = check_decoded_violations(decoded_points[0], decoded_points[-1], run.feature_metadata,
                                        run.scaler, run.feature_cols, check_step_horizon=False)
    row["decoded_cvr_hop_ORIGINAL"] = bool(v_hop)
    row["decoded_cvr_endpoint_ORIGINAL"] = bool(v_end)

    # --- FIXED method (Root Cause B fix): real recorded values for real nodes ---
    real_points = []
    for node in path:
        real_points.append(x0 if node == N else run.X_train[node])
    v_hop_f = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        v_hop_f = v_hop_f or v
    v_end_f, _ = check_decoded_violations(real_points[0], real_points[-1], run.feature_metadata,
                                          run.scaler, run.feature_cols, check_step_horizon=False)
    row["decoded_cvr_hop_FIXED"] = bool(v_hop_f)
    row["decoded_cvr_endpoint_FIXED"] = bool(v_end_f)
    row["path_len"] = len(path) - 1
    row["path"] = json.dumps([int(p) for p in path])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            t0 = time.perf_counter()
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                          run.scaler, run.feature_cols, k=run.k_neighbors)
            base_csr = make_cell_matrix(edge_table, "riemannian", "hard", lam=1.0)
            patients = run.high_risk_test_idx

            rows = []
            for pidx in patients:
                r = evaluate_patient_fixed(run, edge_table, base_csr, pidx)
                r["dataset"] = dataset
                r["seed"] = seed
                rows.append(r)
            all_rows.extend(rows)

            df = pd.DataFrame(rows)
            ok = df[df["success"] == True]
            cvr_orig = ok["decoded_cvr_endpoint_ORIGINAL"].mean() * 100 if len(ok) else float("nan")
            cvr_fixed = ok["decoded_cvr_endpoint_FIXED"].mean() * 100 if len(ok) else float("nan")
            print(f"[verify-B] {dataset} seed={seed}: n_success={len(ok)}/{len(patients)} "
                  f"decodedCVR_ORIGINAL(decode-every-node)={cvr_orig:.1f}%  "
                  f"decodedCVR_FIXED(real-values-for-real-nodes)={cvr_fixed:.1f}%  "
                  f"[{time.perf_counter()-t0:.1f}s]", flush=True)

            pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_rootcause_b_verification.csv", index=False)

    print("\n[verify-B] === FULL SUMMARY (mean over seeds, R+hard cell, high-risk test patients) ===")
    df_all = pd.DataFrame(all_rows)
    for dataset in args.datasets:
        d = df_all[df_all["dataset"] == dataset]
        ok = d[d["success"] == True]
        per_seed_orig = ok.groupby("seed")["decoded_cvr_endpoint_ORIGINAL"].mean() * 100
        per_seed_fixed = ok.groupby("seed")["decoded_cvr_endpoint_FIXED"].mean() * 100
        per_seed_succ = d.groupby("seed")["success"].mean() * 100
        print(f"{dataset}: success={per_seed_succ.mean():.1f}%±{per_seed_succ.std():.1f}  "
              f"decodedCVR_ORIGINAL={per_seed_orig.mean():.1f}%±{per_seed_orig.std():.1f}  "
              f"decodedCVR_FIXED={per_seed_fixed.mean():.1f}%±{per_seed_fixed.std():.1f}")


if __name__ == "__main__":
    main()
