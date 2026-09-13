"""
Tests the confirmed root cause directly: experiments_v3/lib/graph_build.py's
`build_query_edges` (line 238) gates the query's ENTRY edge with
`gate_violates = hard_count > 0` -- hard-only (immutable + age-decrease),
`check_step_horizon=False` -- while every INTERIOR graph edge is gated by the FULL
`check_clinical_violations` predicate (hard OR directional, check_step_horizon=True).
`dir_count` is already computed at that call site and simply not used for the gate.

This script reruns experiments_v3's exact R+hard cell (same pipeline, same seeds, same
patients) with ONE line changed: `gate_violates = (hard_count + dir_count) > 0`, using
`check_step_horizon=True` to match interior edges exactly. Reports, for both the
ORIGINAL (relaxed) gate and the FIXED (strict, interior-edge-matching) gate:
  - success rate
  - decoded-space CVR (current v2-v5 method: decode every node)
  - real-value CVR (Root Cause B's fix: use real recorded X for real nodes)

Usage:
  .venv/bin/python -m experiments_v6_audit.verify_entry_gate_fix --datasets uci nhanes_real --seeds 0 1 2 3 4
"""
import argparse
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
from experiments_v3.lib.graph_build import (
    build_edge_table, make_cell_matrix, augmented_matrix, unscale_matrix,
    vectorized_violation_components,
)
from experiments_v3.lib.riemannian_batch import batched_riemannian_distances
from experiments_v3.lib.search_v2 import find_path_augmented
from experiments_v3.lib.constraints_ext import check_decoded_violations, decode_node
import scipy.sparse as sp

RESULTS_DIR = REPO_ROOT / "experiments_v6_audit" / "results"


def build_query_edges_variant(vae_model, z0, x0, Z_train, X_train, feature_metadata, scaler, feature_cols,
                              strict_gate: bool, device):
    """Copy of experiments_v3.lib.graph_build.build_query_edges, parametrized on the
    ONE line under test: `strict_gate=False` reproduces the original relaxed gate
    exactly (hard_count>0, check_step_horizon=False); `strict_gate=True` uses
    (hard_count+dir_count)>0 with check_step_horizon=True, matching interior edges."""
    N = len(Z_train)
    neighbor_idx = np.arange(N)
    z0t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    zjt = torch.tensor(Z_train, dtype=torch.float32)
    dist_r = batched_riemannian_distances(vae_model, z0t, zjt, device=device).numpy()
    dist_e = np.linalg.norm(Z_train - z0.reshape(1, -1), axis=1)

    X_combined = np.vstack([np.tile(x0.reshape(1, -1), (N, 1)), X_train])
    unscaled = unscale_matrix(X_combined, scaler, feature_cols, feature_metadata)
    src_idx, tgt_idx = np.arange(N), np.arange(N, 2 * N)
    horizon = True if strict_gate else False
    hard_count, dir_count = vectorized_violation_components(unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon=horizon)
    gate_violates = (hard_count + dir_count) > 0 if strict_gate else (hard_count > 0)
    v_count = hard_count + dir_count
    return neighbor_idx, dist_r, dist_e, gate_violates, v_count


def make_query_row_variant(neighbor_idx, dist_r, dist_e, gate_violates, v_count, N, metric="riemannian", lam=1.0):
    dist = dist_r if metric == "riemannian" else dist_e
    keep = ~gate_violates
    cols, w = neighbor_idx[keep], dist[keep]
    rows = np.zeros(len(cols), dtype=np.int64)
    return sp.csr_matrix((w, (rows, cols)), shape=(1, N))


def evaluate(run, edge_table, base_csr, patient_idx, strict_gate):
    x0 = run.X_test[patient_idx]
    with torch.no_grad():
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
    N = edge_table.N
    neighbor_idx, dist_r, dist_e, gate_violates, v_count = build_query_edges_variant(
        run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols,
        strict_gate=strict_gate, device=run.device)
    qrow = make_query_row_variant(neighbor_idx, dist_r, dist_e, gate_violates, v_count, N)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    row = {"patient_idx": int(patient_idx), "success": False}
    try:
        path, cost = find_path_augmented(aug, N, target_mask)
    except ValueError:
        return row
    row["success"] = True
    row["path_len"] = len(path) - 1

    decoded_points = [decode_node(run.vae, z0 if n == N else run.Z_train[n], device=run.device) for n in path]
    real_points = [x0 if n == N else run.X_train[n] for n in path]

    v_dec, _ = check_decoded_violations(decoded_points[0], decoded_points[-1], run.feature_metadata,
                                        run.scaler, run.feature_cols, check_step_horizon=False)
    v_real, _ = check_decoded_violations(real_points[0], real_points[-1], run.feature_metadata,
                                         run.scaler, run.feature_cols, check_step_horizon=False)
    row["decoded_cvr_endpoint"] = bool(v_dec)
    row["real_cvr_endpoint"] = bool(v_real)
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

            for gate_name, strict in [("original_relaxed_gate", False), ("fixed_strict_gate", True)]:
                for pidx in patients:
                    r = evaluate(run, edge_table, base_csr, pidx, strict_gate=strict)
                    r.update({"dataset": dataset, "seed": seed, "gate": gate_name})
                    all_rows.append(r)

            df = pd.DataFrame(all_rows)
            d = df[(df.dataset == dataset) & (df.seed == seed)]
            for gate_name in ["original_relaxed_gate", "fixed_strict_gate"]:
                g = d[d.gate == gate_name]
                ok = g[g.success == True]
                succ = g["success"].mean() * 100
                dcvr = ok["decoded_cvr_endpoint"].mean() * 100 if len(ok) else float("nan")
                rcvr = ok["real_cvr_endpoint"].mean() * 100 if len(ok) else float("nan")
                print(f"[verify-gate] {dataset} seed={seed} {gate_name:22s}: success={succ:5.1f}% "
                      f"decodedCVR={dcvr:5.1f}% realCVR={rcvr:5.1f}%", flush=True)
            print(f"  [{time.perf_counter()-t0:.1f}s]", flush=True)
            pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_entry_gate_fix.csv", index=False)

    print("\n[verify-gate] === FULL SUMMARY (mean +/- std over seeds) ===")
    df_all = pd.DataFrame(all_rows)
    for dataset in args.datasets:
        for gate_name in ["original_relaxed_gate", "fixed_strict_gate"]:
            d = df_all[(df_all.dataset == dataset) & (df_all.gate == gate_name)]
            per_seed_succ = d.groupby("seed")["success"].mean() * 100
            ok = d[d.success == True]
            per_seed_dcvr = ok.groupby("seed")["decoded_cvr_endpoint"].mean() * 100 if len(ok) else pd.Series(dtype=float)
            per_seed_rcvr = ok.groupby("seed")["real_cvr_endpoint"].mean() * 100 if len(ok) else pd.Series(dtype=float)
            print(f"{dataset:12s} {gate_name:22s}: success={per_seed_succ.mean():.1f}+/-{per_seed_succ.std():.1f}%  "
                  f"decodedCVR={per_seed_dcvr.mean():.1f}+/-{per_seed_dcvr.std():.1f}%  "
                  f"realCVR={per_seed_rcvr.mean():.1f}+/-{per_seed_rcvr.std():.1f}%")


if __name__ == "__main__":
    main()
