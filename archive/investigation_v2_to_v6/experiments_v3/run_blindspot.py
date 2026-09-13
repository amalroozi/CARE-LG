"""
Phase 4: blindspot vs. true-infeasibility decomposition of R+hard abstentions
(riemannian metric, hard constraints — the actual CARE-LG cell).

For every high-risk test patient where R+hard finds no path:
  1. Exhaustive feature-space check: does ANY low-risk train node satisfy the
     pairwise source->target constraints directly (check_clinical_violations,
     check_step_horizon=False -- a single conceptual "is this pairing
     feasible at all" test, not a bounded-time single hop)?
       - none exists  -> CERTIFIED INFEASIBLE
       - >=1 exists   -> BLINDSPOT CANDIDATE
  2. For blindspot candidates, two densification probes, each rerun as a
     fresh hard-mode graph search:
       (a) doubled intra-graph k (k -> 2k), same node set
       (b) M=500 synthetic nodes sampled from the VAE prior z~N(0,I) and
           decoded; a synthetic node is eligible as a TERMINAL (low-risk
           target) only if the classifier verifies decoded risk < gamma;
           synthetic nodes may otherwise be used as pass-through hops.
     A blindspot candidate resolved by either probe is a CONFIRMED BLINDSPOT.
     One resolved by neither is UNRESOLVED/UNKNOWN.

Writes: experiments_v3/results/{dataset}_blindspot.csv (per-patient) and
prints the mean+-std decomposition across seeds.
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

import experiments_v3.lib.dataset_registration  # noqa: F401 -- registers 'nhanes_real' key
from experiments_v3.lib.pipeline_v3 import run_dataset_seed, LOW_RISK_THRESHOLD
from experiments_v3.lib.graph_build import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from experiments_v3.lib.search_v2 import find_path_augmented
from experiments_v3.lib.constraints_ext import violates as violates_fn
from experiments_v3.lib.seeding import set_all_seeds

M_SYNTHETIC = 500
METRIC = "riemannian"  # the actual CARE-LG metric


def try_search(vae, z0, x0, Z_nodes, X_nodes, feature_metadata, scaler, feature_cols, k, base_csr, low_risk_mask):
    N = len(Z_nodes)
    qedges = build_query_edges(vae, z0, x0, Z_nodes, X_nodes, feature_metadata, scaler, feature_cols, k=k)
    qrow = make_query_row(qedges, METRIC, "hard", N)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([low_risk_mask, [False]])
    try:
        find_path_augmented(aug, N, target_mask)
        return True
    except ValueError:
        return False


def run_one(dataset: str, seed: int):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard = make_cell_matrix(edge_table, METRIC, "hard")

    # 1. find R+hard abstentions
    abstained = []
    for pidx in run.high_risk_test_idx:
        x0 = run.X_test[pidx]
        with torch.no_grad():
            mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))
        z0 = mu.squeeze(0).numpy()
        ok = try_search(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols,
                         run.k_neighbors, base_hard, run.low_risk_mask)
        if not ok:
            abstained.append((pidx, z0))

    rows = []
    n_abstain = len(abstained)
    if n_abstain == 0:
        return rows, run

    low_risk_idxs = np.where(run.low_risk_mask)[0]

    # Pre-build densification graphs only if needed
    set_all_seeds(seed + 100000)  # distinct sub-seed for synthetic sampling, still reproducible
    edge_table_2k = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols, k=2 * run.k_neighbors)
    base_hard_2k = make_cell_matrix(edge_table_2k, METRIC, "hard")

    with torch.no_grad():
        z_synth = torch.randn(M_SYNTHETIC, run.vae.latent_dim)
        x_synth = run.vae.decode(z_synth).numpy()
        risk_synth = run.classifier(torch.tensor(x_synth, dtype=torch.float32)).numpy().squeeze()
    synth_low_risk = risk_synth < LOW_RISK_THRESHOLD
    Z_combined = np.vstack([run.Z_train, z_synth.numpy()])
    X_combined = np.vstack([run.X_train, x_synth])
    low_risk_combined = np.concatenate([run.low_risk_mask, synth_low_risk])
    edge_table_synth = build_edge_table(run.vae, Z_combined, X_combined, run.feature_metadata, run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard_synth = make_cell_matrix(edge_table_synth, METRIC, "hard")

    for pidx, z0 in abstained:
        x0 = run.X_test[pidx]
        valid_targets = [
            j for j in low_risk_idxs
            if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)
        ]
        certified_infeasible = len(valid_targets) == 0

        resolved_2k = None
        resolved_synth = None
        if not certified_infeasible:
            resolved_2k = try_search(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols,
                                      2 * run.k_neighbors, base_hard_2k, run.low_risk_mask)
            resolved_synth = try_search(run.vae, z0, x0, Z_combined, X_combined, run.feature_metadata, run.scaler, run.feature_cols,
                                         run.k_neighbors, base_hard_synth, low_risk_combined)

        rows.append({
            "dataset": dataset, "seed": seed, "patient_idx": int(pidx),
            "n_valid_targets_exhaustive": len(valid_targets),
            "certified_infeasible": certified_infeasible,
            "resolved_by_2k": resolved_2k,
            "resolved_by_synth": resolved_synth,
            "confirmed_blindspot": (not certified_infeasible) and bool(resolved_2k or resolved_synth),
            "unresolved": (not certified_infeasible) and not bool(resolved_2k or resolved_synth),
        })

    return rows, run


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
    with open(out_dir / "run_blindspot_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "m_synthetic": M_SYNTHETIC, "metric": METRIC,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    for dataset in args.datasets:
        all_rows = []
        seed_summaries = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            rows, run = run_one(dataset, seed)
            t1 = time.perf_counter()
            n_high_risk = len(run.high_risk_test_idx)
            n_abstain = len(rows)
            print(f"[blindspot] {dataset} seed={seed}: {n_high_risk} high-risk patients, "
                  f"{n_abstain} R+hard abstentions ({100.0*n_abstain/max(1,n_high_risk):.1f}%) in {t1-t0:.1f}s", flush=True)
            all_rows.extend(rows)
            if n_abstain > 0:
                df_s = pd.DataFrame(rows)
                n_cert = df_s["certified_infeasible"].sum()
                n_conf = df_s["confirmed_blindspot"].sum()
                n_unres = df_s["unresolved"].sum()
                seed_summaries.append({
                    "seed": seed, "n_high_risk": n_high_risk, "n_abstain": n_abstain,
                    "pct_abstain": 100.0 * n_abstain / max(1, n_high_risk),
                    "pct_certified_infeasible": 100.0 * n_cert / n_abstain,
                    "pct_confirmed_blindspot": 100.0 * n_conf / n_abstain,
                    "pct_unresolved": 100.0 * n_unres / n_abstain,
                })
            else:
                seed_summaries.append({
                    "seed": seed, "n_high_risk": n_high_risk, "n_abstain": 0,
                    "pct_abstain": 0.0, "pct_certified_infeasible": np.nan,
                    "pct_confirmed_blindspot": np.nan, "pct_unresolved": np.nan,
                })

        if all_rows:
            pd.DataFrame(all_rows).to_csv(out_dir / f"{dataset}_blindspot.csv", index=False)
        df_sum = pd.DataFrame(seed_summaries)
        df_sum.to_csv(out_dir / f"{dataset}_blindspot_seed_summary.csv", index=False)
        print(f"\n[blindspot] === {dataset.upper()} summary across seeds ===")
        print(df_sum.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
