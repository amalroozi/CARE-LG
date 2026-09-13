"""
Phase 2: certified-infeasibility vs. blindspot decomposition of R+hard
abstentions under the CORRECTED (B1+B2) v6 pipeline.

Ports experiments_v3/run_blindspot.py's protocol unchanged, onto
experiments_v6/lib (B1-fixed entry gate). One thing is clarified, not
changed, relative to the brief's Phase 2 framing: `find_path_augmented`
(experiments_v6/lib/search_v2.py) runs a SINGLE Dijkstra call over the
whole augmented graph (train nodes + one ephemeral query source). It has
no separate code path for "entry edge missing" vs. "entry succeeded but no
route from there to any low-risk node" -- both manifest identically as
"no finite distance from the query to any target node" -> ValueError. So
this protocol's "abstention" criterion was ALREADY unifying those two
failure modes before this session (the search machinery cannot tell them
apart), not something Phase 2 needs to newly construct. What Phase 2 adds
is running this exhaustive-check + densification protocol AGAINST THE
CORRECTED B1/B2 PIPELINE for the first time (v3's own blindspot run used
the pre-fix relaxed entry gate), and reporting what fraction of the
Phase-1 success-rate drop this decomposition explains.

For every high-risk test patient where R+hard (B1-fixed gate) finds no path:
  1. Exhaustive feature-space check: does ANY low-risk train node satisfy the
     pairwise source->target constraints directly (check_clinical_violations,
     check_step_horizon=False)?
       - none exists  -> CERTIFIED INFEASIBLE
       - >=1 exists   -> BLINDSPOT CANDIDATE
  2. For blindspot candidates, two densification probes, each rerun as a
     fresh hard-mode graph search (still B1-fixed entry gate):
       (a) doubled intra-graph k (k -> 2k), same node set
       (b) M=500 synthetic nodes sampled from the VAE prior z~N(0,I) and
           decoded; eligible as a TERMINAL only if classifier-verified
           decoded risk < gamma; usable as pass-through hops otherwise.
     Resolved by either probe -> CONFIRMED BLINDSPOT. Resolved by neither
     -> UNRESOLVED/UNKNOWN.

Writes: experiments_v6/results/{dataset}_blindspot.csv (per-patient) and
{dataset}_blindspot_seed_summary.csv, plus a printed decomposition of what
fraction of the Phase-1 success drop each category explains.
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
from src.pipeline import run_dataset_seed, LOW_RISK_THRESHOLD
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.graph.constraints_ext import violates as violates_fn
from src.seeding import set_all_seeds

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

    # 1. find R+hard abstentions (B1-fixed entry gate)
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
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
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
