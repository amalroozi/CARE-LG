"""
Phase 1 (of the age-ceiling follow-up plan): checks whether a genuine
2-real-hop chain exists for currently "certified infeasible" patients, that
the actual graph search missed.

Why this can find something Stage 1 missed: Stage 1's real search restricts
INTERIOR edges to each node's k nearest Riemannian neighbors (k=40 UCI,
k=80 NHANES) for computational tractability. A valid 2-hop chain
(x0 -> real patient m -> real low-risk patient j, both hops individually
admissible under the SAME strict, horizon-capped predicate every graph edge
uses) can exist even if m isn't among any node's k-nearest-neighbor set --
the edge would simply never have been built into the graph. This check is
NOT k-NN-restricted: it tests the full real-patient population as possible
intermediate hops.

Method, per certified-infeasible query x0:
  1. R1 = { real training patients m : x0 -> m is admissible } (strict,
     check_step_horizon=True, same predicate as every real graph edge --
     NOT the looser check_step_horizon=False used by the existing
     certified-infeasibility test, which only checks DIRECT feasibility to
     a low-risk target).
  2. can_reach_low_risk[m] = True if m has an admissible (strict) edge to
     ANY real low-risk training patient -- precomputed ONCE per (dataset,
     seed) for all N train nodes, not per query.
  3. If R1 intersects {m : can_reach_low_risk[m]}, a genuine 2-hop real
     chain exists. Record it.

Scope, disclosed: capped at 2 hops (not exhaustively deeper) for this
session's compute budget.

Usage:
  .venv/bin/python -m scripts.run_multihop_check
"""
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

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import unscale_matrix, vectorized_violation_components

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DATASETS = ["uci", "nhanes_real"]
SEEDS = [0, 1, 2, 3, 4]


def precompute_can_reach_low_risk(X_train, feature_metadata, scaler, feature_cols, low_risk_mask):
    """For every train node m, can it reach ANY low-risk node in one admissible (strict) hop?"""
    N = len(X_train)
    low_risk_idx = np.where(low_risk_mask)[0]
    can_reach = np.zeros(N, dtype=bool)
    if len(low_risk_idx) == 0:
        return can_reach
    # batched to bound memory: for each low-risk target, check all N sources at once
    for j in low_risk_idx:
        X_combined = np.vstack([X_train, np.tile(X_train[j].reshape(1, -1), (N, 1))])
        unscaled = unscale_matrix(X_combined, scaler, feature_cols, feature_metadata)
        src_idx, tgt_idx = np.arange(N), np.arange(N, 2 * N)
        hard_count, dir_count = vectorized_violation_components(unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon=True)
        admissible = (hard_count + dir_count) == 0
        can_reach |= admissible
        if can_reach.all():
            break
    return can_reach


def admissible_from_query(x0, X_train, feature_metadata, scaler, feature_cols):
    N = len(X_train)
    X_combined = np.vstack([np.tile(x0.reshape(1, -1), (N, 1)), X_train])
    unscaled = unscale_matrix(X_combined, scaler, feature_cols, feature_metadata)
    src_idx, tgt_idx = np.arange(N), np.arange(N, 2 * N)
    hard_count, dir_count = vectorized_violation_components(unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon=True)
    return (hard_count + dir_count) == 0


def main():
    all_rows = []
    for dataset in DATASETS:
        blindspot = pd.read_csv(TABLES_DIR / f"{dataset}_blindspot.csv")
        cert = blindspot[blindspot.certified_infeasible == True]

        for seed in SEEDS:
            cert_seed = cert[cert.seed == seed]
            if len(cert_seed) == 0:
                continue
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            t0 = time.perf_counter()
            can_reach = precompute_can_reach_low_risk(run.X_train, run.feature_metadata, run.scaler, run.feature_cols, run.low_risk_mask)
            t1 = time.perf_counter()

            n_found = 0
            for pidx in cert_seed.patient_idx:
                x0 = run.X_test[int(pidx)]
                r1 = admissible_from_query(x0, run.X_train, run.feature_metadata, run.scaler, run.feature_cols)
                chain_exists = bool(np.any(r1 & can_reach))
                n_found += int(chain_exists)
                all_rows.append({"dataset": dataset, "seed": seed, "patient_idx": int(pidx), "two_hop_chain_found": chain_exists})
            t2 = time.perf_counter()
            print(f"[multihop] {dataset} seed={seed}: {len(cert_seed)} certified-infeasible checked, "
                  f"{n_found} have a real 2-hop chain ({100.0*n_found/max(1,len(cert_seed)):.1f}%) "
                  f"[precompute {t1-t0:.1f}s, check {t2-t1:.1f}s]", flush=True)

    df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "multihop_check.csv", index=False)

    print("\n[multihop] === SUMMARY ===")
    summary = {}
    for dataset in DATASETS:
        d = df[df.dataset == dataset]
        per_seed = d.groupby("seed")["two_hop_chain_found"].mean() * 100
        summary[dataset] = {"n_checked": len(d), "pct_found_mean": float(per_seed.mean()), "pct_found_std": float(per_seed.std())}
        print(f"{dataset:12s} n={len(d):4d}  2-hop chain found: {per_seed.mean():.1f}% +/- {per_seed.std():.1f}%")

    with open(TABLES_DIR / "multihop_check_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[multihop] wrote results/tables/multihop_check.csv and _summary.json")


if __name__ == "__main__":
    main()
