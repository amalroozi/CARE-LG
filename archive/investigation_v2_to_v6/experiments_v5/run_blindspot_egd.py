"""
Phase 3: certified-infeasibility decomposition on EGD's abstentions. Reuses
experiments_v3/v4's exact methodology: exhaustive feature-space feasibility check
(does ANY true low-risk train node satisfy the pairwise constraint from this query,
checked in TRUE recorded values -- unaffected by EGD, since EGD's guarantee is
architectural, not a pruning predicate to re-derive) + two densification probes
(doubled k; +500 VAE-prior synthetic nodes, classifier-verified low-risk).

EGD's failure mode is different from every prior session's, and the decomposition
must reflect that: EGD never prunes an edge for constraint reasons (its guarantee is
architectural, not pruning-based), so an abstention here is EITHER (a) 'no_path' --
the k-NN latent graph has no route at all to any low-risk-mask node (should be rare,
given the graph is fully unpruned), or (b) 'risk_check' -- a path exists, but every
tried candidate's SEQUENTIALLY DECODED terminal state fails to cross the risk
threshold (the actual, dominant failure mode observed in Phase 2 smoke tests,
especially on real NHANES). The densification probes test whether EITHER failure mode
resolves with a denser graph or synthetic low-risk nodes as additional, reachable
targets (more candidates to try decoding through).

Usage:
  .venv/bin/python -m experiments_v5.run_blindspot_egd --datasets uci nhanes_real --seeds 0 1 2 3 4
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

import experiments_v5.lib.dataset_registration  # noqa: F401
from experiments_v5.lib.constraints_ext import violates as raw_violates
from experiments_v5.lib.egd_graph import build_egd_edge_table
from experiments_v5.lib.pipeline_v5 import LOW_RISK_THRESHOLD, run_dataset_seed_v5
from experiments_v5.lib.search_egd import find_recourse_egd

RESULTS_DIR = REPO_ROOT / "experiments_v5" / "results"


def exhaustive_feasibility(run, pidx):
    x0 = run.X_test[pidx]
    low_risk_idx = np.where(run.low_risk_mask)[0]
    for j in low_risk_idx:
        if not raw_violates(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols,
                            check_step_horizon=False):
            return True, j
    return False, None


def build_densified_probes(run, k_multiplier, n_synthetic, seed):
    """
    Builds BOTH densification probe graphs ONCE per (dataset, seed) -- they do not
    depend on which query patient is being tested. An earlier version of this function
    rebuilt them per-patient (the same mistake, and the same fix, as
    experiments_v4/run_blindspot_dsr.py's CHANGES.md entry): on real NHANES, with
    dsr_full-equivalent abstention rates near 100%, that made a single seed take >70s
    with no sign of finishing; building once and reusing cut this to a few seconds.

    Returns (table_2k, Z_aug, X_aug, mask_aug, table_synth) -- the synthetic-node
    pieces are None if no synthetic sample was classifier-verified low-risk.
    """
    rng = np.random.default_rng(seed)
    table_2k = build_egd_edge_table(run.egd, run.Z_train, run.X_train, run.k_neighbors * k_multiplier, device=run.device)

    z_synth = rng.standard_normal((n_synthetic, run.Z_train.shape[1])).astype(np.float32)
    with torch.no_grad():
        # A synthetic node has no "true source" -- decode it relative to an all-zero
        # (population-mean, since features are standardized) reference, the only
        # generic reference point available for a prior sample with no real identity.
        x_synth = run.egd.decode_step(torch.tensor(z_synth, dtype=torch.float32),
                                      torch.zeros(n_synthetic, run.X_train.shape[1])).numpy()
        risk_synth = run.classifier(torch.tensor(x_synth, dtype=torch.float32)).numpy().squeeze()
    synth_low_risk = risk_synth < LOW_RISK_THRESHOLD
    if synth_low_risk.sum() == 0:
        return table_2k, None, None, None, None

    Z_aug = np.vstack([run.Z_train, z_synth[synth_low_risk]])
    X_aug = np.vstack([run.X_train, x_synth[synth_low_risk]])
    mask_aug = np.concatenate([run.low_risk_mask, np.ones(int(synth_low_risk.sum()), dtype=bool)])
    table_synth = build_egd_edge_table(run.egd, Z_aug, X_aug, run.k_neighbors, device=run.device)
    return table_2k, Z_aug, X_aug, mask_aug, table_synth


def densify_and_retry(run, table_2k, Z_aug, mask_aug, table_synth, pidx):
    """Checks whether the precomputed densified graphs resolve ONE abstaining patient.
    Note: a synthetic node's immutable feature is undefined (it has no real recorded
    identity) -- since find_recourse_egd uses x0 (the query's TRUE values) as the
    sequential-decode anchor throughout, and EGD's masked-immutable construction always
    copies the SOURCE's value forward regardless of which node's z is being decoded,
    the query's own immutables are preserved through any path (real or through a
    synthetic node) automatically; a synthetic node's OWN immutable value from
    build_densified_probes is only ever used as that node's raw X entry for the
    low_risk_mask/graph bookkeeping, never as a decode source for a DIFFERENT patient."""
    x0 = run.X_test[pidx]
    with torch.no_grad():
        z0 = run.egd.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()

    res_k = find_recourse_egd(run.egd, run.classifier, x0, z0, table_2k, run.Z_train, run.low_risk_mask,
                              low_risk_threshold=LOW_RISK_THRESHOLD, device=run.device)
    if res_k.success:
        return True, False

    if table_synth is None:
        return False, False
    res_synth = find_recourse_egd(run.egd, run.classifier, x0, z0, table_synth, Z_aug, mask_aug,
                                  low_risk_threshold=LOW_RISK_THRESHOLD, device=run.device)
    return False, bool(res_synth.success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-abstentions-per-seed", type=int, default=60,
                    help="Cap for the densification-probe analysis (each probe rebuilds an edge table); "
                         "the TRUE abstention rate is always reported in full, uncapped.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, summary_rows = [], []

    for dataset in args.datasets:
        for seed in args.seeds:
            run = run_dataset_seed_v5(dataset, seed, device=torch.device("cpu"))
            table = build_egd_edge_table(run.egd, run.Z_train, run.X_train, run.k_neighbors, device=run.device)
            patients = run.high_risk_test_idx[:20] if args.smoke else run.high_risk_test_idx

            t0 = time.perf_counter()
            abstentions, stages = [], {}
            for pidx in patients:
                x0 = run.X_test[pidx]
                with torch.no_grad():
                    z0 = run.egd.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
                res = find_recourse_egd(run.egd, run.classifier, x0, z0, table, run.Z_train, run.low_risk_mask,
                                        low_risk_threshold=LOW_RISK_THRESHOLD, device=run.device)
                if not res.success:
                    abstentions.append(pidx)
                    stages[pidx] = res.abstain_stage

            n_found = len(abstentions)
            sample = abstentions
            if n_found > args.max_abstentions_per_seed:
                rng = np.random.default_rng(seed)
                sample = list(rng.choice(abstentions, size=args.max_abstentions_per_seed, replace=False))
                print(f"[blindspot-egd]   NOTE: {n_found} abstentions found, analyzing sample of "
                      f"{args.max_abstentions_per_seed}", flush=True)

            print(f"[blindspot-egd] {dataset} seed={seed}: {len(patients)} high-risk, {n_found} true abstentions "
                  f"({100*n_found/max(len(patients),1):.1f}%), analyzing {len(sample)} in {time.perf_counter()-t0:.1f}s",
                  flush=True)

            n_cert, n_bs_k, n_bs_s, n_unres = 0, 0, 0, 0
            densified = None   # lazy: only pay for the two probe graphs if a feasible target ever turns up
            for pidx in sample:
                feasible, target = exhaustive_feasibility(run, pidx)
                if not feasible:
                    n_cert += 1
                    cls, rk, rs = "certified_infeasible", False, False
                else:
                    if densified is None:
                        t_d = time.perf_counter()
                        densified = build_densified_probes(run, 2, 500, seed)
                        print(f"[blindspot-egd]   built densification probes in {time.perf_counter()-t_d:.1f}s "
                              f"(reused for all remaining abstentions this seed)", flush=True)
                    table_2k, Z_aug, X_aug, mask_aug, table_synth = densified
                    rk, rs = densify_and_retry(run, table_2k, Z_aug, mask_aug, table_synth, pidx)
                    if rk:
                        n_bs_k += 1; cls = "blindspot_resolved_by_k"
                    elif rs:
                        n_bs_s += 1; cls = "blindspot_resolved_by_synthetic"
                    else:
                        n_unres += 1; cls = "unresolved"
                all_rows.append({"dataset": dataset, "seed": seed, "patient_idx": int(pidx),
                                 "abstain_stage": stages[pidx], "feasible_target_exists": feasible,
                                 "classification": cls, "resolved_by_2k": bool(rk), "resolved_by_synthetic": bool(rs)})

            n_analyzed = len(sample)
            summary_rows.append({
                "dataset": dataset, "seed": seed, "n_high_risk": len(patients),
                "n_abstain_true": n_found, "n_abstain_analyzed": n_analyzed,
                "pct_abstain": 100 * n_found / max(len(patients), 1),
                "was_subsampled": n_found > args.max_abstentions_per_seed,
                "pct_certified_infeasible": 100 * n_cert / max(n_analyzed, 1),
                "pct_confirmed_blindspot": 100 * (n_bs_k + n_bs_s) / max(n_analyzed, 1),
                "pct_unresolved": 100 * n_unres / max(n_analyzed, 1),
                "pct_abstain_stage_no_path": 100 * sum(1 for p in abstentions if stages[p] == "no_path") / max(n_found, 1),
                "pct_abstain_stage_risk_check": 100 * sum(1 for p in abstentions if stages[p] == "risk_check") / max(n_found, 1),
            })
            pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_blindspot.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"{dataset}_blindspot_summary.csv", index=False)

    print("[blindspot-egd] === summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
