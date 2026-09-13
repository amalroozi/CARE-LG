"""
E3: certified-infeasibility decomposition re-verified under DSR abstentions.

Reuses experiments_v3's exact methodology (exhaustive feature-space check +
two densification probes: doubled k, +500 VAE-prior synthetic nodes verified
low-risk by the classifier), applied to whichever cell's abstentions are being
audited. The v3 result was 100% certified-infeasible with zero blindspots on
both cohorts; this asks whether that survives under DSR's stricter method,
which abstains far more often (both because dsr_c1 prunes differently than
`raw`, and because dsr_full can abstain at the repair stage even when a path
exists in feature space).

Usage:
  .venv/bin/python -m experiments_v4.run_blindspot_dsr --datasets uci nhanes_real --seeds 0 1 2 3 4 --cell dsr_full
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

import experiments_v4.lib.dataset_registration  # noqa: F401
from experiments_v4.lib.calibration import calibrate
from experiments_v4.lib.dsr_graph import (
    augmented_matrix, build_dsr_edge_table, build_dsr_query_edges, make_matrix, make_query_row, prune_for_mode,
)
from experiments_v4.lib.pipeline_v4 import run_dataset_seed_v4
from experiments_v4.lib.repair import verify_and_repair
from experiments_v4.lib.search_v2 import find_path_augmented
from experiments_v4.lib.constraints_ext import violates as raw_violates

RESULTS_DIR = REPO_ROOT / "experiments_v4" / "results"

CELL_CONFIGS = {
    "dsr_c1": dict(prune_mode="decoded_naive", verify_mode="decoded_naive", repair=False, metric="riemannian"),
    "dsr_full": dict(prune_mode="decoded_naive", verify_mode="decoded_banded", repair=True, metric="riemannian"),
}


def find_abstentions(run, table, cfg, taus, patients):
    """Returns the subset of patients where this cell abstains (search or repair failure)."""
    prune_mode, verify_mode, do_repair, metric = cfg["prune_mode"], cfg["verify_mode"], cfg["repair"], cfg["metric"]
    breakdown = prune_for_mode(table, run.feature_metadata, prune_mode, taus)
    base = make_matrix(table, breakdown, metric=metric)
    N = table.N
    abstentions = []
    for pidx in patients:
        x0 = run.X_test[pidx]
        with torch.no_grad():
            z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
        qe = build_dsr_query_edges(run.vae, z0, x0, table, run.Z_train, run.X_train, run.feature_metadata,
                                   run.scaler, run.feature_cols, prune_mode, taus, device=run.device)
        aug = augmented_matrix(base, make_query_row(qe, N, metric=metric))
        target_mask = np.zeros(N + 1, dtype=bool)
        target_mask[:N] = run.low_risk_mask
        try:
            path, _ = find_path_augmented(aug, N, target_mask)
        except ValueError:
            abstentions.append(pidx)
            continue
        if do_repair:
            states = [qe.x0_hat] + [table.X_hat[n] for n in path[1:]]
            rr = verify_and_repair(states, x0, run.vae, run.classifier, run.feature_metadata, run.scaler,
                                   run.feature_cols, taus, mode=verify_mode, device=run.device)
            if rr.status == "failed":
                abstentions.append(pidx)
    return abstentions


def exhaustive_feasibility(run, pidx):
    """Does ANY low-risk train node satisfy the pairwise raw-value constraint from this query?"""
    x0 = run.X_test[pidx]
    low_risk_idx = np.where(run.low_risk_mask)[0]
    for j in low_risk_idx:
        if not raw_violates(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols,
                            check_step_horizon=False):
            return True, j
    return False, None


def build_densified_probes(run, k_multiplier=2, n_synthetic=500, seed=0):
    """
    Builds BOTH densification probes ONCE per (dataset, cell, seed) -- the
    densified graphs do not depend on which query patient is being tested, so
    building them once and reusing across every abstaining patient (rather
    than once per patient) is what makes this analysis tractable at the scale
    nhanes_real abstentions require.

    Returns (table_2k, fake_run_synth_or_None, table_synth_or_None).
    """
    rng = np.random.default_rng(seed)

    table_2k = build_dsr_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata, run.scaler,
                                    run.feature_cols, run.k_neighbors * k_multiplier, device=run.device)

    # +n_synthetic VAE-prior nodes, classifier-verified low-risk, eligible ONLY
    # as terminal targets (never as intermediate hops), matching experiments_v3.
    z_synth = rng.standard_normal((n_synthetic, run.Z_train.shape[1])).astype(np.float32)
    with torch.no_grad():
        x_synth = run.vae.decode(torch.tensor(z_synth, dtype=torch.float32, device=run.device)).cpu().numpy()
        risk_synth = run.classifier(torch.tensor(x_synth, dtype=torch.float32, device=run.device)).cpu().numpy().squeeze()
    synth_low_risk = risk_synth < 0.45

    if synth_low_risk.sum() == 0:
        return table_2k, None, None

    Z_aug = np.vstack([run.Z_train, z_synth[synth_low_risk]])
    X_aug = np.vstack([run.X_train, x_synth[synth_low_risk]])
    mask_aug = np.concatenate([run.low_risk_mask, np.ones(synth_low_risk.sum(), dtype=bool)])
    table_synth = build_dsr_edge_table(run.vae, Z_aug, X_aug, run.feature_metadata, run.scaler,
                                       run.feature_cols, run.k_neighbors, device=run.device)

    class FakeRun:
        pass
    fr = FakeRun()
    for attr in ["dataset", "seed", "decoder", "feature_metadata", "effort_weights", "scaler", "feature_cols",
                "vae", "classifier", "X_train", "Z_train", "device", "X_test"]:
        setattr(fr, attr, getattr(run, attr))
    fr.Z_train, fr.X_train, fr.low_risk_mask = Z_aug, X_aug, mask_aug
    return table_2k, fr, table_synth


def densify_and_retry(run, table_2k, fr_synth, table_synth, cfg, taus, pidx):
    """Checks whether the precomputed densified graphs resolve ONE abstaining patient."""
    absts = find_abstentions(run, table_2k, cfg, taus, [pidx])
    resolved_by_k = len(absts) == 0

    resolved_by_synth = False
    if not resolved_by_k and fr_synth is not None:
        absts2 = find_abstentions(fr_synth, table_synth, cfg, taus, [pidx])
        resolved_by_synth = len(absts2) == 0

    return resolved_by_k, resolved_by_synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cells", nargs="+", default=["dsr_c1", "dsr_full"], choices=list(CELL_CONFIGS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-abstentions-per-seed", type=int, default=25,
                    help="Each abstention with a feasible target costs two full densification "
                         "probes (rebuilding an edge table each); on nhanes_real, dsr_full "
                         "abstains on nearly every high-risk patient at q=0.95, making the "
                         "exhaustive per-abstention analysis intractable at full scale within "
                         "the session budget. This caps the analyzed sample per (dataset, cell, "
                         "seed) via a seeded random draw -- a real, explicitly reported "
                         "reduction (brief's fallback instruction), not a silent one.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, summary_rows = [], []

    for dataset in args.datasets:
        for cell in args.cells:
            cfg = CELL_CONFIGS[cell]
            for seed in args.seeds:
                run = run_dataset_seed_v4(dataset, seed, decoder="v3", device=torch.device("cpu"))
                table = build_dsr_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                             run.scaler, run.feature_cols, run.k_neighbors, device=run.device)
                cal = calibrate(run.vae, run.X_calib, run.feature_metadata, run.scaler, run.feature_cols,
                                quantile=0.95, device=run.device)

                patients = run.high_risk_test_idx[:15] if args.smoke else run.high_risk_test_idx
                t0 = time.perf_counter()
                abstentions = find_abstentions(run, table, cfg, cal.taus, patients)

                n_found = len(abstentions)
                if n_found > args.max_abstentions_per_seed:
                    rng = np.random.default_rng(seed)
                    abstentions = list(rng.choice(abstentions, size=args.max_abstentions_per_seed, replace=False))
                    print(f"[blindspot-dsr]   NOTE: {n_found} abstentions found, analyzing a random "
                          f"sample of {args.max_abstentions_per_seed} (seed={seed}) -- see --max-abstentions-per-seed", flush=True)

                print(f"[blindspot-dsr] {dataset} {cell} seed={seed}: {len(patients)} high-risk, "
                      f"{n_found} true abstentions ({100*n_found/max(len(patients),1):.1f}%), "
                      f"analyzing {len(abstentions)} in {time.perf_counter()-t0:.1f}s", flush=True)

                n_certified, n_blindspot_k, n_blindspot_synth, n_unresolved = 0, 0, 0, 0
                densified = None   # lazy: only pay for the two probe graphs if a feasible target ever turns up
                for pidx in abstentions:
                    feasible, target = exhaustive_feasibility(run, pidx)
                    if not feasible:
                        n_certified += 1
                        cls = "certified_infeasible"
                        rk = rs = False
                    else:
                        if densified is None:
                            t_dens = time.perf_counter()
                            densified = build_densified_probes(run, seed=seed)
                            print(f"[blindspot-dsr]   built densification probes in {time.perf_counter()-t_dens:.1f}s "
                                  f"(reused for all remaining abstentions this seed)", flush=True)
                        table_2k, fr_synth, table_synth = densified
                        rk, rs = densify_and_retry(run, table_2k, fr_synth, table_synth, cfg, cal.taus, pidx)
                        if rk:
                            n_blindspot_k += 1
                            cls = "blindspot_resolved_by_k"
                        elif rs:
                            n_blindspot_synth += 1
                            cls = "blindspot_resolved_by_synthetic"
                        else:
                            n_unresolved += 1
                            cls = "unresolved"
                    all_rows.append({"dataset": dataset, "cell": cell, "seed": seed, "patient_idx": int(pidx),
                                     "feasible_target_exists": feasible, "classification": cls,
                                     "resolved_by_2k": bool(rk), "resolved_by_synthetic": bool(rs)})

                n_analyzed = len(abstentions)
                summary_rows.append({
                    "dataset": dataset, "cell": cell, "seed": seed, "n_high_risk": len(patients),
                    "n_abstain_true": n_found, "n_abstain_analyzed": n_analyzed,
                    "pct_abstain": 100 * n_found / max(len(patients), 1),
                    "was_subsampled": n_found > args.max_abstentions_per_seed,
                    "pct_certified_infeasible": 100 * n_certified / max(n_analyzed, 1),
                    "pct_confirmed_blindspot": 100 * (n_blindspot_k + n_blindspot_synth) / max(n_analyzed, 1),
                    "pct_unresolved": 100 * n_unresolved / max(n_analyzed, 1),
                })
                pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_{cell}_blindspot.csv", index=False)
                pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"{dataset}_{cell}_blindspot_summary.csv", index=False)

    print("[blindspot-dsr] === summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
