"""
Main DSR runner: experiments E1 (component comparison), E2 (tolerance sweep),
E4 (Riemannian vs Euclidean under DSR), and E5 (decoder dependence).

Per (dataset, seed, decoder) the expensive work -- model training, k-NN
structure, batched Riemannian distances, and the one-shot decoded node cache --
happens ONCE and is shared by every method cell, so cells differ only in the
constraint predicate (and, for E4, which distance array is used as the weight).

Evaluation metric note (important for comparability): decoded-space CVR is
computed with experiments_v3's ORIGINAL checker
(`constraints_ext.check_decoded_violations`), unchanged, for every method
including the old one. The method changes; the yardstick does not.

Usage:
  .venv/bin/python -m experiments_v4.run_dsr --datasets uci nhanes_real --seeds 0 1 2 3 4
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

import experiments_v4.lib.dataset_registration  # noqa: F401 -- registers 'nhanes_real'
from benchmarks.metrics import compute_kde_density
from src.graph.clinical_constraints import compute_clinical_effort
from experiments_v4.lib.calibration import calibrate
from experiments_v4.lib.constraints_ext import check_decoded_violations, decode_nodes_batch
from experiments_v4.lib.dsr_graph import (
    augmented_matrix, build_dsr_edge_table, build_dsr_query_edges, make_matrix, make_query_row, prune_for_mode,
)
from experiments_v4.lib.pipeline_v4 import LOW_RISK_THRESHOLD, run_dataset_seed_v4
from experiments_v4.lib.repair import verify_and_repair
from experiments_v4.lib.search_v2 import find_path_augmented

RESULTS_DIR = REPO_ROOT / "experiments_v4" / "results"

# E1 method cells. `prune_mode` gates which edges exist at all; `verify_mode`
# is the standard Component 3's verify-and-repair stage checks against (only
# meaningful when repair=True; otherwise verification uses prune_mode).
#
# DESIGN NOTE -- a real discrepancy found and resolved, documented in full in
# CHANGES.md: applying Component 2's calibrated bands as a hard PRE-FILTER
# (prune_mode='decoded_banded') was tested first, exactly as a literal reading
# of the brief's component ordering suggests. It collapses the graph to zero
# surviving edges at every tau quantile from 0.05 to 0.99 on UCI. Diagnosis
# (experiments_v4/results/tau_vs_eps_diagnostic.csv): for a subset of features
# -- cholesterol and oldpeak on UCI -- the calibrated tau EXCEEDS the original
# epsilon even at the loosest quantile tested (q=0.05), because those features'
# decoder reconstruction error is intrinsically larger than the tight tolerance
# the original codebase assumed. Because a directional violation is an OR over
# ALL directional features, this one poisoned feature alone forces every edge
# to "violate," making pre-filtering with bands useless as a graph-connectivity
# rule regardless of confidence level -- not a coding bug, a property of this
# decoder's actual (measured) fidelity on those two features.
#
# Resolution: `dsr_full`'s GRAPH uses Component 1 only (prune_mode=
# 'decoded_naive') for connectivity -- so a path can still be found -- and
# Component 2's calibrated bands are applied at the VERIFY-AND-REPAIR stage
# (verify_mode='decoded_banded'), where a violation can be REPAIRED (clip the
# offending feature to the band boundary, re-encode) rather than making the
# edge's mere existence impossible. `dsr_c1c2` (bands as a pure pre-filter, no
# repair) is KEPT as its own cell precisely so this collapse is a visible,
# reported ablation result (Fig 2) rather than something worked around
# silently -- it is the direct evidence for the claim above.
METHOD_CELLS = [
    ("old_latent_pruning", dict(prune_mode="raw", verify_mode="raw", repair=False, metric="riemannian")),
    ("dsr_c1", dict(prune_mode="decoded_naive", verify_mode="decoded_naive", repair=False, metric="riemannian")),
    ("dsr_c1c2", dict(prune_mode="decoded_banded", verify_mode="decoded_banded", repair=False, metric="riemannian")),
    ("dsr_full", dict(prune_mode="decoded_naive", verify_mode="decoded_banded", repair=True, metric="riemannian")),
    # E4: same as dsr_full but Euclidean latent distance as the edge weight.
    ("dsr_full_euclidean", dict(prune_mode="decoded_naive", verify_mode="decoded_banded", repair=True, metric="euclidean")),
    # Isolates how much of C2 comes from bands vs. from the identity rule (pure pre-filter, no repair).
    ("dsr_bands_no_identity", dict(prune_mode="banded_no_identity", verify_mode="banded_no_identity", repair=False, metric="riemannian")),
]

# E2: tolerance sweep (quantile levels for tau). 0.95 is the nominal operating
# point. The low end (0.05-0.25) is included because a first run showed tau at
# q=0.95 is very large relative to the constraint epsilons (e.g. cholesterol
# tau=80 mg/dL vs. eps=1 mg/dL), which collapses success to zero -- the sweep
# has to span the range where the trade-off actually bites to be informative.
TAU_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.80, 0.90, 0.95, 0.99]


def evaluate_path_states(states, run, effort_weights):
    """Decoded-space CVR (v3 checker, unchanged), per-type flags, KDE, effort."""
    any_v_hop = False
    agg = {"immutable": False, "non_decreasing": False, "age_horizon": False, "directional": False}
    for a, b in zip(states[:-1], states[1:]):
        v, flags = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols,
                                            check_step_horizon=True)
        any_v_hop |= v
        for kk in agg:
            agg[kk] |= flags[kk]

    v_end, flags_end = check_decoded_violations(states[0], states[-1], run.feature_metadata, run.scaler,
                                                run.feature_cols, check_step_horizon=False)
    kde = float(compute_kde_density(np.vstack(states), run.X_train, bandwidth=0.5))
    effort = float(compute_clinical_effort(states[0], states[-1], effort_weights, run.feature_metadata,
                                           run.scaler, run.feature_cols))
    return any_v_hop, agg, v_end, flags_end, kde, effort


def run_cell(run, table, cell_name, cfg, taus, tau_level, patients, warmup=True):
    """Runs one method cell over all high-risk test patients for one (dataset, seed)."""
    prune_mode, verify_mode, do_repair, metric = cfg["prune_mode"], cfg["verify_mode"], cfg["repair"], cfg["metric"]

    t_prune = time.perf_counter()
    breakdown = prune_for_mode(table, run.feature_metadata, prune_mode, taus)
    base = make_matrix(table, breakdown, metric=metric)
    prune_seconds = time.perf_counter() - t_prune
    n_pruned = int(breakdown.any.sum())

    N = table.N
    rows = []

    # One untimed warm-up query before the timed loop (JIT / cache warm), run on
    # the first patient and discarded entirely -- NOT popped mid-loop, so every
    # cell evaluates the identical patient set and the paired tests line up.
    if warmup and len(patients) > 0:
        _warm = run_cell(run, table, cell_name, cfg, taus, tau_level, patients[:1], warmup=False)
        del _warm

    for pidx in patients:
        x0 = run.X_test[pidx]
        with torch.no_grad():
            z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))[0]
            z0 = z0.squeeze(0).cpu().numpy()

        t0 = time.perf_counter()
        qe = build_dsr_query_edges(run.vae, z0, x0, table, run.Z_train, run.X_train, run.feature_metadata,
                                   run.scaler, run.feature_cols, prune_mode, taus, device=run.device)
        aug = augmented_matrix(base, make_query_row(qe, N, metric=metric))
        target_mask = np.zeros(N + 1, dtype=bool)
        target_mask[:N] = run.low_risk_mask

        row = {
            "dataset": run.dataset, "seed": run.seed, "decoder": run.decoder, "cell": cell_name,
            "tau_level": tau_level, "patient_idx": int(pidx), "prune_mode": prune_mode, "verify_mode": verify_mode,
            "metric": metric,
            "repair_enabled": do_repair, "n_edges_pruned": n_pruned, "n_edges_total": len(table.edge_i),
            "prune_seconds": prune_seconds, "graph_build_seconds": table.build_seconds,
            "decode_seconds": table.decode_seconds,
        }

        try:
            path, cost = find_path_augmented(aug, N, target_mask)
        except ValueError:
            row.update({"success": False, "abstained": True, "abstain_stage": "search",
                        "repair_status": None, "latency_sec": time.perf_counter() - t0})
            rows.append(row)
            continue

        # Decoded trajectory: query's own reconstruction, then each graph node's.
        states = [qe.x0_hat] + [table.X_hat[n] for n in path[1:]]

        repair_status, n_steps_repaired, n_repair_attempts, fail_reason = "not_attempted", 0, 0, None
        if do_repair:
            rr = verify_and_repair(states, x0, run.vae, run.classifier, run.feature_metadata, run.scaler,
                                   run.feature_cols, taus, mode=verify_mode,
                                   low_risk_threshold=LOW_RISK_THRESHOLD, device=run.device)
            repair_status, n_steps_repaired, n_repair_attempts = rr.status, rr.n_steps_repaired, rr.n_repair_attempts
            fail_reason = rr.failure_reason
            trig = rr.triggering_flags or {}
            if rr.status == "failed":
                row.update({"success": False, "abstained": True, "abstain_stage": "repair",
                            "repair_status": repair_status, "n_steps_repaired": n_steps_repaired,
                            "n_repair_attempts": n_repair_attempts, "repair_failure_reason": fail_reason,
                            "repair_trigger_immutable": trig.get("immutable"),
                            "repair_trigger_non_decreasing": trig.get("non_decreasing"),
                            "repair_trigger_age_horizon": trig.get("age_horizon"),
                            "repair_trigger_directional": trig.get("directional"),
                            "latency_sec": time.perf_counter() - t0})
                rows.append(row)
                continue
            row.update({"repair_trigger_immutable": trig.get("immutable"),
                       "repair_trigger_non_decreasing": trig.get("non_decreasing"),
                       "repair_trigger_age_horizon": trig.get("age_horizon"),
                       "repair_trigger_directional": trig.get("directional")})
            states = rr.states

        latency = time.perf_counter() - t0
        v_hop, agg, v_end, flags_end, kde, effort = evaluate_path_states(states, run, run.effort_weights)

        row.update({
            "success": True, "abstained": False, "abstain_stage": None,
            "repair_status": repair_status, "n_steps_repaired": n_steps_repaired,
            "n_repair_attempts": n_repair_attempts, "repair_failure_reason": fail_reason,
            "path": json.dumps([int(p) for p in path]), "path_len": len(path), "path_cost": cost,
            "decoded_cvr_hop": bool(v_hop), "decoded_cvr_endpoint": bool(v_end),
            "viol_immutable": bool(agg["immutable"]), "viol_non_decreasing": bool(agg["non_decreasing"]),
            "viol_age_horizon": bool(agg["age_horizon"]), "viol_directional": bool(agg["directional"]),
            "kde": kde, "effort": effort, "latency_sec": latency,
        })
        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--decoders", nargs="+", default=["v3"], choices=["v2", "v3"])
    ap.add_argument("--tau-sweep", action="store_true", help="also run the E2 tolerance sweep")
    ap.add_argument("--smoke", action="store_true", help="5 patients per cell, for a fast sanity check")
    ap.add_argument("--output-suffix", default="", help="appended to output filenames (avoids clobbering another run's files, e.g. the E5 decoder-dependence run)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"

    meta = {
        "git_sha": sha, "seeds": args.seeds, "datasets": args.datasets, "decoders": args.decoders,
        "method_cells": [c for c, _ in METHOD_CELLS], "tau_levels": TAU_LEVELS if args.tau_sweep else [0.95],
        "operating_tau_quantile": 0.95, "calib_frac": 0.15,
        "low_risk_threshold": LOW_RISK_THRESHOLD, "device": "cpu",
        "torch_version": torch.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke": args.smoke,
    }
    with open(RESULTS_DIR / f"run_metadata{args.output_suffix}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dsr] metadata: {json.dumps(meta, indent=2)}", flush=True)

    for dataset in args.datasets:
        all_rows, calib_rows = [], []
        for decoder in args.decoders:
            for seed in args.seeds:
                t0 = time.perf_counter()
                run = run_dataset_seed_v4(dataset, seed, decoder=decoder, device=torch.device("cpu"))
                table = build_dsr_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                             run.scaler, run.feature_cols, run.k_neighbors, device=run.device)
                cal95 = calibrate(run.vae, run.X_calib, run.feature_metadata, run.scaler, run.feature_cols,
                                  quantile=0.95, device=run.device)
                calib_rows.extend([dict(r, decoder=decoder) for r in cal95.as_rows(dataset, seed)])

                patients = run.high_risk_test_idx[:5] if args.smoke else run.high_risk_test_idx
                print(f"[dsr] {dataset} seed={seed} decoder={decoder}: setup {time.perf_counter()-t0:.1f}s, "
                      f"N_train={len(run.X_train)} N_calib={len(run.X_calib)} "
                      f"N_high_risk={len(patients)} N_low_risk={int(run.low_risk_mask.sum())}", flush=True)
                print(f"[dsr]   tau@0.95: " + ", ".join(f"{k}={v:.3f}" for k, v in cal95.taus.items()), flush=True)

                for cell_name, cfg in METHOD_CELLS:
                    tc = time.perf_counter()
                    rows = run_cell(run, table, cell_name, cfg, cal95.taus, 0.95, patients)
                    all_rows.extend(rows)
                    ok = sum(r["success"] for r in rows)
                    cvr = np.mean([r["decoded_cvr_endpoint"] for r in rows if r["success"]]) * 100 if ok else float("nan")
                    print(f"[dsr]   {cell_name:26s} success={ok:4d}/{len(rows):<4d} "
                          f"decodedCVR={cvr:5.1f}% in {time.perf_counter()-tc:.1f}s", flush=True)

                if args.tau_sweep:
                    for q in TAU_LEVELS:
                        if q == 0.95:
                            continue   # already run above as the operating point
                        cal_q = calibrate(run.vae, run.X_calib, run.feature_metadata, run.scaler,
                                          run.feature_cols, quantile=q, device=run.device)
                        calib_rows.extend([dict(r, decoder=decoder) for r in cal_q.as_rows(dataset, seed)])
                        rows = run_cell(run, table, "dsr_full",
                                        dict(prune_mode="decoded_naive", verify_mode="decoded_banded",
                                             repair=True, metric="riemannian"),
                                        cal_q.taus, q, patients)
                        all_rows.extend(rows)
                        ok = sum(r["success"] for r in rows)
                        cvr = np.mean([r["decoded_cvr_endpoint"] for r in rows if r["success"]]) * 100 if ok else float("nan")
                        print(f"[dsr]   dsr_full @tau_q={q:<5} success={ok:4d}/{len(rows):<4d} decodedCVR={cvr:5.1f}%", flush=True)

                pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_per_patient{args.output_suffix}.csv", index=False)
                pd.DataFrame(calib_rows).to_csv(RESULTS_DIR / f"{dataset}_calibration{args.output_suffix}.csv", index=False)

        print(f"[dsr] wrote {len(all_rows)} rows for {dataset}", flush=True)


if __name__ == "__main__":
    main()
