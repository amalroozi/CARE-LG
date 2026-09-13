"""
Main EGD runner: Phase 2 (graph+search) and Phase 4's controlled EGD-vs-old comparison.

"old_latent_pruning" here is a freshly-trained, same-split, same-seed, same-classifier
comparison cell (exactly mirroring experiments_v4's practice of re-running the old
method inside the new pipeline for a controlled comparison, rather than quoting
experiments_v3's published numbers directly) using the type-aware (v3) decoder and
v3's original hard-pruning rule (constraints_ext.violates, 'raw'/encoded values).

Usage:
  .venv/bin/python -m experiments_v5.run_egd --datasets uci nhanes_real --seeds 0 1 2 3 4
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

import experiments_v5.lib.dataset_registration  # noqa: F401
from benchmarks.metrics import compute_kde_density
from src.graph.clinical_constraints import compute_clinical_effort
from experiments_v5.lib.constraints_ext import check_decoded_violations, decode_nodes_batch
from experiments_v5.lib.egd_graph import build_egd_edge_table
from experiments_v5.lib.graph_build import build_edge_table, build_query_edges, make_cell_matrix, make_query_row
from experiments_v5.lib.pipeline_v5 import LOW_RISK_THRESHOLD, run_dataset_seed_v5
from experiments_v5.lib.search_egd import find_recourse_egd
from experiments_v5.lib.search_v2 import find_path_augmented

RESULTS_DIR = REPO_ROOT / "experiments_v5" / "results"


def run_egd_cell(run, table, patients, metric="riemannian"):
    rows = []
    for pidx in patients:
        x0 = run.X_test[pidx]
        t0 = time.perf_counter()
        with torch.no_grad():
            z0 = run.egd.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
        res = find_recourse_egd(run.egd, run.classifier, x0, z0, table, run.Z_train, run.low_risk_mask,
                                low_risk_threshold=LOW_RISK_THRESHOLD, metric=metric, device=run.device)
        latency = time.perf_counter() - t0

        row = {"dataset": run.dataset, "seed": run.seed, "cell": f"egd_{metric}", "patient_idx": int(pidx),
              "success": res.success, "abstain_stage": res.abstain_stage, "latency_sec": latency,
              "n_candidates_tried": res.n_candidates_tried}
        if res.success:
            full_traj = [x0] + res.decoded_states
            any_v = False
            for a, b in zip(full_traj[:-1], full_traj[1:]):
                v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
                any_v |= v
            v_end, _ = check_decoded_violations(full_traj[0], full_traj[-1], run.feature_metadata, run.scaler,
                                                run.feature_cols, check_step_horizon=False)
            kde = float(compute_kde_density(np.vstack(full_traj), run.X_train, bandwidth=0.5))
            effort = float(compute_clinical_effort(full_traj[0], full_traj[-1], run.effort_weights,
                                                   run.feature_metadata, run.scaler, run.feature_cols))
            row.update({"path_len": len(res.path), "decoded_cvr_hop": bool(any_v), "decoded_cvr_endpoint": bool(v_end),
                       "kde": kde, "effort": effort, "terminal_risk": res.terminal_risk})
        rows.append(row)
    return rows


def run_old_cell(run, patients):
    """Old (v3 latent-pruning) method, freshly trained in this same pipeline for a controlled comparison."""
    old_edge_table = build_edge_table(run.old_vae, run.Z_train_old, run.X_train, run.feature_metadata,
                                      run.scaler, run.feature_cols, run.k_neighbors)
    base = make_cell_matrix(old_edge_table, metric="riemannian", constraints="hard")
    N = old_edge_table.N
    X_hat_old = decode_nodes_batch(run.old_vae, run.Z_train_old, device=run.device)

    rows = []
    for pidx in patients:
        x0 = run.X_test[pidx]
        t0 = time.perf_counter()
        with torch.no_grad():
            z0 = run.old_vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
        qe = build_query_edges(run.old_vae, z0, x0, run.Z_train_old, run.X_train, run.feature_metadata,
                               run.scaler, run.feature_cols, run.k_neighbors, nbrs=old_edge_table.nbrs)
        qrow = make_query_row(qe, "riemannian", "hard", N)
        from experiments_v5.lib.graph_build import augmented_matrix as v3_augmented_matrix
        aug = v3_augmented_matrix(base, qrow)
        target_mask = np.zeros(N + 1, dtype=bool)
        target_mask[:N] = run.low_risk_mask

        row = {"dataset": run.dataset, "seed": run.seed, "cell": "old_latent_pruning", "patient_idx": int(pidx)}
        try:
            path, cost = find_path_augmented(aug, N, target_mask)
        except ValueError:
            row.update({"success": False, "latency_sec": time.perf_counter() - t0})
            rows.append(row)
            continue

        with torch.no_grad():
            x0_hat = decode_nodes_batch(run.old_vae, z0.reshape(1, -1), device=run.device)[0]
        states = [x0_hat] + [X_hat_old[n] for n in path[1:]]
        latency = time.perf_counter() - t0

        any_v = False
        for a, b in zip(states[:-1], states[1:]):
            v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
            any_v |= v
        v_end, _ = check_decoded_violations(states[0], states[-1], run.feature_metadata, run.scaler,
                                            run.feature_cols, check_step_horizon=False)
        kde = float(compute_kde_density(np.vstack(states), run.X_train, bandwidth=0.5))
        effort = float(compute_clinical_effort(states[0], states[-1], run.effort_weights, run.feature_metadata,
                                               run.scaler, run.feature_cols))
        row.update({"success": True, "path_len": len(path), "decoded_cvr_hop": bool(any_v),
                   "decoded_cvr_endpoint": bool(v_end), "kde": kde, "effort": effort, "latency_sec": latency})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"
    meta = {"git_sha": sha, "seeds": args.seeds, "datasets": args.datasets,
           "cells": ["old_latent_pruning", "egd_riemannian", "egd_euclidean"],
           "low_risk_threshold": LOW_RISK_THRESHOLD, "device": "cpu",
           "torch_version": torch.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": args.smoke}
    with open(RESULTS_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[egd] metadata: {json.dumps(meta, indent=2)}", flush=True)

    for dataset in args.datasets:
        all_rows = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            run = run_dataset_seed_v5(dataset, seed, device=torch.device("cpu"))
            patients = run.high_risk_test_idx[:15] if args.smoke else run.high_risk_test_idx
            print(f"[egd] {dataset} seed={seed}: setup {time.perf_counter()-t0:.1f}s, "
                  f"N_train={len(run.X_train)} N_high_risk={len(patients)} N_low_risk={int(run.low_risk_mask.sum())}", flush=True)

            tc = time.perf_counter()
            table = build_egd_edge_table(run.egd, run.Z_train, run.X_train, run.k_neighbors, device=run.device)
            print(f"[egd]   EGD edge table built in {time.perf_counter()-tc:.1f}s", flush=True)

            for metric in ["riemannian", "euclidean"]:
                tc = time.perf_counter()
                rows = run_egd_cell(run, table, patients, metric=metric)
                all_rows.extend(rows)
                ok = sum(r["success"] for r in rows)
                cvr = np.mean([r["decoded_cvr_endpoint"] for r in rows if r["success"]]) * 100 if ok else float("nan")
                print(f"[egd]   egd_{metric:12s} success={ok:4d}/{len(rows):<4d} decodedCVR={cvr:5.1f}% "
                      f"in {time.perf_counter()-tc:.1f}s", flush=True)

            tc = time.perf_counter()
            rows = run_old_cell(run, patients)
            all_rows.extend(rows)
            ok = sum(r["success"] for r in rows)
            cvr_vals = [r.get("decoded_cvr_endpoint") for r in rows if r["success"]]
            cvr = np.mean(cvr_vals) * 100 if cvr_vals else float("nan")
            print(f"[egd]   old_latent_pruning success={ok:4d}/{len(rows):<4d} decodedCVR={cvr:5.1f}% "
                  f"in {time.perf_counter()-tc:.1f}s", flush=True)

            pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{dataset}_per_patient.csv", index=False)

        print(f"[egd] wrote {len(all_rows)} rows for {dataset}", flush=True)


if __name__ == "__main__":
    main()
