"""
experiments_v6 main grid runner -- the CORRECTED baseline pipeline (Phase 1).

Adapts experiments_v3/run_grid.py's structure, with the two confirmed root
causes from experiments_v6_audit fixed as the ONLY mode (not an ablation
toggle -- RECOMMENDATION.md already established these as the corrected
foundation, not an experimental variant):

  B1 (entry-gate asymmetry): experiments_v6/lib/graph_build.py::build_query_edges
  now gates the query's entry edge with the SAME full predicate as every
  interior edge -- (hard_count + dir_count) > 0, check_step_horizon=True --
  instead of v2/v3's hard-only, no-horizon relaxed gate. See that file's
  docstring and CHANGES.md for the full evidence trail.

  B2 (unnecessary decoding of known real nodes): this script computes CVR,
  clinical effort, and path-plausibility (KDE) directly from RECORDED
  feature vectors (run.X_test[patient_idx], run.X_train[node]) for every
  real graph node, matching the ORIGINAL pipeline's own patient-facing
  function (src/recourse/search.py::decode_recourse_trajectory, which
  never calls vae_model.decode() despite accepting it as a parameter).
  VAE decoding is reserved for genuinely synthetic/interpolated nodes,
  none of which appear in this baseline grid (Phase 1 has no densification
  step -- that is Phase 2's certified-infeasibility protocol, which DOES
  probe synthetic nodes and decodes only those). Decoded-space values are
  ALSO computed and reported side-by-side, labeled clearly, purely for
  continuity with v2-v5's historical headline metric -- they are not used
  for any pass/fail or success determination in this script.

For each dataset in {uci, nhanes_real} x seed in SEEDS x grid cell in
GRID_CELLS: trains fresh seeded VAE + classifier, builds the shared k-NN
edge table, reweights per cell, runs recourse search for every high-risk
test patient (B1-fixed entry attachment), and records success, path,
REAL-value CVR (primary), decoded-space CVR (legacy, reported alongside),
per-category violation flags, KDE log-likelihood, clinical effort, and
per-patient latency (one untimed warm-up query per dataset/seed/cell).

Writes:
  experiments_v6/results/run_metadata.json   (git SHA, seeds, cells, versions)
  experiments_v6/results/{dataset}_per_patient.csv  (long format, all seeds/cells)

Usage:
  .venv/bin/python -m experiments_v6.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4
  .venv/bin/python -m experiments_v6.run_grid --datasets uci --seeds 0 --smoke   # fast sanity check
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

from src.graph.clinical_constraints import compute_clinical_effort
from benchmarks.metrics import compute_kde_density
import src.data.dataset_registration  # noqa: F401 -- registers 'nhanes_real' key
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.graph.constraints_ext import decode_node, check_decoded_violations, count_violations, violates as violates_fn

SOFT_LAMBDAS = [0.1, 1.0, 10.0, 100.0]


def build_grid_cells():
    cells = []
    for metric in ("riemannian", "euclidean"):
        cells.append((metric, "hard", None))
        for lam in SOFT_LAMBDAS:
            cells.append((metric, "soft", lam))
        cells.append((metric, "none", None))
    return cells


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def cell_label(metric, constraints, lam):
    if constraints == "soft":
        return f"{metric}_{constraints}_lam{lam}"
    return f"{metric}_{constraints}"


def evaluate_patient(run, edge_table, base_csr, metric, constraints, lam, patient_idx):
    """Runs recourse search for one high-risk test patient in one grid cell. Returns a result dict."""
    x0 = run.X_test[patient_idx]
    with torch.no_grad():
        mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))
    z0 = mu.squeeze(0).cpu().numpy()

    N = edge_table.N
    t0 = time.perf_counter()
    qedges = build_query_edges(
        run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols,
        k=run.k_neighbors, nbrs=edge_table.nbrs,
    )
    qrow = make_query_row(qedges, metric, constraints, N, lam=lam if lam is not None else 1.0)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    row = {
        "patient_idx": int(patient_idx), "success": False, "path_len": None, "path": None,
        "dijkstra_cost": None, "cvr_events_original": None,
        "real_cvr_hop": None, "real_cvr_endpoint": None, "real_violation_categories": None,
        "decoded_cvr_hop": None, "decoded_cvr_endpoint": None, "decoded_violation_categories": None,
        "kde": None, "effort": None, "latency_sec": None,
    }
    try:
        path, cost = find_path_augmented(aug, N, target_mask)
        success = True
    except ValueError:
        success = False
        path, cost = None, None
    t1 = time.perf_counter()
    row["latency_sec"] = t1 - t0
    row["success"] = success

    if not success:
        return row

    row["path_len"] = len(path) - 1
    row["path"] = json.dumps([int(p) for p in path])
    row["dijkstra_cost"] = cost

    # --- Original-space CVR events on path (consecutive hops), using the
    # already-computed edge-table / query-edge violation flags (B1-fixed
    # entry gate) where available; for hard cells this should always be 0
    # by construction.
    cvr_events = 0
    for a, b in zip(path[:-1], path[1:]):
        if a == N:  # query -> first train node
            pos = np.where(qedges.neighbor_idx == b)[0]
            if len(pos) > 0:
                cvr_events += int(qedges.violates[pos[0]])
            else:
                x_src = run.X_test[patient_idx]
                cvr_events += int(violates_fn(x_src, run.X_train[b], run.feature_metadata, run.scaler, run.feature_cols))
        else:
            mask = (edge_table.edge_i == a) & (edge_table.edge_j == b)
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                cvr_events += int(edge_table.violates[idxs[0]])
            else:
                cvr_events += int(violates_fn(run.X_train[a], run.X_train[b], run.feature_metadata, run.scaler, run.feature_cols))
    row["cvr_events_original"] = cvr_events

    # --- B2 fix: REAL-value path states. Every node on this baseline grid's
    # path is either the query itself (x0) or a real, already-known training
    # patient (run.X_train[node]) -- never a synthetic/interpolated point.
    # No VAE decode call is needed or made for these; this matches
    # src/recourse/search.py::decode_recourse_trajectory, the ORIGINAL
    # pipeline's own patient-facing function.
    real_points = [x0 if node == N else run.X_train[node] for node in path]

    any_hop_violation = False
    cat_flags = {"immutable": False, "non_decreasing": False, "age_horizon": False, "directional": False}
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, flags = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        any_hop_violation = any_hop_violation or v
        for k in cat_flags:
            cat_flags[k] = cat_flags[k] or flags[k]
    row["real_cvr_hop"] = any_hop_violation

    endpoint_violation, endpoint_flags = check_decoded_violations(
        real_points[0], real_points[-1], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False
    )
    row["real_cvr_endpoint"] = endpoint_violation
    for k in cat_flags:
        cat_flags[k] = cat_flags[k] or endpoint_flags[k]
    row["real_violation_categories"] = json.dumps(cat_flags)

    # --- Clinical effort: weighted L1 between REAL source and terminal states ---
    row["effort"] = float(compute_clinical_effort(
        real_points[0], real_points[-1], run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols
    ))

    # --- KDE log-likelihood of the REAL trajectory ---
    traj_matrix = np.vstack(real_points)
    row["kde"] = float(compute_kde_density(traj_matrix, run.X_train, bandwidth=0.5))

    # --- Legacy decoded-space measurement, reported alongside for
    # continuity with v2-v5's historical headline metric only. Not used
    # for success/failure or any other determination in this script.
    decoded_points = []
    for node in path:
        z = z0 if node == N else run.Z_train[node]
        decoded_points.append(decode_node(run.vae, z, device=run.device))

    dec_any_hop = False
    dec_cat_flags = {"immutable": False, "non_decreasing": False, "age_horizon": False, "directional": False}
    for a, b in zip(decoded_points[:-1], decoded_points[1:]):
        v, flags = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        dec_any_hop = dec_any_hop or v
        for k in dec_cat_flags:
            dec_cat_flags[k] = dec_cat_flags[k] or flags[k]
    row["decoded_cvr_hop"] = dec_any_hop

    dec_endpoint_violation, dec_endpoint_flags = check_decoded_violations(
        decoded_points[0], decoded_points[-1], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False
    )
    row["decoded_cvr_endpoint"] = dec_endpoint_violation
    for k in dec_cat_flags:
        dec_cat_flags[k] = dec_cat_flags[k] or dec_endpoint_flags[k]
    row["decoded_violation_categories"] = json.dumps(dec_cat_flags)

    return row


def run_one(dataset: str, seed: int, cells, device: torch.device, smoke: bool = False):
    print(f"[grid] dataset={dataset} seed={seed} : training models + building edge table...", flush=True)
    t0 = time.perf_counter()
    run = run_dataset_seed(dataset, seed, device=device)
    edge_table = build_edge_table(
        run.vae, run.Z_train, run.X_train, run.feature_metadata, run.scaler, run.feature_cols, k=run.k_neighbors
    )
    t1 = time.perf_counter()
    print(f"[grid]   setup done in {t1-t0:.1f}s. N_train={edge_table.N}, "
          f"N_high_risk_test={len(run.high_risk_test_idx)}, N_low_risk_train={int(run.low_risk_mask.sum())}", flush=True)

    patient_ids = run.high_risk_test_idx
    if smoke:
        patient_ids = patient_ids[:5]
        cells = cells[:3]

    rows = []
    for metric, constraints, lam in cells:
        label = cell_label(metric, constraints, lam)
        base_csr = make_cell_matrix(edge_table, metric, constraints, lam=lam if lam is not None else 1.0)

        if len(patient_ids) > 0:
            # one untimed warm-up query, discarded entirely
            evaluate_patient(run, edge_table, base_csr, metric, constraints, lam, patient_ids[0])

        t_cell0 = time.perf_counter()
        n_success = 0
        n_real_cvr = 0
        for pidx in patient_ids:
            r = evaluate_patient(run, edge_table, base_csr, metric, constraints, lam, pidx)
            r.update({"dataset": dataset, "seed": seed, "metric": metric, "constraints": constraints, "lambda": lam, "cell": label})
            rows.append(r)
            n_success += int(r["success"])
            if r["success"]:
                n_real_cvr += int(bool(r["real_cvr_endpoint"]) or bool(r["real_cvr_hop"]))
        t_cell1 = time.perf_counter()
        real_cvr_pct = 100.0 * n_real_cvr / max(1, n_success)
        print(f"[grid]   cell={label:28s} n={len(patient_ids):4d} success={n_success:4d} "
              f"({100.0*n_success/max(1,len(patient_ids)):5.1f}%) real_cvr={real_cvr_pct:5.1f}% in {t_cell1-t_cell0:.1f}s", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="Fast sanity check: 1 seed, 5 patients, 3 cells.")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # see experiments_v2/REPO_MAP.md D7: CPU is faster than MPS for this workload

    cells = build_grid_cells()
    metadata = {
        "git_sha": git_sha(),
        "seeds": args.seeds,
        "datasets": args.datasets,
        "grid_cells": [cell_label(*c) for c in cells],
        "k_neighbors": {"uci": 40, "nhanes_real": 80},
        "high_risk_threshold": 0.55,
        "low_risk_threshold": 0.45,
        "device": str(device),
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke": args.smoke,
        "corrected_pipeline": {
            "B1_entry_gate_fix": "gate_violates=(hard_count+dir_count)>0, check_step_horizon=True (matches interior edges)",
            "B2_no_unnecessary_decode": "real_points use X_test/X_train directly; decoded_* fields kept only for legacy comparison",
        },
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("[grid] metadata:", json.dumps(metadata, indent=2))

    for dataset in args.datasets:
        all_rows = []
        for seed in args.seeds:
            rows = run_one(dataset, seed, cells, device, smoke=args.smoke)
            all_rows.extend(rows)
            # incremental checkpoint after every seed, so a crash doesn't lose prior seeds
            df = pd.DataFrame(all_rows)
            csv_path = out_dir / f"{dataset}_per_patient.csv"
            df.to_csv(csv_path, index=False)
            print(f"[grid] checkpointed {len(df)} rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
