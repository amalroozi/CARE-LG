"""
Phase 8 -- FACE rerun under the Phase 1 corrected evaluation protocol.

FACE (Poyiadzi et al., AIES 2020): shortest-path recourse over a k-NN graph
built directly in ORIGINAL feature space (no latent embedding, no VAE
involved at all, and -- per the paper and this project's existing
`benchmarks/run_benchmarks.py::run_face` -- no admissibility/constraint
gating on edges; every k-NN edge exists regardless of clinical plausibility).
Because FACE never touches the VAE, there is no decoded-vs-real CVR
distinction for it the way there was for the B1/B2 bug -- FACE's recourse
target has always been a real recorded training patient
(`benchmarks/run_benchmarks.py:87`, `recourse_x = X_train[best_target]`).
"Rerun under the corrected protocol" here means: evaluate it on the SAME
v6 corrected test cohort, query-attachment convention (ephemeral ended
node, full attachment -- not k-restricted, matching `graph_build.py`'s D8
design so the comparison is apples-to-apples on search breadth), and
real-value CVR metric as Phase 1, rather than reusing the older
`benchmarks/run_benchmarks.py` numbers (100-instance subset, different
success/CVR bookkeeping, no seeding loop).

This is FACE's own unconstrained design, faithfully reproduced -- it is
expected to show much higher success (nothing is ever pruned) and much
higher real-value CVR (nothing prevents a clinically inadmissible hop)
than the corrected v6 CARE-LG. That contrast is the entire point of the
comparison table in FINDINGS.md.

Usage:
  .venv/bin/python -m experiments_v6.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4
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
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import torch
from sklearn.neighbors import NearestNeighbors

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.constraints_ext import check_decoded_violations

K_NEIGHBORS_FACE = {"uci": 40, "nhanes_real": 80}  # same k as v6's own graph, for a fair comparison


def build_face_graph(X_train: np.ndarray, k: int):
    N = len(X_train)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    ei, ej, w = [], [], []
    for i in range(N):
        for idx, nb in enumerate(indices[i]):
            if nb == i:
                continue
            ei.append(i); ej.append(nb); w.append(distances[i][idx])
    return sp.csr_matrix((w, (ei, ej)), shape=(N, N)), nbrs


def evaluate_patient_face(x0, X_train, base_csr, low_risk_mask, feature_metadata, scaler, feature_cols):
    N = len(X_train)
    # Ephemeral query node, full attachment (no k-restriction, no gating), Euclidean
    # distance in ORIGINAL feature space -- FACE's own construction, unmodified.
    t0 = time.perf_counter()
    dists = np.linalg.norm(X_train - x0.reshape(1, -1), axis=1)
    q_row = sp.csr_matrix((dists, (np.zeros(N, dtype=np.int64), np.arange(N))), shape=(1, N))
    last_col = sp.csr_matrix((N, 1))
    top = sp.hstack([base_csr, last_col], format="csr")
    bottom = sp.hstack([q_row, sp.csr_matrix((1, 1))], format="csr")
    aug = sp.vstack([top, bottom], format="csr")

    row = {"success": False, "real_cvr_hop": None, "real_cvr_endpoint": None, "latency_sec": None}
    try:
        distances, predecessors = csgraph.dijkstra(csgraph=aug, directed=True, indices=N, return_predecessors=True)
        target_indices = np.where(low_risk_mask)[0]
        target_dists = distances[target_indices]
        finite_mask = np.isfinite(target_dists)
        if not np.any(finite_mask):
            row["latency_sec"] = time.perf_counter() - t0
            return row
        best_target = target_indices[np.argmin(target_dists)]
        path = []
        curr = best_target
        while curr != N and curr != -9999:
            path.append(curr)
            curr = predecessors[curr]
        path.append(N)
        path.reverse()
    except Exception:
        row["latency_sec"] = time.perf_counter() - t0
        return row
    row["latency_sec"] = time.perf_counter() - t0
    row["success"] = True

    real_points = [x0 if n == N else X_train[n] for n in path]
    any_hop = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, feature_metadata, scaler, feature_cols, check_step_horizon=True)
        any_hop = any_hop or v
    row["real_cvr_hop"] = any_hop
    endpoint_v, _ = check_decoded_violations(real_points[0], real_points[-1], feature_metadata, scaler, feature_cols, check_step_horizon=False)
    row["real_cvr_endpoint"] = endpoint_v
    return row


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

    all_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            base_csr, _ = build_face_graph(run.X_train, K_NEIGHBORS_FACE[dataset])
            t0 = time.perf_counter()
            n_success, n_cvr = 0, 0
            for pidx in run.high_risk_test_idx:
                r = evaluate_patient_face(run.X_test[pidx], run.X_train, base_csr, run.low_risk_mask,
                                           run.feature_metadata, run.scaler, run.feature_cols)
                r.update({"dataset": dataset, "seed": seed, "patient_idx": int(pidx)})
                all_rows.append(r)
                n_success += int(r["success"])
                if r["success"]:
                    n_cvr += int(bool(r["real_cvr_endpoint"]) or bool(r["real_cvr_hop"]))
            t1 = time.perf_counter()
            n = len(run.high_risk_test_idx)
            print(f"[face] {dataset} seed={seed}: success={n_success}/{n} ({100.0*n_success/max(1,n):.1f}%) "
                  f"real_cvr={100.0*n_cvr/max(1,n_success):.1f}% in {t1-t0:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "face_rerun_per_patient.csv", index=False)
    with open(out_dir / "face_rerun_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "k_neighbors": K_NEIGHBORS_FACE, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print("\n[face] === SUMMARY (mean +/- std over seeds) ===")
    for dataset in args.datasets:
        d = df[df.dataset == dataset]
        per_seed_succ = d.groupby("seed")["success"].mean() * 100
        ok = d[d.success == True]
        real_viol = ok["real_cvr_endpoint"].astype(bool) | ok["real_cvr_hop"].astype(bool)
        per_seed_cvr = real_viol.groupby(ok["seed"]).mean() * 100 if len(ok) else pd.Series(dtype=float)
        print(f"{dataset:12s} success={per_seed_succ.mean():.1f}+/-{per_seed_succ.std():.1f}%  "
              f"real_cvr={per_seed_cvr.mean():.1f}+/-{per_seed_cvr.std():.1f}%")


if __name__ == "__main__":
    main()
