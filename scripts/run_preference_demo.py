"""
Phase 6 demonstration: same patient, same admissible edge set (safety
completely unchanged from Phase 1's B1-fixed hard-mode graph), three
different synthetic preference profiles, showing how the SELECTED path and
its feature-change composition differ purely from cost reweighting. Small
and illustrative by design (per the brief) -- a handful of test patients
per dataset, not a full ablation.

Usage:
  .venv/bin/python -m experiments_v6.run_preference_demo --datasets uci nhanes_real --seeds 0 --n_patients 5
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
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table, build_query_edges, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.recourse.preference import preference_profiles, make_cell_matrix_preference, make_query_row_preference

MU = 0.5  # weight of the preference-cost term relative to latent distance
METRIC = "riemannian"


def run_patient(run, edge_table, pidx, profiles):
    x0 = run.X_test[pidx]
    with torch.no_grad():
        mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))
    z0 = mu.squeeze(0).numpy()
    N = edge_table.N
    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    rows = []
    for profile_name, weights in profiles.items():
        base_csr = make_cell_matrix_preference(edge_table, run.X_train, METRIC, weights,
                                                run.feature_metadata, run.feature_cols, mu=MU)
        qrow = make_query_row_preference(qedges, x0, run.X_train, METRIC, weights,
                                          run.feature_metadata, run.feature_cols, N, mu=MU)
        aug = augmented_matrix(base_csr, qrow)
        try:
            path, cost = find_path_augmented(aug, N, target_mask)
        except ValueError:
            rows.append({"patient_idx": int(pidx), "profile": profile_name, "success": False})
            continue
        target_node = path[-1]
        x_target = run.X_train[target_node]
        deltas = {f: float(x_target[run.feature_cols.index(f)] - x0[run.feature_cols.index(f)])
                  for f in run.feature_metadata['continuous_mutable']}
        rows.append({
            "patient_idx": int(pidx), "profile": profile_name, "success": True,
            "path_len": len(path) - 1, "target_node": int(target_node), "cost": float(cost),
            "feature_deltas": json.dumps(deltas),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n_patients", type=int, default=5)
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
            edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                          run.scaler, run.feature_cols, k=run.k_neighbors)
            profiles = preference_profiles(dataset, run.effort_weights)
            # Only demo on patients where R+hard actually succeeds (Phase 1's admissible
            # set) -- preference reweighting has nothing to show on an abstention.
            base_hard_qonly_candidates = run.high_risk_test_idx[: args.n_patients * 4]  # small oversample
            demoed = 0
            for pidx in base_hard_qonly_candidates:
                if demoed >= args.n_patients:
                    break
                rows = run_patient(run, edge_table, pidx, profiles)
                if any(r["success"] for r in rows):
                    for r in rows:
                        r.update({"dataset": dataset, "seed": seed})
                    all_rows.extend(rows)
                    demoed += 1
            print(f"[preference] {dataset} seed={seed}: demoed {demoed} patients x {len(profiles)} profiles", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "preference_demo.csv", index=False)
    with open(out_dir / "preference_demo_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "n_patients": args.n_patients, "mu": MU, "metric": METRIC,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\n[preference] wrote {len(df)} rows to {out_dir / 'preference_demo.csv'}")

    # Print a compact human-readable summary: for each demoed patient, which target
    # node (and path length) each profile selected -- shows whether preferences
    # actually changed the outcome, not just the reported cost.
    for (dataset, seed, pidx), g in df.groupby(["dataset", "seed", "patient_idx"]):
        print(f"\n  patient {dataset}/seed{seed}/#{pidx}:")
        for _, row in g.iterrows():
            if row["success"]:
                print(f"    {row['profile']:20s} -> target_node={row['target_node']:5d}  path_len={row['path_len']}  cost={row['cost']:.3f}")
            else:
                print(f"    {row['profile']:20s} -> (no admissible path under this cost reweighting)")


if __name__ == "__main__":
    main()
