"""
Phase 1.5 (of the age-ceiling follow-up plan): redefines "low risk" (the
recourse TARGET pool) per age bracket instead of one global 0.45 threshold
for every patient regardless of age -- directly targeting the ~25% of
Real-NHANES certified-infeasible patients the age-ceiling diagnostic found
have a genuine, best-case-still-fails ceiling under the global threshold.

Method: bucket the REAL training population into age quartiles. Within each
bracket, a training patient counts as "low risk" if their predicted risk is
below that bracket's OWN 25th-percentile risk cutoff (i.e. "low risk for
someone your age," not "low risk on an absolute scale set by the youngest
patients"). This is a real, defensible clinical framing (age-stratified risk
categories are standard practice), not an invented shortcut.

WHO gets evaluated (high_risk_test_idx, the classifier's own 0.55 cutoff) is
left UNCHANGED, for continuity with every other published number -- only
which real training patients count as valid recourse TARGETS changes.

Same graph, same B1/B2-fixed search code, same exhaustive certified-
infeasibility protocol as the main results -- only the low_risk_mask input
differs.

Usage:
  .venv/bin/python -m scripts.run_age_stratified_experiment
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
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix, unscale_matrix
from src.recourse.search_augmented import find_path_augmented
from src.graph.constraints_ext import check_decoded_violations, violates as violates_fn
from src.seeding import set_all_seeds

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DATASETS = ["uci", "nhanes_real"]
SEEDS = [0, 1, 2, 3, 4]
METRIC = "riemannian"
M_SYNTHETIC = 500
N_QUARTILES = 4
BRACKET_RISK_PERCENTILE = 25  # "low risk" = bottom quartile of risk, WITHIN each age bracket


def age_stratified_low_risk_mask(run):
    ages = unscale_matrix(run.X_train, run.scaler, run.feature_cols, run.feature_metadata)["age"]
    edges = np.percentile(ages, np.linspace(0, 100, N_QUARTILES + 1))
    edges[0] -= 1e-6
    brackets = np.digitize(ages, edges[1:-1])  # 0..N_QUARTILES-1
    mask = np.zeros(len(ages), dtype=bool)
    bracket_thresholds = {}
    for b in range(N_QUARTILES):
        idx = np.where(brackets == b)[0]
        if len(idx) == 0:
            continue
        thr = np.percentile(run.train_risk[idx], BRACKET_RISK_PERCENTILE)
        bracket_thresholds[b] = {"age_range": [float(edges[b]), float(edges[b + 1])], "n": len(idx), "risk_threshold": float(thr)}
        mask[idx] = run.train_risk[idx] < thr
    return mask, bracket_thresholds


def try_search(run, edge_table, base_csr, pidx, low_risk_mask):
    x0 = run.X_test[pidx]
    with torch.no_grad():
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
    N = edge_table.N
    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)
    qrow = make_query_row(qedges, METRIC, "hard", N)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([low_risk_mask, [False]])
    try:
        path, cost = find_path_augmented(aug, N, target_mask)
    except ValueError:
        return None
    real_points = [x0 if n == N else run.X_train[n] for n in path]
    any_hop = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        any_hop = any_hop or v
    endpoint_v, _ = check_decoded_violations(real_points[0], real_points[-1], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)
    return {"success": True, "real_cvr": bool(any_hop or endpoint_v)}


def certified_check(run, low_risk_mask, seed, abstained_pidx):
    if not abstained_pidx:
        return []
    low_risk_idxs = np.where(low_risk_mask)[0]
    set_all_seeds(seed + 100000)
    edge_table_2k = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                      run.scaler, run.feature_cols, k=2 * run.k_neighbors)
    base_hard_2k = make_cell_matrix(edge_table_2k, METRIC, "hard")
    with torch.no_grad():
        z_synth = torch.randn(M_SYNTHETIC, run.vae.latent_dim)
        x_synth = run.vae.decode(z_synth).numpy()
        risk_synth = run.classifier(torch.tensor(x_synth, dtype=torch.float32)).numpy().squeeze()
    # a synthetic node counts as low-risk here if it clears the LOOSEST (youngest-bracket) threshold --
    # conservative in the direction of NOT overstating densification's power under this new definition.
    loosest_thr = 0.45
    synth_low_risk = risk_synth < loosest_thr
    Z_combined = np.vstack([run.Z_train, z_synth.numpy()])
    X_combined = np.vstack([run.X_train, x_synth])
    low_risk_combined = np.concatenate([low_risk_mask, synth_low_risk])
    edge_table_synth = build_edge_table(run.vae, Z_combined, X_combined, run.feature_metadata,
                                        run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard_synth = make_cell_matrix(edge_table_synth, METRIC, "hard")

    rows = []
    for pidx in abstained_pidx:
        x0 = run.X_test[pidx]
        valid_targets = [j for j in low_risk_idxs
                         if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)]
        certified_infeasible = len(valid_targets) == 0
        resolved_2k = resolved_synth = None
        if not certified_infeasible:
            resolved_2k = try_search(run, edge_table_2k, base_hard_2k, pidx, low_risk_mask) is not None
            resolved_synth = try_search(run, edge_table_synth, base_hard_synth, pidx, low_risk_combined) is not None
        rows.append({
            "certified_infeasible": certified_infeasible,
            "confirmed_blindspot": (not certified_infeasible) and bool(resolved_2k or resolved_synth),
            "unresolved": (not certified_infeasible) and not bool(resolved_2k or resolved_synth),
        })
    return rows


def run_one(dataset, label, low_risk_mask_fn, seed):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard = make_cell_matrix(edge_table, METRIC, "hard")

    if label == "original":
        low_risk_mask = run.low_risk_mask
        bracket_info = None
    else:
        low_risk_mask, bracket_info = age_stratified_low_risk_mask(run)

    high_risk_idx = run.high_risk_test_idx  # UNCHANGED -- same cohort as every other result
    rows, abstained = [], []
    for pidx in high_risk_idx:
        r = try_search(run, edge_table, base_hard, pidx, low_risk_mask)
        if r is None:
            abstained.append(pidx)
            rows.append({"success": False, "real_cvr": None})
        else:
            rows.append({"success": True, "real_cvr": r["real_cvr"]})

    cert_rows = certified_check(run, low_risk_mask, seed, abstained)
    n = len(high_risk_idx)
    n_success = sum(r["success"] for r in rows)
    n_cvr = sum(1 for r in rows if r["success"] and r["real_cvr"])
    n_cert = sum(1 for c in cert_rows if c["certified_infeasible"])

    return {
        "dataset": dataset, "label": label, "seed": seed,
        "n_high_risk": n, "n_low_risk_train": int(low_risk_mask.sum()),
        "n_success": n_success, "success_pct": 100.0 * n_success / max(1, n),
        "real_cvr_pct": 100.0 * n_cvr / max(1, n_success) if n_success else float("nan"),
        "n_abstain": len(abstained),
        "certified_infeasible_pct": 100.0 * n_cert / max(1, len(cert_rows)) if cert_rows else float("nan"),
        "bracket_info": bracket_info,
    }


def main():
    all_rows = []
    bracket_infos = {}
    for dataset in DATASETS:
        for label in ["original", "age_stratified"]:
            for seed in SEEDS:
                t0 = time.perf_counter()
                r = run_one(dataset, label, None, seed)
                bi = r.pop("bracket_info")
                if bi and dataset not in bracket_infos:
                    bracket_infos[dataset] = bi
                all_rows.append(r)
                print(f"[age-strat] {dataset:12s} {label:15s} seed={seed}: n_high={r['n_high_risk']:4d} "
                      f"n_low_risk_train={r['n_low_risk_train']:4d} success={r['success_pct']:5.1f}% "
                      f"real_cvr={r['real_cvr_pct']:5.1f}% certified_infeasible={r['certified_infeasible_pct']:5.1f}% "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "age_stratified_per_seed.csv", index=False)
    with open(TABLES_DIR / "age_stratified_bracket_info.json", "w") as f:
        json.dump(bracket_infos, f, indent=2)
    print(f"\n[age-strat] wrote results/tables/age_stratified_per_seed.csv and bracket_info.json")


if __name__ == "__main__":
    main()
