"""
Stage 2 of the approved Stage 1 / Stage 2 hybrid plan: a DIRECTED SYNTHETIC
search, run ONLY for patients Stage 1 (the real-graph search, using the
age-stratified target definition from Phase 1.5) certifies infeasible.

Stage 1 is completely unchanged by this script -- same code path as
run_age_stratified_experiment.py's "age_stratified" arm (0% real-value CVR,
intact). Stage 2 is a separate, clearly-labeled sub-result layered on top,
never silently merged into Stage 1's own numbers.

Method (Riemannian natural-gradient latent traversal, per Pegios et al.,
NeurIPS 2024 Workshop, arXiv:2411.02259 -- same pullback metric this
project's graph edges already use, G(z) = J_decode(z)^T J_decode(z), but
here driving continuous optimization instead of discrete graph search):

  1. Start at z0 = encode(x0).
  2. Each step: compute the ordinary gradient of classifier risk w.r.t. z
     (through the decoder), then precondition it with the local inverse
     Riemannian metric G(z)^-1 (a natural-gradient / Gauss-Newton step),
     so the step respects the VAE manifold's actual local geometry instead
     of treating latent space as flat Euclidean.
  3. Decode the new z, then CLAMP the decoded point back onto the
     admissible region relative to x0: immutable features and age are
     frozen exactly at x0's value ("age frozen by design" -- this was the
     explicit point of Stage 2, per the approved plan); directional_reduce
     features can only move down from x0; directional_increase features
     can only move up. This is a hard projection, not a soft penalty.
  4. Re-encode the clamped point and repeat, tracking the best (lowest
     clamped-decoded) risk seen.
  5. Success = best risk drops below the SAME per-patient target threshold
     Stage 1 used (the age-stratified bracket threshold for that patient's
     own age bracket -- age is frozen, so the bracket never changes mid-
     search).

Because every candidate is hard-clamped onto the admissible region at every
step, constraint violations are checked (not assumed) at the end via
check_decoded_violations and reported honestly, same as every other real
CVR number in this project.

Usage:
  .venv/bin/python -m scripts.run_stage2_synthetic_search
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
from torch.func import jacrev

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.clinical_constraints import unscale_features
from src.graph.constraints_ext import check_decoded_violations, violates as violates_fn
from scripts.run_age_stratified_experiment import age_stratified_low_risk_mask, try_search
from src.graph.query_attachment import build_edge_table, make_cell_matrix, unscale_matrix

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DATASETS = ["uci", "nhanes_real"]
SEEDS = [0, 1, 2, 3, 4]
METRIC = "riemannian"
MAX_ITERS = 50
LR = 0.12
REG = 1e-2


def scale_features(x_dict, scaler, feature_cols, feature_metadata):
    continuous_cols = feature_metadata['continuous_mutable'] + feature_metadata['non_decreasing']
    x = np.zeros(len(feature_cols))
    if scaler is not None and continuous_cols:
        cont_vals = np.array([[x_dict[c] for c in continuous_cols]])
        scaled_cont = np.atleast_1d(scaler.transform(cont_vals).squeeze())
        for i, c in enumerate(continuous_cols):
            x[feature_cols.index(c)] = scaled_cont[i]
    for c in feature_cols:
        if c not in continuous_cols:
            x[feature_cols.index(c)] = x_dict[c]
    return x


def clamp_to_constraints(x_hat_scaled, x0_scaled, feature_metadata, scaler, feature_cols):
    """Hard-projects a decoded candidate onto the admissible region relative
    to x0: immutables + age frozen exactly, directional features one-way
    only. Every feature column is covered by one of these categories
    (checked once at pipeline build time by validate_exhaustive_column_classification)."""
    hat = unscale_features(x_hat_scaled, scaler, feature_cols, feature_metadata)
    x0d = unscale_features(x0_scaled, scaler, feature_cols, feature_metadata)
    out = dict(hat)
    for c in feature_metadata.get('immutable', []):
        out[c] = x0d[c]
    for c in feature_metadata.get('non_decreasing', []):  # age: frozen, not just non-decreasing
        out[c] = x0d[c]
    for c in feature_metadata.get('directional_reduce', []):
        out[c] = min(out[c], x0d[c])
    for c in feature_metadata.get('directional_increase', []):
        out[c] = max(out[c], x0d[c])
    return scale_features(out, scaler, feature_cols, feature_metadata)


def bracket_threshold_for_age(age, bracket_thresholds):
    for b, info in bracket_thresholds.items():
        lo, hi = info["age_range"]
        if lo <= age <= hi:
            return info["risk_threshold"]
    # test-set age outside train quartile edges (rare): clamp to nearest bracket
    keys = sorted(bracket_thresholds.keys(), key=lambda b: bracket_thresholds[b]["age_range"][0])
    if age < bracket_thresholds[keys[0]]["age_range"][0]:
        return bracket_thresholds[keys[0]]["risk_threshold"]
    return bracket_thresholds[keys[-1]]["risk_threshold"]


def stage2_search(run, x0_scaled, target_threshold):
    vae, classifier = run.vae, run.classifier
    d = vae.latent_dim
    x0_t = torch.tensor(x0_scaled, dtype=torch.float32)
    with torch.no_grad():
        z = vae.encode(x0_t.unsqueeze(0))[0].squeeze(0)
        risk0 = float(classifier(x0_t.unsqueeze(0)).item())

    def decode_single(zz):
        return vae.decode(zz.unsqueeze(0)).squeeze(0)

    best_risk, best_x = risk0, x0_scaled.copy()
    for it in range(MAX_ITERS):
        z_req = z.clone().requires_grad_(True)
        x_hat = decode_single(z_req)
        risk = classifier(x_hat.unsqueeze(0)).squeeze()
        grad_z, = torch.autograd.grad(risk, z_req)
        with torch.no_grad():
            J = jacrev(decode_single)(z)  # (D, d)
            G = J.T @ J + REG * torch.eye(d)
            nat_grad = torch.linalg.solve(G, grad_z.detach())
            z_next = z - LR * nat_grad
            x_hat_next = decode_single(z_next).numpy()
        x_clamped = clamp_to_constraints(x_hat_next, x0_scaled, run.feature_metadata, run.scaler, run.feature_cols)
        with torch.no_grad():
            risk_clamped = float(classifier(torch.tensor(x_clamped, dtype=torch.float32).unsqueeze(0)).item())
        if risk_clamped < best_risk:
            best_risk, best_x = risk_clamped, x_clamped
        with torch.no_grad():
            z = vae.encode(torch.tensor(x_clamped, dtype=torch.float32).unsqueeze(0))[0].squeeze(0)
        if best_risk < target_threshold:
            break

    success = best_risk < target_threshold
    violation, flags = check_decoded_violations(x0_scaled, best_x, run.feature_metadata, run.scaler,
                                                 run.feature_cols, check_step_horizon=False)
    return {
        "risk_before": risk0, "risk_after": best_risk, "iters": it + 1,
        "success": bool(success), "constraint_violation": bool(violation), "violation_flags": flags,
    }


def main():
    all_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            run = run_dataset_seed(dataset, seed)
            low_risk_mask, bracket_thresholds = age_stratified_low_risk_mask(run)
            low_risk_idx = np.where(low_risk_mask)[0]

            edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                          run.scaler, run.feature_cols, k=run.k_neighbors)
            base_hard = make_cell_matrix(edge_table, METRIC, "hard")

            ages_test = unscale_matrix(run.X_test, run.scaler, run.feature_cols, run.feature_metadata)["age"]

            n_stage1_success, certified_infeasible_pidx = 0, []
            for pidx in run.high_risk_test_idx:
                r = try_search(run, edge_table, base_hard, pidx, low_risk_mask)
                if r is not None:
                    n_stage1_success += 1
                else:
                    x0 = run.X_test[pidx]
                    valid_targets = [j for j in low_risk_idx
                                     if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler,
                                                         run.feature_cols, check_step_horizon=False)]
                    if len(valid_targets) == 0:
                        certified_infeasible_pidx.append(pidx)

            n_stage2_success, n_stage2_cvr = 0, 0
            for pidx in certified_infeasible_pidx:
                x0 = run.X_test[pidx]
                thr = bracket_threshold_for_age(ages_test[pidx], bracket_thresholds)
                res = stage2_search(run, x0, thr)
                res.update({"dataset": dataset, "seed": seed, "patient_idx": int(pidx), "target_threshold": thr})
                all_rows.append(res)
                if res["success"]:
                    n_stage2_success += 1
                    if res["constraint_violation"]:
                        n_stage2_cvr += 1

            n_high_risk = len(run.high_risk_test_idx)
            n_cert = len(certified_infeasible_pidx)
            print(f"[stage2] {dataset:12s} seed={seed}: n_high_risk={n_high_risk:4d} "
                  f"stage1_success={n_stage1_success:4d} certified_infeasible={n_cert:4d} "
                  f"stage2_success={n_stage2_success:4d}/{n_cert:4d} stage2_cvr={n_stage2_cvr:4d} "
                  f"combined={100.0*(n_stage1_success+n_stage2_success)/max(1,n_high_risk):5.1f}% "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["violation_flags"]).to_csv(TABLES_DIR / "stage2_per_patient.csv", index=False)

    summary_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            age_df = pd.read_csv(TABLES_DIR / "age_stratified_per_seed.csv")
            stage1 = age_df[(age_df.dataset == dataset) & (age_df.seed == seed) & (age_df.label == "age_stratified")]
            if len(stage1) == 0:
                continue
            n_high_risk = int(stage1.n_high_risk.iloc[0])
            n_stage1_success = int(round(stage1.success_pct.iloc[0] / 100.0 * n_high_risk))
            d = df[(df.dataset == dataset) & (df.seed == seed)]
            n_stage2_attempted, n_stage2_success = len(d), int(d.success.sum())
            n_stage2_cvr = int((d.success & d.constraint_violation).sum())
            combined_success = n_stage1_success + n_stage2_success
            summary_rows.append({
                "dataset": dataset, "seed": seed, "n_high_risk": n_high_risk,
                "stage1_success_pct": stage1.success_pct.iloc[0],
                "n_stage2_attempted": n_stage2_attempted, "n_stage2_success": n_stage2_success,
                "stage2_success_pct_of_attempted": 100.0 * n_stage2_success / max(1, n_stage2_attempted),
                "stage2_cvr_pct": 100.0 * n_stage2_cvr / max(1, n_stage2_success) if n_stage2_success else 0.0,
                "combined_success_pct": 100.0 * combined_success / max(1, n_high_risk),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "stage2_summary_per_seed.csv", index=False)

    print("\n[stage2] === SUMMARY (mean +/- std over seeds) ===")
    overall = {}
    for dataset in DATASETS:
        d = summary_df[summary_df.dataset == dataset]
        overall[dataset] = {
            "stage1_success_pct_mean": float(d.stage1_success_pct.mean()),
            "stage2_attempted_mean": float(d.n_stage2_attempted.mean()),
            "stage2_success_pct_of_attempted_mean": float(d.stage2_success_pct_of_attempted.mean()),
            "stage2_success_pct_of_attempted_std": float(d.stage2_success_pct_of_attempted.std()),
            "stage2_cvr_pct_mean": float(d.stage2_cvr_pct.mean()),
            "combined_success_pct_mean": float(d.combined_success_pct.mean()),
            "combined_success_pct_std": float(d.combined_success_pct.std()),
        }
        print(f"{dataset:12s} stage1={d.stage1_success_pct.mean():5.1f}%  "
              f"stage2_of_attempted={d.stage2_success_pct_of_attempted.mean():5.1f}%+/-{d.stage2_success_pct_of_attempted.std():4.1f}  "
              f"stage2_cvr={d.stage2_cvr_pct.mean():5.1f}%  combined={d.combined_success_pct.mean():5.1f}%+/-{d.combined_success_pct.std():4.1f}")

    with open(TABLES_DIR / "stage2_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print("\n[stage2] wrote results/tables/stage2_per_patient.csv, stage2_summary_per_seed.csv, stage2_summary.json")


if __name__ == "__main__":
    main()
