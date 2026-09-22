"""
v10: the full CARE-LG architecture (type-aware VAE, Riemannian pullback-
metric graph search, B1/B2-fixed hard clinical constraint gate, age-
stratified target definition, certified-infeasibility check, Stage 2
Riemannian natural-gradient synthetic fallback) run top-to-bottom on a
THIRD, much larger real dataset: the Kaggle Cardiovascular Disease dataset
(experiments_v10/data_cardio.py), instead of UCI Heart or NHANES.

Separate, self-contained experiment tree (experiments_v10/), src/ and
scripts/ untouched. Stage 1's target definition and Stage 2's algorithm are
reused UNMODIFIED by importing the exact functions the main pipeline uses
(scripts.run_age_stratified_experiment.age_stratified_low_risk_mask,
scripts.run_stage2_synthetic_search.stage2_search / bracket_threshold_for_age)
against a locally-built DatasetRunV10 object -- those functions are written
against duck-typed attribute access (run.vae, run.X_train, run.feature_metadata,
...), not a UCI/NHANES-specific class, so no copy-pasting of the search logic
itself was needed.

Stage 1's PATH SEARCH uses find_path_sra (src.graph.source_relative_admissibility),
not scripts.run_age_stratified_experiment.try_search's find_path_augmented --
the first cardio run used the latter and came back with a nonzero real-value
CVR (~1.5%), which was the exact compositional constraint-drift leak
find_path_sra exists to close (see src/graph/source_relative_admissibility.py's
module docstring), now confirmed on a THIRD independent dataset, not just
UCI/no-age/NHANES. try_search_sra() below is the local, cardio-specific
equivalent of run_grid_sra.py's evaluate_patient_sra(), calling find_path_sra
instead of find_path_augmented -- same drop-in-replacement pattern used
everywhere else in this project's SRA-fixed reruns.

Scale note (disclosed, not hidden): the raw Kaggle file has 70,000 rows;
after dropping physiologically implausible rows (experiments_v10/
data_cardio.py) ~68,590 remain. Each seed stratified-subsamples
N_SUBSAMPLE of those before the usual 80/20 train/test split -- a
computational-tractability scoping decision (graph edges scale as
N_train * k_neighbors, each edge requiring a VAE-Jacobian Riemannian
distance), the same category of decision as this project's already-
disclosed FACE/NHANES subsampling. N_SUBSAMPLE=20000 is still ~4.4x
NHANES's full 4,540 rows and ~67x UCI's 298.

Usage:
  .venv/bin/python -m experiments_v10.run_v10_cardio
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from experiments_v10.data_cardio import get_dataloaders_cardio, FEATURE_METADATA, CLINICAL_EFFORT_WEIGHTS
from src.classifier.blackbox_model import RiskClassifier, train_blackbox_model
from src.vae.type_aware import TabularVAETypeAware, train_vae_type_aware
from src.seeding import set_all_seeds
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, unscale_matrix
from src.graph.source_relative_admissibility import find_path_sra
from src.graph.constraints_ext import check_decoded_violations, violates as violates_fn
from scripts.run_age_stratified_experiment import age_stratified_low_risk_mask
from scripts.run_stage2_synthetic_search import stage2_search, bracket_threshold_for_age

RESULTS_DIR = REPO_ROOT / "experiments_v10" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET = "cardio"
SEEDS = [0, 1, 2, 3, 4]
METRIC = "riemannian"
N_SUBSAMPLE = 20000
K_NEIGHBORS = 100
HIGH_RISK_THRESHOLD = 0.55
LOW_RISK_THRESHOLD = 0.45


@dataclass
class DatasetRunV10:
    dataset: str
    seed: int
    feature_metadata: dict
    effort_weights: dict
    scaler: object
    feature_cols: List[str]
    vae: TabularVAETypeAware
    classifier: RiskClassifier
    Z_train: np.ndarray
    X_train: np.ndarray
    train_risk: np.ndarray
    low_risk_mask: np.ndarray
    X_test: np.ndarray
    test_risk: np.ndarray
    high_risk_test_idx: np.ndarray
    k_neighbors: int
    device: torch.device


def run_dataset_seed_v10(seed: int) -> DatasetRunV10:
    set_all_seeds(seed)
    device = torch.device("cpu")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders_cardio(seed=seed, n_subsample=N_SUBSAMPLE)
    input_dim = len(feature_cols)

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
    vae = train_vae_type_aware(train_loader, input_dim=input_dim, feature_cols=feature_cols,
                                feature_metadata=FEATURE_METADATA, latent_dim=4, epochs=25,
                                age_loss_weight=8.0, device=device)
    vae.eval(); classifier.eval()

    all_z, all_x = [], []
    with torch.no_grad():
        for Xb, _ in train_loader:
            mu, _ = vae.encode(Xb.to(device))
            all_z.append(mu.cpu().numpy()); all_x.append(Xb.numpy())
    Z_train, X_train = np.vstack(all_z), np.vstack(all_x)
    with torch.no_grad():
        train_risk = classifier(torch.tensor(X_train, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    low_risk_mask = train_risk < LOW_RISK_THRESHOLD

    all_xt = [Xb.numpy() for Xb, _ in test_loader]
    X_test = np.vstack(all_xt)
    with torch.no_grad():
        test_risk = classifier(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    high_risk_test_idx = np.where(test_risk > HIGH_RISK_THRESHOLD)[0]

    return DatasetRunV10(dataset=DATASET, seed=seed, feature_metadata=FEATURE_METADATA,
                          effort_weights=CLINICAL_EFFORT_WEIGHTS, scaler=scaler, feature_cols=feature_cols,
                          vae=vae, classifier=classifier, Z_train=Z_train, X_train=X_train, train_risk=train_risk,
                          low_risk_mask=low_risk_mask, X_test=X_test, test_risk=test_risk,
                          high_risk_test_idx=high_risk_test_idx, k_neighbors=K_NEIGHBORS, device=device)


def try_search_sra(run, edge_table, base_csr, pidx, low_risk_mask):
    """SRA-fixed counterpart to scripts.run_age_stratified_experiment.try_search
    -- identical except the final path search is find_path_sra (restricts the
    whole search graph to nodes individually admissible from the query's own
    real baseline) instead of find_path_augmented, closing the compositional
    constraint-drift leak (see module docstring)."""
    x0 = run.X_test[pidx]
    with torch.no_grad():
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
    N = edge_table.N
    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)
    qrow = make_query_row(qedges, METRIC, "hard", N)
    target_mask = np.concatenate([low_risk_mask, [False]])
    try:
        path, cost = find_path_sra(base_csr, qedges, qrow, N, target_mask)
    except ValueError:
        return None
    real_points = [x0 if n == N else run.X_train[n] for n in path]
    any_hop = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        any_hop = any_hop or v
    endpoint_v, _ = check_decoded_violations(real_points[0], real_points[-1], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)
    return {"success": True, "real_cvr": bool(any_hop or endpoint_v)}


def run_one(seed: int):
    t0 = time.perf_counter()
    run = run_dataset_seed_v10(seed)
    low_risk_mask, bracket_thresholds = age_stratified_low_risk_mask(run)
    low_risk_idx = np.where(low_risk_mask)[0]

    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard = make_cell_matrix(edge_table, METRIC, "hard")
    ages_test = unscale_matrix(run.X_test, run.scaler, run.feature_cols, run.feature_metadata)["age"]

    n_stage1_success, n_stage1_cvr, certified_infeasible_pidx = 0, 0, []
    for pidx in run.high_risk_test_idx:
        r = try_search_sra(run, edge_table, base_hard, pidx, low_risk_mask)
        if r is not None:
            n_stage1_success += 1
            if r["real_cvr"]:
                n_stage1_cvr += 1
        else:
            x0 = run.X_test[pidx]
            valid_targets = [j for j in low_risk_idx
                             if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler,
                                                 run.feature_cols, check_step_horizon=False)]
            if len(valid_targets) == 0:
                certified_infeasible_pidx.append(pidx)

    stage2_rows = []
    n_stage2_success, n_stage2_cvr = 0, 0
    for pidx in certified_infeasible_pidx:
        x0 = run.X_test[pidx]
        thr = bracket_threshold_for_age(ages_test[pidx], bracket_thresholds)
        res = stage2_search(run, x0, thr)
        res.update({"dataset": DATASET, "seed": seed, "patient_idx": int(pidx), "target_threshold": thr})
        stage2_rows.append(res)
        if res["success"]:
            n_stage2_success += 1
            if res["constraint_violation"]:
                n_stage2_cvr += 1

    n_high_risk = len(run.high_risk_test_idx)
    n_cert = len(certified_infeasible_pidx)
    summary = {
        "dataset": DATASET, "seed": seed, "n_high_risk": n_high_risk,
        "n_low_risk_train": int(low_risk_mask.sum()),
        "n_stage1_success": n_stage1_success,
        "stage1_success_pct": 100.0 * n_stage1_success / max(1, n_high_risk),
        "stage1_real_cvr_pct": 100.0 * n_stage1_cvr / max(1, n_stage1_success) if n_stage1_success else 0.0,
        "n_certified_infeasible": n_cert,
        "n_stage2_attempted": n_cert, "n_stage2_success": n_stage2_success,
        "stage2_success_pct_of_attempted": 100.0 * n_stage2_success / max(1, n_cert),
        "stage2_cvr_pct": 100.0 * n_stage2_cvr / max(1, n_stage2_success) if n_stage2_success else 0.0,
        "combined_success_pct": 100.0 * (n_stage1_success + n_stage2_success) / max(1, n_high_risk),
    }
    print(f"[v10-cardio] seed={seed}: n_high_risk={n_high_risk:4d} n_low_risk_train={summary['n_low_risk_train']:5d} "
          f"stage1={summary['stage1_success_pct']:5.1f}% stage1_cvr={summary['stage1_real_cvr_pct']:4.1f}% "
          f"certified_infeasible={n_cert:4d} stage2={summary['stage2_success_pct_of_attempted']:5.1f}%_of_attempted "
          f"combined={summary['combined_success_pct']:5.1f}% ({time.perf_counter()-t0:.1f}s)", flush=True)
    return summary, stage2_rows


def main():
    all_summaries, all_stage2_rows = [], []
    for seed in SEEDS:
        summary, stage2_rows = run_one(seed)
        all_summaries.append(summary)
        all_stage2_rows.extend(stage2_rows)

    per_seed_df = pd.DataFrame(all_summaries)
    per_seed_df.to_csv(RESULTS_DIR / "v10_summary_per_seed.csv", index=False)

    stage2_df = pd.DataFrame(all_stage2_rows)
    if len(stage2_df):
        stage2_df.drop(columns=["violation_flags"]).to_csv(RESULTS_DIR / "v10_stage2_per_patient.csv", index=False)

    overall = {
        "n_subsample": N_SUBSAMPLE, "k_neighbors": K_NEIGHBORS,
        "stage1_success_pct_mean": float(per_seed_df.stage1_success_pct.mean()),
        "stage1_success_pct_std": float(per_seed_df.stage1_success_pct.std()),
        "stage1_real_cvr_pct_mean": float(per_seed_df.stage1_real_cvr_pct.mean()),
        "n_high_risk_mean": float(per_seed_df.n_high_risk.mean()),
        "n_low_risk_train_mean": float(per_seed_df.n_low_risk_train.mean()),
        "n_certified_infeasible_mean": float(per_seed_df.n_certified_infeasible.mean()),
        "stage2_attempted_mean": float(per_seed_df.n_stage2_attempted.mean()),
        "stage2_success_pct_of_attempted_mean": float(per_seed_df.stage2_success_pct_of_attempted.mean()),
        "stage2_success_pct_of_attempted_std": float(per_seed_df.stage2_success_pct_of_attempted.std()),
        "stage2_cvr_pct_mean": float(per_seed_df.stage2_cvr_pct.mean()),
        "combined_success_pct_mean": float(per_seed_df.combined_success_pct.mean()),
        "combined_success_pct_std": float(per_seed_df.combined_success_pct.std()),
    }
    with open(RESULTS_DIR / "v10_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\n[v10-cardio] === SUMMARY (mean +/- std over seeds) ===")
    print(f"stage1={overall['stage1_success_pct_mean']:.1f}%+/-{overall['stage1_success_pct_std']:.1f}  "
          f"stage1_real_cvr={overall['stage1_real_cvr_pct_mean']:.2f}%  "
          f"combined={overall['combined_success_pct_mean']:.1f}%+/-{overall['combined_success_pct_std']:.1f}")
    print(f"\n[v10-cardio] wrote experiments_v10/results/v10_summary_per_seed.csv, "
          f"v10_stage2_per_patient.csv, v10_summary.json")


if __name__ == "__main__":
    main()
