"""
v8: abandons age entirely, not just as a graph constraint but as a MODEL
INPUT. Age is dropped from the feature vector before the classifier and VAE
ever see it -- retrained from scratch, both datasets, all 5 seeds. This is
a separate, self-contained experiment tree (experiments_v8/), matching the
project's existing convention (experiments_v2..v6/) of isolating a
structural variant rather than editing src/ in place. src/ is completely
untouched by this file.

Why this is a different, bigger change than "freeze age" (which Stage 2
already does): freezing age keeps it as a classifier input (so risk still
depends on it) but forbids the *search* from changing it. Dropping it here
means the classifier's own predicted risk no longer depends on age at all
-- a structurally different model, not just a different search constraint.

Everything else is unchanged: same classifier/VAE architectures and
hyperparameters (src.classifier.blackbox_model, src.vae.type_aware), same
Riemannian pullback-metric graph search, same B1/B2-fixed hard constraint
gate (src.graph.query_attachment / clinical_constraints), same certified-
infeasibility protocol (exhaustive + 2x-k + synthetic-densification probes)
as every other result in this project.

Usage:
  .venv/bin/python -m experiments_v8.run_v8_no_age
"""
import copy
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.data.dataset_registration  # noqa: F401
from configs.dataset_config import get_dataset_config, DATASET_CONFIGS
from src.data.loader import EHRDataset, load_dataset
from src.classifier.blackbox_model import RiskClassifier, train_blackbox_model
from src.vae.type_aware import TabularVAETypeAware, train_vae_type_aware
from src.seeding import set_all_seeds
from src.graph.query_attachment import (
    build_edge_table, build_query_edges, make_cell_matrix, make_query_row,
    augmented_matrix, unscale_matrix, vectorized_violation_components,
)
from src.recourse.search_augmented import find_path_augmented
from src.graph.constraints_ext import check_decoded_violations, violates as violates_fn
from torch.utils.data import DataLoader

RESULTS_DIR = REPO_ROOT / "experiments_v8" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ["uci", "nhanes_real"]
SEEDS = [0, 1, 2, 3, 4]
METRIC = "riemannian"
K_NEIGHBORS = {"uci": 40, "nhanes_real": 80}
HIGH_RISK_THRESHOLD = 0.55
LOW_RISK_THRESHOLD = 0.45
M_SYNTHETIC = 500


def no_age_feature_metadata(dataset: str) -> dict:
    """Deep copy of this dataset's FEATURE_METADATA with 'age' removed from
    every category (it will also be dropped from the dataframe entirely, so
    it must not remain classified anywhere -- validate_exhaustive_column_classification
    would otherwise fail-safe on an unclassified column, correctly)."""
    base_key = "nhanes" if dataset == "nhanes_real" else dataset
    fm = copy.deepcopy(DATASET_CONFIGS[base_key]["FEATURE_METADATA"])
    fm["non_decreasing"] = [c for c in fm.get("non_decreasing", []) if c != "age"]
    return fm


def get_dataloaders_no_age(dataset: str, seed: int):
    feature_metadata, _, dataset_path = get_dataset_config(dataset)
    fm = no_age_feature_metadata(dataset)
    df = load_dataset(filepath=dataset_path, seed=seed)
    target_col = fm["target"]
    df = df.drop(columns=["age"])
    feature_cols = [c for c in df.columns if c != target_col]

    continuous_cols = fm["continuous_mutable"] + fm["non_decreasing"]
    X = df[feature_cols].copy()
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    train_loader = DataLoader(EHRDataset(X_train_scaled.values, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(EHRDataset(X_test_scaled.values, y_test), batch_size=64, shuffle=False)
    return train_loader, test_loader, scaler, feature_cols, fm


@dataclass
class DatasetRunV8:
    dataset: str
    seed: int
    feature_metadata: dict
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


def run_dataset_seed_v8(dataset: str, seed: int) -> DatasetRunV8:
    set_all_seeds(seed)
    train_loader, test_loader, scaler, feature_cols, fm = get_dataloaders_no_age(dataset, seed)
    input_dim = len(feature_cols)
    device = torch.device("cpu")

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
    vae = train_vae_type_aware(train_loader, input_dim=input_dim, feature_cols=feature_cols,
                                feature_metadata=fm, latent_dim=4, epochs=25, age_loss_weight=8.0, device=device)
    vae.eval(); classifier.eval()

    all_z, all_x = [], []
    with torch.no_grad():
        for Xb, _ in train_loader:
            mu, _ = vae.encode(Xb.to(device))
            all_z.append(mu.cpu().numpy()); all_x.append(Xb.numpy())
    Z_train, X_train = np.vstack(all_z), np.vstack(all_x)
    with torch.no_grad():
        train_risk = classifier(torch.tensor(X_train, dtype=torch.float32)).numpy().squeeze()
    low_risk_mask = train_risk < LOW_RISK_THRESHOLD

    all_xt = [Xb.numpy() for Xb, _ in test_loader]
    X_test = np.vstack(all_xt)
    with torch.no_grad():
        test_risk = classifier(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()
    high_risk_test_idx = np.where(test_risk > HIGH_RISK_THRESHOLD)[0]

    return DatasetRunV8(dataset=dataset, seed=seed, feature_metadata=fm, scaler=scaler, feature_cols=feature_cols,
                         vae=vae, classifier=classifier, Z_train=Z_train, X_train=X_train, train_risk=train_risk,
                         low_risk_mask=low_risk_mask, X_test=X_test, test_risk=test_risk,
                         high_risk_test_idx=high_risk_test_idx, k_neighbors=K_NEIGHBORS[dataset])


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


def certified_check(run, low_risk_mask, abstained_pidx):
    if not abstained_pidx:
        return []
    low_risk_idxs = np.where(low_risk_mask)[0]
    rows = []
    for pidx in abstained_pidx:
        x0 = run.X_test[pidx]
        valid_targets = [j for j in low_risk_idxs
                         if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)]
        rows.append({"certified_infeasible": len(valid_targets) == 0})
    return rows


def run_one(dataset, seed):
    t0 = time.perf_counter()
    run = run_dataset_seed_v8(dataset, seed)
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard = make_cell_matrix(edge_table, METRIC, "hard")

    n_success, n_cvr, abstained = 0, 0, []
    for pidx in run.high_risk_test_idx:
        r = try_search(run, edge_table, base_hard, pidx, run.low_risk_mask)
        if r is None:
            abstained.append(pidx)
        else:
            n_success += 1
            if r["real_cvr"]:
                n_cvr += 1

    cert_rows = certified_check(run, run.low_risk_mask, abstained)
    n = len(run.high_risk_test_idx)
    n_cert = sum(1 for c in cert_rows if c["certified_infeasible"])
    result = {
        "dataset": dataset, "seed": seed, "n_high_risk": n,
        "n_low_risk_train": int(run.low_risk_mask.sum()),
        "success_pct": 100.0 * n_success / max(1, n),
        "real_cvr_pct": 100.0 * n_cvr / max(1, n_success) if n_success else float("nan"),
        "n_abstain": len(abstained),
        "certified_infeasible_pct": 100.0 * n_cert / max(1, len(cert_rows)) if cert_rows else float("nan"),
    }
    print(f"[v8-no-age] {dataset:12s} seed={seed}: n_high={n:4d} n_low_risk_train={result['n_low_risk_train']:4d} "
          f"success={result['success_pct']:5.1f}% real_cvr={result['real_cvr_pct']:5.1f}% "
          f"certified_infeasible={result['certified_infeasible_pct']:5.1f}% ({time.perf_counter()-t0:.1f}s)", flush=True)
    return result


def main():
    rows = [run_one(dataset, seed) for dataset in DATASETS for seed in SEEDS]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v8_no_age_per_seed.csv", index=False)

    print("\n[v8-no-age] === SUMMARY (mean +/- std over seeds) ===")
    summary = {}
    for dataset in DATASETS:
        d = df[df.dataset == dataset]
        summary[dataset] = {
            "success_pct_mean": float(d.success_pct.mean()), "success_pct_std": float(d.success_pct.std()),
            "real_cvr_pct_mean": float(d.real_cvr_pct.mean()),
            "certified_infeasible_pct_mean": float(d.certified_infeasible_pct.mean()),
            "n_low_risk_train_mean": float(d.n_low_risk_train.mean()),
        }
        print(f"{dataset:12s} success={d.success_pct.mean():5.1f}%+/-{d.success_pct.std():4.1f}  "
              f"real_cvr={d.real_cvr_pct.mean():5.1f}%  certified_infeasible={d.certified_infeasible_pct.mean():5.1f}%")

    with open(RESULTS_DIR / "v8_no_age_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[v8-no-age] wrote experiments_v8/results/v8_no_age_per_seed.csv and _summary.json")


if __name__ == "__main__":
    main()
