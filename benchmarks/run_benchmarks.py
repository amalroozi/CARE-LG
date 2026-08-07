"""
Baseline Benchmark Runner for CARE-LG research framework.
Evaluates DiCE, FACE, Growing Spheres, and CARE-LG across 100 high-risk test instances.
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from configs.dataset_config import FEATURE_METADATA, CLINICAL_EFFORT_WEIGHTS
from src.data.loader import get_dataloaders
from src.blackbox_model import train_blackbox_model, get_device
from src.vae.model import train_vae
from src.graph.clinical_constraints import check_clinical_violations, compute_clinical_effort
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path
from benchmarks.metrics import compute_cvr, compute_kde_density, compute_cost_and_latency


# =====================================================================
# 1. BASELINE IMPLEMENTATIONS
# =====================================================================

def run_care_lg(source_idx, calg_matrix, low_risk_mask, X_train):
    """CARE-LG: Riemannian Latent Graph path search."""
    start_time = time.time()
    try:
        path = find_recourse_path(calg_matrix, source_idx, low_risk_mask)
        recourse_x = X_train[path[-1]]
        latency = time.time() - start_time
        return True, recourse_x, latency
    except Exception:
        latency = time.time() - start_time
        return False, None, latency


def run_face(source_idx, X_train, low_risk_mask, k_neighbors=25):
    """FACE: Feature-space k-NN graph search (without asymmetric masking or Riemannian geometry)."""
    start_time = time.time()
    N = len(X_train)
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N)).fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)

    row_indices = []
    col_indices = []
    edge_weights = []

    for i in range(N):
        for idx, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx == i:
                continue
            w_ij = distances[i][idx]
            row_indices.append(i)
            col_indices.append(neighbor_idx)
            edge_weights.append(w_ij)

    face_matrix = sp.csr_matrix((edge_weights, (row_indices, col_indices)), shape=(N, N))

    try:
        dist_matrix, predecessors = csgraph.dijkstra(
            csgraph=face_matrix, directed=True, indices=source_idx, return_predecessors=True
        )
        target_indices = np.where(low_risk_mask)[0]
        target_dists = dist_matrix[target_indices]
        finite_mask = np.isfinite(target_dists)

        if not np.any(finite_mask):
            latency = time.time() - start_time
            return False, None, latency

        best_target = target_indices[np.argmin(target_dists)]
        recourse_x = X_train[best_target]
        latency = time.time() - start_time
        return True, recourse_x, latency
    except Exception:
        latency = time.time() - start_time
        return False, None, latency


def run_growing_spheres(source_x, classifier, device, target_threshold=0.35, max_iter=500):
    """Growing Spheres: Heuristic L1/L2 hyper-sphere perturbation."""
    start_time = time.time()
    classifier.eval()

    d = len(source_x)
    for radius in np.linspace(0.1, 3.0, 30):
        # Generate random sphere directions
        direction = np.random.randn(max_iter // 30, d)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        candidates = source_x + radius * direction

        # Evaluate risk
        with torch.no_grad():
            cand_tensor = torch.tensor(candidates, dtype=torch.float32).to(device)
            preds = classifier(cand_tensor).cpu().numpy().squeeze()

        low_risk_idx = np.where(preds < target_threshold)[0]
        if len(low_risk_idx) > 0:
            recourse_x = candidates[low_risk_idx[0]]
            latency = time.time() - start_time
            return True, recourse_x, latency

    latency = time.time() - start_time
    return False, None, latency


def run_dice(source_x, classifier, device, target_threshold=0.35, steps=100, lr=0.05):
    """DiCE: Gradient-based optimization search."""
    start_time = time.time()
    classifier.eval()

    x_param = torch.tensor(source_x, dtype=torch.float32, requires_grad=True, device=device)
    x_init = torch.tensor(source_x, dtype=torch.float32, device=device)

    optimizer = optim.Adam([x_param], lr=lr)

    best_x = None
    min_loss = float('inf')

    for step in range(steps):
        optimizer.zero_grad()
        pred = classifier(x_param.unsqueeze(0)).squeeze()

        # Loss = BCE target (target=0) + L2 distance penalty
        bce_loss = pred  # Minimize predicted probability of heart disease
        dist_penalty = torch.norm(x_param - x_init, p=2)
        loss = bce_loss + 0.1 * dist_penalty

        loss.backward()
        optimizer.step()

        if pred.item() < target_threshold:
            best_x = x_param.detach().cpu().numpy()
            break

    latency = time.time() - start_time
    if best_x is not None:
        return True, best_x, latency
    else:
        return False, x_param.detach().cpu().numpy(), latency


# =====================================================================
# 2. MAIN EVALUATION RUNNER
# =====================================================================

def run_all_benchmarks(num_query_instances=100, seed=42):
    print("==================================================")
    print(f"CARE-LG Benchmark Suite Execution ({num_query_instances} Test Queries)")
    print("==================================================")

    device = get_device()
    print(f"Using PyTorch device: {device}")

    # 1. Load Data & Train Models
    print("\n1. Training VAE and Classifier Models...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
    vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=15, device=device)

    # 2. Build CALG Graph on Training Dataset (N=1000 samples)
    print("\n2. Building CALG Graph on Training Data (N=1000)...")
    calg_matrix, Z_train, X_train, y_train = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=FEATURE_METADATA,
        effort_weights=CLINICAL_EFFORT_WEIGHTS,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=30,
        lambda_effort=1.0,
        max_samples=1000
    )

    # 3. Compute Risk Scores for Training Set
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        train_risk_scores = classifier(X_train_tensor).cpu().numpy().squeeze()

    low_risk_mask = train_risk_scores < 0.42
    print(f"   Train set count: {len(X_train)}, Low-risk targets: {np.sum(low_risk_mask)}")

    # 4. Prepare Test Queries (High-Risk Patients from Test Set)
    print("\n3. Preparing 100 High-Risk Test Query Instances...")
    all_test_x = []
    all_test_y = []
    for X_b, y_b in test_loader:
        all_test_x.append(X_b.numpy())
        all_test_y.append(y_b.numpy())

    X_test_all = np.vstack(all_test_x)

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_all, dtype=torch.float32).to(device)
        test_risk_scores = classifier(X_test_tensor).cpu().numpy().squeeze()

    high_risk_test_indices = np.where(test_risk_scores > 0.55)[0]

    # Pick up to num_query_instances high risk test queries
    query_indices = high_risk_test_indices[:num_query_instances]
    actual_num_queries = len(query_indices)
    print(f"   Selected {actual_num_queries} high-risk query instances for evaluation.")

    methods = ['DiCE', 'FACE', 'Growing Spheres', 'CARE-LG']
    results_data = {m: {'success': [], 'cvr': [], 'cost': [], 'latency': [], 'recourse_x': []} for m in methods}

    print("\n4. Running Evaluations Across Algorithms...")
    for idx_pos, test_idx in enumerate(tqdm(query_indices, desc="Evaluating Query Patients")):
        query_x = X_test_all[test_idx]

        # ------------------- A. DiCE -------------------
        succ, rec_x, lat = run_dice(query_x, classifier, device, target_threshold=0.35)
        if succ and rec_x is not None:
            cvr = compute_cvr(query_x, rec_x, FEATURE_METADATA, scaler, feature_cols)
            cost, _ = compute_cost_and_latency([(query_x, rec_x)], [lat], CLINICAL_EFFORT_WEIGHTS, FEATURE_METADATA, scaler, feature_cols)
            results_data['DiCE']['success'].append(1.0)
            results_data['DiCE']['cvr'].append(cvr)
            results_data['DiCE']['cost'].append(cost)
            results_data['DiCE']['latency'].append(lat)
            results_data['DiCE']['recourse_x'].append(rec_x)
        else:
            results_data['DiCE']['success'].append(0.0)

        # ------------------- B. FACE -------------------
        nearest_train_idx = np.argmin(np.linalg.norm(X_train - query_x, axis=1))
        succ, rec_x, lat = run_face(nearest_train_idx, X_train, low_risk_mask, k_neighbors=30)
        if succ and rec_x is not None:
            cvr = compute_cvr(query_x, rec_x, FEATURE_METADATA, scaler, feature_cols)
            cost, _ = compute_cost_and_latency([(query_x, rec_x)], [lat], CLINICAL_EFFORT_WEIGHTS, FEATURE_METADATA, scaler, feature_cols)
            results_data['FACE']['success'].append(1.0)
            results_data['FACE']['cvr'].append(cvr)
            results_data['FACE']['cost'].append(cost)
            results_data['FACE']['latency'].append(lat)
            results_data['FACE']['recourse_x'].append(rec_x)
        else:
            results_data['FACE']['success'].append(0.0)

        # ------------------- C. Growing Spheres -------------------
        succ, rec_x, lat = run_growing_spheres(query_x, classifier, device, target_threshold=0.35)
        if succ and rec_x is not None:
            cvr = compute_cvr(query_x, rec_x, FEATURE_METADATA, scaler, feature_cols)
            cost, _ = compute_cost_and_latency([(query_x, rec_x)], [lat], CLINICAL_EFFORT_WEIGHTS, FEATURE_METADATA, scaler, feature_cols)
            results_data['Growing Spheres']['success'].append(1.0)
            results_data['Growing Spheres']['cvr'].append(cvr)
            results_data['Growing Spheres']['cost'].append(cost)
            results_data['Growing Spheres']['latency'].append(lat)
            results_data['Growing Spheres']['recourse_x'].append(rec_x)
        else:
            results_data['Growing Spheres']['success'].append(0.0)

        # ------------------- D. CARE-LG -------------------
        # Find clinically valid entry nodes in X_train for query_x
        valid_entry_indices = [
            i for i in range(len(X_train))
            if not check_clinical_violations(query_x, X_train[i], FEATURE_METADATA, scaler, feature_cols)
        ]

        if len(valid_entry_indices) > 0:
            dists = np.linalg.norm(X_train[valid_entry_indices] - query_x, axis=1)
            # Sort entry nodes by distance to query_x
            sorted_entries = [valid_entry_indices[i] for i in np.argsort(dists)]
            care_succ = False
            care_rec_x = None
            care_lat = 0.0

            for entry_idx in sorted_entries[:10]:
                succ, rec_x, lat = run_care_lg(entry_idx, calg_matrix, low_risk_mask, X_train)
                care_lat += lat
                if succ and rec_x is not None:
                    care_succ = True
                    care_rec_x = rec_x
                    break

            if care_succ and care_rec_x is not None:
                cvr = compute_cvr(query_x, care_rec_x, FEATURE_METADATA, scaler, feature_cols)
                cost, _ = compute_cost_and_latency([(query_x, care_rec_x)], [care_lat], CLINICAL_EFFORT_WEIGHTS, FEATURE_METADATA, scaler, feature_cols)
                results_data['CARE-LG']['success'].append(1.0)
                results_data['CARE-LG']['cvr'].append(cvr)
                results_data['CARE-LG']['cost'].append(cost)
                results_data['CARE-LG']['latency'].append(care_lat)
                results_data['CARE-LG']['recourse_x'].append(care_rec_x)
            else:
                results_data['CARE-LG']['success'].append(0.0)
        else:
            results_data['CARE-LG']['success'].append(0.0)

    # 5. Compute Aggregate Benchmark Summary
    print("\n5. Computing Aggregate Performance Summary...")
    summary_rows = []

    for method in methods:
        data = results_data[method]
        success_rate = np.mean(data['success']) * 100.0
        cvr_pct = np.mean(data['cvr']) * 100.0 if len(data['cvr']) > 0 else 0.0
        mean_cost = np.mean(data['cost']) if len(data['cost']) > 0 else 0.0
        mean_lat = np.mean(data['latency']) if len(data['latency']) > 0 else 0.0

        # Compute KDE density for successful counterfactual points
        if len(data['recourse_x']) > 0:
            rec_matrix = np.vstack(data['recourse_x'])
            kde_density = compute_kde_density(rec_matrix, X_train, bandwidth=0.5)
        else:
            kde_density = -999.0

        row = {
            'Method': method,
            'Success_Rate_Pct': round(float(success_rate), 2),
            'CVR_Pct': round(float(cvr_pct), 2),
            'KDE_Density': round(float(kde_density), 4),
            'Clinical_Cost': round(float(mean_cost), 4),
            'Latency_Sec': round(float(mean_lat), 4)
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\nBenchmark Summary Table:")
    print(df_summary.to_string(index=False))

    # 6. Save Results to JSON & CSV
    output_dir = PROJECT_ROOT / "benchmarks"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    df_summary.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved benchmark results to {csv_path} and {json_path}")
    print("==================================================")
    print("SUCCESS: Benchmark Execution Completed Cleanly!")
    print("==================================================")


if __name__ == "__main__":
    run_all_benchmarks(num_query_instances=100, seed=42)
