"""
Day 2 verification script for CARE-LG research framework.
Tests Riemannian metric tensor, clinical constraints masking, CALG graph construction, Dijkstra pathfinding, and recourse trajectory decoding.
Supports multi-dataset selection via --dataset uci / nhanes.
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from configs.dataset_config import get_dataset_config
from src.data.loader import get_dataloaders
from src.blackbox_model import train_blackbox_model, get_device
from src.vae.model import train_vae
from src.graph.riemannian import compute_decoder_jacobian, compute_metric_tensor, riemannian_distance
from src.graph.clinical_constraints import check_hard_violations, check_clinical_violations, compute_clinical_effort, unscale_features
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path, decode_recourse_trajectory, format_clinical_explanation_report


def test_day2(dataset_name="uci"):
    print("==================================================")
    print(f"CARE-LG Day 2 Graph Engine Verification ({dataset_name.upper()} Dataset)")
    print("==================================================")

    feature_metadata, effort_weights, dataset_path = get_dataset_config(dataset_name)

    device = get_device()
    print(f"Using PyTorch device: {device}")

    # 1. Load Data & Train Core Models
    print(f"\n1. Loading {dataset_name.upper()} Data & Training VAE + Classifier Models...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=dataset_name, batch_size=64, seed=42)
    input_dim = len(feature_cols)

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=15, device=device)
    vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=10, device=device)

    # 2. Test Riemannian Metric Engine
    print("\n2. Testing Riemannian Metric Engine...")
    sample_z1 = torch.randn(4)
    sample_z2 = torch.randn(4)

    J = compute_decoder_jacobian(vae, sample_z1)
    G = compute_metric_tensor(J)
    r_dist = riemannian_distance(sample_z1, sample_z2, vae)

    print(f"   Jacobian shape: {J.shape} (expected: ({input_dim}, 4))")
    print(f"   Metric Tensor G shape: {G.shape} (expected: (4, 4))")
    print(f"   Riemannian Distance d_R(z1, z2): {r_dist:.4f}")

    assert J.shape == (input_dim, 4)
    assert G.shape == (4, 4)
    assert r_dist >= 0.0

    # 3. Test Clinical Constraints & Asymmetric Masking
    print("\n3. Testing Clinical Constraint Masking...")
    X_sample, _ = next(iter(train_loader))
    x_src = X_sample[0].numpy()

    # Create a target with age reduction (violation)
    x_violate_age = x_src.copy()
    src_unscaled = unscale_features(x_src, scaler, feature_cols, feature_metadata)
    age_idx = feature_cols.index('age')
    cont_cols = feature_metadata['continuous_mutable'] + feature_metadata['non_decreasing']
    age_cont_idx = cont_cols.index('age')

    violating_age_unscaled = src_unscaled['age'] - 5.0
    cont_vals = np.array([src_unscaled[c] for c in cont_cols]).reshape(1, -1)
    cont_vals[0, age_cont_idx] = violating_age_unscaled
    scaled_cont = scaler.transform(cont_vals).squeeze()
    for i, c in enumerate(cont_cols):
        x_violate_age[feature_cols.index(c)] = scaled_cont[i]

    has_age_violation = check_clinical_violations(x_src, x_violate_age, feature_metadata, scaler, feature_cols)
    print(f"   Age reduction violation detected: {has_age_violation} (expected: True)")
    assert has_age_violation is True

    # Create a target with sex change (immutable violation)
    x_violate_sex = x_src.copy()
    sex_idx = feature_cols.index('sex')
    x_violate_sex[sex_idx] = 1.0 - x_src[sex_idx]
    has_sex_violation = check_clinical_violations(x_src, x_violate_sex, feature_metadata, scaler, feature_cols)
    print(f"   Immutable attribute violation detected: {has_sex_violation} (expected: True)")
    assert has_sex_violation is True

    # 4. Build CALG Graph
    print("\n4. Building CALG Graph for N=500 samples...")
    calg_matrix, Z, X, y = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=feature_metadata,
        effort_weights=effort_weights,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=60,
        lambda_effort=1.0,
        max_samples=500
    )
    print(f"   CALG Sparse Matrix Shape: {calg_matrix.shape}, Non-zero edges: {calg_matrix.nnz}")
    assert calg_matrix.shape == (len(Z), len(Z))
    assert calg_matrix.nnz > 0

    # 5. Predict Model Risk Scores for CALG Nodes
    print("\n5. Computing Model Risk Predictions across CALG Nodes...")
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        risk_scores = classifier(X_tensor).cpu().numpy().squeeze()

    print(f"   Risk Scores - Min: {risk_scores.min():.4f}, Max: {risk_scores.max():.4f}, Mean: {risk_scores.mean():.4f}")

    high_risk_candidates = np.where(risk_scores > 0.55)[0]
    low_risk_mask = risk_scores < 0.45

    assert len(high_risk_candidates) > 0, "No high risk candidates found."
    assert np.sum(low_risk_mask) > 0, "No low risk target nodes found."
    print(f"   Count of High-Risk Candidates: {len(high_risk_candidates)}")
    print(f"   Count of Low-Risk Target Nodes: {np.sum(low_risk_mask)}")

    # 6. Run Constrained Dijkstra Search & Decode Recourse Path
    print("\n6. Running Dijkstra Pathfinding over CALG...")
    path = None
    source_idx = None
    for cand_idx in high_risk_candidates:
        try:
            path = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
            source_idx = cand_idx
            break
        except ValueError:
            continue

    assert path is not None, "Could not find a feasible recourse path for any high-risk candidate."
    target_idx = path[-1]
    print(f"   Selected High-Risk Source Patient Index: {source_idx} (Initial Risk: {risk_scores[source_idx]:.4f})")
    print(f"   Feasible Recourse Path Found! Path length: {len(path)} steps: {path}")
    print(f"   Target Node Index: {target_idx} (Final Risk: {risk_scores[target_idx]:.4f})")

    # Decode recourse trajectory
    trajectory_df = decode_recourse_trajectory(vae, Z, path, X, scaler, feature_cols, feature_metadata)
    print("\n   Decoded Step-by-Step Recourse Trajectory Summary:")
    print(trajectory_df.head(len(path)))

    # 7. Format & Print Plain-English Clinical Explanation Report
    print("\n7. Generating Plain-English Clinical Recourse Report...")
    report_str = format_clinical_explanation_report(trajectory_df, risk_scores, feature_metadata)
    print("\n" + report_str)

    # 8. Verify Zero Hard Violations Along Recourse Path
    print("\n8. Verifying Constraint Integrity Along Recourse Path...")
    for k in range(len(path) - 1):
        idx_curr = path[k]
        idx_next = path[k + 1]
        x_curr = X[idx_curr]
        x_next = X[idx_next]

        violation = check_hard_violations(x_curr, x_next, feature_metadata, scaler, feature_cols)
        print(f"   Transition step {k} -> {k+1} (Node {idx_curr} -> {idx_next}): Hard Violation = {violation}")
        assert violation is False, f"Clinical violation found on recourse transition step {k} -> {k+1}!"

    print("\n==================================================")
    print(f"SUCCESS: Day 2 Graph Engine & Recourse Verified for {dataset_name.upper()} with 0 Errors!")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE-LG Day 2 Graph Verification")
    parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "nhanes"], help="Dataset to evaluate (uci or nhanes)")
    args = parser.parse_args()

    test_day2(dataset_name=args.dataset)
