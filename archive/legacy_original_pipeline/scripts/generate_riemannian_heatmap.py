"""
Riemannian Latent Geometry Heatmap Generator for CARE-LG research framework.
Generates 300 DPI publication figure (fig3_{dataset}_riemannian_heatmap.png) showing:
- 50x50 metric pullback log-determinant volume distortion log(det(G(z))).
- Patient risk scatter points.
- Smooth CARE-LG geodesic recourse path overlay.
Supports multi-dataset selection via --dataset uci / nhanes.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm

from configs.dataset_config import get_dataset_config
from src.data.loader import get_dataloaders
from src.blackbox_model import RiskClassifier, train_blackbox_model, get_device
from src.vae.model import TabularVAE, train_vae
from src.graph.riemannian import compute_decoder_jacobian, compute_metric_tensor
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path


def generate_riemannian_heatmap(dataset_name="uci", seed=42):
    print("==================================================")
    print(f"CARE-LG Riemannian Latent Geometry Heatmap Generator ({dataset_name.upper()} Dataset)")
    print("==================================================")

    device = get_device()
    print(f"Using PyTorch device: {device}")

    feature_metadata, effort_weights, dataset_path = get_dataset_config(dataset_name)

    # 1. Load Data & Models
    print(f"\n1. Loading {dataset_name.upper()} Dataset & DataLoaders...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=dataset_name, batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    models_dir = PROJECT_ROOT / "models" / dataset_name
    models_dir.mkdir(parents=True, exist_ok=True)
    vae_path = models_dir / "vae_model.pt"
    clf_path = models_dir / "classifier_model.pt"

    vae = TabularVAE(input_dim=input_dim, latent_dim=4).to(device)
    classifier = RiskClassifier(input_dim=input_dim).to(device)

    if vae_path.exists():
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=device)
        torch.save(vae.state_dict(), vae_path)

    if clf_path.exists():
        classifier.load_state_dict(torch.load(clf_path, map_location=device))
        classifier.eval()
    else:
        classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
        torch.save(classifier.state_dict(), clf_path)

    # 2. Encode Dataset to Latent Space & Fit PCA
    print("\n2. Projecting Dataset into 4D Latent Space...")
    Z_list = []
    X_list = []

    vae.eval()
    with torch.no_grad():
        for x_b, _ in train_loader:
            x_b = x_b.to(device)
            mu, _ = vae.encode(x_b)
            Z_list.append(mu.cpu().numpy())
            X_list.append(x_b.cpu().numpy())

    Z = np.vstack(Z_list)
    X = np.vstack(X_list)

    pca = PCA(n_components=2, random_state=seed)
    Z_pca = pca.fit_transform(Z)

    # 3. Create 50x50 Meshgrid over PCA Domain
    print("\n3. Generating 50x50 Meshgrid over PCA Domain...")
    grid_res = 50
    x_min, x_max = Z_pca[:, 0].min() - 0.5, Z_pca[:, 0].max() + 0.5
    y_min, y_max = Z_pca[:, 1].min() - 0.5, Z_pca[:, 1].max() + 0.5

    z1_grid = np.linspace(x_min, x_max, grid_res)
    z2_grid = np.linspace(y_min, y_max, grid_res)
    Z1, Z2 = np.meshgrid(z1_grid, z2_grid)

    grid_pca_2d = np.c_[Z1.ravel(), Z2.ravel()]
    grid_latent_4d = pca.inverse_transform(grid_pca_2d)

    # 4. Compute Pullback Metric Tensor G(z) & Log-Determinants over Grid
    print("\n4. Computing Riemannian Pullback Metric Tensors G(z) & Log-Determinants...")
    log_det_grid = np.zeros(len(grid_latent_4d))

    for idx in tqdm(range(len(grid_latent_4d)), desc="Riemannian Metric Grid"):
        z_pt = torch.tensor(grid_latent_4d[idx], dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(vae, z_pt)
        G = compute_metric_tensor(J)

        # Compute log(det(G(z)))
        det_G = torch.det(G).item()
        log_det_grid[idx] = np.log(max(det_G, 1e-8))

    LOG_DET_2D = log_det_grid.reshape(grid_res, grid_res)

    # 5. Compute Recourse Trajectory Overlay
    print("\n5. Computing CARE-LG Recourse Trajectory for Overlay...")
    calg_matrix, Z_calg, X_calg, y_calg = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=feature_metadata,
        effort_weights=effort_weights,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=60,
        lambda_effort=1.0
    )

    with torch.no_grad():
        X_tensor = torch.tensor(X_calg, dtype=torch.float32).to(device)
        risk_scores = classifier(X_tensor).cpu().numpy().squeeze()

    low_risk_mask = risk_scores < np.percentile(risk_scores, 45)
    high_risk_candidates = np.argsort(risk_scores)[::-1]

    path = None
    source_idx = None
    for cand_idx in high_risk_candidates:
        if risk_scores[cand_idx] <= 0.50:
            break
        try:
            p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
            if p is not None and len(p) >= 2:
                path = p
                source_idx = cand_idx
                break
            elif p is not None and path is None:
                path = p
                source_idx = cand_idx
        except Exception:
            continue

    if path is None:
        low_risk_mask = risk_scores < np.median(risk_scores)
        for cand_idx in high_risk_candidates:
            try:
                p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
                if p is not None:
                    path = p
                    source_idx = cand_idx
                    break
            except Exception:
                continue

    assert path is not None, "Could not find a valid recourse trajectory for visualization."

    path_Z = Z_calg[path]
    path_Z_pca = pca.transform(path_Z)

    # 6. Render Publication Figure
    benchmarks_dir = PROJECT_ROOT / "benchmarks" / dataset_name
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    fig_path = benchmarks_dir / f"fig3_{dataset_name}_riemannian_heatmap.png"

    print(f"\n6. Rendering Publication Figure ({fig_path.name})...")
    sns.set_theme(style="white", font="sans-serif")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Contour heatmap of log(det(G(z)))
    contour = ax.contourf(
        Z1, Z2, LOG_DET_2D,
        levels=20,
        cmap="magma",
        alpha=0.85
    )

    cbar = fig.colorbar(contour, ax=ax, shrink=0.88, aspect=20)
    cbar.set_label(r"Geometric Magnification $\log(\det(G(z)))$", fontsize=11, fontweight='bold', labelpad=10)

    # Overlay patient risk scatter points
    scatter = ax.scatter(
        Z_pca[:, 0], Z_pca[:, 1],
        c=risk_scores[:len(Z_pca)],
        cmap="coolwarm",
        s=25,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.9,
        label='Patients (Risk Score)'
    )

    # Overlay CARE-LG Recourse Trajectory
    ax.plot(
        path_Z_pca[:, 0], path_Z_pca[:, 1],
        color='#00ffff',
        linewidth=3.0,
        linestyle='-',
        marker='o',
        markersize=8,
        zorder=6,
        label='CARE-LG Geodesic Trajectory'
    )

    # Highlight source patient (X) and target patient (*)
    ax.scatter(path_Z_pca[0, 0], path_Z_pca[0, 1], color='#f1c40f', s=160, marker='X', edgecolor='black', linewidth=1.5, zorder=7, label='Source Patient (High Risk)')
    ax.scatter(path_Z_pca[-1, 0], path_Z_pca[-1, 1], color='#2ecc71', s=200, marker='*', edgecolor='black', linewidth=1.5, zorder=7, label='Target Patient (Low Risk)')

    ax.set_title(f"Riemannian Latent Manifold Geometry ({dataset_name.upper()} Cohort)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Latent PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Latent PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSuccessfully generated figure: {fig_path}")
    print("==================================================")
    print(f"SUCCESS: Riemannian Latent Geometry Heatmap Completed for {dataset_name.upper()}!")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE-LG Riemannian Heatmap Generator")
    parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "nhanes"], help="Dataset to process (uci or nhanes)")
    args = parser.parse_args()

    generate_riemannian_heatmap(dataset_name=args.dataset, seed=42)
