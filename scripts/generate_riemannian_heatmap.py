"""
Riemannian Latent Geometry Heatmap Generator for CARE-LG research framework.
Computes pullback metric tensor G(z) = J_g(z)^T J_g(z), evaluates volume distortion log(det(G(z))),
and plots 2D PCA manifold heatmap with patient risk scores and CARE-LG recourse path overlay.
"""

import sys
import os
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

from configs.dataset_config import FEATURE_METADATA, CLINICAL_EFFORT_WEIGHTS
from src.data.loader import get_dataloaders
from src.blackbox_model import RiskClassifier, train_blackbox_model, get_device
from src.vae.model import TabularVAE, train_vae
from src.graph.riemannian import compute_decoder_jacobian, compute_metric_tensor
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path


def get_or_train_models(train_loader, input_dim, device):
    """
    Loads saved TabularVAE model from models/vae_model.pt or trains and saves a new checkpoint.
    Also trains the RiskClassifier model.
    """
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    vae_path = models_dir / "vae_model.pt"
    vae = TabularVAE(input_dim=input_dim, latent_dim=4).to(device)

    if vae_path.exists():
        print(f"Loading trained Tabular VAE model checkpoint from {vae_path}...")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        print(f"Training Tabular VAE model (25 epochs)...")
        vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=device)
        torch.save(vae.state_dict(), vae_path)
        print(f"Saved Tabular VAE model to {vae_path}")

    print("Training Risk Classifier model...")
    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)

    return vae, classifier


def generate_riemannian_heatmap(seed=42):
    print("==================================================")
    print("CARE-LG Riemannian Latent Geometry Heatmap Generator")
    print("==================================================")

    device = get_device()
    print(f"Using PyTorch device: {device}")

    # 1. Setup & Data Loading
    print("\n1. Loading UCI Heart Disease Dataset & DataLoaders...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_path="data/uci_heart.csv", batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    vae, classifier = get_or_train_models(train_loader, input_dim, device)

    # 2. Latent Projection
    print("\n2. Projecting Dataset into 4D Latent Space...")
    vae.eval()
    all_z = []
    all_x = []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            X_batch_dev = X_batch.to(device)
            mu, _ = vae.encode(X_batch_dev)
            all_z.append(mu.cpu().numpy())
            all_x.append(X_batch.numpy())

    Z_4d = np.vstack(all_z)
    X_scaled = np.vstack(all_x)

    # Predict risk scores across dataset
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        risk_scores = classifier(X_tensor).cpu().numpy().squeeze()

    # Fit 2D PCA on 4D Latent Space Z
    pca = PCA(n_components=2, random_state=seed)
    Z_2d = pca.fit_transform(Z_4d)

    z1_min, z1_max = Z_2d[:, 0].min() - 0.8, Z_2d[:, 0].max() + 0.8
    z2_min, z2_max = Z_2d[:, 1].min() - 0.8, Z_2d[:, 1].max() + 0.8

    # 3. Metric Grid Generation (50x50 Meshgrid)
    print("\n3. Generating 50x50 Meshgrid over PCA Domain...")
    grid_size = 50
    grid_z1 = np.linspace(z1_min, z1_max, grid_size)
    grid_z2 = np.linspace(z2_min, z2_max, grid_size)
    Z1_mesh, Z2_mesh = np.meshgrid(grid_z1, grid_z2)

    grid_2d = np.column_stack([Z1_mesh.ravel(), Z2_mesh.ravel()])
    grid_4d = pca.inverse_transform(grid_2d)

    # 4. Riemannian Pullback Metric Tensor & Volume Magnification Calculation
    print("\n4. Computing Riemannian Pullback Metric Tensors G(z) & Log-Determinants...")
    log_det_list = []

    for z_point in tqdm(grid_4d, desc="Riemannian Metric Grid"):
        z_tensor = torch.tensor(z_point, dtype=torch.float32).to(device)
        J = compute_decoder_jacobian(vae, z_tensor)
        G = compute_metric_tensor(J)

        # Compute determinant det(G)
        det_G = torch.det(G).item()
        log_det = np.log(max(det_G, 1e-8))
        log_det_list.append(log_det)

    log_det_grid = np.array(log_det_list).reshape(grid_size, grid_size)

    # 5. Build CALG & Compute CARE-LG Recourse Trajectory
    print("\n5. Computing CARE-LG Recourse Trajectory for Overlay...")
    calg_matrix, Z_calg, X_calg, y_calg = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=FEATURE_METADATA,
        effort_weights=CLINICAL_EFFORT_WEIGHTS,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=30,
        lambda_effort=1.0
    )

    with torch.no_grad():
        X_calg_tensor = torch.tensor(X_calg, dtype=torch.float32).to(device)
        calg_risk_scores = classifier(X_calg_tensor).cpu().numpy().squeeze()

    high_risk_indices = np.where(calg_risk_scores > 0.55)[0]
    low_risk_mask = calg_risk_scores < 0.45

    path = None
    source_idx = None
    for cand_idx in high_risk_indices:
        try:
            p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
            if p is not None:
                path = p
                source_idx = cand_idx
                if len(p) >= 2:
                    break
        except Exception:
            continue

    assert path is not None, "Could not find a feasible recourse path for visualization."

    path_4d = Z_calg[path]
    path_2d = pca.transform(path_4d)

    # 6. Generate Publication Figure (fig3_riemannian_heatmap.png)
    print("\n6. Rendering Publication Figure (fig3_riemannian_heatmap.png)...")
    sns.set_theme(style="white", font="sans-serif")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Background Riemannian Magnification Heatmap
    contour = ax.contourf(
        Z1_mesh, Z2_mesh, log_det_grid,
        levels=30, cmap="viridis", alpha=0.85
    )

    cbar = fig.colorbar(contour, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label(r"Geometric Magnification $\log(\det(G(z)))$", fontsize=11, fontweight="bold")

    # Overlay Patient Samples (colored by Risk)
    scatter = ax.scatter(
        Z_2d[:, 0], Z_2d[:, 1],
        c=risk_scores, cmap="coolwarm",
        edgecolors="k", linewidths=0.5, s=40, alpha=0.85,
        zorder=3
    )

    # Overlay CARE-LG Geodesic Recourse Trajectory Path
    ax.plot(
        path_2d[:, 0], path_2d[:, 1],
        color="magenta", linewidth=3.5, linestyle="-",
        marker="o", markersize=9, zorder=5,
        label="CARE-LG Geodesic Recourse Path"
    )

    # Mark Start & End Nodes
    ax.scatter(path_2d[0, 0], path_2d[0, 1], color="yellow", edgecolors="black", s=180, marker="X", zorder=6, label="Source High-Risk Patient")
    ax.scatter(path_2d[-1, 0], path_2d[-1, 1], color="lime", edgecolors="black", s=220, marker="*", zorder=6, label="Recourse Target Patient")

    ax.set_title("Riemannian Latent Manifold Geometry", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"Latent Component 1 (PCA {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Latent Component 2 (PCA {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, fontweight="bold")

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    output_fig_path = benchmarks_dir / "fig3_riemannian_heatmap.png"

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSuccessfully generated figure: {output_fig_path}")
    print("==================================================")
    print("SUCCESS: Riemannian Latent Geometry Heatmap Completed!")
    print("==================================================")


if __name__ == "__main__":
    generate_riemannian_heatmap(seed=42)
