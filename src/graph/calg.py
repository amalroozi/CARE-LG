"""
Clinical Adaptive Latent Graph (CALG) construction module for CARE-LG research framework.
Constructs k-NN directed latent graphs with Riemannian distances, weighted clinical effort, and asymmetric masking.
"""

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.graph.riemannian import riemannian_distance
from src.graph.clinical_constraints import check_hard_violations, check_soft_violations, check_clinical_violations, compute_clinical_effort


def build_calg_graph(
    vae_model,
    train_loader,
    feature_metadata,
    effort_weights,
    scaler,
    feature_cols,
    k_neighbors=30,
    lambda_effort=1.0,
    max_samples=None
):
    """
    Constructs the Clinical Adaptive Latent Graph (CALG).

    Args:
        vae_model (nn.Module): Trained TabularVAE model.
        train_loader (DataLoader): PyTorch DataLoader.
        feature_metadata (dict): Feature schema configuration.
        effort_weights (dict): Clinical effort weights.
        scaler (StandardScaler): Scaler for continuous features.
        feature_cols (list): List of feature column names.
        k_neighbors (int): Number of nearest neighbors per node (default: 30).
        lambda_effort (float): Trade-off parameter between Riemannian distance and clinical effort.
        max_samples (int, optional): Subset max samples for faster graph building if specified.

    Returns:
        calg_matrix (scipy.sparse.csr_matrix): Sparse directed adjacency matrix (N x N) of edge weights.
        latent_nodes (np.ndarray): Encoded latent points array Z of shape (N, d).
        scaled_features (np.ndarray): Original scaled feature array X of shape (N, D).
        targets (np.ndarray): Target labels array y of shape (N,).
    """
    vae_model.eval()
    device = next(vae_model.parameters()).device

    all_z = []
    all_x = []
    all_y = []

    count = 0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch_dev = X_batch.to(device)
            mu, _ = vae_model.encode(X_batch_dev)
            all_z.append(mu.cpu().numpy())
            all_x.append(X_batch.numpy())
            all_y.append(y_batch.numpy())
            count += X_batch.size(0)
            if max_samples is not None and count >= max_samples:
                break

    Z = np.vstack(all_z)
    X = np.vstack(all_x)
    y = np.concatenate(all_y).squeeze()

    if max_samples is not None and len(Z) > max_samples:
        Z = Z[:max_samples]
        X = X[:max_samples]
        y = y[:max_samples]

    N = len(Z)

    # Adaptive k-NN scaling for larger datasets (NHANES N >= 1000)
    if N >= 1000 and k_neighbors < 75:
        k_neighbors = 100

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N), algorithm='ball_tree').fit(Z)
    distances, indices = nbrs.kneighbors(Z)

    row_indices = []
    col_indices = []
    edge_weights = []

    print(f"Building CALG graph over N={N} nodes with k={k_neighbors} neighbors...")
    for i in tqdm(range(N), desc="CALG Edges"):
        z_i = Z[i]
        x_i = X[i]

        for neighbor_idx in indices[i]:
            if neighbor_idx == i:
                continue

            z_j = Z[neighbor_idx]
            x_j = X[neighbor_idx]

            # 1. HARD Mask Check (Immutable attributes & Age horizon cap > 3.0 yrs)
            if check_hard_violations(x_i, x_j, feature_metadata, scaler, feature_cols):
                continue  # Hard infinity (blocked edge)

            # 2. Compute Riemannian Geodesic Distance
            r_dist = riemannian_distance(z_i, z_j, vae_model)

            # 3. Compute Clinical Effort
            c_effort = compute_clinical_effort(x_i, x_j, effort_weights, feature_metadata, scaler, feature_cols)

            # 4. Base Weight
            w_ij = r_dist + lambda_effort * c_effort

            # 5. SOFT Directional Penalty (1e5x multiplier for soft non-monotonic directional shifts)
            if check_soft_violations(x_i, x_j, feature_metadata, scaler, feature_cols):
                w_ij = w_ij * 100000.0 + 10000.0

            row_indices.append(i)
            col_indices.append(neighbor_idx)
            edge_weights.append(w_ij)

    calg_matrix = sp.csr_matrix((edge_weights, (row_indices, col_indices)), shape=(N, N))
    return calg_matrix, Z, X, y
