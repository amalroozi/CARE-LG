"""
Constrained Pathfinding & Recourse Decoder for CARE-LG research framework.
Uses Dijkstra's algorithm over CALG sparse matrix to find optimal feasible recourse paths.
"""

import numpy as np
import pandas as pd
import scipy.sparse.csgraph as csgraph
from src.graph.clinical_constraints import unscale_features


def find_recourse_path(calg_matrix, source_idx, target_mask):
    """
    Executes Dijkstra shortest path search on CALG sparse adjacency matrix from source_idx
    to the closest feasible target node in target_mask.

    Args:
        calg_matrix (scipy.sparse.csr_matrix): Directed adjacency matrix.
        source_idx (int): Index of query high-risk patient node.
        target_mask (np.ndarray or list): Boolean mask indicating candidate target nodes (low risk).

    Returns:
        path_indices (list): Ordered list of node indices along recourse trajectory [source_idx, ..., target_idx].

    Raises:
        ValueError: If no feasible recourse path exists in the CALG graph.
    """
    distances, predecessors = csgraph.dijkstra(
        csgraph=calg_matrix,
        directed=True,
        indices=source_idx,
        return_predecessors=True
    )

    # Filter target nodes that are reachable
    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        raise ValueError("No target nodes specified in target_mask.")

    target_distances = distances[target_indices]
    finite_mask = np.isfinite(target_distances)

    if not np.any(finite_mask):
        raise ValueError(f"No feasible recourse path found from source node {source_idx} to any target node.")

    # Select target node with minimum shortest path cost
    best_target_idx = target_indices[np.argmin(target_distances)]

    # Reconstruct path from predecessors
    path = []
    curr = best_target_idx
    while curr != source_idx and curr != -9999:  # SciPy uses -9999 for unreached/source predecessor
        path.append(curr)
        curr = predecessors[curr]

    if curr != source_idx:
        raise ValueError(f"Failed to trace path back to source node {source_idx}.")

    path.append(source_idx)
    path.reverse()
    return path


def decode_recourse_trajectory(vae_model, latent_nodes, path_indices, scaled_features, scaler, feature_cols, feature_metadata):
    """
    Decodes latent nodes along recourse path back into step-by-step unscaled clinical feature recommendations.

    Args:
        vae_model (nn.Module): Trained TabularVAE model.
        latent_nodes (np.ndarray): Encoded latent array Z of shape (N, d).
        path_indices (list): List of node indices along recourse path.
        scaled_features (np.ndarray): Original scaled feature array X of shape (N, D).
        scaler (StandardScaler): Scaler object.
        feature_cols (list): Feature column names.
        feature_metadata (dict): Feature schema configuration.

    Returns:
        pd.DataFrame: Step-by-step DataFrame of clinical feature values along recourse trajectory.
    """
    rows = []
    for step_idx, node_idx in enumerate(path_indices):
        x_scaled = scaled_features[node_idx]
        unscaled_dict = unscale_features(x_scaled, scaler, feature_cols, feature_metadata)
        unscaled_dict['step'] = step_idx
        unscaled_dict['node_idx'] = node_idx
        rows.append(unscaled_dict)

    df_trajectory = pd.DataFrame(rows)

    # Reorder columns with step and node_idx first
    cols = ['step', 'node_idx'] + [c for c in df_trajectory.columns if c not in ['step', 'node_idx']]
    return df_trajectory[cols]
