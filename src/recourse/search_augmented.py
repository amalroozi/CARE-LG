# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Dijkstra search over the (N+1, N+1) augmented graph (train nodes + one
ephemeral query source node at index N). Mirrors src/recourse/search.py's
find_recourse_path but operates on the augmented matrix and returns the
shortest-path cost alongside the path.
"""
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


def find_path_augmented(aug_matrix: sp.csr_matrix, source_idx: int, target_mask: np.ndarray) -> Tuple[List[int], float]:
    """
    Args:
        aug_matrix: (N+1, N+1) sparse directed adjacency (query is row/col N).
        source_idx: N (the query node index).
        target_mask: boolean array of length N+1; entry N should be False.

    Returns:
        (path, cost): path is [source_idx, ..., target_idx]; cost is the
        shortest-path distance to the chosen target.

    Raises:
        ValueError if no feasible path exists.
    """
    distances, predecessors = csgraph.dijkstra(
        csgraph=aug_matrix, directed=True, indices=source_idx, return_predecessors=True
    )
    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        raise ValueError("No target nodes specified.")

    target_distances = distances[target_indices]
    finite_mask = np.isfinite(target_distances)
    if not np.any(finite_mask):
        raise ValueError(f"No feasible recourse path found from source {source_idx}.")

    best_target = target_indices[np.argmin(target_distances)]
    path = []
    curr = best_target
    while curr != source_idx and curr != -9999:
        path.append(curr)
        curr = predecessors[curr]
    if curr != source_idx:
        raise ValueError(f"Failed to trace path back to source {source_idx}.")
    path.append(source_idx)
    path.reverse()
    return path, float(distances[best_target])
