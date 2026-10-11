"""
FACE baseline. Poyiadzi, Sokol, Santos-Rodriguez, De Bie, Flach, "FACE:
Feasible and Actionable Counterfactual Explanations" (AIES 2020).

Canonical home for FACE, promoted here from `scripts/run_face_rerun.py`
(built and already correctly cited in a prior session) for consistency with
this project's other baselines (`src/baselines/dice.py`,
`src/baselines/growing_spheres.py`) now that all three live under
`src/baselines/`. `scripts/run_face_rerun.py` now imports from here rather
than containing the algorithm itself -- ONE authoritative implementation,
not two. This is the AUTHORITATIVE FACE implementation for this project;
`benchmarks/run_benchmarks.py::run_face` is superseded legacy (see that
file's own superseded-notice docstring and docs/HISTORY.md).

Faithful to the paper's core mechanism: shortest-path recourse over a
density/distance-weighted k-NN graph built directly in ORIGINAL feature
space (no latent embedding, no VAE, no admissibility/constraint gating on
edges -- every k-NN edge exists regardless of clinical plausibility, exactly
as the published method specifies; FACE's own contribution is feasibility
via graph connectivity to real, dense regions of the data, not any notion
of hard clinical admissibility). Evaluated here under this project's
corrected protocol: full (not k-restricted) ephemeral query attachment
matching `src/graph/query_attachment.py`'s D8 design, so the comparison
against CARE-LG is apples-to-apples on search breadth, and REAL recorded
feature values throughout (FACE never touches a VAE at all, so there was
never a decoded-vs-real distinction for it the way there was for CARE-LG's
B1/B2 bug -- its recourse target has always been a real recorded training
patient).
"""
import time
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from sklearn.neighbors import NearestNeighbors

K_NEIGHBORS_FACE = {"uci": 40, "nhanes_real": 80}  # same k as CARE-LG's own graph, for a fair comparison


def build_face_graph(X_train: np.ndarray, k: int) -> Tuple[sp.csr_matrix, NearestNeighbors]:
    N = len(X_train)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    ei, ej, w = [], [], []
    for i in range(N):
        for idx, nb in enumerate(indices[i]):
            if nb == i:
                continue
            ei.append(i); ej.append(nb); w.append(distances[i][idx])
    return sp.csr_matrix((w, (ei, ej)), shape=(N, N)), nbrs


def run_face_query(x0: np.ndarray, X_train: np.ndarray, base_csr: sp.csr_matrix, low_risk_mask: np.ndarray):
    """
    Returns (success, path_node_indices_or_None, cost, latency_sec).
    `path_node_indices` uses N (len(X_train)) as the ephemeral query node's
    own index, matching this project's augmented-graph convention elsewhere.
    """
    N = len(X_train)
    t0 = time.perf_counter()
    dists = np.linalg.norm(X_train - x0.reshape(1, -1), axis=1)
    q_row = sp.csr_matrix((dists, (np.zeros(N, dtype=np.int64), np.arange(N))), shape=(1, N))
    last_col = sp.csr_matrix((N, 1))
    top = sp.hstack([base_csr, last_col], format="csr")
    bottom = sp.hstack([q_row, sp.csr_matrix((1, 1))], format="csr")
    aug = sp.vstack([top, bottom], format="csr")

    try:
        distances, predecessors = csgraph.dijkstra(csgraph=aug, directed=True, indices=N, return_predecessors=True)
        target_indices = np.where(low_risk_mask)[0]
        target_dists = distances[target_indices]
        finite_mask = np.isfinite(target_dists)
        if not np.any(finite_mask):
            return False, None, None, time.perf_counter() - t0
        best_target = target_indices[np.argmin(target_dists)]
        path = []
        curr = best_target
        while curr != N and curr != -9999:
            path.append(curr)
            curr = predecessors[curr]
        path.append(N)
        path.reverse()
        return True, path, float(distances[best_target]), time.perf_counter() - t0
    except Exception:
        return False, None, None, time.perf_counter() - t0
