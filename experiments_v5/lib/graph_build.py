"""
Ablation-ready CALG construction (Phase 1).

Builds the k-NN edge STRUCTURE exactly once per (dataset, seed) — identical
across every grid cell, per the brief — then computes, once, for every edge:
  - the Riemannian distance (vectorized, see riemannian_batch.py)
  - the plain Euclidean latent distance
  - the constraint violation count v(x_i,x_j) and boolean violates(x_i,x_j)
    (vectorized reimplementation of src.graph.clinical_constraints logic,
    verified to agree with the original scalar functions on a random sample)

A grid cell (metric, constraints, lambda) is then just a cheap reweighting
of this cached edge table (`make_cell_matrix`), so the expensive part (k-NN
search + Jacobian evaluation) runs once per seed, not once per cell.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from experiments_v5.lib.riemannian_batch import batched_riemannian_distances
from experiments_v5.lib.constraints_ext import violates as violates_fn, count_violations, _DIRECTIONAL_EPS, _AGE_DECREASE_TOL, _AGE_HORIZON_CAP, _IMMUTABLE_TOL


def unscale_matrix(X: np.ndarray, scaler, feature_cols, feature_metadata) -> Dict[str, np.ndarray]:
    """
    Vectorized equivalent of calling unscale_features() on every row of X.

    NOTE on dtype: sklearn's StandardScaler.inverse_transform PRESERVES the
    input dtype (float32 in, float32 out) rather than upcasting to float64.
    Since X here is float32 (it comes from a torch float32 tensor), we
    replicate that float32 arithmetic exactly (verified bit-exact against
    scaler.inverse_transform on sample data) so that boundary comparisons
    against the epsilon thresholds below agree with the scalar
    check_hard_violations/check_soft_violations to the same precision,
    rather than silently upcasting to float64 and shifting values by ~1e-7
    right at threshold boundaries (e.g. resting_bp delta of exactly 5.0).
    """
    continuous_cols = feature_metadata['continuous_mutable'] + feature_metadata['non_decreasing']
    cont_idx = [feature_cols.index(c) for c in continuous_cols]
    X_cont_scaled = X[:, cont_idx]
    if scaler is not None and len(cont_idx) > 0:
        orig_dtype = X_cont_scaled.dtype
        X_cont_unscaled = X_cont_scaled.astype(orig_dtype) * scaler.scale_.astype(orig_dtype) + scaler.mean_.astype(orig_dtype)
    else:
        X_cont_unscaled = X_cont_scaled

    out: Dict[str, np.ndarray] = {}
    for i, col in enumerate(continuous_cols):
        out[col] = X_cont_unscaled[:, i]
    for col in feature_cols:
        if col not in out:
            out[col] = X[:, feature_cols.index(col)]
    return out


def vectorized_violation_components(
    unscaled: Dict[str, np.ndarray], src_idx: np.ndarray, tgt_idx: np.ndarray, feature_metadata, check_step_horizon: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized breakdown of violations into (hard_count, directional_count):
      hard_count = immutable changes + age-decrease + (age-horizon-cap if check_step_horizon)
                   -- matches src.graph.clinical_constraints.check_hard_violations
      directional_count = directional_reduce/increase epsilon breaches
                   -- matches src.graph.clinical_constraints.check_soft_violations
    full violation = check_clinical_violations = (hard_count + directional_count) > 0.
    Kept separate so query-attachment edges (see build_query_edges) can gate
    on hard-only (matching the original benchmark harness's entry-node rule,
    REPO_MAP.md D8) while still reporting full severity for the soft penalty.
    """
    n_edges = len(src_idx)
    hard_count = np.zeros(n_edges, dtype=np.int32)
    dir_count = np.zeros(n_edges, dtype=np.int32)

    for col in feature_metadata.get('immutable', []):
        d = np.abs(unscaled[col][tgt_idx] - unscaled[col][src_idx])
        hard_count += (d > _IMMUTABLE_TOL).astype(np.int32)

    for col in feature_metadata.get('non_decreasing', []):
        hard_count += (unscaled[col][tgt_idx] < unscaled[col][src_idx] - _AGE_DECREASE_TOL).astype(np.int32)

    if check_step_horizon and 'age' in unscaled:
        hard_count += ((unscaled['age'][tgt_idx] - unscaled['age'][src_idx]) > _AGE_HORIZON_CAP).astype(np.int32)

    for col in feature_metadata.get('directional_reduce', []):
        eps = _DIRECTIONAL_EPS.get(col, 0.1)
        dir_count += ((unscaled[col][tgt_idx] - unscaled[col][src_idx]) > eps).astype(np.int32)

    for col in feature_metadata.get('directional_increase', []):
        eps = _DIRECTIONAL_EPS.get(col, 1.0)
        dir_count += ((unscaled[col][tgt_idx] - unscaled[col][src_idx]) < -eps).astype(np.int32)

    return hard_count, dir_count


def vectorized_violation_counts(
    unscaled: Dict[str, np.ndarray], src_idx: np.ndarray, tgt_idx: np.ndarray, feature_metadata, check_step_horizon: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Full v(x_i,x_j) count (hard+directional) and violates() boolean. Matches check_clinical_violations."""
    hard_count, dir_count = vectorized_violation_components(unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon)
    count = hard_count + dir_count
    return count, count > 0


@dataclass
class EdgeTable:
    N: int
    k: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray
    violates: np.ndarray
    v_count: np.ndarray
    nbrs: NearestNeighbors  # fitted once on Z_train; reused for query-node attachment


def build_edge_table(
    vae_model: torch.nn.Module,
    Z: np.ndarray,
    X: np.ndarray,
    feature_metadata: dict,
    scaler,
    feature_cols,
    k: int,
    verify_sample: int = 25,
) -> EdgeTable:
    """Builds the shared k-NN edge structure + all per-edge quantities, once per (dataset, seed)."""
    N = len(Z)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="ball_tree").fit(Z)
    _, indices = nbrs.kneighbors(Z)

    edge_i, edge_j = [], []
    for i in range(N):
        for nb in indices[i]:
            if nb == i:
                continue
            edge_i.append(i)
            edge_j.append(nb)
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)

    Zt = torch.tensor(Z, dtype=torch.float32)
    dist_r = batched_riemannian_distances(vae_model, Zt[edge_i], Zt[edge_j]).numpy()
    dist_e = np.linalg.norm(Z[edge_i] - Z[edge_j], axis=1)

    unscaled = unscale_matrix(X, scaler, feature_cols, feature_metadata)
    v_count, viol = vectorized_violation_counts(unscaled, edge_i, edge_j, feature_metadata, check_step_horizon=True)

    if verify_sample > 0:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(edge_i), size=min(verify_sample, len(edge_i)), replace=False)
        for idx in sample:
            ref = violates_fn(X[edge_i[idx]], X[edge_j[idx]], feature_metadata, scaler, feature_cols, check_step_horizon=True)
            assert bool(viol[idx]) == bool(ref), (
                f"Vectorized violation check disagrees with original check_clinical_violations at edge "
                f"({edge_i[idx]},{edge_j[idx]}): vectorized={viol[idx]} original={ref}"
            )
            ref_count = count_violations(X[edge_i[idx]], X[edge_j[idx]], feature_metadata, scaler, feature_cols, check_step_horizon=True)
            assert int(v_count[idx]) == int(ref_count)

    return EdgeTable(N=N, k=k, edge_i=edge_i, edge_j=edge_j, dist_riemannian=dist_r, dist_euclidean=dist_e, violates=viol, v_count=v_count, nbrs=nbrs)


@dataclass
class QueryEdges:
    """Directed out-edges from ONE external query node (a test patient) into the train graph."""
    neighbor_idx: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray
    violates: np.ndarray
    v_count: np.ndarray


def build_query_edges(
    vae_model: torch.nn.Module,
    z0: np.ndarray,
    x0: np.ndarray,
    Z_train: np.ndarray,
    X_train: np.ndarray,
    feature_metadata: dict,
    scaler,
    feature_cols,
    k: int,
    nbrs: Optional[NearestNeighbors] = None,
) -> QueryEdges:
    """
    Attaches a test patient as an ephemeral source node with out-edges to
    EVERY train node (full attachment, not k-restricted) — a deliberate
    design choice, documented here and in CHANGES.md / REPO_MAP.md D8.

    Rationale: the intra-graph k-NN structure (identical across cells, per
    the brief) exists to define smooth local transitions BETWEEN existing
    patients on the manifold. A held-out query patient was never part of
    that curation, so restricting its own out-degree to its k nearest
    Z-neighbors is an artificial locality constraint the original code never
    had either: the original benchmark harness (benchmarks/run_benchmarks.py)
    searches the ENTIRE train set for a valid entry node, not just a local
    k-NN neighborhood. We verified empirically that k-restricted attachment
    (even with the corrected hard-only/no-horizon entry gate below) leaves
    many patients cut off from entries that exist elsewhere in the graph:
    on a seed-0 UCI pilot, 25 high-risk patients had a mean of 34.2 valid
    entries across the full 237-node graph but only ~1 within their 40
    nearest Z-neighbors. Full attachment removes this artifact while
    leaving Phase 4's k-densification probe fully meaningful (that probe
    is about the INTRA-graph k, i.e. how well-connected entries are to
    low-risk targets once reached, not about the entry step itself).

    Gating convention (REPO_MAP.md D8): the entry-edge PRUNE decision
    (`violates`, used only by hard mode) uses check_hard_violations with
    check_step_horizon=False, matching the original benchmark harness's own
    entry-node rule (no directional check, no per-step age-horizon cap,
    since the entry hop represents "which existing case most resembles this
    patient," not a bounded time step). `v_count` (used by the soft-penalty
    sweep) still reflects the FULL hard+directional severity. Interior graph
    hops are unaffected and keep the original check_step_horizon=True full
    definition (check_clinical_violations, unchanged from src/graph/calg.py).
    """
    N = len(Z_train)
    neighbor_idx = np.arange(N)

    z0t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(len(neighbor_idx), 1)
    zjt = torch.tensor(Z_train[neighbor_idx], dtype=torch.float32)
    dist_r = batched_riemannian_distances(vae_model, z0t, zjt).numpy()
    dist_e = np.linalg.norm(Z_train[neighbor_idx] - z0.reshape(1, -1), axis=1)

    X_pair_src = np.tile(x0.reshape(1, -1), (len(neighbor_idx), 1))
    X_pair_tgt = X_train[neighbor_idx]
    X_combined = np.vstack([X_pair_src, X_pair_tgt])
    unscaled = unscale_matrix(X_combined, scaler, feature_cols, feature_metadata)
    n = len(neighbor_idx)
    src_idx = np.arange(n)
    tgt_idx = np.arange(n, 2 * n)
    hard_count, dir_count = vectorized_violation_components(unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon=False)
    gate_violates = hard_count > 0          # hard-only, no horizon: matches original entry-rule convention
    v_count = hard_count + dir_count        # full severity for the soft-penalty sweep

    return QueryEdges(neighbor_idx=neighbor_idx, dist_riemannian=dist_r, dist_euclidean=dist_e, violates=gate_violates, v_count=v_count)


def make_cell_matrix(edge_table: EdgeTable, metric: str, constraints: str, lam: float = 1.0) -> sp.csr_matrix:
    """Reweights the cached edge table into the (N,N) sparse graph for one grid cell."""
    assert metric in ("riemannian", "euclidean")
    assert constraints in ("hard", "soft", "none")

    dist = edge_table.dist_riemannian if metric == "riemannian" else edge_table.dist_euclidean

    if constraints == "hard":
        keep = ~edge_table.violates
        ei, ej, w = edge_table.edge_i[keep], edge_table.edge_j[keep], dist[keep]
    elif constraints == "soft":
        ei, ej = edge_table.edge_i, edge_table.edge_j
        w = dist + lam * edge_table.v_count
    else:  # none
        ei, ej = edge_table.edge_i, edge_table.edge_j
        w = dist

    return sp.csr_matrix((w, (ei, ej)), shape=(edge_table.N, edge_table.N))


def make_query_row(qedges: QueryEdges, metric: str, constraints: str, N: int, lam: float = 1.0) -> sp.csr_matrix:
    """Builds a (1, N) sparse row of out-edges from the query node, for one grid cell."""
    dist = qedges.dist_riemannian if metric == "riemannian" else qedges.dist_euclidean

    if constraints == "hard":
        keep = ~qedges.violates
        cols, w = qedges.neighbor_idx[keep], dist[keep]
    elif constraints == "soft":
        cols, w = qedges.neighbor_idx, dist + lam * qedges.v_count
    else:
        cols, w = qedges.neighbor_idx, dist

    rows = np.zeros(len(cols), dtype=np.int64)
    return sp.csr_matrix((w, (rows, cols)), shape=(1, N))


def augmented_matrix(base_csr: sp.csr_matrix, query_row_csr: sp.csr_matrix) -> sp.csr_matrix:
    """
    Appends the query node as row/col index N of an (N+1, N+1) matrix: the
    query has out-edges (last row) but no in-edges (last column all-zero),
    since nothing should route recourse *through* another patient's query.
    """
    N = base_csr.shape[0]
    last_col = sp.csr_matrix((N, 1))
    top = sp.hstack([base_csr, last_col], format="csr")
    bottom = sp.hstack([query_row_csr, sp.csr_matrix((1, 1))], format="csr")
    return sp.vstack([top, bottom], format="csr")
