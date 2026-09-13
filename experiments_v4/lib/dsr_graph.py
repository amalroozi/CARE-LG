"""
Decode-Safe Recourse — graph construction under any of the four constraint modes.

Reuses experiments_v3's expensive machinery unchanged: the k-NN edge STRUCTURE
and the batched Riemannian/Euclidean distances are computed ONCE per
(dataset, seed) and shared by every mode, so modes differ only in which edges
get pruned. This keeps the ablation exact (identical geometry, identical
neighbourhoods, only the predicate changes) and keeps the runtime tractable.

Component 1's cost note: every node is decoded ONCE here (`decode_nodes_batch`,
one forward pass, O(N)), never inside the per-edge loop.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from experiments_v4.lib.constraints_ext import decode_nodes_batch
from experiments_v4.lib.dsr_rules import ViolationBreakdown, violation_breakdown
from experiments_v4.lib.graph_build import unscale_matrix
from experiments_v4.lib.riemannian_batch import batched_riemannian_distances


@dataclass
class DSREdgeTable:
    """Shared geometry (mode-independent) + decoded node cache."""
    N: int
    k: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray
    X_hat: np.ndarray                  # decoded reconstruction of every graph node
    unscaled_true: Dict[str, np.ndarray]
    unscaled_decoded: Dict[str, np.ndarray]
    nbrs: NearestNeighbors
    build_seconds: float
    decode_seconds: float


def build_dsr_edge_table(
    vae: torch.nn.Module, Z: np.ndarray, X: np.ndarray, feature_metadata: dict, scaler,
    feature_cols, k: int, device: torch.device = torch.device("cpu"),
) -> DSREdgeTable:
    """Builds the shared k-NN structure, distances, and the one-shot decoded node cache."""
    import time

    t0 = time.perf_counter()
    N = len(Z)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="ball_tree").fit(Z)
    _, indices = nbrs.kneighbors(Z)

    edge_i, edge_j = [], []
    for i in range(N):
        for nb in indices[i]:
            if nb != i:
                edge_i.append(i)
                edge_j.append(nb)
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)

    Zt = torch.tensor(Z, dtype=torch.float32)
    dist_r = batched_riemannian_distances(vae, Zt[edge_i], Zt[edge_j]).numpy()
    dist_e = np.linalg.norm(Z[edge_i] - Z[edge_j], axis=1)

    t_dec = time.perf_counter()
    X_hat = decode_nodes_batch(vae, Z, device=device)      # Component 1: O(N), once
    decode_seconds = time.perf_counter() - t_dec

    unscaled_true = unscale_matrix(X, scaler, feature_cols, feature_metadata)
    unscaled_decoded = unscale_matrix(X_hat, scaler, feature_cols, feature_metadata)

    return DSREdgeTable(
        N=N, k=k, edge_i=edge_i, edge_j=edge_j, dist_riemannian=dist_r, dist_euclidean=dist_e,
        X_hat=X_hat, unscaled_true=unscaled_true, unscaled_decoded=unscaled_decoded, nbrs=nbrs,
        build_seconds=time.perf_counter() - t0, decode_seconds=decode_seconds,
    )


def prune_for_mode(
    table: DSREdgeTable, feature_metadata: dict, mode: str, taus: Optional[Dict[str, float]] = None
) -> ViolationBreakdown:
    """Evaluates the constraint predicate over all interior edges in the given mode."""
    return violation_breakdown(
        table.unscaled_true, table.unscaled_decoded, table.edge_i, table.edge_j,
        feature_metadata, mode, taus, check_step_horizon=True,
    )


def make_matrix(table: DSREdgeTable, breakdown: ViolationBreakdown, metric: str = "riemannian") -> sp.csr_matrix:
    """Hard-prunes violating edges (W_ij = inf by omission) and returns the (N,N) graph."""
    dist = table.dist_riemannian if metric == "riemannian" else table.dist_euclidean
    keep = ~breakdown.any
    return sp.csr_matrix(
        (dist[keep], (table.edge_i[keep], table.edge_j[keep])), shape=(table.N, table.N)
    )


@dataclass
class DSRQueryEdges:
    neighbor_idx: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray
    breakdown: ViolationBreakdown
    x0_hat: np.ndarray                 # decoded reconstruction of the query patient


def build_dsr_query_edges(
    vae: torch.nn.Module, z0: np.ndarray, x0: np.ndarray, table: DSREdgeTable, Z_train: np.ndarray,
    X_train: np.ndarray, feature_metadata: dict, scaler, feature_cols, mode: str,
    taus: Optional[Dict[str, float]] = None, device: torch.device = torch.device("cpu"),
) -> DSRQueryEdges:
    """
    Attaches a held-out query patient as an ephemeral source node.

    Attachment topology and the entry-gate convention (no per-step age-horizon
    cap on the entry hop) are inherited UNCHANGED from experiments_v3 --
    see experiments_v3/lib/graph_build.py::build_query_edges for the full
    rationale. The only v4 change is which VALUES the predicate reads, exactly
    as for interior edges.
    """
    N = len(Z_train)
    neighbor_idx = np.arange(N)

    z0t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    zjt = torch.tensor(Z_train[neighbor_idx], dtype=torch.float32)
    dist_r = batched_riemannian_distances(vae, z0t, zjt).numpy()
    dist_e = np.linalg.norm(Z_train[neighbor_idx] - z0.reshape(1, -1), axis=1)

    x0_hat = decode_nodes_batch(vae, z0.reshape(1, -1), device=device)[0]

    # Index 0 = query, indices 1..N = train nodes, for both value sources.
    true_stack = np.vstack([x0.reshape(1, -1), X_train])
    dec_stack = np.vstack([x0_hat.reshape(1, -1), table.X_hat])
    unscaled_true = unscale_matrix(true_stack, scaler, feature_cols, feature_metadata)
    unscaled_dec = unscale_matrix(dec_stack, scaler, feature_cols, feature_metadata)

    src_idx = np.zeros(N, dtype=np.int64)
    tgt_idx = np.arange(1, N + 1, dtype=np.int64)
    breakdown = violation_breakdown(
        unscaled_true, unscaled_dec, src_idx, tgt_idx, feature_metadata, mode, taus, check_step_horizon=False
    )

    return DSRQueryEdges(
        neighbor_idx=neighbor_idx, dist_riemannian=dist_r, dist_euclidean=dist_e,
        breakdown=breakdown, x0_hat=x0_hat,
    )


def make_query_row(qedges: DSRQueryEdges, N: int, metric: str = "riemannian") -> sp.csr_matrix:
    dist = qedges.dist_riemannian if metric == "riemannian" else qedges.dist_euclidean
    keep = ~qedges.breakdown.any
    cols = qedges.neighbor_idx[keep]
    rows = np.zeros(len(cols), dtype=np.int64)
    return sp.csr_matrix((dist[keep], (rows, cols)), shape=(1, N))


def augmented_matrix(base_csr: sp.csr_matrix, query_row_csr: sp.csr_matrix) -> sp.csr_matrix:
    """Query node appended as row/col N: out-edges only, no in-edges."""
    N = base_csr.shape[0]
    top = sp.hstack([base_csr, sp.csr_matrix((N, 1))], format="csr")
    bottom = sp.hstack([query_row_csr, sp.csr_matrix((1, 1))], format="csr")
    return sp.vstack([top, bottom], format="csr")
