"""Phase 6: CALG k-NN latent graph, rule-pruned edges (reuse src/graph/query_attachment.py).

Reused unchanged: the EdgeTable/QueryEdges containers, make_cell_matrix, make_query_row,
augmented_matrix (src/graph/query_attachment.py) and batched_riemannian_distances
(src/graph/riemannian_batch.py). Re-implemented: only the per-edge violation predicate, which here is
the integer-scaled CARE-LG HARD tier (src/knowledge_layer.py::hard_violates) evaluated on RECORDED values,
so that (a) per-dataset noise tolerances come from the adapter (src/graph/constraints_ext.py's
_DIRECTIONAL_EPS has no entries for cardio's binary features, and its 1.0 default for 'increase'
would let `active` drop 1 -> 0 unflagged) and (b) graph, entry gate, SRA and clingo share one predicate.
Entry gate == interior predicate (CARE-LG B1): full hard tier with the age-horizon component.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.graph.query_attachment import EdgeTable, QueryEdges
from src.graph.riemannian_batch import batched_riemannian_distances
from experiment_final.src.knowledge_layer import Schema, hard_components


def build_edge_table(vae, Z: np.ndarray, X_raw: np.ndarray, schema: Schema, k: int) -> EdgeTable:
    """k-NN edges on latent means, Riemannian + Euclidean weights, hard-tier pruning flags (P6)."""
    N = len(Z)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm="ball_tree").fit(Z)
    _, ind = nbrs.kneighbors(Z)
    ei = np.repeat(np.arange(N), ind.shape[1])
    ej = ind.reshape(-1)
    keep = ei != ej
    ei, ej = ei[keep].astype(np.int64), ej[keep].astype(np.int64)
    Zt = torch.tensor(Z, dtype=torch.float32)
    d_r = batched_riemannian_distances(vae, Zt[ei], Zt[ej]).numpy()
    d_e = np.linalg.norm(Z[ei] - Z[ej], axis=1)
    v = np.empty(len(ei), dtype=np.int32)
    for a in range(0, len(ei), 1_000_000):   # chunked to bound memory on large cohorts; identical result
        comp = hard_components(schema, X_raw[ei[a:a + 1_000_000]], X_raw[ej[a:a + 1_000_000]], horizon=True)
        v[a:a + 1_000_000] = sum(comp.values())
    return EdgeTable(N=N, k=k, edge_i=ei, edge_j=ej, dist_riemannian=d_r, dist_euclidean=d_e,
                     violates=v > 0, v_count=v, nbrs=nbrs)


def query_edges(vae, z0: np.ndarray, x0_raw: np.ndarray, Z: np.ndarray, X_raw: np.ndarray, schema: Schema) -> QueryEdges:
    """Full attachment of one query to every train node (CARE-LG D8) with the B1 gate (P6/P7)."""
    N = len(Z)
    z0t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    d_r = batched_riemannian_distances(vae, z0t, torch.tensor(Z, dtype=torch.float32)).numpy()
    d_e = np.linalg.norm(Z - z0.reshape(1, -1), axis=1)
    comp = hard_components(schema, x0_raw, X_raw, horizon=True)
    v = sum(comp.values()).astype(np.int32)
    return QueryEdges(neighbor_idx=np.arange(N), dist_riemannian=d_r, dist_euclidean=d_e, violates=v > 0, v_count=v)


def edge_change_percentile(et: EdgeTable, X_raw: np.ndarray, schema: Schema, pct: float) -> dict:
    """p-th percentile of |delta| over admissible real CALG edges per mutable feature (config soft_change_percentile) (P2/P6)."""
    ok = ~et.violates
    d = np.abs(X_raw[et.edge_j[ok]] - X_raw[et.edge_i[ok]])
    return {f: float(np.percentile(d[:, schema.idx(f)], pct)) for f in schema.mutable}
