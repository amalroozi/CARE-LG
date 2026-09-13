"""
Graph construction and search under the Exact-Guarantee Decoder (EGD) — Phase 2.

Key structural difference from every prior session's graph code: EGD's decode is a
function of (z_target, x_source), not z alone (egd_decoder.py's module docstring). A
graph "node" therefore has no single canonical decoded value — its decoded value
depends on which path reached it. This module resolves that split cleanly:

  1. GRAPH STRUCTURE AND EDGE WEIGHTS (k-NN neighbours, Riemannian/Euclidean distance)
     are computed exactly as in every prior session, in pure LATENT space (encoder mu),
     using each edge's own TRUE source node value for the one-step Jacobian
     (`batched_riemannian_distances_egd`) — this is well-defined per edge, independent
     of path history, and is used ONLY to guide Dijkstra toward geometrically sensible
     candidate paths, exactly its role in every prior session.

  2. PRUNING no longer needs to check immutability or monotonicity at all — the decoder
     architecture makes violating those structurally impossible regardless of which
     edges exist. The graph is therefore UNPRUNED (every k-NN edge kept); the only
     thing edges cannot yet guarantee is RISK REDUCTION, which is not architectural and
     is checked post-hoc.

  3. RISK-BASED TARGET SELECTION AND VERIFICATION happens AFTER Dijkstra returns a
     ranked list of candidate paths, per query: each candidate is SEQUENTIALLY decoded
     (query's true x0 -> hop 1 -> hop 2 -> ...) via `egd.decode_path`, and the
     classifier is evaluated on the ACTUAL decoded terminal state (which may differ from
     the target node's own true recorded risk, since the decoded state is synthetic).
     The `low_risk_mask` used to pick Dijkstra's TARGET SET is still based on graph
     nodes' TRUE recorded values (exactly matching every prior session — this was never
     decoder-dependent even in v2/v3/v4), so it needs no change; only the VERIFICATION
     of whichever candidate is reached needs the new sequential-decode-then-classify
     step, since that is the one guarantee EGD does not provide architecturally.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import torch
from sklearn.neighbors import NearestNeighbors
from torch.func import jacrev, vmap

from experiments_v5.lib.egd_decoder import ExactGuaranteeDecoder


def batched_riemannian_distances_egd(
    egd: ExactGuaranteeDecoder, z_i: torch.Tensor, z_j: torch.Tensor, x_src: torch.Tensor,
    device: torch.device = torch.device("cpu"), chunk: int = 20000,
) -> torch.Tensor:
    """
    Riemannian distance under EGD: for each edge k, the local metric tensor is
    G = J^T J where J = d/dz [decode_step(z, x_src[k])] evaluated at z=(z_i[k]+z_j[k])/2,
    with x_src[k] HELD FIXED (edge k's own true source node value) -- the Jacobian is
    only ever taken w.r.t. the latent argument, matching the original riemannian_batch.py
    exactly except for this extra, non-differentiated conditioning argument.
    """
    egd = egd.to(device)
    egd.eval()
    zi, zj, xs = z_i.to(device), z_j.to(device), x_src.to(device)
    zmid = (zi + zj) / 2.0

    def decode_func(z, x_source):
        return egd.decode_step(z.unsqueeze(0), x_source.unsqueeze(0)).squeeze(0)

    batch_jac = vmap(jacrev(decode_func, argnums=0), in_dims=(0, 0))

    out = []
    with torch.no_grad():
        for s in range(0, len(zmid), chunk):
            zm, xsrc_chunk = zmid[s:s + chunk], xs[s:s + chunk]
            J = batch_jac(zm, xsrc_chunk)
            G = torch.einsum("bdi,bdj->bij", J, J)
            dz = (zj[s:s + chunk] - zi[s:s + chunk]).unsqueeze(-1)
            sq = torch.einsum("bki,bij,bjl->bkl", dz.transpose(1, 2), G, dz).squeeze(-1).squeeze(-1)
            out.append(torch.sqrt(torch.clamp(sq, min=0.0)).cpu())
    return torch.cat(out)


@dataclass
class EGDEdgeTable:
    N: int
    k: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray
    nbrs: NearestNeighbors


def build_egd_edge_table(egd: ExactGuaranteeDecoder, Z: np.ndarray, X: np.ndarray, k: int,
                         device: torch.device = torch.device("cpu")) -> EGDEdgeTable:
    """Builds the k-NN structure + both distance metrics. No pruning: EGD needs none."""
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
    Xt = torch.tensor(X, dtype=torch.float32)
    dist_r = batched_riemannian_distances_egd(egd, Zt[edge_i], Zt[edge_j], Xt[edge_i], device=device).numpy()
    dist_e = np.linalg.norm(Z[edge_i] - Z[edge_j], axis=1)

    return EGDEdgeTable(N=N, k=k, edge_i=edge_i, edge_j=edge_j, dist_riemannian=dist_r, dist_euclidean=dist_e, nbrs=nbrs)


def make_matrix(table: EGDEdgeTable, metric: str = "riemannian") -> sp.csr_matrix:
    """No pruning -- every k-NN edge is kept, since EGD needs no constraint-based masking."""
    dist = table.dist_riemannian if metric == "riemannian" else table.dist_euclidean
    return sp.csr_matrix((dist, (table.edge_i, table.edge_j)), shape=(table.N, table.N))


@dataclass
class EGDQueryEdges:
    neighbor_idx: np.ndarray
    dist_riemannian: np.ndarray
    dist_euclidean: np.ndarray


def build_egd_query_edges(egd: ExactGuaranteeDecoder, z0: np.ndarray, x0: np.ndarray, Z_train: np.ndarray,
                          device: torch.device = torch.device("cpu")) -> EGDQueryEdges:
    """Query attaches to EVERY train node (full attachment, matching v2-v4's documented rationale)."""
    N = len(Z_train)
    z0t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    zjt = torch.tensor(Z_train, dtype=torch.float32)
    x0t = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    dist_r = batched_riemannian_distances_egd(egd, z0t, zjt, x0t, device=device).numpy()
    dist_e = np.linalg.norm(Z_train - z0.reshape(1, -1), axis=1)
    return EGDQueryEdges(neighbor_idx=np.arange(N), dist_riemannian=dist_r, dist_euclidean=dist_e)


def make_query_row(qedges: EGDQueryEdges, N: int, metric: str = "riemannian") -> sp.csr_matrix:
    dist = qedges.dist_riemannian if metric == "riemannian" else qedges.dist_euclidean
    rows = np.zeros(N, dtype=np.int64)
    return sp.csr_matrix((dist, (rows, qedges.neighbor_idx)), shape=(1, N))


def augmented_matrix(base_csr: sp.csr_matrix, query_row_csr: sp.csr_matrix) -> sp.csr_matrix:
    N = base_csr.shape[0]
    top = sp.hstack([base_csr, sp.csr_matrix((N, 1))], format="csr")
    bottom = sp.hstack([query_row_csr, sp.csr_matrix((1, 1))], format="csr")
    return sp.vstack([top, bottom], format="csr")


def k_shortest_targets(aug_matrix: sp.csr_matrix, source_idx: int, target_mask: np.ndarray,
                       max_candidates: int = 20) -> List[Tuple[int, List[int], float]]:
    """
    Returns up to `max_candidates` (target_idx, path, cost) tuples, nearest first, among
    nodes in target_mask reachable from source_idx. Unlike a single-shortest-path search,
    this gives the caller (search_egd.py) a ranked candidate list to sequentially decode
    and risk-check in order, since risk reduction is not architecturally guaranteed and
    the nearest-in-latent-distance candidate can fail that check.
    """
    distances, predecessors = csgraph.dijkstra(csgraph=aug_matrix, directed=True, indices=source_idx,
                                               return_predecessors=True)
    target_indices = np.where(target_mask)[0]
    target_distances = distances[target_indices]
    finite = np.isfinite(target_distances)
    if not np.any(finite):
        return []

    order = np.argsort(target_distances[finite])
    ranked_targets = target_indices[finite][order][:max_candidates]

    results = []
    for tgt in ranked_targets:
        path = []
        curr = int(tgt)
        while curr != source_idx and curr != -9999:
            path.append(curr)
            curr = predecessors[curr]
        if curr != source_idx:
            continue
        path.append(source_idx)
        path.reverse()
        results.append((int(tgt), path, float(distances[tgt])))
    return results
