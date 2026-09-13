"""
Recourse search under EGD — Phase 2/3. Ties together k_shortest_targets (latent-distance
candidate ranking) with sequential decoding and the one check EGD does NOT provide
architecturally: risk reduction.

Algorithm, per query:
  1. Rank up to `max_candidates` low-risk-mask targets by latent-space Dijkstra distance
     (egd_graph.k_shortest_targets) -- this uses the graph's geometry only, unaffected by
     the fact that decode is path-dependent.
  2. For each candidate IN ORDER (nearest first): sequentially decode the path
     (query's true x0 -> hop1 -> hop2 -> ...) via egd.decode_path, then check the
     classifier's risk on the ACTUAL decoded terminal state.
  3. Return the FIRST candidate whose decoded terminal state is genuinely low-risk.
  4. If every candidate in the ranked list fails the risk check, ABSTAIN -- this is a
     real, distinct failure mode from "no path exists at all" (case: the graph found
     candidates, but none of their DECODED trajectories reduce risk enough), and is
     tracked separately (`abstain_stage='risk_check'` vs. `'no_path'`) for Phase 3's
     certified-infeasibility analysis, which needs to know whether the search itself
     ever produced a candidate to test.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from experiments_v5.lib.egd_decoder import ExactGuaranteeDecoder
from experiments_v5.lib.egd_graph import (
    augmented_matrix, build_egd_query_edges, k_shortest_targets, make_matrix, make_query_row,
)


@dataclass
class EGDSearchResult:
    success: bool
    abstain_stage: Optional[str]   # None | 'no_path' | 'risk_check'
    path: Optional[List[int]]
    decoded_states: Optional[List[np.ndarray]]   # hop 1..T decoded states (x0 not included)
    n_candidates_tried: int
    terminal_risk: Optional[float]


def find_recourse_egd(
    egd: ExactGuaranteeDecoder, classifier: torch.nn.Module, x0: np.ndarray, z0: np.ndarray,
    table, Z_train: np.ndarray, low_risk_mask: np.ndarray, low_risk_threshold: float = 0.45,
    metric: str = "riemannian", max_candidates: int = 20, device: torch.device = torch.device("cpu"),
) -> EGDSearchResult:
    N = table.N
    base = make_matrix(table, metric=metric)
    qe = build_egd_query_edges(egd, z0, x0, Z_train, device=device)
    aug = augmented_matrix(base, make_query_row(qe, N, metric=metric))
    target_mask = np.zeros(N + 1, dtype=bool)
    target_mask[:N] = low_risk_mask

    candidates = k_shortest_targets(aug, N, target_mask, max_candidates=max_candidates)
    if not candidates:
        return EGDSearchResult(success=False, abstain_stage="no_path", path=None, decoded_states=None,
                               n_candidates_tried=0, terminal_risk=None)

    x0_t = torch.tensor(x0, dtype=torch.float32, device=device)
    tried = 0
    for tgt, path, cost in candidates:
        tried += 1
        z_path = torch.tensor(Z_train[path[1:]], dtype=torch.float32, device=device)   # exclude source(=N)
        with torch.no_grad():
            states = egd.decode_path(z_path, x0_t)
            terminal_risk = float(classifier(states[-1].unsqueeze(0)).item())
        if terminal_risk < low_risk_threshold:
            return EGDSearchResult(success=True, abstain_stage=None, path=path,
                                   decoded_states=[s.cpu().numpy() for s in states],
                                   n_candidates_tried=tried, terminal_risk=terminal_risk)

    return EGDSearchResult(success=False, abstain_stage="risk_check", path=None, decoded_states=None,
                           n_candidates_tried=tried, terminal_risk=None)
