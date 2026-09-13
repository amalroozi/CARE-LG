# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Phase 6 -- preference elicitation: per-patient reweighting of the clinical-
EFFORT cost term only, never touching admissibility/safety.

DISCREPANCY WITH THE BRIEF, documented per Hard Rule 5 (see CHANGES.md for
the full note): the brief's framing assumes an existing "C_clin edge-cost
term" inside the Dijkstra weight W_ij that Phase 6 can reweight per patient.
Reading `experiments_v6/lib/graph_build.py::make_cell_matrix` /
`make_query_row` shows this is not the case: W_ij in this codebase is
PURE latent-space distance (Riemannian or Euclidean), optionally penalized
by a raw constraint-violation COUNT in "soft" mode -- `compute_clinical_effort`
(src/graph/clinical_constraints.py) is computed only as a POST-HOC reporting
metric on the realized path, never as part of the traversal cost the search
actually optimizes. The most defensible interpretation, adopted here: add a
NEW, clearly-labeled additive cost term, used ONLY by this Phase 6 demo
(`run_preference_demo.py`), on top of the existing hard-mode edge SET
(admissibility, i.e. which edges exist at all, is completely unchanged --
still exactly the B1-fixed `violates` mask from Phase 1). Preferences can
only change WHICH admissible path is cheapest, never make an inadmissible
transition admissible.

    W_ij^pref = dist_metric(z_i, z_j) + mu * weighted_L1_effort(x_i, x_j; profile)

Synthetic preference profiles (illustrative, not derived from real patient
surveys -- documented as such): "no_preference" (baseline
CLINICAL_EFFORT_WEIGHTS, multiplier 1.0 everywhere), "exercise_averse"
(large multiplier on features most associated with lifestyle/exercise-driven
change), "medication_averse" (large multiplier on features most associated
with pharmacological change). The specific feature->category mapping below
is a simplification for demonstration purposes, stated plainly as such.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.sparse as sp

from src.graph.query_attachment import EdgeTable, QueryEdges

PROFILE_MULTIPLIER = 4.0  # how much a profile up-weights its "averse" category, vs. 1.0 baseline

# Illustrative, simplified feature->category mapping (documented above as such).
_EXERCISE_LINKED = {
    "uci": ["max_heart_rate", "oldpeak", "exercise_angina", "slope"],
    "nhanes_real": ["bmi", "glycemic_hba1c"],
}
_MEDICATION_LINKED = {
    "uci": ["cholesterol", "resting_bp"],
    "nhanes_real": ["cholesterol", "systolic_bp", "diastolic_bp"],
}


def preference_profiles(dataset: str, base_effort_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Returns {profile_name: {feature: weight}} for the three demo profiles."""
    base = dict(base_effort_weights)
    profiles = {"no_preference": dict(base)}

    exercise_averse = dict(base)
    for f in _EXERCISE_LINKED.get(dataset, []):
        if f in exercise_averse:
            exercise_averse[f] = exercise_averse[f] * PROFILE_MULTIPLIER
    profiles["exercise_averse"] = exercise_averse

    medication_averse = dict(base)
    for f in _MEDICATION_LINKED.get(dataset, []):
        if f in medication_averse:
            medication_averse[f] = medication_averse[f] * PROFILE_MULTIPLIER
    profiles["medication_averse"] = medication_averse

    return profiles


def _weighted_l1(X_i: np.ndarray, X_j: np.ndarray, profile_weights: Dict[str, float],
                  feature_metadata: dict, feature_cols) -> np.ndarray:
    """Vectorized weighted L1 over mutable columns, real (unscaled-agnostic -- same
    scaled-feature-space convention as compute_clinical_effort) feature space."""
    mutable_cols = feature_metadata['continuous_mutable'] + feature_metadata['categorical_mutable']
    cost = np.zeros(len(X_i), dtype=np.float64)
    for col in mutable_cols:
        if col not in feature_cols:
            continue
        idx = feature_cols.index(col)
        w = profile_weights.get(col, 1.0)
        cost += w * np.abs(X_j[:, idx] - X_i[:, idx])
    return cost


def make_cell_matrix_preference(edge_table: EdgeTable, X: np.ndarray, metric: str,
                                 profile_weights: Dict[str, float], feature_metadata: dict,
                                 feature_cols, mu: float) -> sp.csr_matrix:
    """Same admissible edge SET as hard mode (safety unchanged); cost = dist + mu*weighted_effort."""
    keep = ~edge_table.violates
    dist = edge_table.dist_riemannian if metric == "riemannian" else edge_table.dist_euclidean
    ei, ej = edge_table.edge_i[keep], edge_table.edge_j[keep]
    base_dist = dist[keep]
    effort = _weighted_l1(X[ei], X[ej], profile_weights, feature_metadata, feature_cols)
    w = base_dist + mu * effort
    return sp.csr_matrix((w, (ei, ej)), shape=(edge_table.N, edge_table.N))


def make_query_row_preference(qedges: QueryEdges, x0: np.ndarray, X: np.ndarray, metric: str,
                               profile_weights: Dict[str, float], feature_metadata: dict,
                               feature_cols, N: int, mu: float) -> sp.csr_matrix:
    keep = ~qedges.violates
    dist = qedges.dist_riemannian if metric == "riemannian" else qedges.dist_euclidean
    cols = qedges.neighbor_idx[keep]
    base_dist = dist[keep]
    x0_tiled = np.tile(x0.reshape(1, -1), (len(cols), 1))
    effort = _weighted_l1(x0_tiled, X[cols], profile_weights, feature_metadata, feature_cols)
    w = base_dist + mu * effort
    rows = np.zeros(len(cols), dtype=np.int64)
    return sp.csr_matrix((w, (rows, cols)), shape=(1, N))
