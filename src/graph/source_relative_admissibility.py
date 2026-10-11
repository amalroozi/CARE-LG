"""
Source-Relative Admissibility (SRA) -- fix for the compositional constraint-
drift vulnerability confirmed by scripts/diagnose_compositional_drift.py
(see results/tables/compositional_drift_diagnostic_summary.csv: under the
no_age/nhanes_real regime, 2-hop successful paths carried a 16.2% real
constraint-violation rate at the endpoint, i.e. AMONG a real graph that
NEVER admits an individually-violating edge, even though EVERY individual
hop passed).

Root cause: every directional constraint (BP, cholesterol, ...) is enforced
per EDGE with a fixed noise-tolerance epsilon (src/graph/clinical_constraints.py
check_soft_violations). Two hops that each move a feature by <= epsilon can
compose into a net (source-to-endpoint) delta > epsilon, which the existing
hard-mode edge pruning never catches, because it only ever checks
(previous_node -> candidate), never (original_source -> candidate).

Key simplification that makes this cheap to fix exactly (no need for a
general resource-constrained-shortest-path solver): every node on a CALG
path is a REAL, already-recorded patient with a FIXED feature vector -- not
an accumulating state that depends on the route taken to reach it. So the
cumulative drift from the query's true baseline to any node n is simply
(x_n - x_source), independent of which path reached n. That is EXACTLY the
same predicate build_query_edges() already computes for every node in the
graph (the query's full "entry gate" admissibility vector, B1-fixed,
check_step_horizon=True) -- so no new constraint-checking code is needed,
only a new way of USING what's already computed: restrict the entire
subgraph available to this query's search to nodes that are individually
admissible DIRECTLY from the query's own baseline, not merely from whatever
predecessor the search happens to route through.

Guarantee this gives: for any path returned by find_recourse_path_sra(),
every node n on the path (not just the endpoint) satisfies
violates(x_source, x_n) == False under the SAME full predicate
(hard + directional, check_step_horizon=True) already used for the
project's own entry gate and interior edges -- so real_cvr_endpoint is
provably False, not merely empirically observed to be False. Existing
pairwise (hop-to-hop) admissibility is left fully in place on top of this
(this is an ADDITIONAL restriction, not a replacement), preserving the
existing "each step is individually a plausible clinical transition"
property used by the explanation layer.

One deliberate side effect, stated plainly: the age-horizon-cap component of
the predicate (max 3.05 yrs) now applies to the TOTAL path (source to every
node), not per hop. This is a genuine tightening, not an accident -- the
original rationale for that cap ("a clinically realistic follow-up interval
for risk recalculation") only makes sense as a bound on the total elapsed
time the recourse plan spans, not on each individual hop while allowing
unlimited hops to accumulate additional years.
"""
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from src.graph.query_attachment import QueryEdges


def source_admissible_mask(qedges: QueryEdges, N: int) -> np.ndarray:
    """
    Boolean mask of length N: True for every train node individually
    admissible as a DIRECT one-hop jump from the query's own baseline
    (i.e. violates(x_source, x_n) == False under the full B1-fixed
    predicate). qedges.violates already computed this for the full,
    unrestricted node set (build_query_edges() attaches to every train
    node, not a k-restricted neighborhood) -- SRA simply reuses it as a
    graph-wide filter instead of only an entry-eligibility check.
    """
    mask = np.ones(N, dtype=bool)
    mask[qedges.neighbor_idx] = ~qedges.violates
    return mask


def restrict_to_source_admissible(base_csr: sp.csr_matrix, keep_mask: np.ndarray) -> sp.csr_matrix:
    """
    Zeroes out every edge touching a node NOT in keep_mask (as either
    endpoint), so the search graph handed to Dijkstra can never route
    through -- or terminate at -- a node that isn't directly admissible
    from the query's own baseline. base_csr is the (N, N) interior graph
    (before the query node is appended); keep_mask has length N.
    """
    base_csr = base_csr.tocoo()
    keep_edges = keep_mask[base_csr.row] & keep_mask[base_csr.col]
    restricted = sp.coo_matrix(
        (base_csr.data[keep_edges], (base_csr.row[keep_edges], base_csr.col[keep_edges])),
        shape=base_csr.shape,
    )
    return restricted.tocsr()


def find_path_sra(
    base_csr: sp.csr_matrix, qedges: QueryEdges, query_row_csr: sp.csr_matrix, N: int, target_mask: np.ndarray,
) -> Tuple[list, float]:
    """
    Drop-in replacement for src.recourse.search_augmented.find_path_augmented
    that additionally enforces source-relative admissibility (see module
    docstring). Every node on the returned path -- intermediate hops and the
    final target alike -- is guaranteed individually admissible from the
    query's own real baseline, not merely from its immediate predecessor.
    """
    from src.graph.query_attachment import augmented_matrix
    from src.recourse.search_augmented import find_path_augmented

    keep_mask = source_admissible_mask(qedges, N)
    restricted_base = restrict_to_source_admissible(base_csr, keep_mask)
    # target_mask has length N+1 (query node last, always False); apply the
    # same restriction to targets so an inadmissible node is never selected
    # as the endpoint even if some other pruning bug ever let an edge to it
    # survive.
    restricted_target_mask = target_mask.copy()
    restricted_target_mask[:N] &= keep_mask

    aug = augmented_matrix(restricted_base, query_row_csr)
    return find_path_augmented(aug, N, restricted_target_mask)
