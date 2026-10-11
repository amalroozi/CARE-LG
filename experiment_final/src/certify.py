"""Phase 8: exhaustive 1-hop + 2-hop checks, certified infeasible vs graph-limited, ceiling diagnostic, closest miss, blocking-rule frequency.

Under SRA a tier-1 path may only visit nodes admissible from x0, so its endpoint is itself a 1-hop
admissible target; with full query attachment Dijkstra always sees that direct edge. Hence
tier-1 failure  <=>  no target in T_p is admissible from x0 (exhaustively checked here over ALL train
patients, not the k-NN graph). "Graph-limited" failures are therefore impossible under SRA; we still
run the exhaustive 2-hop check of scripts/run_multihop_check.py (full population as intermediates,
hop-wise admissibility) to quantify what a non-SRA search could add (and at what drift).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np

from experiment_final.src.knowledge_layer import hard_components, hard_set, hard_violates


def exhaustive_one_hop(schema, x0_raw, X_raw, tmask) -> dict:
    """Admissible targets from x0 with and without the age-horizon component (P8)."""
    T = np.where(tmask)[0]
    if len(T) == 0:
        return {"n_targets": 0, "n_admissible_targets": 0, "n_admissible_ignoring_horizon": 0}
    a_h = ~hard_violates(schema, x0_raw, X_raw[T], horizon=True)
    a_nh = ~hard_violates(schema, x0_raw, X_raw[T], horizon=False)
    return {"n_targets": int(len(T)), "n_admissible_targets": int(a_h.sum()), "n_admissible_ignoring_horizon": int(a_nh.sum())}


def exhaustive_two_hop(schema, x0_raw, X_raw, tmask, max_pairs: int = 2_000_000) -> dict:
    """Is there a real chain x0 -> m -> t with both hops admissible (hop-wise, full population)? (P8)."""
    R1 = np.where(~hard_violates(schema, x0_raw, X_raw, horizon=True))[0]
    T = np.where(tmask)[0]
    if len(R1) == 0 or len(T) == 0:
        return {"two_hop_chain_exists": False, "two_hop_endpoint_admissible_from_source": False, "n_r1": int(len(R1))}
    found, sra_ok = False, False
    step = max(1, max_pairs // max(1, len(T)))
    for s in range(0, len(R1), step):
        M = R1[s:s + step]
        src = np.repeat(X_raw[M], len(T), axis=0)
        tgt = np.tile(X_raw[T], (len(M), 1))
        ok = ~hard_violates(schema, src, tgt, horizon=True)
        if ok.any():
            found = True
            ends = T[np.unique(np.where(ok.reshape(len(M), len(T)))[1])]
            sra_ok = bool((~hard_violates(schema, x0_raw, X_raw[ends], horizon=True)).any())
            break
    return {"two_hop_chain_exists": found, "two_hop_endpoint_admissible_from_source": sra_ok, "n_r1": int(len(R1))}


def best_case_corner(schema, x0_raw: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
    """Age/immutables frozen, each directional feature at its best REAL observed value (scripts/run_age_ceiling_diagnostic.py) (P8)."""
    xb = x0_raw.copy()
    for f in schema.mutable:
        j = schema.idx(f)
        if schema.kind[f] == "reduce":
            xb[j] = min(x0_raw[j], X_raw[:, j].min())
        elif schema.kind[f] == "increase":
            xb[j] = max(x0_raw[j], X_raw[:, j].max())
    return xb


def closest_miss(schema, x0_raw, X_raw, tmask, qdist: np.ndarray) -> dict:
    """Target with fewest blocking (kind, feature) pairs, ties by Riemannian distance; blocking-rule frequency (P8)."""
    T = np.where(tmask)[0]
    if len(T) == 0:
        return {"closest_miss": None, "blocking_frequency": {}}
    comps = hard_components(schema, x0_raw, X_raw[T], horizon=True)
    total = sum(comps.values())
    order = np.lexsort((qdist[T], total))
    j = T[order[0]]
    blocks = sorted(hard_set(schema, x0_raw, X_raw[j:j + 1], True)[0])
    freq: Counter = Counter()
    for s in hard_set(schema, x0_raw, X_raw[T], True):
        freq.update({f"{k}:{f}" for k, f in s})
    return {"closest_miss": {"train_pos": int(j), "n_blocking": int(total[order[0]]), "blocking": blocks,
                             "deltas": {f: float(X_raw[j, schema.idx(f)] - x0_raw[schema.idx(f)]) for _, f in blocks}},
            "blocking_frequency": {k: v / len(T) for k, v in freq.most_common()}}


def certify_failure(schema, x0_raw, X_raw, tmask, qdist, oracle_risk_fn, tau_eff: float, two_hop: bool = True) -> Dict:
    """Full P8 record for one tier-1 failure."""
    rec = exhaustive_one_hop(schema, x0_raw, X_raw, tmask)
    rec["certified_infeasible_real"] = rec["n_admissible_targets"] == 0
    rec["blocked_only_by_horizon"] = rec["n_admissible_targets"] == 0 and rec["n_admissible_ignoring_horizon"] > 0
    if two_hop:
        rec.update(exhaustive_two_hop(schema, x0_raw, X_raw, tmask))
    xb = best_case_corner(schema, x0_raw, X_raw)
    rb = float(oracle_risk_fn(xb[None, :])[0])
    rec.update({"best_case_risk": rb, "ceiling_category": "age_ceiling" if rb >= tau_eff else "data_scarcity"})
    rec.update(closest_miss(schema, x0_raw, X_raw, tmask, qdist))
    return rec
