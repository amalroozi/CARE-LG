"""Phase 7: certified real-patient recourse: gate-symmetric attachment, SRA (src/graph/source_relative_admissibility.py), age-stratified targets, Dijkstra, real-value verification, certificate.

Per high-risk query x0 (recorded values) with oracle risk r0 and age-bracket threshold tau_p:
  targets T_p   = train patients with oracle risk < tau_p AND < r0 (a target must actually lower risk;
                  see DECISIONS.md D-TARGET) -- age-stratified per config, or global 0.45;
  entry gate    = full hard tier incl. horizon (B1: identical to interior edges);
  SRA           = search restricted to nodes individually admissible from x0 (reused
                  restrict_to_source_admissible), so EVERY path node, not just each hop, is admissible
                  from the patient's own baseline -> real-value violations are 0 by construction;
  Dijkstra      = src/recourse/search_augmented.py::find_path_augmented on the augmented graph;
  verification  = recorded-value re-check of every node vs x0 (B2) + clingo certificate.
Arms (ablations): metric riemannian|euclidean, sra on|off, soft_as_hard (endpoint soft rules as vetoes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.graph.query_attachment import augmented_matrix, make_query_row
from src.graph.source_relative_admissibility import restrict_to_source_admissible
from src.recourse.search_augmented import find_path_augmented
from experiment_final.src.knowledge_layer import hard_violates, soft_degrees


@dataclass
class Tier1Arm:
    name: str
    metric: str = "riemannian"
    sra: bool = True
    soft_as_hard: bool = False


def target_mask(train_risk: np.ndarray, tau_p: float, r0: float) -> np.ndarray:
    """Real low-risk-for-age targets that also lower the patient's own risk (P7)."""
    return (train_risk < tau_p) & (train_risk < r0)


def search(arm: Tier1Arm, base_csr: dict, qedges, N: int, tmask: np.ndarray, soft_ok: Optional[np.ndarray] = None):
    """One Dijkstra search under an arm; returns (path, cost) or (None, inf) (P7)."""
    qrow = make_query_row(qedges, arm.metric, "hard", N)
    keep = ~qedges.violates
    if arm.soft_as_hard and soft_ok is not None:
        keep = keep & soft_ok
    base = base_csr[arm.metric]
    if arm.sra:
        base = restrict_to_source_admissible(base, keep)
        tm = tmask & keep
    else:
        tm = tmask & (soft_ok if (arm.soft_as_hard and soft_ok is not None) else True)
    if not tm.any():
        return None, float("inf")
    aug = augmented_matrix(base, qrow)
    try:
        path, cost = find_path_augmented(aug, N, np.concatenate([tm, [False]]))
    except ValueError:
        return None, float("inf")
    return [int(n) for n in path[1:]], cost


def verify_path(schema, x0_raw: np.ndarray, X_raw: np.ndarray, path) -> dict:
    """Recorded-value verification (B2): every node vs source, every hop, endpoint (P7)."""
    P = X_raw[path]
    node_v = hard_violates(schema, x0_raw, P, horizon=True)
    chain = np.vstack([x0_raw[None, :], P])
    hop_v = hard_violates(schema, chain[:-1], chain[1:], horizon=True)
    end_v = hard_violates(schema, x0_raw, P[-1:], horizon=False)[0]
    return {"real_violation_any_node": bool(node_v.any()), "real_violation_any_hop": bool(hop_v.any()),
            "real_violation_endpoint": bool(end_v)}


def run_arm(arm: Tier1Arm, ctx: dict, qedges, tmask: np.ndarray, x0_raw: np.ndarray, r0: float, rs=None) -> dict:
    """Search + verification for one patient under one arm (P7)."""
    t0 = time.perf_counter()
    X_raw, N, schema = ctx["X_raw"], ctx["N"], ctx["schema"]
    soft_ok = None
    if arm.soft_as_hard and rs is not None:
        deg = soft_degrees(rs, schema, x0_raw, X_raw)
        soft_ok = (deg["magnitude"].sum(1) == 0) & (deg["target"].sum(1) == 0)
    path, cost = search(arm, ctx["base_csr"], qedges, N, tmask, soft_ok)
    rec = {"arm": arm.name, "tier1_success": path is not None, "latency_s": time.perf_counter() - t0}
    if path is None:
        return rec
    end = path[-1]
    rec.update({"path": path, "n_hops": len(path), "cost": cost, "endpoint": end,
                "endpoint_risk": float(ctx["train_risk"][end]), "risk_drop": float(r0 - ctx["train_risk"][end]),
                **verify_path(schema, x0_raw, X_raw, path)})
    return rec
