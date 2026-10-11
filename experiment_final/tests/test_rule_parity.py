"""Test: rule_parity -- NumPy mirror == clingo, 100% (PIPELINE P2), plus agreement with the CARE-LG src gate."""
import numpy as np

from experiment_final.src.knowledge_layer import (check_clingo, hard_set, resolve_rule_set, soft_degrees, soft_set)


def _parity(b, sch, K, n_src=60, n_cand=20, seed=0):
    X = b.X.to_numpy(float)
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(X), n_src, replace=False):
        for inh in (True, False):
            rs = resolve_rule_set(b.concepts[i], sch, K, inheritance=inh)
            C = X[rng.choice(len(X), n_cand)]
            cr = check_clingo(b.concepts[i], sch, K, X[i], C, inheritance=inh)
            hs, deg = hard_set(sch, X[i], C, True), soft_degrees(rs, sch, X[i], C)
            assert all(cr.hard[j] == hs[j] for j in range(n_cand))
            assert all(cr.soft[j] == soft_set(deg, sch, j) for j in range(n_cand))
            assert {(r, f) for f, fr in rs.rules.items() for r in fr.rule_ids} == cr.selected
            assert all(cr.eff[f] == (rs.rules[f].lo, rs.rules[f].hi) for f in sch.features)
            assert set(rs.contraindicated) == cr.contraindicated
            assert set(rs.derivations) == cr.blocked_via and set(rs.rule_vias) == cr.via


def test_parity_all_datasets(bundles, schemas, K):
    for name in bundles:
        _parity(bundles[name], schemas[name], K)


def test_parity_targets_patients_with_inherited_rules(bundles, schemas, K):
    b, sch = bundles["nhanes"], schemas["nhanes"]
    idx = [i for i, c in enumerate(b.concepts) if {"E11", "J45.909"} <= set(c)][:10]
    assert idx
    X = b.X.to_numpy(float)
    for i in idx:
        rs = resolve_rule_set(b.concepts[i], sch, K)
        cr = check_clingo(b.concepts[i], sch, K, X[i], X[:5])
        assert ("F-SBP-DM", "systolic_bp") in cr.selected or any(r.startswith("F-SBP-") for r, _ in cr.selected)
        key = ("C-BB-J45", "beta_blocker_nonselective", "J45")
        assert key in cr.contraindicated and key in rs.contraindicated   # J45.909 -> J45 by inheritance, both paths


def test_gate_matches_carelg_src_gate_except_exact_boundaries(bundles, schemas):
    """src/graph/query_attachment.py::vectorized_violation_components (float32 unscale) vs the CARE-LG integer gate."""
    import sys
    from sklearn.preprocessing import StandardScaler
    from src.graph.query_attachment import unscale_matrix, vectorized_violation_components
    from src.graph.constraints_ext import _DIRECTIONAL_EPS
    from experiment_final.src.knowledge_layer import hard_violates
    from experiment_final.src.prep import legacy_feature_metadata
    for name in ("nhanes", "uci"):
        b, sch = bundles[name], schemas[name]
        fm = legacy_feature_metadata(b)
        cols = list(b.X.columns)
        cont = fm["continuous_mutable"] + fm["non_decreasing"]
        X = b.X.to_numpy(float)
        sc = StandardScaler().fit(X[:, [cols.index(c) for c in cont]])
        Xs = X.copy(); Xs[:, [cols.index(c) for c in cont]] = sc.transform(X[:, [cols.index(c) for c in cont]])
        rng = np.random.default_rng(1)
        i, j = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
        un = unscale_matrix(np.vstack([Xs[i], Xs[j]]).astype(np.float32), sc, cols, fm)
        h, d = vectorized_violation_components(un, np.arange(len(i)), np.arange(len(i), 2 * len(i)), fm, True)
        ref = (h + d) > 0
        mine = hard_violates(sch, X[i], X[j], horizon=True)
        dis = np.where(ref != mine)[0]
        # any disagreement must sit exactly on a tolerance boundary (|delta| == eps or the 3.05 y horizon)
        for k in dis:
            delta = X[j[k]] - X[i[k]]
            on_edge = [abs(abs(delta[cols.index(f)]) - _DIRECTIONAL_EPS.get(f, 0.1)) < 1e-6 for f in fm["directional_reduce"] + fm["directional_increase"]]
            assert any(on_edge), (name, k, delta)
        assert len(dis) / len(i) < 0.01
