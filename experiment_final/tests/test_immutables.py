"""Test: immutables -- tier-2 construction never moves immutables/age and keeps directional features one-way (P9);
tier-1 results of finished runs carry zero real-value violations (P7)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from experiment_final.src.knowledge_layer import hard_violates, indicated_classes, resolve_rule_set
from experiment_final.src.model import fit_mlp
from experiment_final.src.prep import mad_vector, make_split
from experiment_final.src.tier2 import GenConfig, Problem, describe, generate_ga
from experiment_final.tests.conftest import finished_seed_dirs


@pytest.mark.parametrize("name", ["nhanes", "uci", "cardio"])
def test_ga_counterfactuals_respect_hard_tier(bundles, schemas, K, name):
    b, sch = bundles[name], schemas[name]
    sp_ = make_split(b, 0)
    orc = fit_mlp(sp_.Xtr[sp_.fit], sp_.ytr[sp_.fit], 0)
    mu, sd = sp_.Xtr_raw.mean(0), np.where(sp_.Xtr_raw.std(0) > 0, sp_.Xtr_raw.std(0), 1)
    knn = NearestNeighbors(n_neighbors=5).fit((sp_.Xtr_raw - mu) / sd)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit((sp_.Xtr_raw - mu) / sd)
    risk = orc.predict(sp_.Xte)
    for i in np.argsort(-risk)[:8]:
        x0 = sp_.Xte_raw[i]
        rs = resolve_rule_set(b.concepts[sp_.te[i]], sch, K)
        ind = indicated_classes(rs, sch, K, dict(zip(sch.features, x0)), {})
        P = Problem(x0, float(risk[i]), 0.45, sch, rs, K, orc, sp_, knn, mu, sd, mad_vector(sp_.Xtr_raw),
                    [c for c, t in ind.items() if t], sorted(rs.contra_classes), GenConfig(pop=40, gens=10))
        g = generate_ga(P, np.random.default_rng(int(i)))
        X = P.full(g["Xm"])
        for f in sch.features:
            j = sch.idx(f)
            if sch.kind[f] in ("equal", "non_decreasing"):
                assert (X[:, j] == x0[j]).all()
            elif sch.kind[f] == "reduce":
                assert (X[:, j] <= x0[j] + 1e-9).all()
            elif sch.kind[f] == "increase":
                assert (X[:, j] >= x0[j] - 1e-9).all()
            if sch.ftype[f] != "continuous":
                assert np.allclose(X[:, j], np.round(X[:, j]))
        assert not hard_violates(sch, x0, X, horizon=True).any()
        assert not any(d["hard_violation"] for d in describe(P, g["Xm"], lof))


def test_finished_runs_have_zero_tier1_real_violations():
    dirs = [d for d in finished_seed_dirs() if (d / "patients.csv").stat().st_size > 1]
    if not dirs:
        pytest.skip("no finished run yet")
    for d in dirs:
        p = pd.read_csv(d / "patients.csv")
        t1 = p[p.tier == 1]
        assert not t1.real_violation_any_node.astype(bool).any(), d
        assert not t1.real_violation_endpoint.astype(bool).any(), d
        assert (t1.cert_ok == True).all(), d  # noqa: E712
        assert (p.tier1_success.astype(bool) == (p.tier1_one_hop_targets > 0)).all(), d  # SRA identity
