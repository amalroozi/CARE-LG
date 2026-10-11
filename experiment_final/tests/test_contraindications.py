"""Test: contraindications -- inherited down ICD-10-CM, never recommended, removed per patient (P2/P10)."""
import json

import numpy as np
import pandas as pd
import pytest

from experiment_final.src.knowledge_layer import indicated_classes, resolve_rule_set
from experiment_final.src.recommend import build_card
from experiment_final.tests.conftest import finished_seed_dirs


def test_inheritance_activates_block_level_contraindications(schemas, K):
    sch = schemas["nhanes"]
    rs = resolve_rule_set(["J45.909", "N18.6", "Z99.2", "E10.9"], sch, K)
    blocked = rs.contra_classes
    assert {"beta_blocker_nonselective", "metformin", "sglt2i"} <= blocked
    der = {(c, n, conc) for _, c, n, conc in rs.derivations}
    assert ("beta_blocker_nonselective", "J45", "J45.909") in der      # leaf -> category inheritance
    assert ("sglt2i", "E10", "E10.9") in der


def test_without_inheritance_leaf_codes_miss_parent_rules(schemas, K):
    sch = schemas["nhanes"]
    rs = resolve_rule_set(["J45.909", "E11"], sch, K, inheritance=False)
    assert "beta_blocker_nonselective" not in rs.contra_classes
    assert rs.rules["systolic_bp"].rule_ids == ("F-SBP",)               # E11 alone does not reach E08-E13
    rs2 = resolve_rule_set(["J45.909", "E11"], sch, K, inheritance=True)
    assert rs2.rules["systolic_bp"].rule_ids == ("F-SBP-DM",)


def test_card_never_recommends_contraindicated_or_current_meds(bundles, schemas, K):
    b, sch = bundles["nhanes"], schemas["nhanes"]
    X = b.X.to_numpy(float)
    idx = [i for i, c in enumerate(b.concepts) if "J45.909" in c and X[i, sch.idx("systolic_bp")] > 140][:15]
    assert idx
    for i in idx:
        rs = resolve_rule_set(b.concepts[i], sch, K)
        x0 = X[i]
        tgt = x0.copy()
        for f in sch.mutable:
            tgt[sch.idx(f)] -= 0.2 * abs(x0[sch.idx(f)])
        ind = indicated_classes(rs, sch, K, dict(zip(sch.features, x0)), b.context.iloc[i].to_dict())
        card = build_card(sch, K, rs, x0, tgt, 0.9, lambda Z: np.full(len(Z), 0.5), ind, set(b.current_meds[i]))
        rec = set(card["recommended_classes"])
        assert not (rec & rs.contra_classes)
        assert not (rec & set(b.current_meds[i]))
        assert "beta_blocker_nonselective" not in rec


def test_finished_runs_main_arm_has_zero_contraindicated_cards():
    dirs = [d for d in finished_seed_dirs() if (d / "arms.csv").stat().st_size > 1]
    if not dirs:
        pytest.skip("no finished run yet")
    for d in dirs:
        a = pd.read_csv(d / "arms.csv")
        assert a[a.arm == "main"].contraindicated_recs.sum() == 0, d
