"""Test: adapter_contract (DATASET_ADAPTER.md; PIPELINE P1)."""
import copy

import numpy as np
import pandas as pd
import pytest

from experiment_final import adapters
from experiment_final.adapters.base import AdapterContractError, DatasetBundle, validate_bundle
from experiment_final.src.knowledge_layer import schema_from_meta
from experiment_final.src.prep import make_split


def test_every_configured_adapter_returns_a_valid_bundle(bundles, cfg):
    assert set(bundles) == set(cfg["datasets"])
    for name, b in bundles.items():
        validate_bundle(b)
        assert b.name == name and len(b.X) > 100
        assert set(b.y.unique()) <= {0, 1}
        for f, m in b.feature_meta.items():
            assert m["class"] in ("immutable", "non_decreasing", "mutable") and m["direction"]


def test_no_concept_datasets_degrade_to_generic_rules(bundles):
    for name in ("uci", "cardio"):
        b = bundles[name]
        assert not b.has_ontology and all(c == [] for c in b.concepts) and all(m == [] for m in b.current_meds)


def test_ontology_datasets_have_concepts_and_meds(bundles):
    for name in ("nhanes", "nhanes_pooled"):
        b = bundles[name]
        assert b.has_ontology
        assert np.mean([bool(c) for c in b.concepts]) > 0.5
        assert np.mean([bool(m) for m in b.current_meds]) > 0.2


def test_nhanes_row_order_matches_nhanes_real_csv(bundles):
    slim = pd.read_csv(adapters.get("nhanes").REPO / "data" / "nhanes_real.csv")
    assert np.allclose(bundles["nhanes"].X.to_numpy(float), slim[list(bundles["nhanes"].X.columns)].to_numpy(float))


def test_pooled_2017_2020_slice_reproduces_existing_cohort(bundles):
    b = bundles["nhanes_pooled"]
    full = pd.read_csv(adapters.get("nhanes").REPO / "data" / "nhanes_real_full.csv").set_index("SEQN")
    s = b.context["cycle"] == "2017-2020"
    X = b.X[s.to_numpy()].set_index(b.patient_id[s].to_numpy()).loc[full.index.astype(int)]
    assert len(X) == len(full) == 4539
    assert np.allclose(X.to_numpy(float), full[list(b.X.columns)].to_numpy(float))
    y = pd.Series(b.y[s.to_numpy()].to_numpy(), index=b.patient_id[s].to_numpy()).loc[full.index.astype(int)]
    assert (y.to_numpy() == full["cvd_risk_flag"].to_numpy()).all()


def test_missing_class_or_direction_is_a_fail_safe_stop(bundles):
    b = copy.deepcopy(bundles["uci"])
    del b.feature_meta["cholesterol"]["direction"]
    with pytest.raises(AdapterContractError):
        validate_bundle(b)


def test_feature_without_knowledge_rule_is_a_fail_safe_stop(bundles, K):
    meta = copy.deepcopy(bundles["uci"].feature_meta)
    meta["cholesterol"]["concept"] = "no_such_concept"
    with pytest.raises(ValueError):
        schema_from_meta(meta, K)


def test_new_dataset_runs_with_one_adapter(K):
    """A toy adapter (no concepts) passes the contract, gets a schema and a split -- nothing else to write."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"age": rng.integers(40, 80, n).astype(float), "sex": rng.integers(0, 2, n).astype(float),
                      "systolic_bp": rng.normal(135, 15, n)})
    meta = {"age": dict(type="continuous", **{"class": "non_decreasing"}, direction="none", unit="y", phys_min=18, phys_max=100, concept="age"),
            "sex": dict(type="binary", **{"class": "immutable"}, direction="none", unit="", phys_min=0, phys_max=1, concept="sex"),
            "systolic_bp": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=70, phys_max=250, concept="systolic_bp")}
    b = DatasetBundle(X=X, y=pd.Series((X.systolic_bp > 135).astype(int)), patient_id=pd.Series(range(n)), feature_meta=meta,
                      context=pd.DataFrame({"i": range(n)}), concepts=[[] for _ in range(n)], current_meds=[[] for _ in range(n)],
                      subgroups={"sex": X.sex}, name="toy", label_definition="toy", citation="toy")
    validate_bundle(b)
    sch = schema_from_meta(b.feature_meta, K)
    sp = make_split(b, 0)
    assert sch.mutable == ["systolic_bp"] and len(sp.te) == 60


def test_eicu_demo_stub_is_explicit():
    with pytest.raises(NotImplementedError):
        adapters.get("eicu_demo").load({})
