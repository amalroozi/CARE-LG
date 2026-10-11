"""Test: traceability -- every tag of every generated explanation resolves; tampering is caught (P11)."""
import json

import numpy as np
import pytest

from experiment_final.src.explain import explanation, trace
from experiment_final.src.knowledge_layer import indicated_classes, resolve_rule_set
from experiment_final.src.recommend import build_card
from experiment_final.tests.conftest import finished_seed_dirs


def _example(bundles, schemas, K):
    b, sch = bundles["nhanes"], schemas["nhanes"]
    X = b.X.to_numpy(float)
    i = next(i for i, c in enumerate(b.concepts) if {"E11", "J45.909"} <= set(c) and X[i, sch.idx("systolic_bp")] > 135)
    x0 = X[i]; tgt = x0.copy(); tgt[sch.idx("systolic_bp")] -= 15; tgt[sch.idx("bmi")] -= 1.0
    rs = resolve_rule_set(b.concepts[i], sch, K)
    ctx = b.context.iloc[i].to_dict()
    ind = indicated_classes(rs, sch, K, dict(zip(sch.features, x0)), ctx)
    card = build_card(sch, K, rs, x0, tgt, 0.9, lambda Z: np.full(len(Z), 0.4), ind, set(b.current_meds[i]))
    txt = explanation(2, sch, K, rs, x0, tgt, 0.9, 0.4, 0.45, {"age": 0.2, "systolic_bp": 0.1, "bmi": 0.05}, card, ctx)
    return txt, sch, rs, x0, tgt, card


def test_generated_explanation_is_fully_traceable(bundles, schemas, K):
    txt, sch, rs, x0, tgt, card = _example(bundles, schemas, K)
    assert "[RULE:F-SBP-DM]" in txt and "[NODE:E08-E13]" in txt and "[RULE:C-BB-J45]" in txt
    assert trace(txt, K, sch, rs, x0, tgt, [0.9, 0.4, 0.45], card, rs.contra_classes) == []


def test_tampering_is_detected(bundles, schemas, K):
    txt, sch, rs, x0, tgt, card = _example(bundles, schemas, K)
    bad = txt.replace("<risk=0.900>", "<risk=0.123>").replace("[RULE:F-SBP-DM]", "[RULE:F-SBP-XX]")
    bad += " [NODE:C73] [CITE:NOPE]"
    fails = trace(bad, K, sch, rs, x0, tgt, [0.9, 0.4, 0.45], card, rs.contra_classes)
    assert any("risk 0.123" in f for f in fails) and any("unknown rule" in f for f in fails)
    assert any("not active" in f or "C73" in f for f in fails) and any("citation" in f for f in fails)
    card2 = dict(card, recommendations=card["recommendations"] + [{"class": "beta_blocker_nonselective"}])
    assert any("contraindicated" in f for f in trace(txt, K, sch, rs, x0, tgt, [0.9, 0.4, 0.45], card2, rs.contra_classes))


def test_finished_run_explanations_trace_clean():
    dirs = [d for d in finished_seed_dirs() if (d / "explanations.jsonl").exists()]
    if not dirs:
        pytest.skip("no finished run yet")
    n = 0
    for d in dirs:
        for line in open(d / "explanations.jsonl"):
            e = json.loads(line)
            assert e["trace_failures"] == [], (d, e["test_pos"], e["trace_failures"])
            n += 1
    assert n > 0


def test_tracer_flags_contraindicated_recs_in_unfiltered_ablation():
    """In the ontology-as-features ablation (no symbolic filter) any contraindicated recommendation must be caught."""
    dirs = [d for d in finished_seed_dirs(include_variants=True) if "ontology_as_features" in d.parent.name]
    if not dirs:
        pytest.skip("no ontology_as_features run yet")
    for d in dirs:
        for line in open(d / "explanations.jsonl"):
            e = json.loads(line)
            assert all("contraindicated" in f for f in e["trace_failures"]), (d, e["test_pos"], e["trace_failures"])
