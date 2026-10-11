"""Test: the natural-language plans (src/narrate.py) are fully traceable: every number, rule, code and citation resolves to the card or the knowledge tables."""
import re

import numpy as np
import pytest

from experiment_final.src import narrate as N
from experiment_final.src.explain import TAGS

SEED_DIR = N.EXP_DIR / "results" / "run_full_nhanes" / "nhanes_pooled" / "seed2"
pytestmark = pytest.mark.skipif(not (SEED_DIR / "DONE").exists(), reason="full NHANES rerun not present")


@pytest.fixture(scope="module")
def ctx():
    return N.load_seed("nhanes_pooled", 2)


def _sample(recs):
    out = []
    for tier in (1, 2, 3):
        pool = [r for r in recs if r["tier"] == tier and (tier == 3 or r["card"]["recommendations"])]
        out += pool[:: max(1, len(pool) // 40)][:40]
    return out


def test_every_sampled_card_passes_trace(ctx):
    K, sch, meta, recs, rows, mc = ctx
    for r in _sample(recs):
        assert N.narrate(r, rows.loc[r["test_pos"]], K, sch, meta, mc) is not None, r["test_pos"]


def test_every_number_in_the_text_comes_from_a_tag(ctx):
    K, sch, meta, recs, rows, mc = ctx
    allowed_prose = {"1", "2", "3"}                                   # "Step n" and the step numbers in review sentences
    for r in _sample(recs)[:60]:
        res = N.narrate(r, rows.loc[r["test_pos"]], K, sch, meta, mc)
        card = r["card"]
        tagged = res["tagged"]
        nums = {N._fmt(float(m)) for pat in (TAGS["Q"], TAGS["T"]) for _, m in pat.findall(tagged)}
        nums |= {N._fmt(float(m[1])) for m in N.X_TAG.findall(tagged)} | {N._fmt(float(m[2])) for m in N.D_TAG.findall(tagged)}
        nums |= {f"{100 * float(m):.1f}" for m in TAGS["RISK"].findall(tagged)}
        for s in res["sentences"]:
            t = s["text"]
            for ref in re.findall(r"\[\d+\]", t):
                t = t.replace(ref, "")
            for v in (s.get("verbatim"),):
                if v:
                    t = t.replace(v, "")
            for d in N.load_text()["dose"]:
                t = t.replace(d, "")
            t = re.sub(r"%", "", re.sub(r"Step \d+|step \d+", "", t))
            for n in re.findall(r"\d+\.?\d*", t):
                assert n in nums or n in allowed_prose, (r["test_pos"], n, t)


def test_every_citation_exists_and_is_numbered(ctx):
    K, sch, meta, recs, rows, mc = ctx
    for r in _sample(recs)[:60]:
        res = N.narrate(r, rows.loc[r["test_pos"]], K, sch, meta, mc)
        keys = set(TAGS["CITE"].findall(res["tagged"]))
        assert keys <= set(K.citations)
        assert {x["key"] for x in res["references"]} == keys
        assert sorted(x["n"] for x in res["references"]) == list(range(1, len(keys) + 1))
        for s in res["sentences"]:
            assert not TAGS["CITE"].search(s["text"])


def test_contraindicated_classes_are_never_recommended(ctx):
    K, sch, meta, recs, rows, mc = ctx
    txt = N.load_text()
    for r in _sample(recs):
        res = N.narrate(r, rows.loc[r["test_pos"]], K, sch, meta, mc)
        rs = N.resolve_rule_set(r["concepts"], sch, K, True, True, mc)
        for s in res["sentences"]:
            if s["kind"] == "action":
                assert not any(txt.loc[c, "action_sentence"] == s["text"].split(" [")[0] for c in rs.contra_classes)


def test_tampered_numbers_fail_and_print_nothing(ctx, tmp_path, monkeypatch):
    K, sch, meta, recs, rows, mc = ctx
    r = next(r for r in recs if r["tier"] == 1 and r["card"]["recommendations"])
    monkeypatch.setattr(N, "LOG", tmp_path / "fail.log")
    orig = N.tagged_plan

    def tamper(*a, **k):
        return [dict(x, text=re.sub(r"\{q\.([a-z0-9_]+)=(-?[0-9.]+)\}", lambda m: "{q.%s=%.2f}" % (m.group(1), float(m.group(2)) + 1), x["text"], count=1)) for x in orig(*a, **k)]
    monkeypatch.setattr(N, "tagged_plan", tamper)
    assert N.narrate(r, rows.loc[r["test_pos"]], K, sch, meta, mc) is None
    assert (tmp_path / "fail.log").read_text().strip()


def test_review_interval_and_dose_stay_todo_unless_in_knowledge():
    t = N.load_text()
    assert (t["review_interval"] == "TODO").all() and (t["verify"] == "1").all()
    assert set(t.index[t["dose"] != "TODO"]) <= {"weight_management", "statin"}
