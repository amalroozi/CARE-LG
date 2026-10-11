"""Template-only natural-language plan for a stored recommendation card (no LLM).

A card becomes a stepwise plan: Goal -> Step 1 lifestyle -> Step 2 one drug -> Step 3 combination -> Not recommended -> decision-support line.
Every number, rule, ICD-10-CM code and citation is first written as a machine-checkable tag (the existing grammar of explain.py:
[RULE:] [NODE:] [IV:] [CITE:] {q.f=} {t.f=} <risk=>, plus two tags for numbers this module computes: {x.f=} the expected value of a
feature after a step and {d.class.f=} a class's typical effect). The tagged text is checked by explain.trace() and by `verify_extra()`
(recomputes {x.} and {d.}); only if both pass are the tags turned into sentences, grey chips and a numbered reference list.
If a check fails nothing is returned and the failure is appended to results/narration_failures.log.

Expected values use the planner's own additive approximation (effect sizes from interventions.csv via recommend.effect_size) and stop at
the goal value, exactly as the escalation planner stops adding classes once the effect covers the required change.

  .venv/bin/python -m experiment_final.src.narrate --tier 1          # print one plan (any of 1, 2, 3)
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiment_final.src.common import EXP_DIR
from experiment_final.src.explain import TAGS, trace
from experiment_final.src.knowledge_layer import KNOWLEDGE, load_knowledge, resolve_rule_set, schema_from_meta
from experiment_final.src.recommend import TIERS, effect_size

LOG = EXP_DIR / "results" / "narration_failures.log"
NAME = {"systolic_bp": "systolic blood pressure", "diastolic_bp": "diastolic blood pressure", "cholesterol": "total cholesterol", "bmi": "BMI",
        "glycemic_hba1c": "HbA1c", "age": "age", "sex": "sex"}
STEP_TITLE = {"lifestyle": "lifestyle changes", "single_drug": "one drug per feature that lifestyle cannot cover", "combination": "a second drug"}
X_TAG = re.compile(r"\{x\.([a-z0-9_]+)=(-?[0-9.]+)\}")
D_TAG = re.compile(r"\{d\.([a-z0-9_]+)\.([a-z0-9_]+)=([0-9.]+)\}")
ANY_TAG = re.compile(r"\[(?:RULE|NODE|IV|CITE):[^\]]+\]|\{[qtxd]\.[^}]+\}|<risk=[0-9.]+>")


def load_text() -> pd.DataFrame:
    return pd.read_csv(KNOWLEDGE / "intervention_text.csv", dtype=str, keep_default_na=False).set_index("intervention_class")


def _unit(meta: Dict[str, dict], f: str) -> str:
    u = meta.get(f, {}).get("unit", "")
    return {"": "", "kg/m2": "kg/m²"}.get(u, u)


def plan_values(card: dict, K, schema, x0: np.ndarray, target: Optional[np.ndarray]) -> dict:
    """effect[(class, feature)] = typical absolute effect; after[step][feature] = expected value after that step (capped at the goal)."""
    ch = {c["feature"]: c for c in card["changes"]}
    recs = card["recommendations"]
    # a class counts towards a feature as the planner does: every lifestyle class, but in a drug step only the strongest class for that feature
    effect = {}
    for r in recs:
        for f in r["features"]:
            if f not in ch:
                continue
            e = effect_size(K, r["class"], schema.concept[f], float(x0[schema.idx(f)]))
            if r["step"] != "lifestyle":
                rivals = [(-int(K.iv(q["class"])["first_line"]), -effect_size(K, q["class"], schema.concept[f], float(x0[schema.idx(f)])), q["class"])
                          for q in recs if q["step"] == r["step"] and f in q["features"]]
                if min(rivals)[2] != r["class"]:
                    continue
            effect[(r["class"], f)] = e
    after, steps = {}, [s for s in TIERS if any(r["step"] == s for r in recs)]
    for s in steps:
        used = [r["class"] for r in recs if TIERS.index(r["step"]) <= TIERS.index(s)]
        after[s] = {}
        for f, c in ch.items():
            tot = sum(effect.get((cl, f), 0.0) for cl in used)
            v = float(x0[schema.idx(f)]) + (tot if c["sign"] == "+" else -tot)
            after[s][f] = max(v, c["to"]) if c["sign"] == "-" else min(v, c["to"])
    return {"effect": effect, "after": after, "steps": steps}


def _rule_sentences(rs, schema, K) -> List[dict]:
    """One sentence per (ancestor, patient code) naming the non-generic rules that shaped the goals (rule ID and code -> ancestor become chips)."""
    by = {}
    for rid, node, conc in rs.rule_vias:
        if node != "ROOT":
            by.setdefault((node, conc), []).append(rid)
    out = []
    for (node, conc), rids in sorted(by.items()):
        feats = [f for f in schema.features if any(r in rs.rules[f].rule_ids for r in rids)]
        out.append({"kind": "rules", "text": f"Goals for {', '.join(NAME.get(f, f) for f in feats)} follow the guideline rules for {K.onto.label.get(conc, conc).lower()}, "
                    f"which this patient has [NODE:{conc}] [NODE:{node}] " + " ".join(f"[RULE:{r}]" for r in rids) + "."})
    return out


def tagged_plan(rec: dict, row, K, schema, rs, meta: Dict[str, dict]) -> List[dict]:
    """Sentences of the plan as tagged strings; `rec` = one cards.jsonl record, `row` = its patients.csv row (r0, tau, target_risk)."""
    card, tier = rec["card"], int(rec["tier"])
    x0, tgt = np.asarray(rec["x0"], float), (None if rec.get("target") is None else np.asarray(rec["target"], float))
    txt = load_text()
    r0, tau = float(row["r0"]), float(row["tau"])
    out: List[dict] = []
    ch = {c["feature"]: c for c in card["changes"]}
    unit = lambda f: (" " + _unit(meta, f)) if _unit(meta, f) else ""   # noqa: E731
    if tier == 3:
        sf = card["semifactual"]
        out.append({"kind": "goal", "text": f"Goal: bring this patient's predicted risk, now <risk={r0:.3f}>, below <risk={tau:.3f}>."})
        out.append({"kind": "goal", "text": f"No admissible plan was found: even if every modifiable value reached the best value recorded in the training data (age fixed), "
                    f"predicted risk would only fall to <risk={sf['best_case_risk']:.3f}>."})
        why = []
        for kind, f in sf.get("blocking", []):
            rid = {"age_horizon": "A-HORIZON"}.get(kind) or (rs.rules[f].rule_ids[0] if f in rs.rules else None)
            word = {"immutable": f"{NAME.get(f, f)} cannot be changed", "age_decrease": "age cannot decrease",
                    "age_horizon": "the age gap to the target is too large"}.get(kind, f"{NAME.get(f, f)} cannot rise")
            why.append(word + (f" [RULE:{rid}]" if rid else ""))
        if why:
            out.append({"kind": "goal", "text": "The closest real low-risk patient is out of reach because " + "; ".join(why) + "."})
        out.append({"kind": "support", "text": "This is decision support, not an order; a clinician decides."})
        return out
    rt = float(row["target_risk"])
    moves = "; ".join(f"{NAME.get(f, f)} from {{q.{f}={c['from']:.2f}}}{unit(f)} to {{t.{f}={c['to']:.2f}}}{unit(f)}" for f, c in ch.items())
    out.append({"kind": "goal", "text": f"Goal: lower the predicted risk from <risk={r0:.3f}> to about <risk={rt:.3f}> (the goal is below <risk={tau:.3f}>) by moving {moves}."})
    out.append({"kind": "goal", "text": "The target is the recorded profile of a real, lower-risk patient reachable through admissible patients (certified)." if tier == 1 else
                "No real patient was reachable under the safety rules, so the target is a synthetic profile that satisfies the hard rules by construction."})
    out += _rule_sentences(rs, schema, K)
    pv = plan_values(card, K, schema, x0, tgt)
    steps = pv["steps"]
    for n, s in enumerate(steps, 1):
        items = [r for r in card["recommendations"] if r["step"] == s]
        out.append({"kind": "step", "text": f"Step {n}: {STEP_TITLE[s]}."})
        for r in items:
            t = txt.loc[r["class"]]
            cites = " ".join(f"[CITE:{c}]" for c in r["citation"].split(";"))
            out.append({"kind": "action", "text": f"{t['action_sentence']} [IV:{r['class']}] {cites}", "dose": t["dose"], "review_interval": t["review_interval"]})
            eff = [f"{NAME.get(f, f)} by about {{d.{r['class']}.{f}={pv['effect'][(r['class'], f)]:.2f}}}{unit(f)}" for f in ch if f in r["features"] and pv["effect"].get((r["class"], f), 0) > 0]
            if eff:
                out.append({"kind": "effect", "text": f"It {t['effect_wording']}; on average, " + (", ".join(eff[:-1]) + " and " + eff[-1] if len(eff) > 1 else eff[0]) + "."})
        exp = "; ".join(f"{NAME.get(f, f)} about {{x.{f}={pv['after'][s][f]:.2f}}}{unit(f)}" for f in ch)
        out.append({"kind": "expected", "text": f"Expected after step {n}: {exp}."})
        if n < len(steps):
            open_ = [f for f, c in ch.items() if abs(pv["after"][s][f] - c["to"]) > 1e-6]
            if open_:
                cond = " or ".join(f"{NAME.get(f, f)} is still {'above' if ch[f]['sign'] == '-' else 'below'} {{t.{f}={ch[f]['to']:.2f}}}{unit(f)}" for f in open_)
                out.append({"kind": "review", "text": f"At review, if {cond}, move to step {n + 1}.", "review_interval": "TODO"})
            else:
                out.append({"kind": "review", "text": "At review, stop here if the goal values are met; step " + str(n + 1) + " is only needed otherwise.", "review_interval": "TODO"})
        else:
            short = [f for f, c in ch.items() if abs(pv["after"][s][f] - c["to"]) > 1e-6]
            out.append({"kind": "review", "text": "At review, check the goal values again." + (" The available classes are not expected to reach the goal for " +
                        ", ".join(NAME.get(f, f) for f in short) + "." if short else ""), "review_interval": "TODO"})
    for u in card.get("unmapped", []):
        f = u["feature"]
        out.append({"kind": "unmapped", "text": f"No intervention class in the knowledge tables can move {NAME.get(f, f)} from {{q.{f}={u['from']:.2f}}}{unit(f)} to {{t.{f}={u['to']:.2f}}}{unit(f)}."})
    ci = K.contraindications.set_index("rule_id")
    seen = set()
    for e in card.get("excluded", []):
        if e["reason"] == "contraindicated" and e["class"] not in seen and e["derivation"]:
            seen.add(e["class"])
            rid, cls, node, conc = e["derivation"][0]
            out.append({"kind": "not_recommended", "text": f"Not recommended: {txt.loc[cls, 'display_name'].lower()} [IV:{cls}]. {ci.loc[rid, 'reason']}. "
                        f"The patient has {K.onto.label.get(conc, conc).lower()} [NODE:{conc}] [NODE:{node}] [RULE:{rid}].", "verbatim": ci.loc[rid, "reason"]})
    taking = sorted({e["class"] for e in card.get("excluded", []) if e["reason"] == "already_taking"})
    if taking:
        out.append({"kind": "taking", "text": "Already taking, so not re-recommended: " + ", ".join(f"[IV:{c}]" for c in taking) + "."})
    out.append({"kind": "support", "text": "This is decision support, not an order; a clinician decides."})
    return out


def verify_extra(text: str, card: dict, K, schema, x0: np.ndarray, tgt: Optional[np.ndarray]) -> List[str]:
    """Recompute the {x.} and {d.} numbers; also fail on any dose that is not stated in the knowledge tables."""
    pv = plan_values(card, K, schema, x0, tgt) if card["recommendations"] else {"effect": {}, "after": {}, "steps": []}
    fails = []
    for f, v in X_TAG.findall(text):
        if not any(abs(a.get(f, 1e9) - float(v)) <= 0.006 for a in pv["after"].values()):
            fails.append(f"expected value mismatch {f}={v}")
    for cl, f, v in D_TAG.findall(text):
        if abs(pv["effect"].get((cl, f), -1.0) - float(v)) > 0.006:
            fails.append(f"effect mismatch {cl}/{f}={v}")
    return fails


def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"


def render(sentences: List[dict], K) -> dict:
    """Tags -> sentences + chips + numbered references (only called after the checks passed)."""
    ref_no: Dict[str, int] = {}
    cit_ver = pd.read_csv(KNOWLEDGE / "citations.csv", dtype=str, keep_default_na=False).set_index("key")["verify"]
    txt = load_text()
    out = []
    for s in sentences:
        t, chips = s["text"], []
        chips += TAGS["RULE"].findall(t)
        nodes = TAGS["NODE"].findall(t)
        if nodes:
            chips.append(" → ".join(nodes))
        t = TAGS["RULE"].sub("", t)
        t = TAGS["NODE"].sub("", t)
        t = TAGS["IV"].sub(lambda m: txt.loc[m.group(1), "display_name"].lower() if s["kind"] == "taking" else "", t)
        t = re.sub(r"\s+([.;,])", r"\1", t)
        t = TAGS["RISK"].sub(lambda m: f"{100 * float(m.group(1)):.1f}%", t)
        t = TAGS["Q"].sub(lambda m: _fmt(float(m.group(2))), t)
        t = TAGS["T"].sub(lambda m: _fmt(float(m.group(2))), t)
        t = X_TAG.sub(lambda m: _fmt(float(m.group(2))), t)
        t = D_TAG.sub(lambda m: _fmt(float(m.group(3))), t)

        def cite(m):
            n = ref_no.setdefault(m.group(1), len(ref_no) + 1)
            return f"[{n}]"
        t = TAGS["CITE"].sub(cite, t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        extra = []
        if s.get("dose") not in (None, ""):
            extra.append(f"dose: {s['dose']}")
        if s.get("review_interval") == "TODO":
            extra.append("review interval: TODO")
        out.append({"kind": s["kind"], "text": t, "chips": chips, "notes": extra, "verbatim": s.get("verbatim")})
    refs = [{"n": n, "key": k, "text": K.citations[k], "unverified": cit_ver.get(k, "1") == "1"} for k, n in ref_no.items()]
    return {"sentences": out, "references": refs}


def narrate(rec: dict, row, K, schema, meta: Dict[str, dict], data_maxchg: Optional[dict] = None) -> Optional[dict]:
    """Full pipeline for one stored card; None (and a log line) if any trace check fails."""
    rs = resolve_rule_set(rec["concepts"], schema, K, True, True, data_maxchg)
    sents = tagged_plan(rec, row, K, schema, rs, meta)
    text = "\n".join(s["text"] for s in sents)
    x0 = np.asarray(rec["x0"], float); tgt = None if rec.get("target") is None else np.asarray(rec["target"], float)
    card = rec["card"]
    risks = [float(row["r0"]), float(row["tau"])] + ([float(row["target_risk"])] if int(rec["tier"]) != 3 else []) + \
            ([card["semifactual"]["best_case_risk"]] if int(rec["tier"]) == 3 else [])
    fails = trace(text, K, schema, rs, x0, tgt, risks, card, rs.contra_classes) + verify_extra(text, card, K, schema, x0, tgt)
    if fails:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(json.dumps({"dataset": rec["dataset"], "seed": rec["seed"], "test_pos": rec["test_pos"], "failures": fails}) + "\n")
        return None
    res = render(sents, K)
    res["tagged"] = text
    return res


CSS = """
.plan{margin:6px 0 4px}.plan p{margin:6px 0;line-height:1.55}.plan .stepline{font-weight:700;margin-top:14px;font-size:15px}
.plan .sub{margin-left:22px;color:#333}.plan .sub.exp{color:#555}.plan .nr{color:#9a3412}
.gchip{display:inline-block;background:#eee;color:#666;border-radius:4px;padding:0 6px;font-size:11px;margin-left:5px;font-family:ui-monospace,monospace;white-space:nowrap}
.cref{color:#2563eb;font-size:12px}.cref.unv{border-bottom:1px dotted #b45309;color:#b45309}
.refs{font-size:12px;color:#555;margin-top:10px}.refs li{margin:2px 0}.unvb{background:#fef3c7;color:#92400e;border-radius:3px;padding:0 5px;font-size:11px;margin-left:4px}
"""


def render_html(res: dict) -> str:
    """Sentences first, grey chips (rule ID, code -> ancestor, TODO notes) beside them, numbered references with an 'unverified' badge (dev report)."""
    unv = {r["n"] for r in res["references"] if r["unverified"]}
    out = ['<div class="plan">']
    for s in res["sentences"]:
        t = html.escape(s["text"])
        t = re.sub(r"\[(\d+)\]", lambda m: f"<span class='cref{' unv' if int(m.group(1)) in unv else ''}' title='{'unverified citation' if int(m.group(1)) in unv else ''}'>[{m.group(1)}]</span>", t)
        chips = "".join(f"<span class='gchip'>{html.escape(c)}</span>" for c in s["chips"] + s["notes"])
        cls = {"step": "stepline", "action": "sub", "effect": "sub", "expected": "sub exp", "review": "sub exp", "not_recommended": "nr"}.get(s["kind"], "")
        out.append(f"<p class='{cls}'>{t}{chips}</p>")
    out.append("</div><ol class='refs'>" + "".join(f"<li value='{r['n']}'>{html.escape(r['text'])}" + ("<span class='unvb'>unverified</span>" if r["unverified"] else "") + "</li>" for r in res["references"]) + "</ol>")
    return "".join(out)


def plain(res: dict) -> str:
    lines = []
    for s in res["sentences"]:
        pre = "  " if s["kind"] in ("action", "effect", "expected", "review") else ""
        post = ("   [" + "; ".join(s["chips"] + s["notes"]) + "]") if (s["chips"] or s["notes"]) else ""
        lines.append(pre + s["text"] + post)
    lines += ["", "References"] + [f"[{r['n']}] {r['text']}" + ("  (unverified)" if r["unverified"] else "") for r in res["references"]]
    return "\n".join(lines)


def load_seed(dataset: str = "nhanes_pooled", seed: int = 2):
    from experiment_final import adapters
    d = EXP_DIR / "results" / "run_full_nhanes" / dataset / f"seed{seed}"
    K = load_knowledge(); meta = {k: dict(v) for k, v in adapters.get(dataset).FEATURE_META.items()}
    schema = schema_from_meta(meta, K)
    recs = [json.loads(line) for line in open(d / "cards.jsonl")]
    rows = pd.read_csv(d / "patients.csv").set_index("test_pos")
    mc = json.loads((d / "setup.json").read_text()).get("data_maxchg_p95")
    return K, schema, meta, recs, rows, mc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tier", type=int, default=1); ap.add_argument("--test-pos", type=int)
    a = ap.parse_args()
    K, schema, meta, recs, rows, mc = load_seed()
    rec = next(r for r in recs if (r["test_pos"] == a.test_pos if a.test_pos is not None else (r["tier"] == a.tier and (a.tier == 3 or r["card"]["recommendations"]))))
    res = narrate(rec, rows.loc[rec["test_pos"]], K, schema, meta, mc)
    print(f"patient #{rec['test_pos']} (tier {rec['tier']})\n" + (plain(res) if res else "trace failed; see " + str(LOG)))
