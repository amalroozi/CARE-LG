"""Phase 11: SHAP, guideline derivation, semifactual, statin/BP annotations, four-part template + traceability.

Tag grammar (from archive/experiments_caren/src/explain.py, extended): [RULE:<id>] [NODE:<icd>] [IV:<class>]
[CITE:<key>] {q.<feature>=<value>} {t.<feature>=<value>} <risk=<p>> [TIER:<1|2|3>]. Every tag is
machine-checked by `trace()` against the knowledge tables, the patient's rule set, the card and the
computed numbers; a non-empty failure list means the explanation is not traceable.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

TAGS = {"RULE": re.compile(r"\[RULE:([A-Za-z0-9_\-]+)\]"), "NODE": re.compile(r"\[NODE:([A-Z0-9.\-]+)\]"),
        "IV": re.compile(r"\[IV:([a-z0-9_]+)\]"), "CITE": re.compile(r"\[CITE:([A-Za-z0-9_]+)\]"),
        "Q": re.compile(r"\{q\.([a-z0-9_]+)=(-?[0-9.]+)\}"), "T": re.compile(r"\{t\.([a-z0-9_]+)=(-?[0-9.]+)\}"),
        "RISK": re.compile(r"<risk=([0-9.]+)>")}


def _v(x: float) -> str:
    return f"{x:.2f}"


def _cites(s: str) -> str:
    return " ".join(f"[CITE:{c.strip()}]" for c in str(s).split(";") if c.strip())


def active_annotations(k, rs, context: Dict) -> List[Tuple[str, str, str, str]]:
    """(rule_id, node, description, citation) of non-blocking annotations active for this patient (P11)."""
    out = []
    for r in k.annotations.itertuples():
        if r.attached_icd10_node not in rs.active_nodes:
            continue
        cond = str(r.condition)
        if cond.startswith("ascvd_ge:"):
            v = (context or {}).get("ascvd_10yr_risk_pct")
            if v is None or float(v) < float(cond.split(":")[1]):
                continue
        out.append((r.rule_id, r.attached_icd10_node, r.description, r.citation))
    return out


def explanation(tier: int, schema, k, rs, x0, target, r0, r_target, tau, shap: Dict[str, float], card: dict,
                context: Dict, cert: dict = None) -> str:
    """Four-part template: why high risk / what to change / which intervention / which rules (P11)."""
    top = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:3] if shap else []
    why = ", ".join(f"{f} {{q.{f}={_v(x0[schema.idx(f)])}}}" for f, _ in top) or "n/a"
    parts = [f"WHY HIGH RISK: predicted risk <risk={r0:.3f}> exceeds 0.55; largest SHAP contributors: {why}."]
    if tier in (1, 2) and target is not None:
        ch = "; ".join(f"{c['feature']} {{q.{c['feature']}={_v(c['from'])}}} -> {{t.{c['feature']}={_v(c['to'])}}}" for c in card["changes"]) or "no clinically meaningful change"
        src = "a REAL recorded patient reachable through admissible real patients (certified)" if tier == 1 else "a SYNTHETIC counterfactual (hard rules enforced by construction)"
        parts.append(f"WHAT TO CHANGE [TIER:{tier}]: {ch}. Target is {src}; its predicted risk is <risk={r_target:.3f}> "
                     f"(goal: below <risk={tau:.3f}> = min(age-bracket low-risk cut-off, current risk)).")
    else:
        sf = card.get("semifactual", {})
        parts.append(f"WHAT TO CHANGE [TIER:3]: no admissible recourse found. Even if every modifiable feature reached its best "
                     f"recorded value (age frozen), predicted risk would be <risk={sf.get('best_case_risk', r0):.3f}> "
                     f"vs goal <risk={tau:.3f}>. Closest real low-risk patient is blocked by: "
                     + (", ".join(f"{kd} on {f}" for kd, f in sf.get('blocking', [])) or "n/a") + ".")
    ivs = []
    for it in card.get("recommendations", []):
        ivs.append(f"({it['rank']}) [IV:{it['class']}] ({it['step']}) for {', '.join(it['features'])} {_cites(it['citation'])}")
    ex = sorted({(e["class"], e["reason"]) for e in card.get("excluded", [])})
    ex_txt = "; ".join(f"[IV:{c}] {why_}" for c, why_ in ex)
    parts.append("WHICH INTERVENTION: " + ("; ".join(ivs) if ivs else "none mapped") + "."
                 + (f" Excluded: {ex_txt}." if ex_txt else "") + " Decision support only; clinician judgement required.")
    rules = []
    for f in schema.features:
        fr = rs.rules[f]
        for rid, node in zip(fr.rule_ids, fr.nodes):
            row = k.feature_rules.set_index("rule_id").loc[rid]
            if node == "ROOT":
                continue
            vias = [c for r_, n_, c in rs.rule_vias if r_ == rid and n_ == node]
            rules.append(f"[RULE:{rid}] at [NODE:{node}] via patient concept " + "/".join(f"[NODE:{c}]" for c in vias) + f" {_cites(row['citation'])}")
    for rid, cls, node, conc in rs.derivations:
        cit = k.contraindications.set_index("rule_id").loc[rid, "citation"]
        rules.append(f"[RULE:{rid}] blocks [IV:{cls}] at [NODE:{node}] via [NODE:{conc}] {_cites(cit)}")
    for rid, node, desc, cit in active_annotations(k, rs, context):
        if rid != "A-HORIZON":
            rules.append(f"[RULE:{rid}] {desc} {_cites(cit)}")
    hard = "[RULE:F-AGE] age frozen/non-decreasing; immutable features unchanged; directional features never worsen"
    parts.append("WHICH RULES APPLIED: " + ("; ".join(rules) if rules else "generic population rules only (no patient concepts)")
                 + f". Hard tier: {hard}" + (" (clingo certificate: all path nodes pass)." if cert and cert.get("all_hard_ok") else "."))
    return "\n".join(parts)


def trace(text: str, k, schema, rs, x0, target, risks: Sequence[float], card: dict, true_contra: set) -> List[str]:
    """Machine check of every tag; returns failures (empty = fully traceable) (P11)."""
    fails = []
    known_rules = set(k.feature_rules["rule_id"]) | set(k.contraindications["rule_id"]) | set(k.annotations["rule_id"])
    selected = {r for fr in rs.rules.values() for r in fr.rule_ids}
    for rid in TAGS["RULE"].findall(text):
        if rid not in known_rules:
            fails.append(f"unknown rule {rid}")
        elif rid.startswith("F-") and rid not in selected and rid != "F-AGE":
            fails.append(f"rule {rid} not in patient's rule set")
        elif rid.startswith("C-") and rid not in {r for r, _, _ in rs.contraindicated}:
            fails.append(f"contraindication {rid} not active")
    for n in TAGS["NODE"].findall(text):
        if n not in k.onto:
            fails.append(f"node {n} not in ICD-10-CM")
        elif n not in rs.active_nodes:
            fails.append(f"node {n} not active for patient")
    rec = {it["class"] for it in card.get("recommendations", [])}
    for c in TAGS["IV"].findall(text):
        if c not in set(k.classes()):
            fails.append(f"unknown intervention {c}")
    for c in rec & true_contra:
        fails.append(f"recommended {c} is contraindicated")
    for c in TAGS["CITE"].findall(text):
        if c not in k.citations:
            fails.append(f"unknown citation {c}")
    for f, v in TAGS["Q"].findall(text):
        if f not in schema.features or abs(float(v) - x0[schema.idx(f)]) > 0.006:
            fails.append(f"query value mismatch {f}={v}")
    for f, v in TAGS["T"].findall(text):
        if target is None or f not in schema.features or abs(float(v) - target[schema.idx(f)]) > 0.006:
            fails.append(f"target value mismatch {f}={v}")
    for p in TAGS["RISK"].findall(text):
        if not any(abs(float(p) - r) < 6e-4 for r in risks):
            fails.append(f"risk {p} not computed")
    return fails
