"""Phase 10: acts_on mapping, contraindication filter, skip current meds, escalation-tier ranking, precedent, recommendation cards.

For a recourse target x (tier-1 real endpoint or tier-2 counterfactual) of patient x0:
 1. required changes = mutable features with |x - x0| >= min_meaningful_change (feature_rules.csv);
 2. each change -> intervention classes whose acts_on/expected_direction match (knowledge graph);
 3. FILTER: contraindicated classes (inherited, with derivation) are removed; classes not indicated
    for this patient are removed; classes the patient already takes are skipped ("already taking");
 4. ESCALATION (lifestyle -> one drug -> combination, config.recommendation.escalation): per feature,
    allowed lifestyle classes are used first; if their summed typical effect (interventions.csv,
    additive approximation) does not cover the required change, the best single drug is added
    (first-line, then effect), then a second drug (combination). Items are ranked escalation step ->
    oracle risk drop when only the features the class addresses move -> number of features covered;
 5. card = ranked items + excluded (contraindicated / not indicated / already taking) + unmapped.
The `no_escalation` ablation ranks every allowed class by risk drop then coverage (CARE-N v2 style).
Decision support only, never orders.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from experiment_final.src.tier2 import intervention_map

TIERS = ["lifestyle", "single_drug", "combination"]


def required_changes(schema, rs, x0: np.ndarray, x: np.ndarray) -> List[dict]:
    """Clinically meaningful changes in the admissible direction (P10)."""
    out = []
    for f in schema.mutable:
        j = schema.idx(f)
        d = float(x[j] - x0[j])
        thr = max(rs.rules[f].meaningful, 1e-9)
        if abs(d) >= thr - 1e-9 and abs(d) > 1e-9:
            out.append({"feature": f, "from": float(x0[j]), "to": float(x[j]), "delta": d, "sign": "+" if d > 0 else "-"})
    return out


def effect_size(k, cls: str, concept: str, x0f: float) -> float:
    """Typical absolute effect of `cls` on `concept` at baseline x0f ('-25%' is relative) (P10)."""
    r = k.iv(cls)
    for c, e in zip(r["acts_on_features"].split(";"), r["typical_effect"].split(";")):
        if c == concept:
            e = e.strip()
            return abs(float(e[:-1]) / 100.0 * x0f) if e.endswith("%") else abs(float(e))
    return 0.0


def _partial_risk(risk_fn, x0, x, feats: Sequence[int]) -> float:
    y = x0.copy()
    y[list(feats)] = x[list(feats)]
    return float(risk_fn(y[None, :])[0])


def classify_classes(k, rs_filter, indicated: Dict[str, list], current: Set[str], apply_filter: bool = True) -> Dict[str, str]:
    """Class -> status: allowed | contraindicated | not_indicated | already_taking (P10)."""
    out = {}
    for c in k.classes():
        if apply_filter and c in rs_filter.contra_classes:
            out[c] = "contraindicated"
        elif not indicated.get(c):
            out[c] = "not_indicated"
        elif c in current:
            out[c] = "already_taking"
        else:
            out[c] = "allowed"
    return out


def build_card(schema, k, rs, x0: np.ndarray, x: np.ndarray, r0: float, risk_fn, indicated: Dict[str, list],
               current: Set[str], mode: str = "escalation", apply_filter: bool = True, top: int = 3,
               precedent: Optional[Dict[str, float]] = None) -> dict:
    """Steps 1-5 for one patient/target (P10)."""
    imap = intervention_map(k, schema)
    status = classify_classes(k, rs, indicated, current, apply_filter)
    changes = required_changes(schema, rs, x0, x)
    excluded, unmapped = [], []
    feat_classes: Dict[str, List[str]] = {}
    for ch in changes:
        cands = imap.get((ch["feature"], ch["sign"]), [])
        if not cands:
            unmapped.append({**ch, "reason": "no intervention class in the knowledge graph moves this feature in this direction"})
            continue
        ok = [c for c in cands if status[c] == "allowed"]
        for c in cands:
            if status[c] != "allowed":
                excluded.append({"class": c, "feature": ch["feature"], "reason": status[c],
                                 "derivation": [d for d in rs.derivations if d[1] == c] if status[c] == "contraindicated" else []})
        if ok:
            feat_classes[ch["feature"]] = ok
        else:
            unmapped.append({**ch, "reason": "every class acting on it is " + "/".join(sorted({status[c] for c in cands}))})
    need = {ch["feature"]: abs(ch["delta"]) for ch in changes}
    cls_feats: Dict[str, List[str]] = {}
    for f, cs in feat_classes.items():
        for c in cs:
            cls_feats.setdefault(c, []).append(f)

    def risk_drop(c):
        return r0 - _partial_risk(risk_fn, x0, x, [schema.idx(f) for f in cls_feats[c]])

    tier_of = {c: k.iv(c)["escalation_tier"] for c in cls_feats}
    drops = {c: risk_drop(c) for c in cls_feats}
    # escalation planner: per feature lifestyle -> 1 drug -> 2 drugs
    chosen, step_of, feat_step, residual = set(), {}, {}, {}
    for f, cs in feat_classes.items():
        conc, x0f = schema.concept[f], float(x0[schema.idx(f)])
        life = [c for c in cs if tier_of[c] == "lifestyle"]
        drugs = sorted([c for c in cs if tier_of[c] != "lifestyle"],
                       key=lambda c: (-int(k.iv(c)["first_line"]), -effect_size(k, c, conc, x0f), -drops[c], c))
        eff = sum(effect_size(k, c, conc, x0f) for c in life)
        for c in life:
            chosen.add(c); step_of.setdefault(c, "lifestyle")
        st = "lifestyle"
        for n_drug, c in enumerate(drugs[:2]):
            if eff >= need[f] - 1e-9:
                break
            chosen.add(c)
            st = "single_drug" if n_drug == 0 else "combination"
            step_of[c] = st if TIERS.index(st) > TIERS.index(step_of.get(c, "lifestyle")) else step_of.get(c, st)
            eff += effect_size(k, c, conc, x0f)
        feat_step[f] = st if (life or eff > 0) else "none"
        residual[f] = max(0.0, need[f] - eff)
    min_tier = max((TIERS.index(s) for s in feat_step.values() if s in TIERS), default=0)

    def item(c, step):
        r = k.iv(c)
        return {"class": c, "step": step, "escalation_tier": r["escalation_tier"], "features": sorted(cls_feats[c]),
                "risk_drop": drops[c], "citation": r["citation"], "verify": int(r["verify"]),
                "indication": indicated.get(c, []), "precedent_share_taking": (precedent or {}).get(c, np.nan)}

    if mode == "escalation":
        items = [item(c, step_of[c]) for c in chosen]
        items.sort(key=lambda d: (TIERS.index(d["step"]), -d["risk_drop"], -len(d["features"]), d["class"]))
    else:  # no_escalation ablation: rank all allowed by risk drop, coverage
        items = [item(c, k.iv(c)["escalation_tier"]) for c in cls_feats]
        items.sort(key=lambda d: (-d["risk_drop"], -len(d["features"]), d["class"]))
        items = items[:top]
    for i, it in enumerate(items):
        it["rank"] = i + 1
    rec_classes = [it["class"] for it in items]
    # appropriateness: non-decreasing escalation order AND no drug for a feature lifestyle alone covers
    order_ok = all(TIERS.index(a["escalation_tier"]) <= TIERS.index(b["escalation_tier"]) for a, b in zip(items, items[1:]))
    def lifestyle_covers(f):
        conc, x0f = schema.concept[f], float(x0[schema.idx(f)])
        return sum(effect_size(k, c, conc, x0f) for c in feat_classes.get(f, []) if tier_of[c] == "lifestyle") >= need[f] - 1e-9

    # overtreatment: a drug is recommended although allowed lifestyle classes alone cover EVERY feature it addresses
    over = any(it["escalation_tier"] != "lifestyle" and all(lifestyle_covers(f) for f in it["features"]) for it in items)
    return {"recommendations": items, "covered": bool(items), "n_changes": len(changes), "changes": changes,
            "excluded": excluded, "unmapped": unmapped, "min_sufficient_tier": TIERS[min_tier] if feat_step else None,
            "feature_step": feat_step, "residual_gap": residual,
            "escalation_appropriate": bool(items) and order_ok and not over, "overtreatment": over,
            "actionability": (sum(1 for ch in changes if ch["feature"] in feat_classes) / len(changes)) if changes else np.nan,
            "recommended_classes": rec_classes}


def precedent_table(k, schema, rs_cache, X_raw, contexts, meds, concepts) -> Dict[str, float]:
    """Share of train patients indicated for a class who already take it (descriptive) (P10)."""
    from experiment_final.src.knowledge_layer import indicated_classes
    if not any(meds):
        return {}
    num, den = {}, {}
    for i in range(len(X_raw)):
        rs = rs_cache(tuple(concepts[i]))
        ind = indicated_classes(rs, schema, k, dict(zip(schema.features, X_raw[i])), contexts[i])
        for c, toks in ind.items():
            if toks:
                den[c] = den.get(c, 0) + 1
                num[c] = num.get(c, 0) + int(c in meds[i])
    return {c: num.get(c, 0) / den[c] for c in den}
