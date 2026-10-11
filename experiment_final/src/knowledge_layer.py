"""Phase 2: patient concepts -> rule set (hard/soft/filter) with inheritance; clingo certificates + NumPy mirror.

Tiers (PIPELINE P2; DECISIONS.md D-RULES):
  HARD    the CARE-LG admissibility predicate (src/graph/clinical_constraints.py): immutables equal,
          age never decreases (0.1 y tol) and (tier 1) total horizon <= 3.05 y, directional features never
          worsen beyond the per-feature noise tolerance. Defined by the adapter's feature_meta.
  SOFT    patient-specific guideline target range (resolved down the ICD-10-CM hierarchy: deepest
          attached node wins, envelope on ties, blank node fields inherit the ROOT row) and the
          max-plausible-change magnitude. Penalised with a degree, never a veto (except soft_as_hard).
  FILTER  contraindicated intervention classes (inherited down the hierarchy) with derivations, plus
          indications (concept / above-target / PCE-risk tokens).
All comparisons use integers scaled x1000 (`si`) in BOTH the NumPy mirror and clingo, so parity is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import clingo
import numpy as np
import pandas as pd

from experiment_final.src.common import KNOWLEDGE

SCALE = 1000
ROOT = "ROOT"
AGE_DECREASE_TOL = 0.1      # src/graph/constraints_ext.py::_AGE_DECREASE_TOL
AGE_HORIZON_CAP = 3.05      # src/graph/constraints_ext.py::_AGE_HORIZON_CAP
DEFAULT_EPS = {"reduce": 0.1, "increase": 1.0}   # constraints_ext defaults for unlisted features
HARD_KINDS = ["immutable", "age_decrease", "age_horizon", "direction"]
SOFT_KINDS = ["target", "magnitude"]


def si(x) -> np.ndarray:
    """Scale floats to x1000 integers; identical rounding in NumPy and clingo paths (P2)."""
    return np.round(np.asarray(x, dtype=float) * SCALE).astype(np.int64)


def _num(x) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "" else float(x)


# ------------------------------------------------------------------ tables

@dataclass
class Knowledge:
    """The reviewed CSV tables + ontology (P2)."""
    feature_rules: pd.DataFrame
    interventions: pd.DataFrame
    contraindications: pd.DataFrame
    annotations: pd.DataFrame
    citations: Dict[str, str]
    onto: object

    def root_row(self, concept: str) -> Optional[pd.Series]:
        if not hasattr(self, "_root_cache"):
            fr = self.feature_rules
            self._root_cache = {c: g.iloc[0] for c, g in fr[fr["applies_to_icd10_node"] == ROOT].groupby("feature")}
        return self._root_cache.get(concept)

    def rows_for(self, concept: str) -> List[dict]:
        """All feature-rule rows of one concept as dicts (cached; speed only) (P2)."""
        if not hasattr(self, "_rows_cache"):
            self._rows_cache = {c: g.to_dict("records") for c, g in self.feature_rules.groupby("feature")}
        return self._rows_cache.get(concept, [])

    def classes(self) -> List[str]:
        return list(self.interventions["intervention_class"])

    def iv(self, cls: str) -> pd.Series:
        return self.interventions.set_index("intervention_class").loc[cls]


def _keys(s: str) -> List[str]:
    return [k.strip() for k in str(s).split(";") if k.strip()]


def validate_knowledge(k: Knowledge) -> None:
    """Fail-safe: unique ids, every row cited with resolvable keys, nodes exist in ICD-10-CM (P2)."""
    fr = k.feature_rules
    assert fr["rule_id"].is_unique and k.contraindications["rule_id"].is_unique and k.annotations["rule_id"].is_unique
    for df in (fr, k.interventions, k.contraindications, k.annotations):
        for c in df["citation"]:
            missing = [x for x in _keys(c) if x not in k.citations]
            if pd.isna(c) or missing:
                raise ValueError(f"uncited or unresolved citation key(s) {missing} in a knowledge row")
    nodes = set(fr["applies_to_icd10_node"]) | set(k.contraindications["contraindicated_icd10_node"]) | set(k.annotations["attached_icd10_node"])
    bad = [n for n in nodes if n != ROOT and n not in k.onto]
    if bad:
        raise ValueError(f"knowledge rows attached to nodes missing from ICD-10-CM: {bad}")
    for c in set(fr["feature"]):
        if k.root_row(c) is None:
            raise ValueError(f"FAIL-SAFE: concept '{c}' lacks a ROOT row")


@lru_cache(maxsize=1)
def load_knowledge() -> Knowledge:
    """Load + validate knowledge/*.csv and the ICD-10-CM ontology (P2)."""
    from experiment_final.knowledge.ontology import get_ontology
    rd = lambda n: pd.read_csv(KNOWLEDGE / n, dtype=str, keep_default_na=False)  # noqa: E731
    k = Knowledge(rd("feature_rules.csv"), rd("interventions.csv"), rd("contraindications.csv"), rd("annotations.csv"),
                  dict(zip(*[rd("citations.csv")[c] for c in ("key", "reference")])), get_ontology())
    validate_knowledge(k)
    return k


# ------------------------------------------------------------------ dataset schema -> hard rules

@dataclass(frozen=True)
class Schema:
    """Per-dataset feature schema derived from the adapter's feature_meta (P2/P3)."""
    features: Tuple[str, ...]
    concept: Dict[str, str]
    kind: Dict[str, str]          # equal | non_decreasing | reduce | increase | none
    eps: Dict[str, float]
    ftype: Dict[str, str]
    phys: Dict[str, Tuple[float, float]]

    @property
    def mutable(self) -> List[str]:
        return [f for f in self.features if self.kind[f] in ("reduce", "increase", "none")]

    def idx(self, f: str) -> int:
        return self.features.index(f)


def schema_from_meta(feature_meta: Dict[str, dict], k: Knowledge) -> Schema:
    """Adapter meta -> Schema; fail-safe if a concept lacks a ROOT rule or directions disagree (P2)."""
    kind, eps, concept, ftype, phys = {}, {}, {}, {}, {}
    for f, m in feature_meta.items():
        c = m.get("concept", f)
        root = k.root_row(c)
        if root is None:
            raise ValueError(f"FAIL-SAFE: feature '{f}' (concept '{c}') has no ROOT row in feature_rules.csv")
        kd = {"immutable": "equal", "non_decreasing": "non_decreasing"}.get(m["class"], m["direction"])
        if kd in ("reduce", "increase") and root["direction"] != kd:
            raise ValueError(f"FAIL-SAFE: adapter direction '{kd}' for '{f}' disagrees with knowledge '{root['direction']}'")
        kind[f], concept[f], ftype[f] = kd, c, m["type"]
        eps[f] = float(m.get("noise_tol", DEFAULT_EPS.get(kd, 0.0)))
        phys[f] = (float(m["phys_min"]), float(m["phys_max"]))
    return Schema(tuple(feature_meta), concept, kind, eps, ftype, phys)


# ------------------------------------------------------------------ patient rule set

@dataclass(frozen=True)
class FeatureRule:
    feature: str
    rule_ids: Tuple[str, ...]
    nodes: Tuple[str, ...]
    lo: Optional[int]
    hi: Optional[int]
    maxchg: Optional[int]
    meaningful: float


@dataclass
class PatientRuleSet:
    """Rules active for one source patient (P2)."""
    concepts: FrozenSet[str]
    active_nodes: FrozenSet[str]
    rules: Dict[str, FeatureRule]
    contraindicated: Tuple[Tuple[str, str, str], ...]        # (rule_id, class, node)
    derivations: Tuple[Tuple[str, str, str, str], ...]        # (rule_id, class, node, concept)
    rule_vias: Tuple[Tuple[str, str, str], ...]               # (rule_id, node, concept)
    inheritance: bool = True

    @property
    def contra_classes(self) -> Set[str]:
        return {c for _, c, _ in self.contraindicated}


def _contra_rows(k: "Knowledge") -> Tuple[Tuple[str, str, str], ...]:
    if not hasattr(k, "_contra_cache"):
        ci = k.contraindications
        k._contra_cache = tuple(zip(ci["rule_id"], ci["intervention_class"], ci["contraindicated_icd10_node"]))
    return k._contra_cache


def active_nodes(concepts: Iterable[str], onto, inheritance: bool = True) -> Set[str]:
    """Concepts + (with inheritance) all ancestors + ROOT (P2, C2)."""
    concepts = [c for c in concepts if c in onto]
    return onto.closure(concepts) if inheritance else ({ROOT} | set(concepts))


def resolve_rule_set(concepts: Sequence[str], schema: Schema, k: Knowledge, inheritance: bool = True,
                     use_node_rules: bool = True, data_maxchg: Optional[Dict[str, float]] = None) -> PatientRuleSet:
    """Patient concepts -> active rules; `use_node_rules=False` = generic population rules (P2)."""
    concepts = frozenset(c for c in concepts if c in k.onto)
    act = active_nodes(concepts, k.onto, inheritance)
    fr = k.feature_rules
    rules: Dict[str, FeatureRule] = {}
    vias: Set[Tuple[str, str, str]] = set()
    for f in schema.features:
        c = schema.concept[f]
        root = k.root_row(c)
        rows = [r for r in k.rows_for(c) if r["applies_to_icd10_node"] in act
                and (use_node_rules or r["applies_to_icd10_node"] == ROOT)]
        depth = [0 if r["applies_to_icd10_node"] == ROOT else k.onto.depth(r["applies_to_icd10_node"]) for r in rows]
        sel = [r for r, d in zip(rows, depth) if d == max(depth)]
        los = [_num(r["target_low"]) if _num(r["target_low"]) is not None else _num(root["target_low"]) for r in sel]
        his = [_num(r["target_high"]) if _num(r["target_high"]) is not None else _num(root["target_high"]) for r in sel]
        los, his = [v for v in los if v is not None], [v for v in his if v is not None]
        mc = _num(root["max_plausible_change"])
        if mc is None and data_maxchg:
            mc = data_maxchg.get(f)
        rules[f] = FeatureRule(f, tuple(r["rule_id"] for r in sel), tuple(r["applies_to_icd10_node"] for r in sel),
                               int(si(min(los))) if los else None, int(si(max(his))) if his else None,
                               None if mc is None else int(si(mc)), float(_num(root["min_meaningful_change"]) or 0.0))
        for rid, node in ((r["rule_id"], r["applies_to_icd10_node"]) for r in sel):
            if node == ROOT:
                vias.add((rid, ROOT, ROOT))
                continue
            for cc in concepts:
                if (inheritance and node in k.onto.ancestors(cc)) or (not inheritance and node == cc):
                    vias.add((rid, node, cc))
    contra = tuple(sorted((r, c, n) for r, c, n in _contra_rows(k) if n in act))
    derivs = set()
    for r, c, n in contra:
        for cc in concepts:
            if (inheritance and n in k.onto.ancestors(cc)) or (not inheritance and n == cc):
                derivs.add((r, c, n, cc))
    return PatientRuleSet(concepts, frozenset(act), rules, contra, tuple(sorted(derivs)), tuple(sorted(vias)), inheritance)


# ------------------------------------------------------------------ indications (FILTER tier)

def indicated_classes(rs: PatientRuleSet, schema: Schema, k: Knowledge, x0: Dict[str, float],
                      context: Optional[Dict[str, float]] = None) -> Dict[str, List[str]]:
    """Class -> list of satisfied indication tokens (empty list = not indicated) (P2/P10).

    Tokens: always | concept:<node> (node in active set, i.e. inherited) | above_target:<concept>
    (baseline outside the patient-specific target in the improvable direction) | ascvd_ge:<pct>.
    If no token is assessable on this dataset, `indication_fallback` decides.
    """
    by_concept = {schema.concept[f]: f for f in schema.features}
    out = {}
    for r in k.interventions.itertuples():
        ok, assessable = [], False
        for tok in str(r.indication).split("|"):
            kind, _, arg = tok.partition(":")
            if kind == "always":
                ok.append("always"); assessable = True
            elif kind == "concept":
                if rs.concepts or k.onto is None:
                    assessable = True
                if arg in rs.active_nodes and arg != ROOT:
                    ok.append(tok)
            elif kind == "above_target" and arg in by_concept:
                assessable = True
                f = by_concept[arg]; fr = rs.rules[f]; v = int(si(x0[f]))
                if (schema.kind[f] == "reduce" and fr.hi is not None and v > fr.hi) or \
                   (schema.kind[f] == "increase" and fr.lo is not None and v < fr.lo):
                    ok.append(tok)
            elif kind == "ascvd_ge" and context and context.get("ascvd_10yr_risk_pct") is not None:
                assessable = True
                if float(context["ascvd_10yr_risk_pct"]) >= float(arg):
                    ok.append(tok)
        if not assessable and str(r.indication_fallback) == "1":
            ok.append("fallback:not_assessable")
        out[r.intervention_class] = ok
    return out


# ------------------------------------------------------------------ NumPy mirror (vectorised)

def hard_components(schema: Schema, src: np.ndarray, tgt: np.ndarray, horizon: bool) -> Dict[str, np.ndarray]:
    """kind -> (n,) int counts of hard violations between unscaled src rows and tgt rows (P2/P6/P7).

    src: (n, F) or (F,) unscaled clinical values; tgt: (n, F). Integer x1000 arithmetic.
    """
    S = si(np.broadcast_to(src, tgt.shape)); T = si(tgt)
    n = len(T)
    out = {kd: np.zeros(n, dtype=np.int64) for kd in HARD_KINDS}
    for j, f in enumerate(schema.features):
        kd, d = schema.kind[f], T[:, j] - S[:, j]
        if kd == "equal":
            out["immutable"] += d != 0
        elif kd == "non_decreasing":
            out["age_decrease"] += d < -int(si(AGE_DECREASE_TOL))
            if horizon:
                out["age_horizon"] += d > int(si(AGE_HORIZON_CAP))
        elif kd == "reduce":
            out["direction"] += d > int(si(schema.eps[f]))
        elif kd == "increase":
            out["direction"] += d < -int(si(schema.eps[f]))
    return out


def hard_violates(schema: Schema, src: np.ndarray, tgt: np.ndarray, horizon: bool) -> np.ndarray:
    """(n,) bool: any hard violation (the B1-fixed full predicate when horizon=True) (P6/P7)."""
    return sum(hard_components(schema, src, tgt, horizon).values()) > 0


def hard_set(schema: Schema, src: np.ndarray, tgt: np.ndarray, horizon: bool) -> List[Set[Tuple[str, str]]]:
    """Per-candidate set of (kind, feature) hard violations (parity with clingo `hard/3`)."""
    S = si(src); T = si(np.atleast_2d(tgt))
    res = []
    for t in T:
        s = set()
        for j, f in enumerate(schema.features):
            kd, d = schema.kind[f], int(t[j] - S[j])
            if kd == "equal" and d != 0: s.add(("immutable", f))
            if kd == "non_decreasing" and d < -int(si(AGE_DECREASE_TOL)): s.add(("age_decrease", f))
            if kd == "non_decreasing" and horizon and d > int(si(AGE_HORIZON_CAP)): s.add(("age_horizon", f))
            if kd == "reduce" and d > int(si(schema.eps[f])): s.add(("direction", f))
            if kd == "increase" and d < -int(si(schema.eps[f])): s.add(("direction", f))
        res.append(s)
    return res


def soft_degrees(rs: PatientRuleSet, schema: Schema, src: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
    """kind -> (n, F) int64 soft-rule degrees (x1000 units; 0 = satisfied) (P2/P9)."""
    X = np.atleast_2d(X)
    S, T = si(src), si(X)
    out = {kd: np.zeros(T.shape, dtype=np.int64) for kd in SOFT_KINDS}
    for j, f in enumerate(schema.features):
        r = rs.rules[f]
        if schema.kind[f] in ("equal", "non_decreasing"):
            continue
        if r.lo is not None:
            out["target"][:, j] = np.maximum(r.lo - T[:, j], 0)
        if r.hi is not None:
            out["target"][:, j] = np.maximum(out["target"][:, j], np.maximum(T[:, j] - r.hi, 0))
        if r.maxchg is not None:
            out["magnitude"][:, j] = np.maximum(np.abs(T[:, j] - S[j]) - r.maxchg, 0)
    return out


def soft_set(deg: Dict[str, np.ndarray], schema: Schema, i: int) -> Set[Tuple[str, str, int]]:
    return {(kd, schema.features[j], int(d[i, j])) for kd, d in deg.items() for j in np.where(d[i] > 0)[0]}


# ------------------------------------------------------------------ clingo path

def _q(s: str) -> str:
    return '"' + str(s).replace('"', "") + '"'


@lru_cache(maxsize=1)
def _program() -> str:
    return (KNOWLEDGE / "rules.lp").read_text()


def static_facts(concepts: Sequence[str], schema: Schema, k: Knowledge, inheritance: bool, use_node_rules: bool,
                 data_maxchg: Optional[Dict[str, float]] = None) -> List[str]:
    """Rule-table + ontology facts for one source patient (independent of the NumPy resolver) (P2)."""
    concepts = [c for c in concepts if c in k.onto]
    facts = [f"has_concept({_q(c)})." for c in concepts]
    chains = set()
    for c in concepts:
        a = k.onto.ancestors(c)
        chains.update(zip(a[:-1], a[1:]))
    facts += [f"parent({_q(a)},{_q(b)})." for a, b in sorted(chains)]
    if inheritance:
        facts.append("inherit.")
    fr = k.feature_rules
    for f in schema.features:
        c = schema.concept[f]
        root = k.root_row(c)
        rows = fr[fr["feature"] == c]
        if not use_node_rules:
            rows = rows[rows["applies_to_icd10_node"] == ROOT]
        for r in rows.itertuples():
            n = r.applies_to_icd10_node
            facts.append(f"frule({_q(r.rule_id)},{_q(f)},{_q(n)}). depth({_q(n)},{0 if n == ROOT else k.onto.depth(n)}).")
            lo = _num(r.target_low) if _num(r.target_low) is not None else _num(root["target_low"])
            hi = _num(r.target_high) if _num(r.target_high) is not None else _num(root["target_high"])
            if lo is not None: facts.append(f"lo({_q(r.rule_id)},{int(si(lo))}).")
            if hi is not None: facts.append(f"hi({_q(r.rule_id)},{int(si(hi))}).")
        facts.append(f"kind({_q(f)},{schema.kind[f]}). eps({_q(f)},{int(si(schema.eps[f]))}).")
        mc = _num(root["max_plausible_change"])
        if mc is None and data_maxchg:
            mc = data_maxchg.get(f)
        if mc is not None and schema.kind[f] not in ("equal", "non_decreasing"):
            facts.append(f"maxchg({_q(f)},{int(si(mc))}).")
    facts.append(f"horizon({int(si(AGE_HORIZON_CAP))}).")
    ci = k.contraindications
    facts += [f"contra({_q(r)},{_q(c)},{_q(n)})." for r, c, n in zip(ci["rule_id"], ci["intervention_class"], ci["contraindicated_icd10_node"])]
    return facts


@dataclass
class ClingoResult:
    selected: Set[Tuple[str, str]] = field(default_factory=set)
    eff: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)
    via: Set[Tuple[str, str, str]] = field(default_factory=set)
    hard: Dict[int, Set[Tuple[str, str]]] = field(default_factory=dict)
    soft: Dict[int, Set[Tuple[str, str, int]]] = field(default_factory=dict)
    contraindicated: Set[Tuple[str, str, str]] = field(default_factory=set)
    blocked_via: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    passed: Dict[int, Set[Tuple[str, str]]] = field(default_factory=dict)


def check_clingo(concepts: Sequence[str], schema: Schema, k: Knowledge, src: np.ndarray, cands: np.ndarray,
                 horizon: bool = True, inheritance: bool = True, use_node_rules: bool = True,
                 data_maxchg: Optional[Dict[str, float]] = None) -> ClingoResult:
    """One clingo solve: rule set, hard/soft per candidate, contraindications, derivations (P2/P7)."""
    cands = np.atleast_2d(cands)
    facts = static_facts(concepts, schema, k, inheritance, use_node_rules, data_maxchg)
    if horizon:
        facts.append("use_horizon.")
    S = si(src)
    facts += [f"src({_q(f)},{int(S[j])})." for j, f in enumerate(schema.features)]
    for i, row in enumerate(si(cands)):
        facts.append(f"cand({i}).")
        facts += [f"val({i},{_q(f)},{int(row[j])})." for j, f in enumerate(schema.features)]
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], _program() + "\n" + "\n".join(facts))
    ctl.ground([("base", [])])
    res = ClingoResult(hard={i: set() for i in range(len(cands))}, soft={i: set() for i in range(len(cands))},
                       passed={i: set() for i in range(len(cands))})
    lo, hi = {}, {}
    with ctl.solve(yield_=True) as h:
        for m in h:
            for a in m.symbols(shown=True):
                g = a.arguments
                if a.name == "selected": res.selected.add((g[0].string, g[1].string))
                elif a.name == "eff_lo": lo[g[0].string] = g[1].number
                elif a.name == "eff_hi": hi[g[0].string] = g[1].number
                elif a.name == "via": res.via.add(tuple(x.string for x in g))
                elif a.name == "hard": res.hard[g[2].number].add((g[0].name, g[1].string))
                elif a.name == "soft": res.soft[g[2].number].add((g[0].name, g[1].string, g[3].number))
                elif a.name == "contraindicated": res.contraindicated.add(tuple(x.string for x in g))
                elif a.name == "blocked_via": res.blocked_via.add(tuple(x.string for x in g))
                elif a.name == "passed": res.passed[g[2].number].add((g[0].name, g[1].string))
            break
    res.eff = {f: (lo.get(f), hi.get(f)) for f in schema.features}
    return res


def certificate(concepts: Sequence[str], schema: Schema, k: Knowledge, src: np.ndarray, path_nodes: np.ndarray,
                inheritance: bool = True, data_maxchg=None) -> dict:
    """Tier-1 certificate: clingo re-check of every path node against the SOURCE baseline (P7)."""
    r = check_clingo(concepts, schema, k, src, path_nodes, horizon=True, inheritance=inheritance, data_maxchg=data_maxchg)
    return {"all_hard_ok": all(not v for v in r.hard.values()),
            "hard_violations": {i: sorted(v) for i, v in r.hard.items() if v},
            "passed_checks_endpoint": sorted(r.passed[len(path_nodes) - 1]),
            "selected_rules": sorted(r.selected), "rule_vias": sorted(r.via),
            "contraindicated": sorted(r.contraindicated), "blocked_via": sorted(r.blocked_via),
            "soft_endpoint": sorted(r.soft[len(path_nodes) - 1])}
