"""DatasetBundle + contract validation shared by every adapter (DATASET_ADAPTER.md, PIPELINE P1).

feature_meta[f] keys (required): type (continuous|binary|categorical|ordinal), class
(immutable|non_decreasing|mutable), direction (reduce|increase|range|none), unit, phys_min, phys_max.
Optional keys used downstream: concept (canonical knowledge-table feature name; defaults to f),
noise_tol (per-edge measurement tolerance of the CARE-LG directional gate; defaults to the
CARE-LG table in src/graph/constraints_ext.py, i.e. 0.1 for reduce and 1.0 for increase).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

TYPES = {"continuous", "binary", "categorical", "ordinal"}
CLASSES = {"immutable", "non_decreasing", "mutable"}
DIRECTIONS = {"reduce", "increase", "range", "none"}
REQUIRED_META = ("type", "class", "direction", "unit", "phys_min", "phys_max")


@dataclass
class DatasetBundle:
    """Everything downstream reads (DATASET_ADAPTER.md)."""
    X: pd.DataFrame
    y: pd.Series
    patient_id: pd.Series
    feature_meta: Dict[str, dict]
    context: pd.DataFrame
    concepts: List[List[str]]
    current_meds: List[List[str]]
    subgroups: Dict[str, pd.Series]
    name: str
    label_definition: str
    citation: str
    has_ontology: bool = False
    cohort_flow: List[dict] = field(default_factory=list)


class AdapterContractError(ValueError):
    """Fail-safe stop (DATASET_ADAPTER.md degradation rules)."""


def validate_bundle(b: DatasetBundle) -> None:
    """Checks the contract; raises AdapterContractError (fail-safe stop) on any breach (P1)."""
    n = len(b.X)
    if not (len(b.y) == len(b.patient_id) == len(b.context) == len(b.concepts) == len(b.current_meds) == n):
        raise AdapterContractError(f"{b.name}: row counts disagree")
    if b.patient_id.duplicated().any():
        raise AdapterContractError(f"{b.name}: patient_id not unique")
    if set(pd.unique(b.y)) - {0, 1}:
        raise AdapterContractError(f"{b.name}: y must be binary 0/1")
    if b.X.isna().any().any():
        raise AdapterContractError(f"{b.name}: X has missing values")
    for f in b.X.columns:
        m = b.feature_meta.get(f)
        if m is None or any(k not in m or m[k] is None for k in ("class", "direction")):
            raise AdapterContractError(f"FAIL-SAFE: feature '{f}' missing class/direction")
        for k in REQUIRED_META:
            if k not in m:
                raise AdapterContractError(f"{b.name}: feature '{f}' missing meta '{k}'")
        if m["type"] not in TYPES or m["class"] not in CLASSES or m["direction"] not in DIRECTIONS:
            raise AdapterContractError(f"{b.name}: feature '{f}' has an invalid type/class/direction")
        if m["class"] != "mutable" and m["direction"] not in ("none",):
            raise AdapterContractError(f"{b.name}: non-mutable feature '{f}' must have direction 'none'")
    for k, s in b.subgroups.items():
        if len(s) != n:
            raise AdapterContractError(f"{b.name}: subgroup '{k}' length mismatch")
    if not all(isinstance(c, list) for c in b.concepts) or not all(isinstance(c, list) for c in b.current_meds):
        raise AdapterContractError(f"{b.name}: concepts/current_meds must be lists")


def age_band(age: pd.Series, edges=(40, 50, 60, 70, 80)) -> pd.Series:
    """Fixed decade bands for fairness audits (P14)."""
    labels = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])]
    out = pd.cut(age, bins=list(edges), right=False, labels=labels).astype(str)
    out[age < edges[0]] = f"<{edges[0]}"
    out[age >= edges[-1]] = f">={edges[-1]}"
    return out
