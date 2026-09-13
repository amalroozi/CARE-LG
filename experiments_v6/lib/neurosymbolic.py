"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/graph/neurosymbolic.py."""
from src.graph.neurosymbolic import (
    AdmissibilityResult,
    check_admissibility_symbolic,
    statin_indication,
    bp_target,
    RULE_CITATIONS,
    GUIDELINE_ANNOTATIONS,
    STATIN_THRESHOLD_PCT,
    BP_TARGET_DIABETIC,
    BP_TARGET_NONDIABETIC,
)

__all__ = [
    "AdmissibilityResult", "check_admissibility_symbolic", "statin_indication", "bp_target",
    "RULE_CITATIONS", "GUIDELINE_ANNOTATIONS", "STATIN_THRESHOLD_PCT",
    "BP_TARGET_DIABETIC", "BP_TARGET_NONDIABETIC",
]
