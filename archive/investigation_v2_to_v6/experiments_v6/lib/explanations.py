"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/recourse/explanations.py."""
from src.recourse.explanations import (
    shap_risk_attribution,
    guideline_derivation_text,
    SemifactualExplanation,
    semifactual_abstention_explanation,
    nhanes_guideline_context,
)

__all__ = [
    "shap_risk_attribution", "guideline_derivation_text",
    "SemifactualExplanation", "semifactual_abstention_explanation",
    "nhanes_guideline_context",
]
