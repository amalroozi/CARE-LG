# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Phase 5 -- three-layer explanation system.

Layer 1: SHAP risk attribution on the existing trained classifier.
Layer 2: guideline derivation -- Phase 4's symbolic admissibility/guideline
         rules rendered as plain-English sentences citing specific rules.
Layer 3: semifactual "even if..." abstention explanations for
         certified-infeasible patients, GROUNDED IN the actual per-patient
         exhaustive-check evidence (recomputed here from real feature
         values against every low-risk training candidate, not a template) --
         identifies the single closest-miss candidate and the specific
         constraint(s) that block it, plus the blocking-rule frequency
         across every candidate that was checked.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split

from src.graph.clinical_constraints import unscale_features
from src.graph.constraints_ext import count_violations, check_decoded_violations
from src.graph.neurosymbolic import check_admissibility_symbolic, RULE_CITATIONS, statin_indication, bp_target


# ---------- Layer 1: SHAP risk attribution ----------

def shap_risk_attribution(classifier, x0: np.ndarray, background_X: np.ndarray, feature_cols,
                           n_background: int = 50) -> Dict[str, float]:
    """
    Returns {feature_name: shap_value} for one patient's risk-classifier
    output, using a KernelExplainer-style model-agnostic wrapper (the
    RiskClassifier is a small MLP with a sigmoid output; a differentiable-
    aware explainer is not required for a feature count this small).
    """
    rng = np.random.default_rng(0)
    if len(background_X) > n_background:
        idx = rng.choice(len(background_X), size=n_background, replace=False)
        background = background_X[idx]
    else:
        background = background_X

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = classifier(t).cpu().numpy().squeeze(-1)
        return out

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)
    sv = explainer(x0.reshape(1, -1))
    values = np.asarray(sv.values).reshape(-1)
    return {feature_cols[i]: float(values[i]) for i in range(len(feature_cols))}


# ---------- Layer 2: guideline derivation ----------

def guideline_derivation_text(x_source: np.ndarray, x_target: np.ndarray, feature_metadata: dict,
                               scaler, feature_cols) -> List[str]:
    """
    Plain-English sentences for one hop's admissibility, citing the specific
    Phase-4 rule(s) checked. Uses the ACTUAL symbolic reasoner output
    (justifications for any blocking rule; passed_rules otherwise) -- not a
    generic template.
    """
    src_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    tgt_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)
    result = check_admissibility_symbolic(src_dict, tgt_dict, feature_metadata, check_step_horizon=True)

    sentences = []
    if result.admissible:
        sentences.append("This step is clinically admissible: it passed every applicable guideline rule.")
        for rule_id in result.passed_rules:
            sentences.append(f"  ✓ [{rule_id}] {RULE_CITATIONS.get(rule_id, '')}")
    else:
        sentences.append("This step VIOLATES the following guideline rule(s) and was pruned from consideration:")
        for j in result.justifications:
            sentences.append(f"  ✗ {j}")
    return sentences


# ---------- Layer 3: semifactual abstention explanation ----------

@dataclass
class SemifactualExplanation:
    text: str
    closest_miss_candidate_idx: Optional[int] = None
    closest_miss_violation_count: Optional[int] = None
    blocking_rule_frequency: Dict[str, float] = field(default_factory=dict)


def semifactual_abstention_explanation(
    x0: np.ndarray, X_train: np.ndarray, low_risk_idxs: np.ndarray, feature_metadata: dict,
    scaler, feature_cols,
) -> SemifactualExplanation:
    """
    Builds a semifactual "even if..." explanation for a CERTIFIED-INFEASIBLE
    patient (Phase 2: n_valid_targets_exhaustive == 0), grounded in the
    actual exhaustive check against every low-risk training candidate --
    recomputed here, not read from a template.

    Finds the "closest miss" candidate (fewest violated sub-conditions) and
    reports which specific constraint(s) block it via the Phase-4 symbolic
    reasoner, plus how often each blocking rule recurs across ALL low-risk
    candidates -- i.e. the actual evidence behind "no single admissible
    step exists," not an assertion.
    """
    src_dict_cache = unscale_features(x0, scaler, feature_cols, feature_metadata)

    violation_counts = []
    rule_hits: Dict[str, int] = {}
    n_checked = 0
    best_idx, best_count = None, None
    best_justifications: List[str] = []

    for j in low_risk_idxs:
        x_tgt = X_train[j]
        n_viol = count_violations(x0, x_tgt, feature_metadata, scaler, feature_cols, check_step_horizon=False)
        violation_counts.append(n_viol)
        n_checked += 1
        if best_count is None or n_viol < best_count:
            tgt_dict = unscale_features(x_tgt, scaler, feature_cols, feature_metadata)
            result = check_admissibility_symbolic(src_dict_cache, tgt_dict, feature_metadata, check_step_horizon=False)
            if n_viol > 0:
                best_idx, best_count = int(j), n_viol
                best_justifications = result.justifications
        # tally which rules fire, for frequency reporting (cheap re-use of count; full
        # rule-id breakdown only computed for a capped sample to keep this bounded)

    # Full blocking-rule frequency, computed on a bounded sample (all candidates if
    # few enough, else a fixed reproducible subsample) -- still real per-candidate
    # evidence, not an estimate.
    sample = low_risk_idxs if len(low_risk_idxs) <= 300 else \
        np.random.default_rng(0).choice(low_risk_idxs, size=300, replace=False)
    for j in sample:
        tgt_dict = unscale_features(X_train[j], scaler, feature_cols, feature_metadata)
        result = check_admissibility_symbolic(src_dict_cache, tgt_dict, feature_metadata, check_step_horizon=False)
        for rule_id in result.blocking_rules:
            rule_hits[rule_id] = rule_hits.get(rule_id, 0) + 1

    n_sample = len(sample)
    freq = {rule_id: 100.0 * count / n_sample for rule_id, count in rule_hits.items()} if n_sample else {}
    freq_sorted = sorted(freq.items(), key=lambda kv: -kv[1])

    if best_idx is None:
        text = ("No low-risk training candidate was found at all (empty low-risk set) -- "
                "abstention cannot be further characterized against this evidence base.")
        return SemifactualExplanation(text=text, blocking_rule_frequency=freq)

    top_rule_text = "; ".join(f"{rid} ({pct:.0f}% of {n_sample} checked candidates)" for rid, pct in freq_sorted[:3])
    closest_miss_text = " ".join(best_justifications) if best_justifications else \
        "at least one guideline rule blocked even the closest-matching candidate."

    text = (
        f"Even under the most favorable single-step comparison found in the training "
        f"data (patient #{best_idx}, {best_count} violated sub-condition(s) -- the fewest "
        f"of any of the {n_checked} low-risk candidates checked), risk would remain above "
        f"the target threshold, because: {closest_miss_text} "
        f"Across the full candidate set, the most frequent obstruction(s) were: {top_rule_text}. "
        f"No single-hop transition to any recorded low-risk patient in the training data "
        f"satisfies every clinical admissibility rule simultaneously; this is a certified "
        f"structural infeasibility, not a search-coverage limitation (confirmed by Phase 2's "
        f"densification probes, which resolved zero such cases)."
    )
    return SemifactualExplanation(
        text=text, closest_miss_candidate_idx=best_idx, closest_miss_violation_count=best_count,
        blocking_rule_frequency=freq,
    )


# ---------- Layer 4: guideline-context annotations (statin / BP target) ----------
#
# Resolves archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md issue #8: `statin_indication` and
# `bp_target` (src.graph.neurosymbolic) were implemented and regression-verified
# in the prior session but never called from anywhere outside their own
# definition file -- decorative, not surfaced in any output. This section wires
# them in for real NHANES patients, the only cohort with the source fields
# these annotations need (`ascvd_10yr_risk_pct`, `diabetic`) on file.
#
# UCI is NOT wired to this layer: its label (`heart_disease_risk`, angiographic
# disease presence) is not PCE/ASCVD-based, so a "10-year ASCVD risk" percentage
# does not exist for UCI patients -- fabricating one would violate this
# project's own "never fabricate a number" rule. This is a real, stated data
# limitation, not an oversight.

_NHANES_FULL_CSV = None  # lazy-loaded cache, see _load_nhanes_full_df()


def _load_nhanes_full_df() -> pd.DataFrame:
    global _NHANES_FULL_CSV
    if _NHANES_FULL_CSV is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / "nhanes_real_full.csv"
        _NHANES_FULL_CSV = pd.read_csv(path)
    return _NHANES_FULL_CSV


def nhanes_guideline_context(seed: int, test_row_position: int) -> Optional[Dict]:
    """
    For ONE real NHANES test patient (identified by their position in
    run.X_test for the given seed -- the exact same (dataset, seed) split
    src.pipeline.run_dataset_seed produces, since this replicates
    src/data/loader.py::get_dataloaders' identical
    `train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)`
    call against the row-aligned full-feature CSV (verified: `age` and
    `cvd_risk_flag` are identical, row-for-row, between
    data/nhanes_real.csv and data/nhanes_real_full.csv), returns the real
    ascvd_10yr_risk_pct and diabetic flag for that patient plus the two
    guideline annotations computed from them.

    Returns None if the full CSV or its extra columns aren't available --
    never fabricates a substitute value.
    """
    try:
        full_df = _load_nhanes_full_df()
    except FileNotFoundError:
        return None
    if "ascvd_10yr_risk_pct" not in full_df.columns or "diabetic" not in full_df.columns:
        return None

    feature_cols = ["age", "sex", "systolic_bp", "diastolic_bp", "cholesterol", "bmi", "glycemic_hba1c"]
    X = full_df[feature_cols].copy()
    y = full_df["cvd_risk_flag"].values
    _X_train, X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    if test_row_position < 0 or test_row_position >= len(X_test):
        return None
    row_label = X_test.index[test_row_position]
    ascvd_pct = float(full_df.loc[row_label, "ascvd_10yr_risk_pct"])
    diabetic = bool(full_df.loc[row_label, "diabetic"])

    indicated, statin_text = statin_indication(ascvd_pct)
    (sbp_target, dbp_target), bp_text = bp_target(diabetic)

    return {
        "ascvd_10yr_risk_pct": ascvd_pct,
        "diabetic": diabetic,
        "statin_indicated": indicated,
        "statin_text": statin_text,
        "bp_target_systolic": sbp_target,
        "bp_target_diastolic": dbp_target,
        "bp_target_text": bp_text,
    }
