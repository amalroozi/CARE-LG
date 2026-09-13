# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Phase 4 -- neurosymbolic clinical-guideline layer, built on clingo (ASP).

Two distinct roles, kept deliberately separate so the corrected admissibility
gate from Phase 1 is never accidentally changed by adding this layer:

1. ADMISSIBILITY RULES (`R-*`): a symbolic re-encoding, rule-for-rule, of the
   exact predicate in `src/graph/clinical_constraints.py::check_clinical_violations`
   (immutable / non_decreasing / age_horizon / directional_reduce /
   directional_increase). These are REGRESSION-CHECKED against the hardcoded
   Python predicate (see `regression_check.py`) and must agree exactly on
   every covered constraint type. They exist so that edge admissibility can
   be explained ("which rule blocked this transition"), not to replace the
   fast vectorized numpy gate used inside the hot path of
   `experiments_v6/lib/graph_build.py` for the full ablation grid -- clingo
   is invoked per-edge and is not a substitute for that vectorized code at
   grid scale. It IS used directly for: (a) the regression check itself,
   (b) Phase 5's per-patient guideline-derivation explanations, and (c) the
   Phase 7 UI, all of which operate on one patient/edge at a time.

2. GUIDELINE ANNOTATIONS (`G-*`): non-blocking clinical facts that do not
   gate any edge in the graph (there is no code-level constraint for them
   in `check_clinical_violations`) but justify *why* the corresponding
   directional constraints exist, for patient-facing explanation only:
   statin indication (10-yr ASCVD risk threshold) and BP treatment targets
   differentiated by diabetes status.

Citations (documented once here, referenced by rule id elsewhere):
  [PCE2013]  Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. "2013 ACC/AHA
             Guideline on the Assessment of Cardiovascular Risk." Circulation.
             2014;129(25 Suppl 2):S49-S73. Defines the Pooled Cohort
             Equations and the 7.5% 10-year ASCVD risk threshold for
             statin-therapy consideration -- the same threshold this
             project's own PCE-based label uses (experiments_v3/DATA_PROVENANCE.md
             section on `cvd_risk_flag`).
  [HTN2017]  Whelton PK, Carey RM, Aronow WS, et al. "2017 ACC/AHA/AAPA/ABC/
             ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention,
             Detection, Evaluation, and Management of High Blood Pressure in
             Adults." Hypertension. 2018;71(6):e13-e115. Recommends a BP
             target of <130/80 mmHg for adults with diabetes or established
             ASCVD risk, versus the historically more lenient <140/90 mmHg
             threshold (JNC7/JNC8) used for lower-risk adults without those
             comorbidities -- the differentiation this project's `diabetic`
             flag (data/nhanes_real_full.csv) is used to apply.
  [ADA2023]  American Diabetes Association. "Standards of Care in Diabetes
             -2023." Diabetes Care. 2023;46(Suppl 1). Supports the <130/80
             mmHg BP target and non-worsening HbA1c-direction guidance for
             patients with diabetes.
  [CHOL2018] Grundy SM, Stone NJ, Bailey AL, et al. "2018 AHA/ACC/AACVPR/
             AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the
             Management of Blood Cholesterol." Circulation. 2019;139(25):
             e1082-e1143. Supports non-worsening (reduce-or-hold) direction
             for total cholesterol as a lipid-management goal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import clingo

_IMMUTABLE_TOL = 1e-4
_AGE_DECREASE_TOL = 0.1
_AGE_HORIZON_CAP = 3.05
_DIRECTIONAL_EPS = {
    'cholesterol': 1.0, 'systolic_bp': 1.0, 'diastolic_bp': 1.0, 'resting_bp': 5.0,
    'bmi': 0.1, 'glycemic_hba1c': 0.05, 'oldpeak': 0.05, 'exercise_angina': 0.5,
    'slope': 0.5, 'fasting_blood_sugar': 0.5,
}

RULE_CITATIONS = {
    "R-IMMUTABLE": ("Immutable clinical attribute cannot be altered by a recourse action "
                     "(structural constraint, not a guideline citation)."),
    "R-AGE-MONOTONIC": ("Age cannot decrease between source and target (structural constraint)."),
    "R-AGE-HORIZON": ("Single recourse step limited to <= 3.05 years, a clinically realistic "
                       "follow-up interval for risk recalculation (structural constraint, "
                       "matches the original benchmark's own single-step horizon rule)."),
    "R-BP-DIRECTION": ("Blood pressure must trend downward or hold stable under "
                        "guideline-directed management [HTN2017]."),
    "R-LIPID-DIRECTION": ("Total cholesterol must trend downward or hold stable under "
                           "lipid-management goals [CHOL2018]."),
    "R-GLUCOSE-DIRECTION": ("HbA1c/glycemic markers must trend downward or hold stable "
                             "under glycemic-control goals [ADA2023]."),
    "R-BMI-DIRECTION": "BMI must not increase under a lifestyle-intervention recourse step.",
    "R-DIRECTIONAL-OTHER": "Feature must trend in its clinically admissible direction.",
}

GUIDELINE_ANNOTATIONS = {
    "G-STATIN": ("10-year ASCVD risk >= 7.5% (2013 ACC/AHA Pooled Cohort Equations) meets the "
                 "guideline threshold for statin-therapy consideration [PCE2013]."),
    "G-BP-TARGET-DM": "Target blood pressure <130/80 mmHg for patients with diabetes [HTN2017][ADA2023].",
    "G-BP-TARGET-NODM": "Target blood pressure <140/90 mmHg for patients without diabetes [HTN2017].",
}

STATIN_THRESHOLD_PCT = 7.5  # [PCE2013]
BP_TARGET_DIABETIC = (130.0, 80.0)   # [HTN2017][ADA2023]
BP_TARGET_NONDIABETIC = (140.0, 90.0)  # [HTN2017]


@dataclass
class AdmissibilityResult:
    admissible: bool
    blocking_rules: List[str] = field(default_factory=list)   # rule ids that fired (violations)
    justifications: List[str] = field(default_factory=list)   # human-readable text for each blocking rule
    passed_rules: List[str] = field(default_factory=list)     # rule ids that were checked and did NOT fire


# clingo's ASP-core-2 grounder has no float-literal terms -- every numeric
# fact below is an INTEGER, scaled by _SCALE (millesimal precision, i.e.
# thousandths of a clinical unit -- ample for the coarsest epsilon here,
# 0.05 HbA1c/oldpeak). All comparisons are therefore exact integer
# comparisons on scaled values, equivalent to the float comparisons in
# check_clinical_violations up to _SCALE's precision.
_SCALE = 1000


def _si(x: float) -> int:
    """Scale a float clinical value/threshold to an integer ASP term."""
    return int(round(x * _SCALE))


_ASP_PROGRAM = """
% --- admissibility rules (R-*), mirroring check_clinical_violations exactly ---
% (all numeric facts are pre-scaled integers -- see _si() in the Python wrapper)
violates("R-IMMUTABLE", F) :- immutable(F), abs_delta(F, D), immutable_tol(T), D > T.
violates("R-AGE-MONOTONIC", F) :- non_decreasing(F), delta(F, D), age_decrease_tol(T), D < -T.
violates("R-AGE-HORIZON", F) :- non_decreasing(F), check_step_horizon, delta(F, D), age_horizon_cap(C), D > C.
violates(Rule, F) :- directional_reduce(F), delta(F, D), eps(F, E), D > E, dir_rule_id(F, Rule).
violates(Rule, F) :- directional_increase(F), delta(F, D), eps(F, E), D < -E, dir_rule_id(F, Rule).

#show violates/2.
"""

_DIR_RULE_MAP = {
    'cholesterol': "R-LIPID-DIRECTION",
    'systolic_bp': "R-BP-DIRECTION", 'diastolic_bp': "R-BP-DIRECTION", 'resting_bp': "R-BP-DIRECTION",
    'bmi': "R-BMI-DIRECTION",
    'glycemic_hba1c': "R-GLUCOSE-DIRECTION", 'fasting_blood_sugar': "R-GLUCOSE-DIRECTION",
}


def check_admissibility_symbolic(
    source_dict: Dict[str, float], target_dict: Dict[str, float], feature_metadata: dict,
    check_step_horizon: bool = True,
) -> AdmissibilityResult:
    """
    Queries clingo with the ASP encoding of the admissibility predicate for one
    (source, target) transition. Returns which rule(s), if any, blocked the
    transition, with human-readable justification text.
    """
    facts = []
    checked_rule_ids = set()

    for f in feature_metadata.get('immutable', []):
        if f in source_dict and f in target_dict:
            facts.append(f'immutable("{f}").')
            facts.append(f'abs_delta("{f}", {_si(abs(target_dict[f] - source_dict[f]))}).')
            checked_rule_ids.add("R-IMMUTABLE")
    for f in feature_metadata.get('non_decreasing', []):
        if f in source_dict and f in target_dict:
            d = target_dict[f] - source_dict[f]
            facts.append(f'non_decreasing("{f}").')
            facts.append(f'delta("{f}", {_si(d)}).')
            checked_rule_ids.add("R-AGE-MONOTONIC")
            if check_step_horizon:
                checked_rule_ids.add("R-AGE-HORIZON")
    for f in feature_metadata.get('directional_reduce', []):
        if f in source_dict and f in target_dict:
            d = target_dict[f] - source_dict[f]
            eps = _DIRECTIONAL_EPS.get(f, 0.1)
            facts.append(f'directional_reduce("{f}").')
            facts.append(f'delta("{f}", {_si(d)}).')
            facts.append(f'eps("{f}", {_si(eps)}).')
            rule_id = _DIR_RULE_MAP.get(f, "R-DIRECTIONAL-OTHER")
            facts.append(f'dir_rule_id("{f}", "{rule_id}").')
            checked_rule_ids.add(rule_id)
    for f in feature_metadata.get('directional_increase', []):
        if f in source_dict and f in target_dict:
            d = target_dict[f] - source_dict[f]
            eps = _DIRECTIONAL_EPS.get(f, 1.0)
            facts.append(f'directional_increase("{f}").')
            facts.append(f'delta("{f}", {_si(d)}).')
            facts.append(f'eps("{f}", {_si(eps)}).')
            rule_id = _DIR_RULE_MAP.get(f, "R-DIRECTIONAL-OTHER")
            facts.append(f'dir_rule_id("{f}", "{rule_id}").')
            checked_rule_ids.add(rule_id)

    facts.append(f'immutable_tol({_si(_IMMUTABLE_TOL)}).')
    facts.append(f'age_decrease_tol({_si(_AGE_DECREASE_TOL)}).')
    facts.append(f'age_horizon_cap({_si(_AGE_HORIZON_CAP)}).')
    if check_step_horizon:
        facts.append('check_step_horizon.')

    program = "\n".join(facts) + "\n" + _ASP_PROGRAM
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    fired = []  # (rule_id, feature)
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            for atom in model.symbols(shown=True):
                if atom.name == "violates":
                    rule_id = atom.arguments[0].string
                    feat = atom.arguments[1].string
                    fired.append((rule_id, feat))
            break  # single stable model expected (no disjunction/choice in this program)

    blocking_rules = sorted(set(r for r, _ in fired))
    justifications = []
    for rule_id, feat in fired:
        base = RULE_CITATIONS.get(rule_id, "Clinical admissibility rule violated.")
        justifications.append(f"[{rule_id}] {feat}: {base}")

    passed_rules = sorted(checked_rule_ids - set(blocking_rules))
    return AdmissibilityResult(
        admissible=(len(fired) == 0),
        blocking_rules=blocking_rules,
        justifications=justifications,
        passed_rules=passed_rules,
    )


def statin_indication(ascvd_risk_pct: float) -> Tuple[bool, str]:
    """[PCE2013]: 10-yr ASCVD risk >= 7.5% meets the guideline threshold for statin consideration."""
    indicated = ascvd_risk_pct >= STATIN_THRESHOLD_PCT
    text = (f"10-year ASCVD risk {ascvd_risk_pct:.1f}% "
            f"{'meets' if indicated else 'is below'} the {STATIN_THRESHOLD_PCT}% guideline threshold "
            f"for statin-therapy consideration. {GUIDELINE_ANNOTATIONS['G-STATIN']}")
    return indicated, text


def bp_target(diabetic: Optional[bool]) -> Tuple[Tuple[float, float], str]:
    """[HTN2017][ADA2023]: BP target differentiated by diabetes status."""
    if diabetic:
        return BP_TARGET_DIABETIC, GUIDELINE_ANNOTATIONS["G-BP-TARGET-DM"]
    return BP_TARGET_NONDIABETIC, GUIDELINE_ANNOTATIONS["G-BP-TARGET-NODM"]
