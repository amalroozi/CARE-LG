"""P2 deliverable: results/REVIEW_rules.md -- every knowledge row for human review + self-review notes.

  .venv/bin/python -m experiment_final.src.review_rules
"""
from __future__ import annotations

import pandas as pd

from experiment_final.src.common import KNOWLEDGE, RESULTS, md_table, new_path

SELF_REVIEW = [
    ("F-SBP / F-DBP (ROOT 140/90)", "Generic target uses the JNC8 / HTN2017-stage-2 threshold; HTN2017 itself recommends <130/80 for most adults with "
     "hypertension and >=10% ASCVD risk, and JNC8 allows <150/90 at age >=60. The ROOT row is a deliberate middle ground so that the "
     "patient-specific <130/80 rows (diabetes, CKD, ischaemic heart disease, stroke, HF) actually differ from the default. Reviewer: confirm."),
    ("F-A1C-DM lower bound 6.0%", "ADA gives an upper goal (<7%) and warns about hypoglycaemia; the 6.0% floor is an author choice used only as a SOFT rule."),
    ("F-TC (<200 mg/dL)", "Total-cholesterol target is from ATP III (2002); CHOL2018 targets LDL-C, which NHANES/UCI here do not provide."),
    ("Typical effects (interventions.csv)", "Additive lifestyle effects and the drug effects (LAW2009 ~9/5.5 mmHg per standard-dose class, statin -25% TC, "
     "metformin -1.1% HbA1c, GLP-1 RA BMI -2.0) are approximations used only to choose the escalation step; they never change validity."),
    ("beta_blocker_nonselective", "Not a first-line antihypertensive (HTN2017). Included so the asthma contraindication (C-BB-J45) is exercised; carvedilol and "
     "labetalol are matched as non-selective (they also block alpha-1)."),
    ("C-GLP1-C73", "The FDA boxed warning concerns medullary thyroid carcinoma; ICD-10-CM C73 covers all thyroid cancers, so blocking at C73 over-blocks "
     "(conservative)."),
    ("C-THZ-M10, C-STATIN-K746, C-ACEARB-T783", "Relative contraindications treated as blocking (gout with thiazides; cirrhosis with statins - compensated "
     "cirrhosis may tolerate statins; ACE-inhibitor angioedema applied to the ACEi/ARB class although ARB cross-reactivity is low). Conservative."),
    ("diabetic -> E11", "NHANES does not record diabetes type; mapping all doctor-told diabetes to E11 means C-SGLT2-E10 only fires for patients with an "
     "explicit E10 Rx reason code."),
    ("Indication tokens", "Antihypertensives are indicated when the baseline BP is above the PATIENT-SPECIFIC target (so inheritance changes indication); "
     "statins by PCE >= 7.5%, diabetes, ASCVD or above-target cholesterol; diabetes drugs need a diabetes concept (or above-normal cardio glucose category)."),
]


def build() -> str:
    rd = lambda n: pd.read_csv(KNOWLEDGE / n, dtype=str, keep_default_na=False)  # noqa: E731
    parts = ["# REVIEW_rules.md -- knowledge tables for human review (CARE-LG, PIPELINE P2)", "",
             "Every row below is used by the pipeline. `verify=1` means the citation/value was written from memory and must be "
             "checked against the source document before any clinical use. Citation keys resolve in `knowledge/citations.csv`. "
             "ICD-10-CM nodes were checked to exist in the ICD-10-CM 2027 tabular XML (fail-safe in `src/knowledge_layer.py`).", "",
             "## Self-review notes (author)", ""]
    parts += [f"- **{a}**: {b}" for a, b in SELF_REVIEW]
    for name, title in (("feature_rules.csv", "Feature rules (HARD direction from the adapter; SOFT targets/magnitude, inherited down ICD-10-CM)"),
                        ("interventions.csv", "Intervention classes (escalation tier, acts_on, typical effect, indication)"),
                        ("contraindications.csv", "Contraindications (FILTER tier; inherited down ICD-10-CM)"),
                        ("annotations.csv", "Non-blocking guideline annotations"),
                        ("nhanes_condition_map.csv", "NHANES item -> ICD-10-CM concept map"),
                        ("citations.csv", "Citations")):
        df = rd(name)
        n_ver = int((df["verify"] == "1").sum()) if "verify" in df else 0
        parts += ["", f"## {title} -- `knowledge/{name}` ({len(df)} rows" + (f", {n_ver} marked verify=1" if "verify" in df else "") + ")", "",
                  md_table(df, index=False)]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    p = new_path(RESULTS / "REVIEW_rules.md")
    p.write_text(build())
    print(p)
