# knowledge/ — clinical knowledge tables and ontology

Every row of the guideline tables is marked `verify=1`: drafted from memory, needs clinical review (see `../results/REVIEW_rules.md`).

- `interventions.csv` — the 13 intervention classes: escalation tier, features acted on, typical effect size, indication rule, citation (source of every recommended intervention).
- `intervention_text.csv` — template wording for the natural-language plans (display name, action sentence, dose, effect wording, review interval, citation, verify=1); dose/review interval are `TODO` unless the knowledge tables state them.
- `contraindications.csv` — contraindications attached to ICD-10-CM nodes (inherited by descendant codes).
- `feature_rules.csv` — hard / soft rules per feature and the patient-specific target ranges attached to ICD-10-CM nodes.
- `citations.csv` — citation key → full reference for every rule and intervention row.
- `annotations.csv` — extra annotations used by the explanation text (e.g. statin / blood-pressure notes).
- `nhanes_condition_map.csv` — NHANES questionnaire / prescription fields → ICD-10-CM concepts.
- `nhanes_variable_map.csv` — NHANES variable names per cycle → canonical feature names.
- `rules.lp` — the clingo (answer-set) encoding of the hard rules used for certificates; mirrors the NumPy implementation.
- `ontology.py` — ICD-10-CM hierarchy loader and `ancestors()`; loads `archive/experiments_caren/knowledge/ontology.py` and the ICD-10-CM XML under `archive/experiments_caren/data/raw/`.
