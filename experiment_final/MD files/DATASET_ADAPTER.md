# Dataset adapter contract (plug any dataset in end to end)

Every dataset enters through one adapter in `adapters/<name>.py` exposing:

    load(config) -> DatasetBundle

DatasetBundle fields (everything downstream reads ONLY these):
- `X`            DataFrame, one row per patient, model features (raw clinical units)
- `y`            Series, binary outcome (1 = high-risk / event)
- `patient_id`   Series, unique id (used for splits)
- `feature_meta` dict per feature: type (continuous|binary|categorical), class (immutable|non_decreasing|mutable), direction (reduce|increase|range|none), unit, phys_min, phys_max
- `context`      DataFrame of non-model fields kept for audits/annotations (race, true risk %, flags)
- `concepts`     list of ICD-10-CM codes per patient (may be empty lists)
- `current_meds` list of intervention classes the patient already takes (may be empty)
- `subgroups`    dict name -> Series (sex, age band, race, ...)
- `name`, `label_definition`, `citation`

Degradation rules (pipeline still runs end to end):
- No `concepts` -> ontology layer uses generic (population-level) rules only; inheritance/contraindication ablations are skipped and reported as "not applicable".
- No `current_meds` -> "skip already-taken interventions" is a no-op.
- No intervention links for a feature -> its changes are reported as unmapped (lowers actionability, never silently dropped).
- Any feature missing `class`/`direction` -> fail-safe stop.

Planned adapters: `nhanes.py` (primary, full ontology), `uci.py`, `cardio.py` (rules without ontology), `eicu_demo.py` (static first-24h snapshot; optional), `mimic.py` (later).
