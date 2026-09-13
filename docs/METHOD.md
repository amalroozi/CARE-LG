# METHOD.md — CARE-LG (current)

The corrected pipeline (B1/B2 fix as the sole foundation), a neurosymbolic
clinical-guideline layer, a three-layer explanation system, per-patient
preference elicitation, and a comparison against FACE and a PACE-style
baseline. This is the current, final method description — the canonical
implementation lives in `src/`. All numbers in this document and in
`docs/FINDINGS.md` were originally produced in the `experiments_v6` session
(git SHA `539d87b795b78bb8f9446f99b0fabeded2c84841`) and have since been
reproduced bit-identically from `src/` (see `CHANGES.md` at the repo root
for the reorganization, and `archive/investigation_v2_to_v6/experiments_v6/CHANGES.md`
for the original session's exact reproduction commands).

## Core Assumption (added per `archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md` issue #6)

**Every recourse plan this system produces ends at another real, recorded
training patient's feature vector.** "This plan is valid" means "a real
patient in the training data was recorded in this state" — it does NOT mean
"this state is achievable by any specific new patient who follows the plan."
The graph does not model medication regimen, genetics, treatment adherence,
comorbidities beyond the modeled features, or the time actually elapsed for
the reference patient to reach that state. This is a deliberate, defensible
design choice — it is precisely why CALG's recourse targets are more
plausible (better KDE density, more clinically coherent feature
combinations) than pure gradient/perturbation baselines like DiCE or
Growing Spheres, which can invent feature combinations no real patient ever
exhibited. But it means the method's "validity" guarantee is about the
DATA (a state was recorded), not about ACHIEVABILITY BY A NEW INDIVIDUAL. A
patient told "reach the profile of training patient #225" has no guarantee
that patient #225's path there is reproducible by anyone else. This is
stated here as a limitation of the method's foundational claim, not a
footnote.

## 0. Where this code actually lives

The corrected pipeline, neurosymbolic layer, explanation system, and
preference elicitation are canonically implemented in `src/`
(`src/graph/query_attachment.py`, `src/graph/neurosymbolic.py`,
`src/graph/constraints_ext.py`, `src/graph/riemannian_batch.py`,
`src/recourse/search_augmented.py`, `src/recourse/explanations.py`,
`src/recourse/preference.py`, `src/vae/type_aware.py`, `src/seeding.py`,
`src/data/dataset_registration.py`, `src/pipeline.py`). This wasn't always
true: before the DEEP_AUDIT session, none of the B1/B2 fix, the
query-attachment/entry-gate concept itself, the neurosymbolic layer, or the
explanation/preference code existed anywhere in `src/` —
`src/recourse/search.py::find_recourse_path` has no query-attachment
mechanism at all, and the whole six-session investigation lived entirely in
parallel `experiments_*/` directories, disconnected from the actually-
importable library. That was fixed by merging the corrected pipeline into
`src/`; those `experiments_*/` directories (by then just thin shims
re-exporting `src/`) have since been moved wholesale into
`archive/investigation_v2_to_v6/` as part of a later repo-consolidation
session (see `CHANGES.md` at the repo root) — they are historical record
now, not a runnable code path. `src/` has no dependency on `archive/`.
`ui/server.py` and every script in `scripts/` import directly from `src/`,
so the demoed code path, the reproduction path, and the deployable path are
all the same code. Full verification, repeated after both the original
merge and the later consolidation: the complete 5-seed/both-cohort grid was
rerun end-to-end and produced numbers bit-identical to the already-published
`docs/FINDINGS.md` table (38.6%±13.1% UCI / 37.7%±3.4% NHANES success,
0.00%±0.00% real CVR both cohorts) — see
`archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md` for the full verification
log.

## 1. The corrected pipeline (Phase 1)

`archive/investigation_v2_to_v6/experiments_v6/lib/` adapts `archive/investigation_v2_to_v6/experiments_v3/lib/` with two fixes applied
unconditionally (not as an ablation toggle):

**B1 — entry-gate fix** (`lib/graph_build.py::build_query_edges`): a query
patient's out-edges into the training graph are now admissible under the
exact same predicate as every interior graph edge —
`(hard_count + dir_count) > 0`, `check_step_horizon=True` — instead of
v2/v3's relaxed hard-only, no-horizon gate. Query attachment remains FULL
(every training node is a candidate, not a k-restricted neighborhood — see
that file's docstring for the empirical justification, carried over from
`experiments_v3`/`REPO_MAP.md` D8).

**B2 — no unnecessary decoding** (`run_grid.py::evaluate_patient`): every
path node that is a real, already-recorded patient (the query itself, or any
training-set node) is read directly from `X_test`/`X_train` — never passed
through the VAE decoder — matching `src/recourse/search.py::decode_recourse_trajectory`,
the ORIGINAL pipeline's own patient-facing function. VAE decoding is used
only for genuinely synthetic nodes (Phase 2's densification probes). The
legacy "decode every node" computation is still reported alongside, labeled
`decoded_cvr_*`, purely for continuity with v2-v5's historical metric — it
never determines success/failure.

v4 (DSR) and v5 (EGD) were independently checked for the B1 pattern in their
own query-attachment code (`dsr_graph.py`, `egd_graph.py`) and found NOT to
have it (v4: only a narrower, defensible age-horizon-cap asymmetry; v5: no
constraint-gating predicate exists at all — its guarantee is architectural).
Neither needed fixing. Per `RECOMMENDATION.md`'s conclusion, neither is
carried forward as an active v6 component regardless; their code is kept
for reference only, unmodified.

**Result** (5 seeds, both cohorts, R+hard — the actual CARE-LG operating
point): real-value constraint-violation rate (CVR) is **exactly 0.00% ± 0.00%**
on every seed of both datasets. Success rate: UCI 38.6% ± 13.1%, real NHANES
37.7% ± 3.4%. Full numbers: `FINDINGS.md`.

## 2. Certified infeasibility (Phase 2)

`archive/investigation_v2_to_v6/experiments_v6/run_blindspot.py` ports `archive/investigation_v2_to_v6/experiments_v3/run_blindspot.py`'s
protocol (exhaustive feature-space check + two densification probes: 2×k,
and 500 VAE-prior synthetic nodes classifier-verified low-risk) onto the
B1-fixed pipeline. `find_path_augmented` runs one Dijkstra call over the
whole augmented graph and has no code path distinguishing "no entry edge"
from "entry succeeded but no route onward" — both were always unified
failure modes (see `run_blindspot.py`'s docstring). Result: of Phase-1
abstentions, **87.9% (UCI) / 98.5% (NHANES) are CERTIFIED INFEASIBLE**
(zero low-risk training patient is reachable in one admissible hop, by
exhaustive check), **0% are confirmed blindspots** (densification resolved
none, in any seed, either dataset — consistent with three prior sessions'
identical finding), and the small remainder (12.1% / 1.5%) is unresolved
(not certified infeasible, but not resolved by either probe either).

## 3. Neurosymbolic guideline layer (Phase 4)

`archive/investigation_v2_to_v6/experiments_v6/lib/neurosymbolic.py`, built on `clingo` (ASP). Two kinds of
rules:

**Admissibility rules (`R-*`)** — a rule-for-rule ASP re-encoding of
`src/graph/clinical_constraints.py::check_clinical_violations`:

| Rule ID | Condition | Citation |
|---|---|---|
| `R-IMMUTABLE` | any immutable feature (sex; also `fasting_blood_sugar` for UCI) changes by more than 1e-4 | structural, not a guideline citation |
| `R-AGE-MONOTONIC` | age decreases by more than 0.1 yr | structural |
| `R-AGE-HORIZON` | age increases by more than 3.05 yr in one step | structural (matches the original benchmark's own single-step horizon) |
| `R-BP-DIRECTION` | systolic/diastolic/resting BP increases beyond its epsilon | [HTN2017] |
| `R-LIPID-DIRECTION` | total cholesterol increases beyond its epsilon | [CHOL2018] |
| `R-GLUCOSE-DIRECTION` | HbA1c or fasting blood sugar increases beyond its epsilon | [ADA2023] |
| `R-BMI-DIRECTION` | BMI increases beyond its epsilon | lifestyle-intervention guidance |
| `R-DIRECTIONAL-OTHER` | any other directional feature (oldpeak, exercise_angina, slope, max_heart_rate) moves the wrong way | — |

Every fact is scaled to an integer before being passed to clingo (ASP-core-2
has no float literal terms); scale factor 1000 (millesimal precision),
verified against the float thresholds this replicates (finest epsilon in
this project is 0.05).

**Guideline annotations (`G-*`)** — non-blocking, explanation-only facts,
never gating any edge:
- `G-STATIN`: 10-year ASCVD risk ≥ 7.5% meets the guideline threshold for
  statin-therapy consideration [PCE2013] — the same 7.5% cutoff
  `archive/investigation_v2_to_v6/experiments_v3/DATA_PROVENANCE.md` already uses to define this project's
  own `cvd_risk_flag` label, so this rule is definitionally consistent with
  the data, not an independent judgment call.
- `G-BP-TARGET-DM` / `G-BP-TARGET-NODM`: target BP <130/80 mmHg for patients
  with diabetes, <140/90 mmHg otherwise [HTN2017][ADA2023]. The `diabetic`
  flag is read from `archive/investigation_v2_to_v6/experiments_v3/data/nhanes_real_full.csv` (present in
  the audit-retained full feature set, not the trimmed model-input CSV).

**Citations** (full bibliographic form in `lib/neurosymbolic.py`'s module
docstring):
- [PCE2013] Goff DC Jr et al., "2013 ACC/AHA Guideline on the Assessment of
  Cardiovascular Risk," *Circulation* 2014;129(25 Suppl 2):S49-S73.
- [HTN2017] Whelton PK et al., "2017 ACC/AHA/... Guideline for ... High
  Blood Pressure in Adults," *Hypertension* 2018;71(6):e13-e115.
- [ADA2023] American Diabetes Association, "Standards of Care in Diabetes
  -2023," *Diabetes Care* 2023;46(Suppl 1).
- [CHOL2018] Grundy SM et al., "2018 AHA/ACC/... Guideline on the Management
  of Blood Cholesterol," *Circulation* 2019;139(25):e1082-e1143.

**Regression check** (`run_neurosymbolic_regression.py`): 4,000 sampled
(source, target) pairs (2,000/dataset, half query-entry, half interior),
comparing the symbolic reasoner's decision against the hardcoded predicate.
**Result: 2000/2000 agreement on both datasets (100.0000%), 0 discrepancies.**

**Scope note**: clingo is invoked per-edge for explanation purposes (the
regression check, Phase 5's guideline text, the Phase 7 UI) — it is
deliberately NOT substituted into the vectorized-numpy hot path
(`graph_build.py`) that the full ablation grid runs, since that would slow
the grid by orders of magnitude for no measurement benefit (both layers
re-encode the identical predicate, confirmed above).

**Update (this session, DEEP_AUDIT issue #8)**: `statin_indication` and
`bp_target` above were built and regression-verified previously but never
called from anywhere outside their own definition file. They are now wired
into `src.recourse.explanations.nhanes_guideline_context`, surfaced in the
UI for real NHANES patients (confirmed on patient #5, seed 0: 17.7% ASCVD
risk → statin indicated; non-diabetic → <140/90 mmHg target). Not wired for
UCI, whose label is not ASCVD-based.

## 4. Three-layer explanation system (Phase 5)

`archive/investigation_v2_to_v6/experiments_v6/lib/explanations.py`:

1. **SHAP risk attribution** (`shap_risk_attribution`): `shap.Explainer`
   wrapping the trained `RiskClassifier` as a black-box numpy function, a
   sampled background set from `X_train`. Returns per-feature SHAP values
   for one patient's risk score.
2. **Guideline derivation** (`guideline_derivation_text`): for one hop,
   queries the Phase-4 symbolic reasoner and renders its actual
   `justifications`/`passed_rules` output as plain-English sentences citing
   the specific rule ID and clinical source — not a template with the rule
   name substituted in; the sentence content comes from the reasoner's
   actual per-transition computation.
3. **Semifactual abstention explanation** (`semifactual_abstention_explanation`):
   for a certified-infeasible patient, recomputes the violation count
   against EVERY low-risk training candidate, identifies the single
   closest-miss candidate (fewest violated sub-conditions), gets its
   specific blocking rule(s) from the symbolic reasoner, and separately
   tallies blocking-rule frequency across the full candidate set (or a
   fixed, seeded 300-sample subset for larger cohorts). The generated text
   states the closest-miss patient's actual violated rule(s) and the
   most-frequent obstruction(s) with their real percentages — verified by
   direct inspection this session (see the worked UCI example in
   `FINDINGS.md`) to be grounded in that computation, not boilerplate.

## 5. Preference elicitation (Phase 6)

`archive/investigation_v2_to_v6/experiments_v6/lib/preference.py`. **Documented discrepancy with the
brief** (Hard Rule 5): this codebase's Dijkstra cost W_ij is pure
latent-space distance; `compute_clinical_effort` is a post-hoc reporting
metric only, never part of the optimized cost. A new additive term was
introduced for this demo only:

```
W_ij^pref = dist_metric(z_i, z_j) + mu * weighted_L1_effort(x_i, x_j; profile)
```

applied on top of the UNCHANGED Phase-1 admissible edge set (safety/
admissibility never varies with preference; only which admissible path is
cheapest can change). Three illustrative synthetic profiles: `no_preference`
(baseline `CLINICAL_EFFORT_WEIGHTS`), `exercise_averse` and
`medication_averse` (4× multiplier on a documented, simplified
feature-to-category mapping — see `lib/preference.py`). `mu=0.5`.

**Caveat (per `archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md` issue #9, checked this
session)**: `CLINICAL_EFFORT_WEIGHTS` (`configs/dataset_config.py`) are
unvalidated, illustrative constants. Git history shows no rationale ever
attached beyond the generic comment "representing the difficulty/cost of
changing each feature" — no citation, no derivation from a patient survey
or published treatment-burden instrument. Every clinical-effort number
reported anywhere in this project, and this entire preference-elicitation
mechanism (which only reweights these same base numbers), inherits this
limitation. A principled alternative would derive per-feature weights from
a real burden-of-treatment survey or a published clinical-burden score
(e.g. a treatment-burden questionnaire instrument) rather than hand-set
constants — not implemented this session; noted as future work only.

## Classifier audit (Phase 2 of the DEEP_AUDIT resolution)

Every prior session's claims (0% CVR, certified infeasibility, "this
patient is high-risk") assumed the `RiskClassifier`'s own output was
trustworthy without ever checking it directly. This session's audit
(`archive/investigation_v2_to_v6/experiments_v6/CLASSIFIER_AUDIT.md`) found real, respectable discrimination
(AUC 0.83 UCI / 0.96 NHANES) and no meaningful sex-based fairness gap on
NHANES's large subgroups, but also a large, previously-unreconciled gap
between the classifier's own probability scale and the clinical 7.5%
ASCVD/PCE threshold the data label itself is built from: the pipeline's
operational 0.55/0.45 gate sits roughly 7.3× (NHANES) / 2.2× (UCI) higher on
the probability scale than the point where observed outcomes actually cross
7.5%. This was NOT recalibrated this session — doing so would silently
change every previously-published success/CVR/certified-infeasibility
number in this project, which the brief's own working style explicitly
says to flag rather than do quietly. Full numbers and the reasoning for not
recalibrating: `archive/investigation_v2_to_v6/experiments_v6/CLASSIFIER_AUDIT.md`.

## 6. HTML UI (Phase 7)

See `archive/investigation_v2_to_v6/experiments_v6/ui/` and `HOW_TO_RUN.md`. A minimal local FastAPI
backend (`ui/server.py`) wraps the corrected pipeline (Phases 1-6) and a
static single-page frontend (`ui/index.html`) with the five specified
panels. No screenshots included, per the brief.

## 7. FACE / PACE comparison (Phase 8)

**FACE**: `run_face_rerun.py` reruns FACE's own unmodified, unconstrained
design (feature-space k-NN, no admissibility gating at all — confirmed by
reading `benchmarks/run_benchmarks.py::run_face`) on the v6 test cohort with
the same real-value-CVR metric as Phase 1, for a controlled comparison
(rather than the older, differently-scoped `benchmarks/run_benchmarks.py`
numbers).

**PACE**: the actual paper (arXiv:2607.01306) was read directly this
session (see `CHANGES.md` for the discrepancy this revealed against the
brief's "single-step" description). A minimal, same-domain PACE-style
reimplementation (`run_pace_reimpl.py`) — MLP classifier + Phase-4 ASP
admissibility filter + incremental-budget perturb-and-test search, no graph,
no real-patient lookup — was built and run on this project's own cohorts.
This is explicitly NOT a reproduction of PACE's own reported Adult Income
numbers, which are presented separately and clearly labeled in `FINDINGS.md`.

Full comparison table and numbers: `FINDINGS.md`.
