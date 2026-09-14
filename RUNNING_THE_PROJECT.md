# Running CARE-LG — All Commands by Architecture Layer

All commands assume you're in the repo root (`/Users/amalroozi/.gemini/antigravity/scratch/care_lg`)
and use the project's own virtualenv interpreter directly — no `source .venv/bin/activate` needed,
but it works too if you prefer it.

## 0. Setup

```bash
# one-time: create the venv and install dependencies (skip if .venv/ already exists)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# one-time: fetch the real NHANES cohort (writes data/nhanes_real.csv)
.venv/bin/python -m scripts.download_nhanes
```

## 1. Data → Classifier → VAE → Graph (the core pipeline)

There's no separate command to "train the classifier" or "train the VAE" on their own —
`src/pipeline.py::run_dataset_seed(dataset, seed)` trains both fresh, every time, for a given
(dataset, seed) pair, then encodes the graph. Every script below calls it internally. If you just
want to sanity-check the pipeline trains cleanly with no downstream experiment:

```bash
.venv/bin/python -c "
import src.data.dataset_registration
from src.pipeline import run_dataset_seed
run = run_dataset_seed('uci', 0)
print('trained OK:', run.X_train.shape, 'train patients,', len(run.high_risk_test_idx), 'high-risk test patients')
"
```

## 2. Main grid — CALG search, all metric/constraint combinations

The corrected (B1+B2-fixed) pipeline, full ablation grid (Riemannian/Euclidean × hard/soft-λ/none),
both datasets, all 5 seeds. This is the source of `results/tables/uci_per_patient.csv` and
`results/tables/nhanes_real_per_patient.csv`.

```bash
.venv/bin/python -m scripts.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4
```

## 3. Neurosymbolic layer — regression check against the hardcoded predicate

Verifies the clingo/ASP rule re-encoding agrees exactly with `src/graph/clinical_constraints.py`'s
hardcoded checker (2000 sampled pairs per dataset).

```bash
.venv/bin/python -m scripts.run_neurosymbolic_regression
```

## 4. Certified infeasibility vs. blindspot decomposition

For every abstention under the corrected gate: exhaustive check + 2×-k and synthetic-densification
probes, classifying it as certified infeasible / confirmed blindspot / unresolved.

```bash
.venv/bin/python -m scripts.run_blindspot
```

## 5. Baselines (comparison methods)

```bash
# FACE (density-weighted graph, real target)
.venv/bin/python -m scripts.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4

# DiCE + GrowingSpheres (synthetic targets)
.venv/bin/python -m scripts.run_baselines_rebuilt --datasets uci nhanes_real --seeds 0 1 2 3 4

# PACE (neurosymbolic MLP + ASP, from-scratch reimplementation)
.venv/bin/python -m scripts.run_pace_reimpl --datasets uci nhanes_real --seeds 0 1 2 3 4
```

## 6. Classifier threshold audit

Empirical crossing point between the classifier's own probability and the real 7.5% ACC/AHA clinical
threshold.

```bash
.venv/bin/python -m scripts.run_classifier_audit
```

## 7. Improvement experiments (age-ceiling follow-up)

Run in this order — each one is a prerequisite check before the next:

```bash
# diagnostic: is a certified-infeasible patient's ceiling genuine (age) or data-scarcity?
.venv/bin/python -m scripts.run_age_ceiling_diagnostic

# Phase 1: does a real 2-hop chain exist that the k-NN graph missed? (ruled out — 0% found)
.venv/bin/python -m scripts.run_multihop_check

# Phase 1.5: age-stratified recourse targets (adopted — real success-rate gain)
.venv/bin/python -m scripts.run_age_stratified_experiment

# Stage 2: directed synthetic search (Riemannian natural-gradient, age frozen), only on
# patients Stage 1 certifies infeasible
.venv/bin/python -m scripts.run_stage2_synthetic_search

# disclosed negative-result experiment: recalibrating thresholds to the real 7.5% line
# (made results WORSE — kept as an honest negative finding, not part of the final pipeline)
.venv/bin/python -m scripts.run_recalibrated_threshold_experiment
```

## 8. v8 — age dropped entirely (separate, self-contained experiment tree)

Does not touch `src/`; classifier + VAE retrained from scratch with age removed as a model input.

```bash
.venv/bin/python -m experiments_v8.run_v8_no_age
.venv/bin/python -m experiments_v8.build_v8_report      # writes + opens experiments_v8/v8_report.html
```

## 9. Single-patient report (one patient, full architecture walkthrough)

```bash
.venv/bin/python -m scripts.run_single_patient --dataset uci --seed 0 --patient_idx 0
# add --no-open to write the file without launching a browser
```

## 10. Final combined report (the one deliverable — recommendation + all figures + tables)

Builds `results/final_report.html` and every SVG in `results/figures/`; auto-opens in your browser.

```bash
.venv/bin/python -m scripts.build_final_report
```

## 11. Extra / standalone figures

```bash
# Fig. 13: whole-graph CALG structure overview (distinct from the neighborhood figure
# embedded in the final report)
.venv/bin/python -m scripts.build_fig13_calg_graph
```

## 12. Regression tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## 13. Everything, start to finish (fresh clone → final report)

```bash
.venv/bin/python -m scripts.download_nhanes
.venv/bin/python -m scripts.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m scripts.run_neurosymbolic_regression
.venv/bin/python -m scripts.run_blindspot
.venv/bin/python -m scripts.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m scripts.run_baselines_rebuilt --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m scripts.run_pace_reimpl --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m scripts.run_age_ceiling_diagnostic
.venv/bin/python -m scripts.run_multihop_check
.venv/bin/python -m scripts.run_age_stratified_experiment
.venv/bin/python -m scripts.run_stage2_synthetic_search
.venv/bin/python -m scripts.build_final_report
```

**Note on `experiments_v6.*` in some scripts' own docstrings**: a few files' module docstrings still
say `python -m experiments_v6.run_xyz` from before the repo reorganization moved them into `scripts/`.
The commands above use each file's real current location (`scripts.*` / `experiments_v8.*`) — use
those, not what an individual docstring says.
