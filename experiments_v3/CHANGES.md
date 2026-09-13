# CHANGES.md — experiments_v3 file-by-file explanation

**No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, or `data/` were modified.
`experiments_v2/` is also left completely untouched** (per Hard Rule 3) — every v2 result
file, figure, and finding from the prior session still stands exactly as it was; this
session only adds `experiments_v3/`. No git commands beyond read-only
`status`/`log`/`diff`/`rev-parse` were run; nothing was committed or pushed.

## Phase A — real NHANES data

### `experiments_v3/data_raw/*.xpt`
9 raw NHANES 2017–March 2020 pre-pandemic component files, downloaded via `curl` from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{FILE}.xpt` (the corrected,
currently-live path — see `DATA_PROVENANCE.md` §1 for the wrong-path dead end hit and
resolved first). Kept for audit, not deleted.

### `experiments_v3/build_real_nhanes.py`
Merges the 9 components on `SEQN`, computes the 2013 ACC/AHA Pooled Cohort Equations
10-year ASCVD risk (coefficients cross-checked against a citable open-source reference
implementation — see `DATA_PROVENANCE.md` §3), thresholds at ≥7.5% for `cvd_risk_flag`,
restricts to the PCE-valid age range [40,79], and complete-case-excludes any remaining
missing values. Writes `experiments_v3/data/nhanes_real.csv` (model-facing, exactly
matching the existing 7-feature schema) and `nhanes_real_full.csv` (with provenance
columns). Full missingness/sample-size accounting: `DATA_PROVENANCE.md`.

### `experiments_v3/lib/dataset_registration.py`
Registers a NEW dataset key, `'nhanes_real'`, into `configs.dataset_config.DATASET_CONFIGS`
**at runtime** (not by editing `configs/dataset_config.py` on disk), reusing the existing
`'nhanes'` entry's `FEATURE_METADATA`/`CLINICAL_EFFORT_WEIGHTS` verbatim and pointing
`dataset_path` at the new real CSV. Every v3 script imports this module for its side effect
before calling `get_dataset_config('nhanes_real')`/`get_dataloaders(dataset_name='nhanes_real')`.
The synthetic `'nhanes'` key and `data/nhanes.csv` are completely unaffected and remain
runnable exactly as before.

## Phase B — type-aware VAE decoder

### `experiments_v3/lib/vae_v3.py`
`TabularVAETypeAware` + `train_vae_type_aware`: a new decoder architecture (encoder
unchanged), added ALONGSIDE `src/vae/model.py::TabularVAE` (untouched), not instead of it.
Per-column type auto-detected from `FEATURE_METADATA` + training data (`detect_column_types`):
  - **Binary** columns (immutable/categorical_mutable with ≤2 distinct values: `sex`,
    `fasting_blood_sugar`, `exercise_angina`): dedicated sigmoid head, BCE loss.
  - **Multi-category** columns (categorical_mutable with >2 values: UCI's `slope` ∈
    {1,2,3}): dedicated softmax head over the observed categories, cross-entropy loss on
    the class index. The value exposed through `decode()` is the SOFT EXPECTATION under the
    softmax (Σ class_value×P(class)), not a hard argmax — required so the decoder stays
    differentiable in `z` everywhere (`torch.func.vmap(jacrev(...))`, used by
    `riemannian_batch.py`'s batched Jacobian, needs this).
  - **Age** (`non_decreasing`): a linear regression head like any other continuous
    feature, but its MSE term is weighted `age_loss_weight=8.0`× in the training loss
    (default; not swept, but 8x was chosen as a clearly-more-than-1x emphasis without
    destabilizing the other heads' training — see the reconstruction-error table below for
    the measured effect).
  - **All other continuous_mutable columns**: standard linear regression head, MSE, weight 1×.
`decode(z)` always returns a single `(batch, input_dim)` tensor in the same column order
as the original `TabularVAE.decode()` — a drop-in replacement everywhere in
`experiments_v2/v3` code that calls `vae.decode(z)`. **Implementation pitfall found and
fixed**: an initial version built this output tensor via in-place indexed assignment into
a freshly-allocated `torch.zeros(...)`, which silently breaks under `torch.func.vmap`
(`riemannian_batch.py`'s batched Riemannian distance) with a runtime error about in-place
ops on non-batched tensors; fixed by assembling the output via `torch.stack` over
per-column tensors instead (fully out-of-place), verified working with `vmap(jacrev(...))`.

### `experiments_v3/lib/pipeline_v3.py`
Identical in structure to `experiments_v2/lib/pipeline.py`, but calls
`train_vae_type_aware` instead of the original `train_vae`. Classifier training is
UNCHANGED (`src.blackbox_model.train_blackbox_model`, same hyperparameters), since REVISE
(Phase C) needs to share the exact same trained classifier as CARE-LG.

### `experiments_v3/compare_decoders.py`
Standalone, rerunnable script that produces the before/after comparison below (persisted
as an ad hoc analysis during the session; this is its proper, reproducible form).

### `experiments_v3/results/vae_reconstruction_error_v3.csv`
Direct before/after comparison (same seed, same data, same epoch budget — the only
difference is decoder architecture) of the v2 original decoder vs. the v3 type-aware
decoder, per feature, per dataset:

| Dataset | Feature | v2 original MAE | v3 type-aware MAE | Improvement |
|---|---|---:|---:|---:|
| UCI | age | 6.83 yr | 1.63 yr | **4.2× lower** |
| UCI | sex (decoded accuracy) | 41.8% | 67.5% | improved, still weak |
| UCI | fasting_blood_sugar (accuracy) | 86.9% | 86.9% | unchanged (identical classification decisions; BCE calibration differs but doesn't flip any decision on this imbalanced ~14.5%-positive feature) |
| UCI | exercise_angina (accuracy) | 69.2% | 69.2% | unchanged (same reason) |
| UCI | slope (accuracy) | 47.3% | 54.9% | improved |
| UCI | other continuous (resting_bp, cholesterol, max_heart_rate, oldpeak) | — | — | essentially unchanged (±1-2%), as expected (these heads are architecturally identical to before) |
| Real NHANES | age | 4.12 yr | 0.72 yr | **5.7× lower** |
| Real NHANES | sex (decoded accuracy) | 62.4% | 68.2% | improved, still weak |
| Real NHANES | other continuous | — | — | essentially unchanged |

**Verdict on Phase B (stated plainly, not oversold)**: the age fix worked dramatically — a
4-6× reduction in age reconstruction error, which is exactly the feature the non-decreasing/
age-horizon constraint checks depend on most. The binary-classification heads (sex
especially) improved only modestly and remain barely better than chance at recovering sex
from the 4-dimensional latent code — this appears to be an information bottleneck in the
(unchanged) 4-dimensional encoder rather than something a better decoder head alone can
fix; we did not increase latent dimensionality, per the brief's instruction to make "one
solid attempt" at the decoder and report the residual gap as a limitation, not chase it
indefinitely. See `REPORT.pdf` and `FINDINGS.md` for whether this reduced decoded-space CVR
in the full pipeline (it did, substantially, for age-driven violations; sex-driven
violations persisted at a similar rate — full numbers in FINDINGS.md).

## Phase C — REVISE

### `experiments_v3/lib/revise.py`
`run_revise(x0, vae, classifier, device, lr=0.05, steps=100, lambda_prox=0.01,
target_threshold=0.45)`: faithful implementation of Joshi et al. 2019 (arXiv:1907.09615) —
Adam optimization directly over the latent code `z`, minimizing classifier risk on
`decode(z)` plus a proximity term `lambda_prox * ||z - z0||^2` in LATENT space, sharing the
exact same trained VAE + classifier as CARE-LG for the given (dataset, seed). Full method
description and hyperparameter derivation (including a real dead end found and fixed) in
the module docstring; summary:
  - **First attempt, lambda_prox=0.5 (naive guess): only 15% success on a 20-patient UCI
    pilot.** Diagnosis: the raw gradient of classifier-risk w.r.t. `z` at the starting point
    has norm ≈0.02–0.04 on this model, so a quadratic penalty at weight 0.5 overwhelms it
    almost immediately and `z` barely moves for 100 steps. The only "successes" were the 3
    patients whose VAE-reconstructed starting point (`decode(z0)`, with NO optimization at
    all) already happened to sit below the 0.45 risk threshold — itself a notable, separate
    finding (see below).
  - **Swept `lambda_prox` ∈ {0.001, 0.005, 0.01, 0.05} × `lr` ∈ {0.05, 0.1}** (25 UCI
    seed-0 patients, `experiments_v3/results/revise_lr_sweep.csv`): a sharp transition at
    `lambda_prox=0.05` (64-72% success) vs. `lambda_prox≤0.01` (100% success, mean
    latent-to-feature-space distance ≈3.0). **Final choice: lambda_prox=0.01, lr=0.05,
    steps=100** (steps matched to this repo's existing `run_dice` budget for cross-method
    comparability).

**Related finding surfaced while debugging REVISE (reported honestly, not buried)**: the
classifier's risk prediction on a VAE round-trip reconstruction of a genuinely high-risk
patient (`classifier(decode(encode(x0)))`) is frequently much lower than its prediction on
the true `x0` directly (e.g. one UCI patient: direct risk 0.739, risk-after-reconstruction
0.299 — already "successful" by the 0.45 threshold with ZERO gradient steps). This is a
symptom of the same VAE fidelity issue Phase B targets (reconstructions regress toward the
training distribution's mean, i.e. look less like an outlier high-risk case), but it
manifests differently here: it means part of any VAE-decode-based method's apparent
"success" (REVISE here) can be an artifact of decoder smoothing rather than a genuine
actionable recommendation. This does NOT apply to CARE-LG's own success/failure
determination, which evaluates `low_risk_mask` on TRUE `X_train` feature values, not
decoded reconstructions (see `src/graph/calg.py` / `experiments_v3/lib/pipeline_v3.py`) —
but it DOES apply to REVISE's own success criterion, and is flagged prominently in
`REPORT.pdf` and `FINDINGS.md` rather than left as an unstated confound.

### `experiments_v3/revise_sweep.py`
Standalone, rerunnable script that produces the hyperparameter sweep in
`revise_lr_sweep.csv` (persisted as an ad hoc analysis during the session; this is its
proper, reproducible form).

### `experiments_v3/run_baselines.py`
Reuses `run_dice`/`run_face`/`run_growing_spheres` UNCHANGED from
`benchmarks/run_benchmarks.py` (exactly as `experiments_v2/run_baselines.py` did), and adds
`run_revise` as a genuine fourth baseline, run under the identical seeded pipeline and same
high-risk test patients as the main grid, on both `uci` and `nhanes_real`.

## Phase D — full grid rerun

### `experiments_v3/run_grid.py`, `experiments_v3/lib/{graph_build,constraints_ext,riemannian_batch,search_v2,seeding}.py`
Duplicated from `experiments_v2/lib/` (kept self-contained rather than importing across
the v2/v3 boundary) with only import-path and dataset-key updates — none of the underlying
graph/constraint/search LOGIC changed, since Phase D's job is to rerun the *same* ablation
methodology on corrected inputs (real data, better decoder), not to change the methodology
itself. Datasets: `{uci, nhanes_real}` (the v2 synthetic `nhanes` grid is NOT rerun here —
its results remain valid and available in `experiments_v2/results/` exactly as before).

### `experiments_v3/run_blindspot.py`
Duplicated from `experiments_v2/run_blindspot.py`, same import/dataset-key updates. Rerun
on both `uci` and `nhanes_real` (v2 only had non-trivial abstentions on UCI; Phase D
explicitly re-checks whether real NHANES also produces zero abstentions, per the brief's
instruction not to assume the synthetic-data finding holds).

### `experiments_v3/aggregate.py`, `experiments_v3/make_table1.py`, `experiments_v3/make_figures.py`, `experiments_v3/make_report.py`
Duplicated from `experiments_v2/`, updated for the `nhanes_real` dataset key and to
include REVISE as a genuine additional Table 1 row (labeled as genuinely implemented this
session, not a placeholder). Figure/report captions and the executive summary are
rewritten to describe what was ACTUALLY observed in this session's runs (not copied from
v2's captions), including a new "What changed since v2 and why" opening section.

## Full command list to regenerate every number and figure from scratch

```bash
# Phase A: download raw NHANES files (see DATA_PROVENANCE.md for the exact URLs used)
mkdir -p experiments_v3/data_raw
BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
for f in P_DEMO P_BPXO P_BMX P_TCHOL P_HDL P_GHB P_BPQ P_DIQ P_SMQ; do
  curl -s -o "experiments_v3/data_raw/${f}.xpt" "${BASE}/${f}.xpt"
done

# Phase A: build the real NHANES cohort + PCE label
.venv/bin/python experiments_v3/build_real_nhanes.py

# Phase B: decoder before/after comparison (writes vae_reconstruction_error_v3.csv)
.venv/bin/python -m experiments_v3.compare_decoders

# Phase C: REVISE hyperparameter sweep (writes revise_lr_sweep.csv)
.venv/bin/python -m experiments_v3.revise_sweep

# Phase D: main ablation grid, both datasets, 5 seeds, 12 cells each
.venv/bin/python -m experiments_v3.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4

# Phase D: blindspot vs. true-infeasibility decomposition (both datasets this time)
.venv/bin/python -m experiments_v3.run_blindspot --datasets uci nhanes_real --seeds 0 1 2 3 4

# Phase C+5: baselines (DiCE / FACE / GrowingSpheres / REVISE), same seeds & patients as the grid
.venv/bin/python -m experiments_v3.run_baselines --datasets uci nhanes_real --seeds 0 1 2 3 4

# Aggregate: Table 1 CSVs, paired Wilcoxon tests, RQ summary JSON
.venv/bin/python -m experiments_v3.aggregate

# Final Table 1 (brief's exact row/column spec) as CSV + LaTeX
.venv/bin/python -m experiments_v3.make_table1

# Figures A-E (PNG + PDF)
.venv/bin/python -m experiments_v3.make_figures

# Assemble the PDF report
.venv/bin/python -m experiments_v3.make_report
```

## Interpretation choices flagged for the record (beyond DATA_PROVENANCE.md)

- **Race-group mapping for the PCE label**: `RIDRETH3==4` (Non-Hispanic Black) → African-
  American equation; every other NHANES race/ethnicity category → White equation, per the
  guideline's own recommended approximation for populations without a dedicated equation
  (see `DATA_PROVENANCE.md` §3).
- **Prevalent ASCVD not excluded**: a fully faithful primary-prevention PCE application
  would exclude respondents with a prior heart attack/stroke/CHD diagnosis; we did not
  (would require an additional questionnaire component not otherwise needed by this
  schema) — documented as a limitation, not silently omitted.
- **age_loss_weight=8.0** for the type-aware decoder's age head was chosen as a clear,
  round emphasis factor, not tuned via a grid search (the brief calls for "one solid
  attempt," not an architecture search) — the measured 4-6× MAE improvement (Phase B table
  above) shows it was effective without needing further tuning.
- **REVISE's proximity term lives in latent space** (`||z-z0||^2`), while this repo's
  existing DiCE lives in feature space (`||x-x0||`) — this is not an inconsistency to fix;
  it is the actual methodological difference between the two papers being compared.
