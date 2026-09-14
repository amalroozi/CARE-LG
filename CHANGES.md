# CHANGES.md — repo consolidation

What moved where, in this reorganization session. Nothing was deleted —
every file was moved or copied, never removed. No git commands were run
this session (no add, no commit, no push); the working tree is left for
manual review.

## 1. `src/` internal reorg

- `src/blackbox_model.py` → **`src/classifier/blackbox_model.py`** (canonical).
  `src/blackbox_model.py` is now a 3-line compatibility shim re-exporting
  from the new location, so `scripts/`, `benchmarks/run_benchmarks.py`, and
  any other existing `from src.blackbox_model import ...` keep working
  unchanged. `src/pipeline.py` (the one file inside `src/` that used the old
  path) was updated to import from the canonical location directly.
- Everything else in `src/` (`vae/`, `graph/`, `recourse/`, `reporting/`,
  `data/`, `seeding.py`, `pipeline.py`) was already close to the target
  layout and did not need to move.

## 2. Real NHANES data promoted to top-level `data/`

`src/data/dataset_registration.py` and `src/recourse/explanations.py`'s
`nhanes_guideline_context` previously read `experiments_v3/data/nhanes_real.csv`
and `nhanes_real_full.csv` directly — a functional dependency of the CURRENT
implementation on a directory about to become historical archive. Both
files were copied to **`data/nhanes_real.csv`** and
**`data/nhanes_real_full.csv`**, and both `src/` files were updated to read
from there instead. **`.gitignore`** was given a narrow, explicit exception
(`!data/nhanes_real.csv`, `!data/nhanes_real_full.csv`) so these two files
are actually committed — `data/*` was otherwise fully ignored, which would
have silently broken a fresh clone (the file `src/` now depends on would
never have been checked out). No other `data/*` ignore behavior changed.

## 3. Six investigation directories moved wholesale to archive

`experiments_v2/`, `experiments_v3/`, `experiments_v4/`, `experiments_v5/`,
`experiments_v6/`, `experiments_v6_audit/` → **`archive/investigation_v2_to_v6/`**,
internal structure untouched (per the brief: historical record, not
reorganized inside). These are **not** expected to run from their new
location — several of their own scripts import via `experiments_v6.lib.*`
paths that only resolved when `experiments_v6/` was a top-level package;
`docs/HISTORY.md` states this explicitly. Current reproduction uses `src/`
and `scripts/` instead.

## 4. Current results copied into top-level `results/`

From `archive/investigation_v2_to_v6/experiments_v6/results/` (the final,
corrected numbers, not any intermediate/pre-fix session's output):
`uci_per_patient.csv`, `nhanes_real_per_patient.csv`, both blindspot CSVs +
seed summaries, `classifier_audit.json`, `neurosymbolic_regression_summary.json`,
`face_rerun_per_patient.csv`, `pace_reimpl_per_patient.csv`,
`preference_demo.csv`, and their metadata JSONs → **`results/tables/`**.
`REPORT.pdf` → `results/REPORT.pdf`. `report/index.html` → regenerated as
`results/full_report.html` (see item 6). The two example single-patient
reports → `results/`. Three figures actually cited by the current report
(`figA_ablation_bars.png`, `figB_abstention_decomposition.png`,
`fig1_motivating_gap.png`, originally in `experiments_v3/figures/` and
`experiments_v5/figures/`) → **`results/figures/`**.

## 5. Current docs copied into `docs/`

`experiments_v6/METHOD.md` → `docs/METHOD.md` (header rewritten to describe
`src/` as the canonical implementation; internal path references updated to
point at `archive/investigation_v2_to_v6/...`). `experiments_v3/DATA_PROVENANCE.md`
→ `docs/DATA_PROVENANCE.md`. `experiments_v6/CLASSIFIER_AUDIT.md` →
`docs/CLASSIFIER_AUDIT.md`. **`docs/FINDINGS.md`** and **`docs/HISTORY.md`**
are new consolidations written this session, pulling only numbers already
published in the archived `FINDINGS.md` files — nothing recomputed.

## 6. UI, tests, and reproduction scripts moved to their new top-level homes

- `experiments_v6/ui/{server.py,index.html}` → **`ui/`**. `server.py`
  already imported from `src/` directly (from a prior session); only its
  `REPO_ROOT` depth and `RESULTS_DIR` (split into `RESULTS_TABLES_DIR` for
  reading blindspot CSVs and `REPORT_OUTPUT_DIR` for writing generated
  reports) were updated for the new location.
- `experiments_v6/tests/test_gate_symmetry.py` → **`tests/`**. Already
  imported from `src/` directly; only `REPO_ROOT`'s depth was updated.
- `experiments_v6/run_*.py`, `make_report.py`, and `report/{build_report,
  collect_examples,collect_journey}.py` → **`scripts/`** and
  **`scripts/report/`**. These previously imported via `experiments_v6.lib.*`
  shim modules (which themselves just re-exported `src/`); imports were
  changed to reference `src/` directly, and every `RESULTS_DIR`/`out_dir`/
  `OUT_PDF` default was updated to point at `results/tables/` or `results/`.
  `scripts/report/build_report.py`'s many inline "Source:" citation strings
  (in the HTML it generates) were also updated to cite the new paths.

## 7. Root files

`requirements.txt`, `configs/` untouched in location (`configs/dataset_config.py`
had only its existing DEEP_AUDIT-issue-9 comment's path reference updated
to point at the archive). **`README.md`** rewritten (old version described
an outdated, pre-fix benchmark table and claims that no longer reflect
`docs/FINDINGS.md`'s actual, audited numbers). `benchmarks/` and the
original `scripts/test_day1.py`/`test_day2.py`/`test_pipeline.py`/
`generate_*.py`/`download_nhanes.py` (pre-dating the whole
experiments_v2-v6 investigation, and never part of the brief's explicit
move list) were left in place — they still import `src.blackbox_model`,
`src.vae.model`, `src.graph.calg`, `src.recourse.search`, all of which
still exist unmoved at their original `src/` locations (the original,
pre-fix reference pipeline, kept as-is, not superseded by this
reorganization — only the classifier's file location changed, and that
move is covered by a compatibility shim, so these scripts needed no edits).

## Verification performed this session (see final summary for full output)

- `src/` imports cleanly (every module, including the new `data/nhanes_real*.csv`
  dependency) after each incremental move.
- `pytest tests/` passes, including `test_gate_symmetry.py`, from its new
  top-level location.
- The full 5-seed, both-cohort grid was rerun via `scripts/run_grid.py` and
  reproduced `docs/FINDINGS.md`'s numbers exactly (real-value CVR 0.00% ±
  0.00% both cohorts; success 38.6% ± 13.1% UCI / 37.7% ± 3.4% NHANES).
- `scripts.report.build_report` regenerated `results/full_report.html`
  successfully from the new paths, zero missing sections.
- The UI and `scripts/run_single_patient.py` were exercised against the new
  structure (see final summary).

## Baseline rebuild: DiCE, FACE, Growing Spheres brought to citation standard

`benchmarks/run_benchmarks.py`'s `run_dice`, `run_face`, and
`run_growing_spheres` were confirmed AI-generated prior to this project's
documented investigation phase (git history places them in one of the
repository's first commits), with no source-paper citation and no
algorithm-fidelity documentation anywhere in the file — inconsistent with
every baseline built since (REVISE, PACE, the corrected FACE rerun), which
all cite their source and document their own deviations. **Step 0 check
performed before any change**: grepped `docs/`, `results/`, `scripts/`,
`ui/`, `src/` for references to these three functions or to
`benchmarks/results.csv`/`.json` — FACE's currently-published numbers in
`docs/FINDINGS.md` already came from a prior session's corrected rerun
(`scripts/run_face_rerun.py`), not from `benchmarks/run_benchmarks.py`; no
DiCE or Growing Spheres numbers existed anywhere in current docs/results.
So no previously-published number depended on the original three functions
being touched, and none needed correcting as a result of this rebuild.

**Added**: `src/baselines/` (new package) — `dice.py` (Mothilal, Sharma,
Tan, FAT* 2020, arXiv:1905.07697: the paper's actual joint diverse-
counterfactual loss, k candidates optimized together with a DPP diversity
term, not the original's single L2-penalized counterfactual),
`growing_spheres.py` (Laugel, Lesot, Marsala, Renard, Detyniecki,
arXiv:1712.08443: faithful expanding-spherical-shell sampling with correct
uniform-by-volume shell sampling), and `face.py` (promoted from
`scripts/run_face_rerun.py`, unchanged algorithm, now the single canonical
FACE implementation both that script and any future one import from).
`scripts/run_face_rerun.py` updated to import from `src/baselines/face.py`
rather than containing the algorithm itself, and extended to also report
KDE plausibility and clinical effort (previously only success/CVR/latency),
matching the metrics reported for the other two rebuilt baselines.
`scripts/run_baselines_rebuilt.py` (new): evaluates the new DiCE and
Growing Spheres under this project's own corrected protocol (real-value
CVR, full 5-seed/both-cohort cohorts) AND runs the ORIGINAL legacy
functions on a matched 15-patients-per-seed subset for a direct old-vs-new
comparison.

**Disclosed deviations from the papers** (documented in each module's own
docstring, not silent): DiCE's `max_iter` reduced from the paper's 5000 to
300 for this session's full-cohort compute budget; DiCE's diversity term
uses `logdet` rather than raw `det` for numerical stability (matches the
official `dice-ml` reference implementation's own choice); Growing
Spheres' `n_samples` per layer reduced from the paper's 10,000 to 2,000;
Growing Spheres' paper-described sparsity-reduction post-processing phase
(a separate, optional refinement step) is not implemented, only the
defining shell-search mechanism.

**Real results, executed this session** (raw data:
`results/tables/baselines_rebuilt_per_patient.csv`,
`results/tables/baselines_old_vs_new.csv`,
`results/tables/baselines_rebuilt_metadata.json`; full tables in
`docs/FINDINGS.md` section 7 and `docs/HISTORY.md`). Headline finding,
reported plainly: the original `run_dice` was not a faithful DiCE
implementation by the paper's own definition — no diversity mechanism at
all, DiCE's entire defining contribution. Old-vs-new, same patients, same
seeds: DiCE success rose 66.7%→98.7% (UCI) and 49.3%→97.3% (NHANES); DiCE
real-value CVR fell 98.0%→74.3% (UCI) and 100.0%→69.9% (NHANES) once the
paper's own actionability mechanism (holding immutable/non-decreasing
features fixed) was actually applied. Growing Spheres changed less
(82.7%→94.7% / 98.7%→100.0% success; already ~100% CVR both versions, since
neither has any constraint awareness).

`benchmarks/run_benchmarks.py`'s original three functions were left
UNCHANGED (not deleted, per Hard Rule 3) with a superseded-notice added to
the file's module docstring pointing to `src/baselines/`.
