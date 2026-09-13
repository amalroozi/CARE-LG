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
