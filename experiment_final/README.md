# experiment_final — CARE-LG: tiered, ontology-grounded actionable recourse

The final version of the project: every high-risk patient gets (1) a certified path to a real lower-risk patient, (2) an ontology-guided synthetic counterfactual, or (3) a certified statement that none exists, plus an escalation-ordered, contraindication-free recommendation card.

## Where things are
| folder / file | what |
|---|---|
| `src/`, `adapters/`, `knowledge/`, `tests/` | pipeline code, dataset adapters, clinical knowledge tables, tests (each has a README) |
| `config.yaml` | locked configuration (datasets, seeds, thresholds, paths) |
| `run_full.py` | full NHANES 1999-2023 rerun (all high-risk patients, 5 seeds, resumable); `run_all.py` (runs the other datasets) and `single_patient.py` (print one patient's card) are the other entry points |
| `results/` | final outputs: `report_final.html`, `run_full_nhanes/`, `tables/`, `figures/` (see `results/README.md`) |
| `paper/` | final paper `main_final.tex` / `.pdf` |
| `MD files/` | documentation: `PIPELINE.md`, `DATASET_ADAPTER.md`, `NHANES_POOLED.md`, `DECISIONS.md`, `REVIEW_rules.md`, ... |
| `data/` | `raw/` (downloaded NHANES, not tracked) and `processed/` (caches used by the figures) |
| `../archive/` | every previous version (`MANIFEST.md` lists what moved and why) |

## Common commands (from the repo root `care_lg/`)
```
.venv/bin/python -m pytest experiment_final            # tests
.venv/bin/python -m experiment_final.src.build_report     # rebuild results/report_final.html
open experiment_final/results/report_final.html
.venv/bin/python -m experiment_final.src.narrate --tier 2 # print a plan for one card (tiers 1, 2, 3)
```
The full build order for tables, figures, report and paper is in `src/README.md`.
