# results/ — final outputs

- `report_final.html` — the final report (patient #139 recommendation card, architecture, CALG figure, final results). Open it in a browser.
- `run_full_nhanes/` — the full NHANES 1999-2023 rerun: 5 seeds × {main, ontology_as_features}, all high-risk test patients (`logs/`, `run_log.json`, per-seed `arms.csv`, `cards.jsonl`, `patients.csv`, ...).
- `run_full_nhanes.orchestrate.sh` — script that ran the rerun (2 seeds in parallel, retry with 1 on failure); `run_full_nhanes.orchestrator.log` — its progress log.
- `run_view/` — symlink view the build reads: NHANES 1999-2023 → `run_full_nhanes`, Kaggle CVD-8k / NHANES 2017-2020 / UCI → the archived report_v3 runs.
- `tables/` — all final tables (CSV; `labelled/` has display names; `full_rerun_changes.csv` is the machine-readable change list).
- `figures/` — all final figures (SVG/PNG/PDF); `fig_calg` (6.5 in, report) and `fig_calg_col` (3.5 in, paper) are the CALG patient figure, `fig_ablation_grid` (2x3 ablation grid), `fig_blocking_rules` figure, `fig01_architecture_notitle` and `fig03_ontology_inheritance` the paper's other figures.

Documentation (`FULL_RERUN_CHANGES.md`, `DECISIONS.md`, `DATA_PROVENANCE_pooled.md`, `REVIEW_rules.md`) is in `../MD files/`.

Older reports, tables, figures and runs are in `../../archive/` (see `archive/MANIFEST.md`).
