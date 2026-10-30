# MD files — project documentation

- `PIPELINE.md` — the phased pipeline specification (P0-P15) the code follows.
- `DATASET_ADAPTER.md` — the `DatasetBundle` contract every dataset adapter must meet.
- `NHANES_POOLED.md` — how the pooled NHANES 1999-2023 cohort is downloaded and harmonised.
- `DECISIONS.md` — log of every judgement call made while building the pipeline.
- `DATA_PROVENANCE_pooled.md` — participant counts at every filter for the pooled NHANES cohort.
- `REVIEW_rules.md` — every knowledge row listed for human review (all `verify=1`).
- `FULL_RERUN_CHANGES.md` — every NHANES 1999-2023 number that changed from report_v3 (old → new) and the Tier-1 / contraindication flags.

Folder READMEs (`src/`, `adapters/`, `knowledge/`, `results/`, `paper/`) and the repo-root `CLAUDE.md` / `AGENTS.md` stay where they are.
