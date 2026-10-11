"""Shared paths, config, seeding, run provenance and never-overwrite helpers (PIPELINE P0)."""
from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parent
KNOWLEDGE = EXP_DIR / "knowledge"
RESULTS = EXP_DIR / "results"
# The patient shown in the report (Example output, CALG figure): dataset, seed, and the patient's position in that seed's test set (0 .. n_test-1).
# Change the number here, or try one without editing: CARELG_EXAMPLE_POS=350 .venv/bin/python -m experiment_final.src.build_report
EXAMPLE_DATASET, EXAMPLE_SEED, EXAMPLE_POS = "nhanes_pooled", 2, int(os.environ.get("CARELG_EXAMPLE_POS", 139))
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RUNS = RESULTS / "runs"          # runs written by run_all.py (created on demand)
# Archived inputs of the final report/tests (moved by the repo clean-up, see archive/MANIFEST.md)
ARCHIVE = REPO / "archive" / "experiment_final_prev"
OLD_RUNS = ARCHIVE / "v3" / "runs"                 # report_v3 runs: Kaggle CVD-8k, NHANES 2017-2020, UCI results + test fixtures
OLD_TABLES = ARCHIVE / "v3" / "tables"             # report_v3 tables (old side of FULL_RERUN_CHANGES.md, learning curve, cycle-held-out, mortality)
OLD_TABLES_V4 = ARCHIVE / "v4" / "tables_v4"       # accuracy_backfill_v3runs.csv

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_config() -> Dict[str, Any]:
    """Locked config.yaml (P0)."""
    with open(EXP_DIR / "config.yaml") as fh:
        return yaml.safe_load(fh)


def git_sha() -> str:
    """Current HEAD SHA (read-only git) for provenance (P0)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def new_path(path: Path) -> Path:
    """Never overwrite: returns `path` if free, else path with _v2, _v3 ... suffix (P0)."""
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while True:
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def run_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def md_table(df, index: bool = True, floatfmt: str = "{:.2f}") -> str:
    """Minimal GitHub-markdown table (no tabulate dependency) (P15)."""
    import pandas as pd
    d = df.reset_index() if index else df
    fmt = lambda v: (floatfmt.format(v) if isinstance(v, float) and not pd.isna(v) else ("" if pd.isna(v) else str(v)))  # noqa: E731
    rows = [[str(c) for c in d.columns]] + [[fmt(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])
