# --- Canonical home in src/ (see CHANGES.md for the repo consolidation this file is part of). ---
"""
Registers the real NHANES cohort as a new dataset key, 'nhanes_real', in
configs.dataset_config.DATASET_CONFIGS at runtime -- WITHOUT editing
configs/dataset_config.py on disk (that file, and the synthetic 'nhanes'
entry it already defines, are left completely untouched; see
docs/DATA_PROVENANCE.md section 8).

Import this module (for its side effect) before calling
get_dataset_config('nhanes_real') or get_dataloaders(dataset_name='nhanes_real').
"""
from pathlib import Path

from configs.dataset_config import DATASET_CONFIGS

# Real NHANES CSV lives at top-level data/ (promoted here from
# experiments_v3/data/ during the repo consolidation -- see CHANGES.md and
# .gitignore's explicit exception for this file, since data/* is otherwise
# ignored). A historical copy remains at
# archive/investigation_v2_to_v6/experiments_v3/data/nhanes_real.csv but
# src/ must not depend on archive/, so this reads the top-level copy.
_REAL_NHANES_CSV = str(Path(__file__).resolve().parents[2] / "data" / "nhanes_real.csv")

if "nhanes_real" not in DATASET_CONFIGS:
    DATASET_CONFIGS["nhanes_real"] = {
        # Identical feature schema/effort weights as the existing synthetic 'nhanes' entry --
        # reused verbatim (not re-derived), since Phase A confirmed every model feature has
        # a direct, one-to-one NHANES source (DATA_PROVENANCE.md section 2).
        "FEATURE_METADATA": DATASET_CONFIGS["nhanes"]["FEATURE_METADATA"],
        "CLINICAL_EFFORT_WEIGHTS": DATASET_CONFIGS["nhanes"]["CLINICAL_EFFORT_WEIGHTS"],
        "dataset_path": _REAL_NHANES_CSV,
    }
