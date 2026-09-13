"""
Registers the real NHANES cohort as a new dataset key, 'nhanes_real', in
configs.dataset_config.DATASET_CONFIGS at runtime -- WITHOUT editing
configs/dataset_config.py on disk (that file, and the synthetic 'nhanes'
entry it already defines, are left completely untouched; see
DATA_PROVENANCE.md section 8).

Import this module (for its side effect) before calling
get_dataset_config('nhanes_real') or get_dataloaders(dataset_name='nhanes_real').
"""
from pathlib import Path

from configs.dataset_config import DATASET_CONFIGS

_REAL_NHANES_CSV = str(Path(__file__).resolve().parents[1] / "data" / "nhanes_real.csv")

if "nhanes_real" not in DATASET_CONFIGS:
    DATASET_CONFIGS["nhanes_real"] = {
        # Identical feature schema/effort weights as the existing synthetic 'nhanes' entry --
        # reused verbatim (not re-derived), since Phase A confirmed every model feature has
        # a direct, one-to-one NHANES source (DATA_PROVENANCE.md section 2).
        "FEATURE_METADATA": DATASET_CONFIGS["nhanes"]["FEATURE_METADATA"],
        "CLINICAL_EFFORT_WEIGHTS": DATASET_CONFIGS["nhanes"]["CLINICAL_EFFORT_WEIGHTS"],
        "dataset_path": _REAL_NHANES_CSV,
    }
