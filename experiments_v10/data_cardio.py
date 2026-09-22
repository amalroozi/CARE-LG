"""
v10: registers the Kaggle "Cardiovascular Disease dataset" (sulianova,
external_datasets/cardio_disease_kaggle/) as a third, self-contained CARE-LG
dataset -- a separate experiment tree (experiments_v10/), src/ untouched,
matching the project's existing convention (experiments_v8/, experiments_v9/)
of isolating a new variant rather than editing configs/dataset_config.py or
src/data/dataset_registration.py in place.

Source: 70,000 real patient records, 11 usable features + binary target
(cardio: 1 = has cardiovascular disease, 34,979 positive / 35,021 negative
-- near-perfectly balanced). Raw columns: id, age (days), gender, height,
weight, ap_hi/ap_lo (systolic/diastolic BP), cholesterol (1/2/3 ordinal),
gluc (1/2/3 ordinal), smoke, alco, active, cardio.

Cleaning (disclosed, not hidden): the raw file has known data-entry errors
common to self-reported survey data -- ap_hi/ap_lo occasionally negative,
zero, or in the thousands; height/weight occasional implausible outliers.
Rows outside a physiologically plausible range are dropped before any
modeling. Every SURVIVING row is a real, unmodified patient record.

Feature classification (feature_metadata), same categories/semantics the
rest of this project already uses:
  - age: non_decreasing (converted from days to years)
  - gender: immutable (not a recourse target; remapped to 0/1 -- the raw
    file encodes it 1/2, and this project's type-aware VAE decoder head
    for 'immutable'/'categorical_mutable' columns assumes a binary column
    is already 0/1, same convention as UCI's `sex`/NHANES's `sex`)
  - height: DROPPED entirely (not used as a model input). It doesn't change
    in adults, so it isn't a recourse target either way, and keeping it
    would force it into the type-aware VAE's high-cardinality categorical
    decoder head (it has ~80 distinct raw values, unlike every other
    immutable column in this project, which are all binary) -- adding a
    ~80-way softmax head for a column recourse never touches is pure
    modeling overhead for no benefit. Same category of simplification as
    v8 dropping age entirely, disclosed not hidden.
  - weight, ap_hi, ap_lo, cholesterol, gluc: continuous_mutable,
    directional_reduce (lower is better for all five)
  - smoke, alco: categorical_mutable, directional_reduce (quitting is the
    only admissible direction)
  - active: categorical_mutable, directional_increase (becoming more
    active is the only admissible direction)
  - target: cardio
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.loader import EHRDataset

RAW_CSV = (REPO_ROOT / "external_datasets" / "cardio_disease_kaggle" / "datasets" /
           "sulianova" / "cardiovascular-disease-dataset" / "versions" / "1" / "cardio_train.csv")

# Physiologically plausible ranges used to drop known bad rows (disclosed
# cleaning step, not a change to any row that survives it).
_BOUNDS = {
    "age_years": (18, 100),
    "height": (120, 210),   # cm (used only for row-cleaning, then dropped)
    "weight": (35, 200),    # kg
    "ap_hi": (80, 220),     # mmHg systolic
    "ap_lo": (40, 160),     # mmHg diastolic
}

FEATURE_METADATA = {
    "continuous_mutable": ["weight", "ap_hi", "ap_lo", "cholesterol", "gluc"],
    "categorical_mutable": ["smoke", "alco", "active"],
    "non_decreasing": ["age"],
    "immutable": ["gender"],
    "directional_reduce": ["weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco"],
    "directional_increase": ["active"],
    "target": "cardio",
}

# Illustrative, hand-set effort weights -- same uncited-constant caveat as
# every other CLINICAL_EFFORT_WEIGHTS dict in this project (see
# configs/dataset_config.py's note on 'uci'/'nhanes'): NOT derived from a
# published treatment-burden instrument.
CLINICAL_EFFORT_WEIGHTS = {
    "weight": 2.0, "ap_hi": 1.5, "ap_lo": 1.5, "cholesterol": 2.0,
    "gluc": 2.5, "smoke": 3.0, "alco": 2.0, "active": 1.0,
}


def load_clean_cardio_df() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, sep=";")
    df = df.drop(columns=["id"])
    df["age"] = df["age"] / 365.25  # days -> years
    df["gender"] = df["gender"] - 1  # raw file: 1/2 -> 0/1 (binary VAE head convention)
    df = df[df.ap_hi >= df.ap_lo]  # systolic must exceed diastolic
    for col, (lo, hi) in _BOUNDS.items():
        c = "age" if col == "age_years" else col
        df = df[(df[c] >= lo) & (df[c] <= hi)]
    df = df.drop(columns=["height"])
    return df.reset_index(drop=True)


def get_dataloaders_cardio(seed: int, n_subsample: int = 8000, batch_size: int = 64):
    """Loads + cleans the raw CSV, stratified-subsamples to n_subsample rows
    (computational-tractability scoping decision, same category as this
    project's existing disclosed FACE/NHANES subsampling -- see
    external_baselines/experiment_comparison.html -- not hidden), then
    splits/scales exactly like src.data.loader.get_dataloaders."""
    df = load_clean_cardio_df()
    if n_subsample and len(df) > n_subsample:
        df, _ = train_test_split(df, train_size=n_subsample, random_state=seed, stratify=df["cardio"])
        df = df.reset_index(drop=True)

    target_col = FEATURE_METADATA["target"]
    feature_cols = [c for c in df.columns if c != target_col]
    continuous_cols = FEATURE_METADATA["continuous_mutable"] + FEATURE_METADATA["non_decreasing"]

    X = df[feature_cols].copy()
    y = df[target_col].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    train_loader = DataLoader(EHRDataset(X_train_scaled.values, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(EHRDataset(X_test_scaled.values, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler, feature_cols
