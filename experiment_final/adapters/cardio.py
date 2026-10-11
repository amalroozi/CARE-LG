"""Adapter: cardio -> DatasetBundle (see DATASET_ADAPTER.md). PIPELINE P1.

Kaggle Cardiovascular Disease dataset via ../archive/experiments_v10/data_cardio.py::load_clean_cardio_df
(imported, unchanged: same cleaning, age in years, gender 0/1, height dropped). Stratified subsample
of `cardio_n_subsample` rows (default 8000, the v10 loader default; fixed sampling seed so every
model seed sees the same cohort, see DECISIONS.md). No diagnoses/medications -> generic rules only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_final.adapters.base import DatasetBundle, age_band, validate_bundle

EXP_DIR = Path(__file__).resolve().parents[1]
HAS_ONTOLOGY = False
REPO = EXP_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FEATURE_META = {
    "age": dict(type="continuous", **{"class": "non_decreasing"}, direction="none", unit="years", phys_min=18, phys_max=100, concept="age"),
    "gender": dict(type="binary", **{"class": "immutable"}, direction="none", unit="1=male (raw 2)", phys_min=0, phys_max=1, concept="sex"),
    "weight": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="kg", phys_min=35, phys_max=200, concept="body_weight", noise_tol=0.1),
    "ap_hi": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=80, phys_max=220, concept="systolic_bp", noise_tol=1.0),
    "ap_lo": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=40, phys_max=160, concept="diastolic_bp", noise_tol=1.0),
    "cholesterol": dict(type="ordinal", **{"class": "mutable"}, direction="reduce", unit="1/2/3", phys_min=1, phys_max=3, concept="cholesterol_category", noise_tol=0.5),
    "gluc": dict(type="ordinal", **{"class": "mutable"}, direction="reduce", unit="1/2/3", phys_min=1, phys_max=3, concept="glucose_category", noise_tol=0.5),
    "smoke": dict(type="binary", **{"class": "mutable"}, direction="reduce", unit="1=yes", phys_min=0, phys_max=1, concept="smoking", noise_tol=0.5),
    "alco": dict(type="binary", **{"class": "mutable"}, direction="reduce", unit="1=yes", phys_min=0, phys_max=1, concept="alcohol", noise_tol=0.5),
    "active": dict(type="binary", **{"class": "mutable"}, direction="increase", unit="1=yes", phys_min=0, phys_max=1, concept="physical_activity", noise_tol=0.5),
}


def load(config: dict) -> DatasetBundle:
    """Cardio bundle (P1)."""
    from archive.experiments_v10.data_cardio import load_clean_cardio_df
    df = load_clean_cardio_df()
    flow = [{"step": "cardio_train.csv after v10 plausibility cleaning", "n": len(df)}]
    n_sub = int(config.get("cardio_n_subsample", 8000))
    if n_sub and len(df) > n_sub:
        df, _ = train_test_split(df, train_size=n_sub, random_state=12345, stratify=df["cardio"])
        flow.append({"step": f"stratified subsample (fixed random_state=12345)", "n": n_sub})
    df = df.reset_index(drop=True)
    X = df[list(FEATURE_META)].astype(float)
    pid = pd.Series(range(len(df)), name="row_id")
    b = DatasetBundle(
        X=X, y=df["cardio"].astype(int), patient_id=pid, feature_meta={k: dict(v) for k, v in FEATURE_META.items()},
        context=pd.DataFrame({"row_id": pid}), concepts=[[] for _ in range(len(df))], current_meds=[[] for _ in range(len(df))],
        subgroups={"sex": X["gender"].map({1.0: "Male", 0.0: "Female"}), "age_band": age_band(X["age"], (30, 40, 50, 60, 70))},
        name="cardio", has_ontology=False, cohort_flow=flow,
        label_definition="Presence of cardiovascular disease (Kaggle 'cardio' target)", citation="CARDIO_KAGGLE",
    )
    validate_bundle(b)
    return b
