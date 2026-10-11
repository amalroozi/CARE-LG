"""Adapter: uci -> DatasetBundle (see DATASET_ADAPTER.md). PIPELINE P1.

UCI Heart Disease (Cleveland, 297 complete rows) as used throughout CARE-LG (../data/uci_heart.csv).
Feature classes/directions and per-edge noise tolerances are copied from configs/dataset_config.py
and src/graph/constraints_ext.py::_DIRECTIONAL_EPS so the integer gate equals the original CARE-LG gate.
No diagnoses or medications -> concepts/current_meds are empty (generic population rules only).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_final.adapters.base import DatasetBundle, age_band, validate_bundle

EXP_DIR = Path(__file__).resolve().parents[1]
HAS_ONTOLOGY = False

FEATURE_META = {
    "age": dict(type="continuous", **{"class": "non_decreasing"}, direction="none", unit="years", phys_min=18, phys_max=100, concept="age"),
    "sex": dict(type="binary", **{"class": "immutable"}, direction="none", unit="1=male", phys_min=0, phys_max=1, concept="sex"),
    "fasting_blood_sugar": dict(type="binary", **{"class": "immutable"}, direction="none", unit="1=FBS>120", phys_min=0, phys_max=1, concept="fasting_glucose_flag"),
    "resting_bp": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=70, phys_max=250, concept="systolic_bp", noise_tol=5.0),
    "cholesterol": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mg/dL", phys_min=70, phys_max=600, concept="total_cholesterol", noise_tol=1.0),
    "max_heart_rate": dict(type="continuous", **{"class": "mutable"}, direction="increase", unit="bpm", phys_min=60, phys_max=220, concept="max_heart_rate", noise_tol=1.0),
    "oldpeak": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mm", phys_min=0, phys_max=7, concept="st_depression", noise_tol=0.05),
    "exercise_angina": dict(type="binary", **{"class": "mutable"}, direction="reduce", unit="1=yes", phys_min=0, phys_max=1, concept="exercise_angina", noise_tol=0.5),
    "slope": dict(type="categorical", **{"class": "mutable"}, direction="reduce", unit="1/2/3", phys_min=1, phys_max=3, concept="st_slope", noise_tol=0.5),
}


def load(config: dict) -> DatasetBundle:
    """UCI bundle (P1)."""
    df = pd.read_csv(EXP_DIR / config["paths"]["uci_csv"]).reset_index(drop=True)
    X = df[list(FEATURE_META)].astype(float)
    pid = pd.Series(range(len(df)), name="row_id")
    b = DatasetBundle(
        X=X, y=df["heart_disease_risk"].astype(int), patient_id=pid, feature_meta={k: dict(v) for k, v in FEATURE_META.items()},
        context=pd.DataFrame({"row_id": pid}), concepts=[[] for _ in range(len(df))], current_meds=[[] for _ in range(len(df))],
        subgroups={"sex": X["sex"].map({1.0: "Male", 0.0: "Female"}), "age_band": age_band(X["age"], (30, 40, 50, 60, 70, 80))},
        name="uci", has_ontology=False, cohort_flow=[{"step": "uci_heart.csv complete rows", "n": len(df)}],
        label_definition="Angiographic coronary artery disease (>50% diameter narrowing), num>0", citation="UCI1989",
    )
    validate_bundle(b)
    return b
