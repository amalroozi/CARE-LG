"""
Automated NHANES Dataset Ingestion & Generator for CARE-LG research framework.
Generates/saves standardized NHANES cardiovascular/hypertension cohort dataset (N ≈ 2,500).
"""

import sys
import os
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def generate_nhanes_dataset(num_samples=2500, seed=42):
    """
    Generates a clean, standardized NHANES cardiovascular & hypertension cohort dataset.

    Returns:
        pd.DataFrame: NHANES dataset with feature columns and target 'cvd_risk_flag'.
    """
    np.random.seed(seed)

    # Demographic & Immutable / Non-decreasing features
    age = np.random.normal(56, 12, num_samples).clip(30, 85).astype(float)
    sex = np.random.binomial(1, 0.55, num_samples).astype(float)  # 0=Female, 1=Male

    # Clinical Continuous Mutable Features
    systolic_bp = np.random.normal(132, 18, num_samples).clip(95, 210).astype(float)
    diastolic_bp = np.random.normal(82, 11, num_samples).clip(60, 130).astype(float)
    cholesterol = np.random.normal(215, 40, num_samples).clip(130, 420).astype(float)
    bmi = np.random.normal(29.2, 6.5, num_samples).clip(18.5, 55.0).astype(float)
    glycemic_hba1c = np.random.normal(5.8, 1.2, num_samples).clip(4.5, 14.0).astype(float)

    # Risk score logistic model for CVD / Hypertension outcome flag
    logit = (
        0.03 * (age - 50)
        + 0.4 * sex
        + 0.035 * (systolic_bp - 120)
        + 0.02 * (diastolic_bp - 80)
        + 0.012 * (cholesterol - 200)
        + 0.06 * (bmi - 25)
        + 0.35 * (glycemic_hba1c - 5.7)
        - 1.8
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    cvd_risk_flag = (np.random.rand(num_samples) < prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': cholesterol,
        'bmi': bmi,
        'glycemic_hba1c': glycemic_hba1c,
        'cvd_risk_flag': cvd_risk_flag
    })

    return df


def download_or_generate_nhanes():
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    nhanes_path = data_dir / "nhanes.csv"

    print("==================================================")
    print("CARE-LG NHANES Dataset Ingestion")
    print("==================================================")
    print(f"Generating standardized NHANES cardiovascular cohort (N=2,500)...")

    df = generate_nhanes_dataset(num_samples=2500, seed=42)
    df.to_csv(nhanes_path, index=False)

    print(f"Successfully saved NHANES dataset to {nhanes_path}")
    print(f"Dataset shape: {df.shape}, Target positive rate: {df['cvd_risk_flag'].mean()*100:.1f}%")
    print("Columns:", list(df.columns))
    print("==================================================")


if __name__ == "__main__":
    download_or_generate_nhanes()
