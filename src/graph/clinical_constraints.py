"""
Asymmetric Masking & Clinical Cost Engine for CARE-LG research framework.
Checks clinical recourse constraints (non-decreasing & immutable) and computes weighted clinical effort.
"""

import numpy as np
import pandas as pd
import torch


def unscale_features(x, scaler, feature_cols, feature_metadata):
    """
    Unscales feature vector x back to original clinical units using scaler.

    Args:
        x (np.ndarray or torch.Tensor or pd.Series): Feature values vector.
        scaler (StandardScaler): Fitted scaler object.
        feature_cols (list): List of feature column names.
        feature_metadata (dict): Metadata containing feature classification.

    Returns:
        dict: Mapping of feature column names to unscaled values.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(x, pd.Series):
        x_dict = x.to_dict()
    else:
        x = np.asarray(x).squeeze()
        x_dict = dict(zip(feature_cols, x))

    continuous_cols = feature_metadata['continuous_mutable'] + feature_metadata['non_decreasing']

    # Extract continuous feature values in order expected by scaler
    continuous_indices = [feature_cols.index(c) for c in continuous_cols if c in feature_cols]
    if scaler is not None and len(continuous_indices) > 0:
        cont_vals = np.array([x_dict[c] for c in continuous_cols]).reshape(1, -1)
        unscaled_cont_vals = scaler.inverse_transform(cont_vals).squeeze()
        for idx, col in enumerate(continuous_cols):
            x_dict[col] = float(unscaled_cont_vals[idx])

    return x_dict


def check_clinical_violations(x_source, x_target, feature_metadata, scaler, feature_cols, tol=1e-5):
    """
    Checks for clinical rule violations between x_source and x_target:
    1. Non-decreasing features (e.g. age cannot decrease).
    2. Realistic age horizon (max age increase per single step <= 3.0 years).
    3. Immutable attributes (e.g. sex or fasting_blood_sugar cannot change).
    4. Directional medical safeguards:
       - Cholesterol cannot increase (Delta cholesterol > 0 => violation).
       - Exercise angina cannot be acquired (0 -> 1 => violation).
       - Resting BP cannot increase substantially (Delta resting_bp > 5.0 mmHg => violation).

    Args:
        x_source (np.ndarray or torch.Tensor): Source sample features.
        x_target (np.ndarray or torch.Tensor): Target sample features.
        feature_metadata (dict): Feature schema configuration.
        scaler (StandardScaler): Scaler object.
        feature_cols (list): Feature column names.
        tol (float): Tolerance threshold for floating point comparison.

    Returns:
        bool: True if any violation exists, False if transition is clinically valid.
    """
    source_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)

    # 1. Non-decreasing feature check (e.g., age cannot decrease)
    for col in feature_metadata['non_decreasing']:
        if target_dict[col] < source_dict[col] - tol:
            return True

    # 2. Realistic Age Horizon Cap (age increase <= 3.0 years per single step)
    if 'age' in source_dict and 'age' in target_dict:
        if (target_dict['age'] - source_dict['age']) > (3.0 + tol):
            return True

    # 3. Immutable attribute check (e.g., sex, fasting_blood_sugar)
    for col in feature_metadata['immutable']:
        if abs(target_dict[col] - source_dict[col]) > tol:
            return True

    # 4. Directional Medical Safeguards:
    # 4a. Serum Cholesterol cannot increase (both UCI & NHANES)
    if 'cholesterol' in source_dict and 'cholesterol' in target_dict:
        if target_dict['cholesterol'] > source_dict['cholesterol'] + tol:
            return True

    # 4b. Exercise angina cannot be acquired (UCI: 0.0 -> 1.0)
    if 'exercise_angina' in source_dict and 'exercise_angina' in target_dict:
        if source_dict['exercise_angina'] <= 0.5 and target_dict['exercise_angina'] > 0.5:
            return True

    # 4c. Resting blood pressure cannot increase substantially (UCI: Delta > 5.0 mmHg)
    if 'resting_bp' in source_dict and 'resting_bp' in target_dict:
        if (target_dict['resting_bp'] - source_dict['resting_bp']) > (5.0 + tol):
            return True

    # 4d. Systolic blood pressure cannot increase (NHANES: Delta > 0)
    if 'systolic_bp' in source_dict and 'systolic_bp' in target_dict:
        if target_dict['systolic_bp'] > source_dict['systolic_bp'] + tol:
            return True

    # 4e. Body Mass Index (BMI) cannot increase (NHANES: Delta > 0)
    if 'bmi' in source_dict and 'bmi' in target_dict:
        if target_dict['bmi'] > source_dict['bmi'] + tol:
            return True

    # 4f. Glycemic HbA1c cannot increase (NHANES: Delta > 0)
    if 'glycemic_hba1c' in source_dict and 'glycemic_hba1c' in target_dict:
        if target_dict['glycemic_hba1c'] > source_dict['glycemic_hba1c'] + tol:
            return True

    return False


def compute_clinical_effort(x_source, x_target, clinical_effort_weights, feature_metadata, scaler, feature_cols):
    """
    Computes weighted L1 clinical effort distance across mutable features in original physical scale.

    Args:
        x_source (np.ndarray or torch.Tensor): Source sample features.
        x_target (np.ndarray or torch.Tensor): Target sample features.
        clinical_effort_weights (dict): Weight mapping per mutable feature.
        feature_metadata (dict): Feature schema configuration.
        scaler (StandardScaler): Scaler object.
        feature_cols (list): Feature column names.

    Returns:
        float: Weighted clinical effort.
    """
    source_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)

    mutable_cols = feature_metadata['continuous_mutable'] + feature_metadata['categorical_mutable']

    total_effort = 0.0
    for col in mutable_cols:
        weight = clinical_effort_weights.get(col, 1.0)
        delta = abs(target_dict[col] - source_dict[col])
        total_effort += weight * delta

    return total_effort
