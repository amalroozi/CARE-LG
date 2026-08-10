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


def check_hard_violations(x_source, x_target, feature_metadata, scaler, feature_cols, check_step_horizon=True):
    """
    Checks for HARD clinical violations that strictly block graph edges (W_ij = infinity):
    1. Immutable attributes (e.g. sex, fasting_blood_sugar).
    2. Age reduction (age cannot decrease by > 0.1 yrs).
    3. Realistic age horizon cap (single-step age increase > 3.0 yrs, evaluated when check_step_horizon=True).

    Returns:
        bool: True if hard violation exists, False otherwise.
    """
    source_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)

    # 1. Immutable attribute check
    for col in feature_metadata['immutable']:
        if col in source_dict and col in target_dict:
            if abs(target_dict[col] - source_dict[col]) > 1e-4:
                return True

    # 2. Non-decreasing age check
    for col in feature_metadata['non_decreasing']:
        if col in source_dict and col in target_dict:
            if target_dict[col] < source_dict[col] - 0.1:
                return True

    # 3. Age horizon cap (single-step increase > 3.0 yrs during edge evaluation)
    if check_step_horizon and 'age' in source_dict and 'age' in target_dict:
        if (target_dict['age'] - source_dict['age']) > 3.05:
            return True

    return False


def check_soft_violations(x_source, x_target, feature_metadata, scaler, feature_cols):
    """
    Checks for SOFT directional medical violations penalized via heavy edge weight multiplier (1e5x):
    1. Cholesterol increase > 1.0 mg/dL.
    2. Systolic BP increase > 1.0 mmHg.
    3. Resting BP increase > 5.0 mmHg.
    4. BMI increase > 0.1 units.
    5. HbA1c increase > 0.05 units.
    6. Exercise angina acquired (0 -> 1).

    Returns:
        bool: True if soft directional violation exists, False otherwise.
    """
    source_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)

    if 'cholesterol' in source_dict and 'cholesterol' in target_dict:
        if (target_dict['cholesterol'] - source_dict['cholesterol']) > 1.0:
            return True

    if 'exercise_angina' in source_dict and 'exercise_angina' in target_dict:
        if source_dict['exercise_angina'] <= 0.5 and target_dict['exercise_angina'] > 0.5:
            return True

    if 'resting_bp' in source_dict and 'resting_bp' in target_dict:
        if (target_dict['resting_bp'] - source_dict['resting_bp']) > 5.0:
            return True

    if 'systolic_bp' in source_dict and 'systolic_bp' in target_dict:
        if (target_dict['systolic_bp'] - source_dict['systolic_bp']) > 1.0:
            return True

    if 'bmi' in source_dict and 'bmi' in target_dict:
        if (target_dict['bmi'] - source_dict['bmi']) > 0.1:
            return True

    if 'glycemic_hba1c' in source_dict and 'glycemic_hba1c' in target_dict:
        if (target_dict['glycemic_hba1c'] - source_dict['glycemic_hba1c']) > 0.05:
            return True

    return False


def check_clinical_violations(x_source, x_target, feature_metadata, scaler, feature_cols, tol=1e-5):
    """
    Checks for clinical rule violations between x_source and x_target (returns True if hard or soft violation exists).
    """
    return check_hard_violations(x_source, x_target, feature_metadata, scaler, feature_cols) or \
           check_soft_violations(x_source, x_target, feature_metadata, scaler, feature_cols)


def compute_clinical_effort(x_source, x_target, clinical_effort_weights, feature_metadata, scaler, feature_cols):
    """
    Computes weighted L1 clinical effort distance across mutable features in standardized Z-score feature space.

    Args:
        x_source (np.ndarray or torch.Tensor): Source sample features (Z-score standardized).
        x_target (np.ndarray or torch.Tensor): Target sample features (Z-score standardized).
        clinical_effort_weights (dict): Weight mapping per mutable feature.
        feature_metadata (dict): Feature schema configuration.
        scaler (StandardScaler): Scaler object (unused, maintained for signature compatibility).
        feature_cols (list): Feature column names.

    Returns:
        float: Weighted clinical effort in standardized feature space.
    """
    if isinstance(x_source, torch.Tensor):
        x_source = x_source.detach().cpu().numpy()
    if isinstance(x_target, torch.Tensor):
        x_target = x_target.detach().cpu().numpy()

    x_source = np.asarray(x_source).squeeze()
    x_target = np.asarray(x_target).squeeze()

    source_dict = dict(zip(feature_cols, x_source))
    target_dict = dict(zip(feature_cols, x_target))

    mutable_cols = feature_metadata['continuous_mutable'] + feature_metadata['categorical_mutable']

    total_effort = 0.0
    for col in mutable_cols:
        if col in source_dict and col in target_dict:
            weight = clinical_effort_weights.get(col, 1.0)
            delta = abs(target_dict[col] - source_dict[col])
            total_effort += weight * delta

    return total_effort
