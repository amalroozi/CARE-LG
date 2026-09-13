# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Extensions to src/graph/clinical_constraints.py needed for the ablation grid
and decoded-space verification (Phases 1 and 3). Does NOT re-derive the
constraint *decision* logic: `violates()` below calls the existing
`check_clinical_violations` directly, so hard-mode pruning in the new grid
is byte-for-byte the same predicate as the original pipeline uses. The only
new logic is (a) a *count* of how many sub-conditions are violated, for the
soft-penalty cost v(x_i,x_j) = lambda * count, and (b) decoded-space checks
with explicit, documented noise tolerances (Phase 3), since the original
thresholds (e.g. 1e-4 for immutables) are tight enough to be triggered by
ordinary VAE reconstruction error rather than a genuine constraint breach.

Tolerance choices for decoded-space verification (Phase 3), stated once here:
  - Immutable BINARY features (sex, fasting_blood_sugar): round the decoded,
    unscaled value to the nearest of {0, 1} before comparing source vs.
    target. Tolerance = 0.5 (round-to-class). Justification: these features
    are one-hot/binary in the training data; the decoder's continuous output
    is a reconstruction of a binary quantity and small deviations around 0
    or 1 are reconstruction noise, not a semantic sex/FBS change.
  - Non-decreasing (age): identical 0.1-year tolerance as the original
    check_hard_violations, applied to the unscaled decoded age.
  - Directional reduce/increase (BP, cholesterol, BMI, HbA1c, oldpeak,
    exercise_angina, slope): identical per-feature epsilons already defined
    in check_soft_violations (these were already designed as measurement/
    reconstruction noise tolerances in the original code, so we reuse them
    verbatim rather than inventing new ones).
"""
from typing import Dict, Tuple

import numpy as np
import torch

from src.graph.clinical_constraints import (
    check_clinical_violations,
    check_hard_violations,
    check_soft_violations,
    unscale_features,
)

# Same epsilon table as src/graph/clinical_constraints.py::check_soft_violations,
# duplicated here ONLY because the original function is boolean/short-circuit
# and cannot return a per-feature breakdown. Kept in exact sync intentionally.
_DIRECTIONAL_EPS = {
    'cholesterol': 1.0,
    'systolic_bp': 1.0,
    'diastolic_bp': 1.0,
    'resting_bp': 5.0,
    'bmi': 0.1,
    'glycemic_hba1c': 0.05,
    'oldpeak': 0.05,
    'exercise_angina': 0.5,
    'slope': 0.5,
    'fasting_blood_sugar': 0.5,
}

_AGE_DECREASE_TOL = 0.1
_AGE_HORIZON_CAP = 3.05
_IMMUTABLE_TOL = 1e-4


def violates(x_source, x_target, feature_metadata, scaler, feature_cols, check_step_horizon=True) -> bool:
    """Thin wrapper around the existing check_clinical_violations (hard+directional OR)."""
    return check_clinical_violations(x_source, x_target, feature_metadata, scaler, feature_cols, check_step_horizon=check_step_horizon)


def count_violations(x_source, x_target, feature_metadata, scaler, feature_cols, check_step_horizon=True) -> int:
    """
    Counts how many individual constraint sub-conditions are violated between
    x_source and x_target (both in the ORIGINAL scaled/graph feature space,
    same convention as check_clinical_violations). Used as v(x_i,x_j) for the
    soft-penalty edge weight W_ij = d(z_i,z_j) + lambda * v(x_i,x_j).

    Consistency invariant (checked by the caller during edge-table
    construction): count_violations(...) > 0 iff violates(...) is True.
    """
    source_dict = unscale_features(x_source, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target, scaler, feature_cols, feature_metadata)

    n = 0
    for col in feature_metadata.get('immutable', []):
        if col in source_dict and col in target_dict:
            if abs(target_dict[col] - source_dict[col]) > _IMMUTABLE_TOL:
                n += 1

    for col in feature_metadata.get('non_decreasing', []):
        if col in source_dict and col in target_dict:
            if target_dict[col] < source_dict[col] - _AGE_DECREASE_TOL:
                n += 1

    if check_step_horizon and 'age' in source_dict and 'age' in target_dict:
        if (target_dict['age'] - source_dict['age']) > _AGE_HORIZON_CAP:
            n += 1

    for col in feature_metadata.get('directional_reduce', []):
        if col in source_dict and col in target_dict:
            eps = _DIRECTIONAL_EPS.get(col, 0.1)
            if (target_dict[col] - source_dict[col]) > eps:
                n += 1

    for col in feature_metadata.get('directional_increase', []):
        if col in source_dict and col in target_dict:
            eps = _DIRECTIONAL_EPS.get(col, 1.0)
            if (target_dict[col] - source_dict[col]) < -eps:
                n += 1

    return n


def decode_node(vae_model: torch.nn.Module, z, device=None) -> np.ndarray:
    """Decodes a single latent point z (numpy or tensor, shape (d,)) to scaled feature space."""
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32)
    if device is None:
        device = next(vae_model.parameters()).device
    z = z.to(device)
    with torch.no_grad():
        x_hat = vae_model.decode(z.unsqueeze(0) if z.dim() == 1 else z)
    return x_hat.squeeze(0).cpu().numpy()


def decode_nodes_batch(vae_model: torch.nn.Module, Z: np.ndarray, device=None) -> np.ndarray:
    """Decodes a (M, d) array of latent points to a (M, D) array of scaled features."""
    if device is None:
        device = next(vae_model.parameters()).device
    with torch.no_grad():
        zt = torch.tensor(Z, dtype=torch.float32, device=device)
        x_hat = vae_model.decode(zt)
    return x_hat.cpu().numpy()


def check_decoded_violations(
    x_source_decoded, x_target_decoded, feature_metadata, scaler, feature_cols, check_step_horizon=True
) -> Tuple[bool, Dict[str, bool]]:
    """
    Re-checks constraints between two DECODED (VAE reconstruction) feature
    vectors, using the noise tolerances documented in this file's module
    docstring rather than the tight originals meant for exact source data.

    Returns:
        (any_violation, per_category_flags) where per_category_flags has
        keys 'immutable', 'non_decreasing', 'age_horizon', 'directional'.
    """
    source_dict = unscale_features(x_source_decoded, scaler, feature_cols, feature_metadata)
    target_dict = unscale_features(x_target_decoded, scaler, feature_cols, feature_metadata)

    flags = {'immutable': False, 'non_decreasing': False, 'age_horizon': False, 'directional': False}

    binary_immutables = {'sex', 'fasting_blood_sugar'}
    for col in feature_metadata.get('immutable', []):
        if col in source_dict and col in target_dict:
            if col in binary_immutables:
                s_cls = round(float(np.clip(source_dict[col], 0.0, 1.0)))
                t_cls = round(float(np.clip(target_dict[col], 0.0, 1.0)))
                if s_cls != t_cls:
                    flags['immutable'] = True
            else:
                if abs(target_dict[col] - source_dict[col]) > _IMMUTABLE_TOL:
                    flags['immutable'] = True

    for col in feature_metadata.get('non_decreasing', []):
        if col in source_dict and col in target_dict:
            if target_dict[col] < source_dict[col] - _AGE_DECREASE_TOL:
                flags['non_decreasing'] = True

    if check_step_horizon and 'age' in source_dict and 'age' in target_dict:
        if (target_dict['age'] - source_dict['age']) > _AGE_HORIZON_CAP:
            flags['age_horizon'] = True

    for col in feature_metadata.get('directional_reduce', []):
        if col in source_dict and col in target_dict:
            eps = _DIRECTIONAL_EPS.get(col, 0.1)
            if (target_dict[col] - source_dict[col]) > eps:
                flags['directional'] = True

    for col in feature_metadata.get('directional_increase', []):
        if col in source_dict and col in target_dict:
            eps = _DIRECTIONAL_EPS.get(col, 1.0)
            if (target_dict[col] - source_dict[col]) < -eps:
                flags['directional'] = True

    return any(flags.values()), flags
