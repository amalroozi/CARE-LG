"""
Decode-Safe Recourse (DSR) — Components 1 and 2: the constraint predicate.

This module defines the SINGLE place where DSR's constraint rule differs from
the v3 latent-pruning method, so the ablation is exactly attributable.

Four evaluation modes of the same predicate, all vectorized over an edge list:

  'raw'             — v3 / OLD METHOD. Every constraint evaluated on the node's
                      ORIGINAL recorded feature values (X_train / x0). This is
                      what experiments_v3 does, reproduced here unchanged so the
                      old method is a controlled baseline inside the same runner.

  'decoded_naive'   — COMPONENT 1. Every constraint evaluated on the node's
                      DECODED reconstruction x̂ = f_θ(z), which is the
                      representation the patient is actually shown (this is the
                      same representation experiments_v3's Phase-3 decoded-CVR
                      checker evaluates, so pruning is now aligned with the
                      metric it is judged by). Binary immutables are compared
                      after rounding to the nearest class, matching
                      experiments_v3/lib/constraints_ext.py::check_decoded_violations.

  'decoded_banded'  — COMPONENTS 1+2. Same as 'decoded_naive' for the mutable /
                      monotone features, but each threshold is tightened by a
                      per-feature calibrated tolerance τ_k (see calibration.py),
                      so the constraint holds even under worst-case-within-band
                      reconstruction error; AND immutable features are evaluated
                      on the ORIGINAL recorded values instead of the decoded
                      ones (see the "immutability by node identity" note below).

  'banded_no_identity' — an ablation-of-the-ablation used only to isolate how
                      much of C2's benefit comes from the τ bands versus from
                      the identity rule: bands applied, but immutables still
                      checked on decoded values.

## Immutability by node identity (the key methodological point of Component 2)

For a binary immutable such as `sex`, a calibrated numeric tolerance band is the
wrong instrument: experiments_v3 measured decoded sex accuracy at 67.5% (UCI) /
68.2% (real NHANES), i.e. close to chance, so *no* band width makes a decoded
sex comparison informative — a band wide enough to cover the error is wide
enough to admit any transition at all.

The honest treatment is to stop asking the decoder. Every node in the CALG is a
REAL patient whose sex is recorded exactly in the data; the query node is a real
held-out patient whose sex is likewise recorded exactly. So DSR evaluates
immutables on those recorded values and the constraint holds EXACTLY, by node
identity, with no probabilistic qualifier at all.

The scope limit of that claim, stated precisely because it matters: this is a
guarantee about which TRANSITIONS the graph permits, not a claim that the
decoder reconstructs sex correctly. A patient shown the decoded reconstruction
of a step may still see a decoded `sex` value that rounds to the wrong class;
what DSR guarantees is that the step was not *selected* on the basis of a sex
change. Mutable clinical features are different in kind — their decoded value IS
the actionable recommendation ("aim for roughly this blood pressure"), so their
error must be bounded and carried into the constraint, which is what τ does.
METHOD.md states this distinction as the formal scope of the guarantee.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from experiments_v4.lib.constraints_ext import (
    _AGE_DECREASE_TOL,
    _AGE_HORIZON_CAP,
    _DIRECTIONAL_EPS,
    _IMMUTABLE_TOL,
)

MODES = ("raw", "decoded_naive", "decoded_banded", "banded_no_identity")

# Binary immutables get rounded to their nearest class before comparison in the
# decoded modes -- same convention as experiments_v3's decoded-CVR checker.
BINARY_IMMUTABLES = {"sex", "fasting_blood_sugar"}


@dataclass
class ViolationBreakdown:
    """Per-edge violation flags, split by constraint type (for Fig 4)."""
    immutable: np.ndarray
    non_decreasing: np.ndarray
    age_horizon: np.ndarray
    directional: np.ndarray

    @property
    def any(self) -> np.ndarray:
        return self.immutable | self.non_decreasing | self.age_horizon | self.directional

    @property
    def count(self) -> np.ndarray:
        return (self.immutable.astype(np.int32) + self.non_decreasing.astype(np.int32)
                + self.age_horizon.astype(np.int32) + self.directional.astype(np.int32))


def _round_binary(values: np.ndarray) -> np.ndarray:
    """Rounds a decoded binary feature to its nearest class in {0, 1}."""
    return np.round(np.clip(values, 0.0, 1.0))


def violation_breakdown(
    unscaled_true: Dict[str, np.ndarray],
    unscaled_decoded: Dict[str, np.ndarray],
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    feature_metadata: dict,
    mode: str,
    taus: Optional[Dict[str, float]] = None,
    check_step_horizon: bool = True,
) -> ViolationBreakdown:
    """
    Vectorized constraint evaluation over an edge list, in one of the four modes.

    Args:
        unscaled_true: per-feature arrays of ORIGINAL recorded values, unscaled.
        unscaled_decoded: per-feature arrays of DECODED (f_θ(z)) values, unscaled.
            Both dicts must be indexable by the same src_idx / tgt_idx.
        src_idx, tgt_idx: parallel edge index arrays.
        mode: one of MODES.
        taus: per-feature calibrated tolerances (required for banded modes).
        check_step_horizon: apply the single-step age-horizon cap (interior graph
            edges use True; query-attachment edges use False, matching the v3
            entry-gate convention documented in experiments_v3/lib/graph_build.py).

    Returns:
        ViolationBreakdown with one boolean array per constraint type.
    """
    assert mode in MODES, f"unknown mode {mode!r}"
    banded = mode in ("decoded_banded", "banded_no_identity")
    if banded:
        assert taus is not None, "banded modes require calibrated taus"
    taus = taus or {}

    # Which value source each constraint family reads.
    if mode == "raw":
        imm_src, mut_src = unscaled_true, unscaled_true
    elif mode == "decoded_naive":
        imm_src, mut_src = unscaled_decoded, unscaled_decoded
    elif mode == "decoded_banded":
        imm_src, mut_src = unscaled_true, unscaled_decoded   # identity rule for immutables
    else:  # banded_no_identity
        imm_src, mut_src = unscaled_decoded, unscaled_decoded

    n_edges = len(src_idx)
    f_imm = np.zeros(n_edges, dtype=bool)
    f_nondec = np.zeros(n_edges, dtype=bool)
    f_horizon = np.zeros(n_edges, dtype=bool)
    f_dir = np.zeros(n_edges, dtype=bool)

    decoded_immutables = imm_src is unscaled_decoded

    for col in feature_metadata.get("immutable", []):
        if col not in imm_src:
            continue
        s, t = imm_src[col][src_idx], imm_src[col][tgt_idx]
        if decoded_immutables and col in BINARY_IMMUTABLES:
            f_imm |= _round_binary(s) != _round_binary(t)
        else:
            f_imm |= np.abs(t - s) > _IMMUTABLE_TOL

    for col in feature_metadata.get("non_decreasing", []):
        if col not in mut_src:
            continue
        delta = mut_src[col][tgt_idx] - mut_src[col][src_idx]
        if banded:
            # Require a positive margin of tau, so the constraint survives
            # worst-case-within-band reconstruction error (METHOD.md eq. 3).
            f_nondec |= delta < taus.get(col, 0.0)
        else:
            f_nondec |= delta < -_AGE_DECREASE_TOL

    if check_step_horizon and "age" in mut_src:
        delta = mut_src["age"][tgt_idx] - mut_src["age"][src_idx]
        cap = _AGE_HORIZON_CAP - taus.get("age", 0.0) if banded else _AGE_HORIZON_CAP
        f_horizon |= delta > cap

    for col in feature_metadata.get("directional_reduce", []):
        if col not in mut_src:
            continue
        eps = _DIRECTIONAL_EPS.get(col, 0.1)
        thresh = eps - taus.get(col, 0.0) if banded else eps
        f_dir |= (mut_src[col][tgt_idx] - mut_src[col][src_idx]) > thresh

    for col in feature_metadata.get("directional_increase", []):
        if col not in mut_src:
            continue
        eps = _DIRECTIONAL_EPS.get(col, 1.0)
        thresh = -eps + taus.get(col, 0.0) if banded else -eps
        f_dir |= (mut_src[col][tgt_idx] - mut_src[col][src_idx]) < thresh

    return ViolationBreakdown(immutable=f_imm, non_decreasing=f_nondec, age_horizon=f_horizon, directional=f_dir)


def scalar_violation_breakdown(
    x_true_src: np.ndarray, x_true_tgt: np.ndarray,
    x_dec_src: np.ndarray, x_dec_tgt: np.ndarray,
    feature_metadata: dict, scaler, feature_cols,
    mode: str, taus: Optional[Dict[str, float]] = None, check_step_horizon: bool = True,
) -> Tuple[bool, Dict[str, bool]]:
    """
    Single-pair version of violation_breakdown, used by the repair loop
    (Component 3) where states are produced one at a time rather than as an
    edge table. Returns (any_violation, per-category flags).
    """
    from experiments_v4.lib.graph_build import unscale_matrix

    true_pair = unscale_matrix(np.vstack([x_true_src, x_true_tgt]), scaler, feature_cols, feature_metadata)
    dec_pair = unscale_matrix(np.vstack([x_dec_src, x_dec_tgt]), scaler, feature_cols, feature_metadata)
    idx_s, idx_t = np.array([0]), np.array([1])

    b = violation_breakdown(true_pair, dec_pair, idx_s, idx_t, feature_metadata, mode, taus, check_step_horizon)
    flags = {
        "immutable": bool(b.immutable[0]),
        "non_decreasing": bool(b.non_decreasing[0]),
        "age_horizon": bool(b.age_horizon[0]),
        "directional": bool(b.directional[0]),
    }
    return any(flags.values()), flags
