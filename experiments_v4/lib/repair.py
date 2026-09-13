"""
Decode-Safe Recourse — Component 3: verify-and-repair projection.

After Dijkstra returns a path, every node is decoded and every consecutive step
is re-checked in DECODED feature space under the same banded predicate the graph
was pruned with. A step that still violates is repaired by projecting the
offending decoded state onto the constraint boundary (plus band), re-encoding,
and re-decoding; the repair is accepted only if the re-decoded state actually
satisfies the constraint AND preserves monotone risk reduction along the path.
If it cannot, the path is REJECTED and the query ABSTAINS rather than returning
a violating recourse.

Why a projection needs re-encoding at all: clipping happens in feature space,
but the object the method traffics in is a latent code, and the patient is shown
f_θ(z). A clipped feature vector that is never pushed back through enc/dec would
be a value the model never actually predicts -- so we re-encode it, re-decode,
and check the RESULT. That round trip can itself reintroduce error, which is why
the loop iterates (bounded) and why failure is a real, reported outcome rather
than an assumed success.

Immutables during repair: the repaired state is a synthetic point, not a real
patient, so the node-identity argument of Component 2 does not apply to it. We
therefore preserve immutables BY CONSTRUCTION -- the repaired state's immutable
entries are copied verbatim from the query patient's own recorded values, since
a recourse trajectory is by definition a sequence of states for ONE person.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments_v4.lib.constraints_ext import _AGE_HORIZON_CAP, _DIRECTIONAL_EPS
from experiments_v4.lib.dsr_rules import scalar_violation_breakdown
from experiments_v4.lib.graph_build import unscale_matrix

RISK_TOLERANCE = 1e-3   # allowed numerical slack when checking risk monotonicity


@dataclass
class RepairResult:
    status: str                       # 'clean' | 'repaired' | 'failed'
    states: List[np.ndarray] = field(default_factory=list)   # decoded, scaled feature vectors
    n_steps_repaired: int = 0
    n_repair_attempts: int = 0
    failure_reason: Optional[str] = None
    triggering_flags: Optional[Dict[str, bool]] = None   # which constraint type(s) first triggered repair


def _to_scaled(unscaled_row: Dict[str, float], feature_cols, feature_metadata, scaler) -> np.ndarray:
    """Inverse of unscale_matrix for a single row (continuous cols re-standardized)."""
    continuous_cols = feature_metadata["continuous_mutable"] + feature_metadata["non_decreasing"]
    x = np.zeros(len(feature_cols), dtype=np.float32)
    cont_vals = np.array([[unscaled_row[c] for c in continuous_cols]], dtype=np.float32)
    if scaler is not None and len(continuous_cols) > 0:
        cont_scaled = ((cont_vals - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32))[0]
    else:
        cont_scaled = cont_vals[0]
    for i, c in enumerate(continuous_cols):
        x[feature_cols.index(c)] = cont_scaled[i]
    for c in feature_cols:
        if c not in continuous_cols:
            x[feature_cols.index(c)] = unscaled_row[c]
    return x


def _project(
    prev_unscaled: Dict[str, float], cur_unscaled: Dict[str, float], flags: Dict[str, bool],
    feature_metadata: dict, taus: Dict[str, float],
) -> Dict[str, float]:
    """Clips the violating features of `cur` to the constraint boundary plus band."""
    out = dict(cur_unscaled)

    if flags.get("non_decreasing"):
        for col in feature_metadata.get("non_decreasing", []):
            tau = taus.get(col, 0.0)
            if out[col] - prev_unscaled[col] < tau:
                out[col] = prev_unscaled[col] + tau

    if flags.get("age_horizon") and "age" in out:
        cap = _AGE_HORIZON_CAP - taus.get("age", 0.0)
        if out["age"] - prev_unscaled["age"] > cap:
            out["age"] = prev_unscaled["age"] + cap

    if flags.get("directional"):
        for col in feature_metadata.get("directional_reduce", []):
            thresh = _DIRECTIONAL_EPS.get(col, 0.1) - taus.get(col, 0.0)
            if out[col] - prev_unscaled[col] > thresh:
                out[col] = prev_unscaled[col] + thresh
        for col in feature_metadata.get("directional_increase", []):
            thresh = -_DIRECTIONAL_EPS.get(col, 1.0) + taus.get(col, 0.0)
            if out[col] - prev_unscaled[col] < thresh:
                out[col] = prev_unscaled[col] + thresh

    return out


def verify_and_repair(
    decoded_states: List[np.ndarray],
    x0_true: np.ndarray,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    feature_metadata: dict,
    scaler,
    feature_cols: List[str],
    taus: Dict[str, float],
    mode: str = "decoded_banded",
    low_risk_threshold: float = 0.45,
    max_iters: int = 3,
    device: torch.device = torch.device("cpu"),
) -> RepairResult:
    """
    Verifies every consecutive decoded step and repairs violations in place.

    Args:
        decoded_states: decoded (scaled) feature vectors along the path, step 0 first.
        x0_true: the query patient's ORIGINAL recorded features (scaled) -- the
            source of truth for immutables, both for the identity check and for
            construction-time preservation in repaired states.

    Returns:
        RepairResult; status 'failed' means the caller must abstain.
    """
    states = [np.asarray(s, dtype=np.float32).copy() for s in decoded_states]
    immutable_cols = feature_metadata.get("immutable", [])
    imm_idx = [feature_cols.index(c) for c in immutable_cols if c in feature_cols]

    n_repaired = 0
    n_attempts = 0

    first_flags = None
    for t in range(1, len(states)):
        # `x_true_*` supplies immutables under the identity rule. For a repaired
        # (synthetic) predecessor we fall back to the query's own recorded
        # immutables, which is what a repaired state carries by construction.
        step_first_flags = None
        for _ in range(max_iters):
            violated, flags = scalar_violation_breakdown(
                x0_true, x0_true,                      # immutables: same person throughout
                states[t - 1], states[t],
                feature_metadata, scaler, feature_cols, mode=mode, taus=taus,
                check_step_horizon=True,
            )
            if not violated:
                break
            if step_first_flags is None:
                step_first_flags = dict(flags)
                first_flags = first_flags or step_first_flags

            n_attempts += 1
            pair = unscale_matrix(np.vstack([states[t - 1], states[t]]), scaler, feature_cols, feature_metadata)
            prev_u = {c: float(pair[c][0]) for c in feature_cols}
            cur_u = {c: float(pair[c][1]) for c in feature_cols}

            projected = _project(prev_u, cur_u, flags, feature_metadata, taus)
            x_proj = _to_scaled(projected, feature_cols, feature_metadata, scaler)
            for i in imm_idx:
                x_proj[i] = x0_true[i]                 # immutables preserved by construction

            with torch.no_grad():
                mu, _ = vae.encode(torch.tensor(x_proj, dtype=torch.float32, device=device).unsqueeze(0))
                states[t] = vae.decode(mu).squeeze(0).cpu().numpy()
        else:
            return RepairResult(status="failed", states=states, n_steps_repaired=n_repaired,
                                n_repair_attempts=n_attempts, triggering_flags=step_first_flags,
                                failure_reason=f"step {t} still violates after {max_iters} repair "
                                               f"iterations (triggered by {step_first_flags})")

        if n_attempts > 0 and not np.allclose(states[t], decoded_states[t]):
            n_repaired += 1

    if n_attempts == 0:
        return RepairResult(status="clean", states=states)

    # Repair changed the trajectory: re-verify the two global properties it must preserve.
    with torch.no_grad():
        risks = classifier(torch.tensor(np.vstack(states), dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    risks = np.atleast_1d(risks)

    if np.any(np.diff(risks) > RISK_TOLERANCE):
        return RepairResult(status="failed", states=states, n_steps_repaired=n_repaired,
                            n_repair_attempts=n_attempts, triggering_flags=first_flags,
                            failure_reason="repair broke monotone risk reduction along the path")

    if risks[-1] >= low_risk_threshold:
        return RepairResult(status="failed", states=states, n_steps_repaired=n_repaired,
                            n_repair_attempts=n_attempts, triggering_flags=first_flags,
                            failure_reason="repaired terminal state is no longer low-risk")

    return RepairResult(status="repaired", states=states, n_steps_repaired=n_repaired,
                        n_repair_attempts=n_attempts, triggering_flags=first_flags)
