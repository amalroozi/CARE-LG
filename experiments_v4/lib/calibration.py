"""
Decode-Safe Recourse — Component 2: reconstruction-error tolerance bands.

Derives a per-feature tolerance τ_k from a held-out CALIBRATION split that the
VAE and classifier never saw during training (see pipeline_v4.py's 3-way split).

Method (deliberately distribution-free — no Gaussian assumption):
    τ_k(q) = the empirical q-quantile of |unscale(f_θ(enc(x)))_k − unscale(x)_k|
             over the calibration rows,
i.e. the conformal-style absolute-residual quantile for feature k. At q = 0.95,
τ_k is the value that 95% of calibration reconstruction errors fall below.

Coverage claim this supports, stated exactly (and no more):
    For a single feature k and a single node, P(|x̂_k − x_k| ≤ τ_k(q)) ≥ q
    under exchangeability of the calibration and deployment nodes.
A per-STEP constraint compares two nodes, so its error is the sum of two such
residuals; the margin actually applied by dsr_rules.py is a single τ_k, which is
the rule this project's brief specifies. METHOD.md §"Scope of the guarantee"
states both this and the stricter 2τ_k alternative explicitly, and reports which
was used, rather than implying a tighter guarantee than was measured.

Binary/categorical features are handled separately and NOT given a numeric band:
their calibration statistic is a misclassification RATE, and the DSR rule for
immutables is identity-based rather than band-based (see dsr_rules.py's module
docstring for why a band is the wrong instrument there).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from experiments_v4.lib.dsr_rules import BINARY_IMMUTABLES
from experiments_v4.lib.graph_build import unscale_matrix


@dataclass
class Calibration:
    """Per-feature reconstruction-error calibration on a held-out split."""
    quantile: float
    n_calib: int
    taus: Dict[str, float] = field(default_factory=dict)            # continuous features only
    misclass_rate: Dict[str, float] = field(default_factory=dict)   # binary/categorical features
    mae: Dict[str, float] = field(default_factory=dict)             # diagnostic, all features

    def as_rows(self, dataset: str, seed: int) -> List[dict]:
        rows = []
        for col in sorted(set(list(self.taus) + list(self.misclass_rate))):
            rows.append({
                "dataset": dataset, "seed": seed, "quantile": self.quantile, "n_calib": self.n_calib,
                "feature": col,
                "tau": self.taus.get(col, np.nan),
                "misclass_rate": self.misclass_rate.get(col, np.nan),
                "mae": self.mae.get(col, np.nan),
                "band_type": "conformal_quantile" if col in self.taus else "identity_rule_no_band",
            })
        return rows


def calibrate(
    vae: torch.nn.Module,
    X_calib: np.ndarray,
    feature_metadata: dict,
    scaler,
    feature_cols: List[str],
    quantile: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> Calibration:
    """
    Computes per-feature tolerance bands on the calibration split.

    Args:
        X_calib: (M, D) scaled calibration features -- rows the VAE never trained on.
        quantile: the conformal level q; τ_k is the q-quantile of |x̂_k − x_k|.

    Returns:
        Calibration with `taus` (continuous features, in UNSCALED clinical units,
        matching the units dsr_rules.py compares in) and `misclass_rate` for
        binary/categorical features.
    """
    vae.eval()
    with torch.no_grad():
        xt = torch.tensor(X_calib, dtype=torch.float32, device=device)
        mu, _ = vae.encode(xt)
        X_hat = vae.decode(mu).cpu().numpy()

    true_u = unscale_matrix(X_calib, scaler, feature_cols, feature_metadata)
    dec_u = unscale_matrix(X_hat, scaler, feature_cols, feature_metadata)

    categorical = set(feature_metadata.get("immutable", [])) | set(feature_metadata.get("categorical_mutable", []))

    cal = Calibration(quantile=quantile, n_calib=len(X_calib))
    for col in feature_cols:
        err = np.abs(dec_u[col] - true_u[col])
        cal.mae[col] = float(err.mean())
        if col in categorical:
            # Classification error, not a numeric band. Binary features round to
            # the nearest class; multi-category ones snap to the nearest observed
            # level in the calibration data.
            levels = np.unique(np.round(true_u[col], 6))
            if col in BINARY_IMMUTABLES or len(levels) <= 2:
                pred = np.round(np.clip(dec_u[col], 0.0, 1.0))
                truth = np.round(np.clip(true_u[col], 0.0, 1.0))
            else:
                pred = levels[np.argmin(np.abs(dec_u[col][:, None] - levels[None, :]), axis=1)]
                truth = levels[np.argmin(np.abs(true_u[col][:, None] - levels[None, :]), axis=1)]
            cal.misclass_rate[col] = float((pred != truth).mean())
        else:
            cal.taus[col] = float(np.quantile(err, quantile))

    return cal
