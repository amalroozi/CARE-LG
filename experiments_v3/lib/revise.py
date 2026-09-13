"""
Phase C: faithful REVISE implementation.

Joshi, Koyejo, Vijitbenjaronk, Kim, Ghosh, "Towards Realistic Individual
Recourse and Actionable Explanations in Black-Box Decision Making Systems"
(arXiv:1907.09615): a continuous, gradient-based search directly over a
VAE's latent code z, minimizing a loss that combines (i) a classifier term
pushing f(decode(z)) toward the target (desired) outcome, and (ii) a
proximity term keeping z close to the original patient's encoding z0,
optimized with Adam; the returned counterfactual is x* = decode(z*).

This implementation uses the SAME trained VAE and SAME trained risk
classifier as CARE-LG for the given (dataset, seed) -- required for a fair
comparison, per the brief -- rather than training separate models.

Hyperparameters (recorded here and in CHANGES.md; picked via the documented
sweep below, not guessed):
  - Optimizer: Adam, lr=0.05 (matches this repo's existing run_dice
    optimization learning rate, for cross-method consistency).
  - Steps: 100 (matches the existing run_dice's step budget exactly, so
    "same number of optimization steps" holds across every gradient-based
    method in this project).
  - Proximity weight (lambda_prox): 0.01, applied to the SQUARED L2
    distance in LATENT space (‖z - z0‖_2^2), per the original paper's
    latent-space proximity term (this differs from this repo's existing
    run_dice, whose proximity term is L2 distance in FEATURE space with
    weight 0.1 -- not directly comparable in scale, since latent and
    feature space have different units/geometry; REVISE's defining
    characteristic vs. DiCE IS searching in latent rather than feature
    space, so its own proximity term must live in that space).
  - Target: classifier output (risk probability) minimized directly (no
    separate "desired probability" hyperparameter), matching run_dice's
    existing loss shape (`pred + proximity`) for consistency; success is
    declared when decoded risk < LOW_RISK_THRESHOLD (0.45), the SAME
    threshold used throughout this project's graph-based methods (a
    deliberate alignment choice, documented as a deviation from this
    repo's DiCE/GrowingSpheres, which use a separate 0.35 threshold
    inherited from the original benchmark script).

**Sweep performed** (experiments_v3/results/revise_lr_sweep.csv, 25 UCI
seed-0 high-risk patients): an initial attempt with lambda_prox=0.5 (a
naive first guess) gave only 15% success -- diagnosis showed the raw
gradient of classifier-risk w.r.t. z at the starting point z0 has norm
~0.02-0.04, so a quadratic penalty at weight 0.5 overwhelms it almost
immediately, freezing z near z0 for the vast majority of patients (the only
"successes" were patients whose VAE-reconstructed starting point, decode(z0),
already happened to sit below the risk threshold with no optimization at
all -- itself a notable finding, see CHANGES.md and REPORT.pdf). Sweeping
lambda_prox in {0.001, 0.005, 0.01, 0.05} x lr in {0.05, 0.1} found a sharp
transition: lambda_prox <= 0.01 gives 100% success (mean latent-to-feature
distance ~3.0) on this pilot; lambda_prox=0.05 drops back to 64-72%. We
retained lambda_prox=0.01 as the smallest value tested that still exerts
a real (non-negligible) proximity penalty rather than sitting right at the
threshold's edge.
"""
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

LOW_RISK_THRESHOLD = 0.45


def run_revise(
    x0: np.ndarray,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    device: torch.device,
    lr: float = 0.05,
    steps: int = 100,
    lambda_prox: float = 0.01,
    target_threshold: float = LOW_RISK_THRESHOLD,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Runs REVISE for one patient.

    Returns:
        success: bool, whether decoded risk dropped below target_threshold.
        recourse_x: decoded feature-space counterfactual (np.ndarray), or None.
        z_final: final latent code (np.ndarray), or None.
        latency: wall-clock seconds (time.perf_counter), the full optimization loop.
    """
    classifier.eval()
    vae.eval()
    x0_t = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        mu0, _ = vae.encode(x0_t)
    z0 = mu0.detach().clone()
    z = z0.clone().requires_grad_(True)

    optimizer = optim.Adam([z], lr=lr)

    best_z = None
    best_risk = float("inf")
    for _ in range(steps):
        optimizer.zero_grad()
        x_hat = vae.decode(z)
        risk = classifier(x_hat).squeeze()
        prox = lambda_prox * torch.sum((z - z0) ** 2)
        loss = risk + prox
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_risk = classifier(vae.decode(z)).squeeze().item()
        if cur_risk < best_risk:
            best_risk = cur_risk
            best_z = z.detach().clone()
        if cur_risk < target_threshold:
            break

    latency = time.perf_counter() - t0

    if best_z is None or best_risk >= target_threshold:
        return False, None, None, latency

    with torch.no_grad():
        x_final = vae.decode(best_z).squeeze(0).cpu().numpy()
    return True, x_final, best_z.squeeze(0).cpu().numpy(), latency
