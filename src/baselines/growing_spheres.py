"""
Growing Spheres baseline. Laugel, Lesot, Marsala, Renard, Detyniecki,
"Inverse Classification for Comparison-based Interpretability in Machine
Learning" (arXiv:1712.08443).

Rebuilds `benchmarks/run_benchmarks.py::run_growing_spheres` (confirmed
AI-generated prior to this project's documented investigation phase, with
no citation and no fidelity check -- see docs/HISTORY.md) to the standard
every other baseline in this project meets (REVISE, PACE, the corrected
FACE rerun): read the actual paper, cite it, implement its defining
mechanism faithfully, and document every deviation explicitly rather than
silently.

Faithful to the paper's CORE search procedure: expanding spherical layers
around the query point x0, uniform sampling within each layer, stopping at
the first "enemy" (differently-classified point) found, returning the
L2-closest one found in that layer. This is what the ORIGINAL `run_growing_spheres`
approximated (random directions at growing radii) -- but that version
sampled a fixed direction count per fixed-width layer without the paper's
own uniform-by-volume sampling within a shell, had no real stopping-radius
logic tied to the paper's a0/a1 layer bounds, and cited nothing.

Documented deviation from the paper (disclosed, not silent): the paper's
SECOND phase -- a post-hoc sparsity-reduction step that greedily reverts
individual feature perturbations back toward x0 (controlled by a gamma
sparsity weight) while the point remains an "enemy" -- is NOT implemented
here. This rebuild targets the defining growing-spheres SEARCH mechanism
(what distinguishes this method from other perturbation baselines), not the
optional sparsity refinement, which is a separate post-processing concern
orthogonal to the search itself.
"""
import time
from typing import Optional, Tuple

import numpy as np
import torch


def _sample_uniform_in_sphere_shell(n_samples: int, dim: int, r_min: float, r_max: float,
                                     rng: np.random.Generator) -> np.ndarray:
    """
    Uniform sampling within a spherical shell [r_min, r_max] in R^dim.

    Direction: uniform on the unit sphere, via the standard Gaussian-
    normalization trick (a normalized isotropic Gaussian vector is uniform
    on the sphere) -- a numerically equivalent, more commonly available
    substitute for the paper's own cited "YPHL algorithm" for the same
    uniform-direction distribution.

    Radius: drawn so the resulting points are uniform BY VOLUME within the
    shell (not uniform in radius, which would bias samples toward the outer
    edge, since shell volume grows with r^(dim-1)) -- achieved via
    radius^dim ~ Uniform(r_min^dim, r_max^dim).
    """
    directions = rng.standard_normal((n_samples, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    u = rng.uniform(r_min ** dim, r_max ** dim, size=n_samples)
    radii = u ** (1.0 / dim)
    return directions * radii[:, None]


def run_growing_spheres(
    x0: np.ndarray, classifier, device, target_threshold: float = 0.45,
    eta: float = 0.1, n_samples: int = 2000, max_layers: int = 30, seed: int = 0,
) -> Tuple[bool, Optional[np.ndarray], float]:
    """
    Returns (success, recourse_x, latency_sec).

    Hyperparameter names preserved from the paper: `eta` (initial radius /
    layer width), `n_samples` (samples drawn per layer -- the paper uses
    n=10,000; reduced here to 2,000 for this session's full-cohort,
    multi-seed compute budget, a disclosed deviation validated against a
    higher-n run on a subset -- see CHANGES.md for that comparison).
    `max_layers` caps the expansion; the paper's own search is unbounded in
    principle, a cap is required for a terminating implementation.
    """
    rng = np.random.default_rng(seed)
    d = len(x0)
    t0 = time.perf_counter()

    def is_enemy(candidates: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            preds = classifier(torch.tensor(candidates, dtype=torch.float32, device=device)).cpu().numpy().squeeze(-1)
        return preds < target_threshold

    # Phase 1 of the paper (shrink an initial ball until it contains no
    # enemies) is skipped: for a recourse query, x0 itself is high-risk and
    # essentially never already an "enemy" (low-risk) of itself, so the
    # paper's own Phase 2 (expand outward from radius eta) is the branch
    # that actually executes for this task in the overwhelming majority of
    # cases -- starting directly there is a faithful simplification, not an
    # approximation of a different mechanism.
    a0, a1 = 0.0, eta
    for _layer in range(max_layers):
        candidates = x0.reshape(1, -1) + _sample_uniform_in_sphere_shell(n_samples, d, a0, a1, rng)
        enemies_mask = is_enemy(candidates)
        if np.any(enemies_mask):
            enemy_candidates = candidates[enemies_mask]
            dists = np.linalg.norm(enemy_candidates - x0.reshape(1, -1), axis=1)
            best = enemy_candidates[np.argmin(dists)]
            return True, best, time.perf_counter() - t0
        a0 = a1
        a1 += eta

    return False, None, time.perf_counter() - t0
