"""
DiCE baseline. Mothilal, Sharma, Tan, "Explaining Machine Learning
Classifiers through Diverse Counterfactual Explanations" (FAT* 2020,
arXiv:1905.07697).

Rebuilds `benchmarks/run_benchmarks.py::run_dice` (confirmed AI-generated
prior to this project's documented investigation phase, with no citation
and no fidelity check -- see docs/HISTORY.md) to the standard every other
baseline in this project meets. **Finding, reported plainly per this
session's brief**: the original `run_dice` was a single L2-penalized
gradient-descent counterfactual search with NO diversity mechanism at all
-- diversity across multiple simultaneous counterfactuals is DiCE's
entire defining contribution over plain gradient-descent counterfactual
search (which is what CARE-LG's other baseline, REVISE, already is, just
in latent space). Calling that single-counterfactual search "DiCE" was not
faithful to the method; this file implements the paper's actual
joint, diversity-regularized objective.

Loss (paper's Equation 4), optimized jointly over k candidate
counterfactuals c_1..c_k:

    L = (1/k) sum_i hinge_yloss(f(c_i))
      + (lambda1/k) sum_i dist(c_i, x0)
      - lambda2 * dpp_diversity(c_1..c_k)

  hinge_yloss(p) = max(0, 1 - z * logit(p)), z = -1 here (target: reduce
                   predicted risk toward the low-risk class). logit(p) is
                   recovered from the trained model's own sigmoid output
                   via the inverse-sigmoid (log-odds) transform -- the
                   model has no separate pre-sigmoid head, so this is the
                   only way to get a logit-shaped quantity out of it, and
                   it is exact (the sigmoid is invertible) up to the
                   float32 clamp used to avoid log(0).
  dist(c, x0)      = mean of {MAD-normalized absolute distance per
                     continuous feature} union {0/1 indicator per changed
                     categorical feature} -- the paper's own proximity
                     definition, MAD computed from the training set.
  dpp_diversity    = det(K), K_ij = 1 / (1 + dist(c_i, c_j))

Default hyperparameters, taken directly from the paper: lambda1=0.5,
lambda2=1.0 ("based on a grid-search", per the paper), Adam, lr=0.05,
max_iter=5000.

**Disclosed deviations from the paper** (documented here, not silent, the
same standard `revise.py` holds itself to):
  1. `max_iter` reduced from the paper's 5000 to 300 for this session's
     full-cohort, multi-seed compute budget. Validated against a higher-
     iteration run on a patient subset -- see CHANGES.md for that
     comparison and how much it changed the reported numbers, if at all.
  2. `dpp_diversity` is computed as `logdet(K)` rather than raw `det(K)`
     for numerical stability (raw determinants of a k>3 similarity matrix
     with entries in (0,1] underflow/vanish easily in float32) -- this is
     the same substitution the official `dice-ml` reference implementation
     makes, not an ad hoc choice.
  3. DiCE's own "actionability" mechanism (user-specified immutable
     features held fixed during optimization) is applied using this
     project's existing immutable/non_decreasing feature lists. DiCE has
     no equivalent of CARE-LG's DIRECTIONAL monotonicity constraints (e.g.
     "cholesterol must not increase"), so those are not enforced during
     generation -- exactly as the original `run_dice` and every other
     baseline in this project's comparison table handle it: constraint
     satisfaction is measured AFTER generation via real-value CVR, not
     built into the search itself.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _feature_mad(X_train: np.ndarray, continuous_idx: List[int]) -> np.ndarray:
    """Median absolute deviation per continuous feature -- the paper's own
    proximity-term normalizer. Falls back to 1.0 for a zero-MAD feature to
    avoid divide-by-zero (standard practice, matches dice-ml)."""
    if not continuous_idx:
        return np.array([])
    med = np.median(X_train[:, continuous_idx], axis=0)
    mad = np.median(np.abs(X_train[:, continuous_idx] - med), axis=0)
    mad = np.where(mad < 1e-6, 1.0, mad)
    return mad


def _proximity(c: torch.Tensor, x0: torch.Tensor, continuous_idx: List[int], categorical_idx: List[int],
               mad: torch.Tensor) -> torch.Tensor:
    parts = []
    if continuous_idx:
        parts.append(torch.abs(c[:, continuous_idx] - x0[continuous_idx]) / mad)
    if categorical_idx:
        parts.append((c[:, categorical_idx] != x0[categorical_idx]).float())
    if not parts:
        return torch.zeros(c.shape[0], device=c.device)
    return torch.cat(parts, dim=1).mean(dim=1)


def _pairwise_feature_dist(c: torch.Tensor, continuous_idx: List[int], categorical_idx: List[int],
                            mad: torch.Tensor) -> torch.Tensor:
    n_cont, n_cat = len(continuous_idx), len(categorical_idx)
    k = c.shape[0]
    total = max(n_cont + n_cat, 1)
    d_cont = torch.zeros(k, k, device=c.device)
    d_cat = torch.zeros(k, k, device=c.device)
    if n_cont:
        cont = c[:, continuous_idx] / mad
        d_cont = torch.cdist(cont, cont, p=1) / n_cont
    if n_cat:
        cat = c[:, categorical_idx]
        d_cat = (cat.unsqueeze(1) != cat.unsqueeze(0)).float().mean(dim=2)
    return (d_cont * n_cont + d_cat * n_cat) / total


def run_dice_diverse(
    x0: np.ndarray, classifier, device, feature_metadata: dict, feature_cols: List[str],
    X_train: np.ndarray, k: int = 4, target_threshold: float = 0.45,
    lambda1: float = 0.5, lambda2: float = 1.0, lr: float = 0.05, max_iter: int = 300,
    seed: int = 0,
) -> Tuple[bool, Optional[List[np.ndarray]], float]:
    """
    Returns (success, list_of_valid_counterfactuals[<=k], latency_sec).
    success = at least one of the k jointly-optimized candidates achieved
    predicted risk < target_threshold by the end of optimization.
    """
    torch.manual_seed(seed)
    d = len(x0)
    continuous_idx = [feature_cols.index(f) for f in feature_metadata.get('continuous_mutable', []) if f in feature_cols]
    categorical_idx = [feature_cols.index(f) for f in feature_metadata.get('categorical_mutable', []) if f in feature_cols]
    fixed_idx = [i for i in range(d) if i not in continuous_idx and i not in categorical_idx]  # immutable + non_decreasing

    mad_np = _feature_mad(X_train, continuous_idx)
    mad = torch.tensor(mad_np, dtype=torch.float32, device=device) if len(mad_np) else torch.tensor([], device=device)
    x0_t = torch.tensor(x0, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    c = x0_t.unsqueeze(0).repeat(k, 1).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([c], lr=lr)
    z = -1.0  # target: reduce risk (toward the low-risk / class-0 side)

    for _step in range(max_iter):
        optimizer.zero_grad()
        preds = classifier(c).squeeze(-1).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(preds / (1 - preds))
        hinge = torch.clamp(1 - z * logits, min=0)
        yloss = hinge.mean()

        prox = _proximity(c, x0_t, continuous_idx, categorical_idx, mad).mean()

        if k > 1:
            dist_matrix = _pairwise_feature_dist(c, continuous_idx, categorical_idx, mad)
            K_mat = 1.0 / (1.0 + dist_matrix) + torch.eye(k, device=device) * 1e-4
            diversity = torch.logdet(K_mat)
        else:
            diversity = torch.tensor(0.0, device=device)

        loss = yloss + lambda1 * prox - lambda2 * diversity
        loss.backward()
        with torch.no_grad():
            if fixed_idx:
                c.grad[:, fixed_idx] = 0.0  # DiCE's own actionability: hold immutable/non-decreasing features fixed
        optimizer.step()

    with torch.no_grad():
        final_preds = classifier(c).squeeze(-1).cpu().numpy()
    valid_mask = final_preds < target_threshold
    latency = time.perf_counter() - t0
    if np.any(valid_mask):
        valid_c = c.detach().cpu().numpy()[valid_mask]
        return True, list(valid_c), latency
    return False, None, latency
