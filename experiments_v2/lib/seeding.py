"""
Deterministic seeding helper for experiments_v2.

The original repository (src/, benchmarks/) never seeds torch or Python's
`random` module (see experiments_v2/REPO_MAP.md, Discrepancy D4). All new
code in experiments_v2/ calls `set_all_seeds(seed)` before any model
training or randomized sampling so that every run is reproducible given a
(dataset, grid-cell, seed) triple.
"""
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seeds python `random`, numpy, and torch (CPU/CUDA/MPS) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
