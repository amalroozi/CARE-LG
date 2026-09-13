# --- Canonical home since Phase 1 of DEEP_AUDIT resolution (see experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md); experiments_v6/lib/ re-exports this unchanged. ---
"""
Deterministic seeding helper for experiments_v3 (duplicated from
experiments_v2/lib/seeding.py, kept self-contained rather than importing
across the v2/v3 boundary, per Hard Rule 3 -- v3 must not depend on v2
still being present/unmodified to run).
"""
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
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
