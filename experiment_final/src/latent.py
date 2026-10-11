"""Phase 5: type-aware VAE (reuse care_lg/src/vae/type_aware.py) + Riemannian metric (src/graph/riemannian_batch.py)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import EHRDataset
from src.seeding import set_all_seeds
from src.vae.type_aware import TabularVAETypeAware, train_vae_type_aware


def fit_vae(split, seed: int, latent_dim: int = 4, epochs: int = 25, age_loss_weight: float = 8.0) -> TabularVAETypeAware:
    """Train the CARE-LG type-aware VAE on the full (scaled) train set, unchanged hyper-parameters (P5)."""
    set_all_seeds(seed)
    dl = DataLoader(EHRDataset(split.Xtr, split.ytr), batch_size=64, shuffle=True)
    vae = train_vae_type_aware(dl, input_dim=len(split.features), feature_cols=split.features,
                               feature_metadata=split.legacy_meta, latent_dim=latent_dim, epochs=epochs,
                               age_loss_weight=age_loss_weight, device=torch.device("cpu"))
    vae.eval()
    return vae


def encode(vae, Xs: np.ndarray) -> np.ndarray:
    """Posterior means mu(x) (P5)."""
    with torch.no_grad():
        return vae.encode(torch.as_tensor(np.atleast_2d(Xs), dtype=torch.float32))[0].numpy()


def decode(vae, Z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return vae.decode(torch.as_tensor(np.atleast_2d(Z), dtype=torch.float32)).numpy()
