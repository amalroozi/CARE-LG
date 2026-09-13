"""
Per-(dataset, seed) pipeline for experiments_v4.

Differs from experiments_v3/lib/pipeline_v3.py in exactly two ways, both
required by the DSR method and both documented in CHANGES.md:

1. THREE-WAY SPLIT. Component 2 needs a calibration split the decoder never saw.
   The v3 80/20 train/test split is kept EXACTLY as-is (so the held-out test
   patients, and therefore every reported comparison, are the same patients v3
   evaluated on for a given seed); the 80% TRAIN partition is then split again,
   stratified and seeded, into graph-train (85%) and calibration (15%). The VAE
   and classifier are fit on graph-train only, and the CALG is built over
   graph-train nodes only, so calibration rows are genuinely held out from both
   model fitting and the graph.

   Consequence, stated plainly: the old (v3) method re-run here sees ~15% fewer
   training rows than it did in experiments_v3, so its numbers here will not be
   bit-identical to the published v3 numbers. That is why the old method is
   RE-RUN inside this pipeline rather than compared against v3's published
   table -- E1 must differ between rows only by method, never by data.

2. DECODER CHOICE. `decoder='v3'` trains the type-aware decoder (v3 Phase B);
   `decoder='v2'` trains the original undifferentiated TabularVAE from
   src/vae/model.py. E5 uses this to test whether DSR's guarantee is robust to
   decoder quality or merely inherits it.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.dataset_config import get_dataset_config
from src.blackbox_model import RiskClassifier, train_blackbox_model
from src.data.loader import EHRDataset, get_dataloaders
from src.vae.model import train_vae as train_vae_original
from experiments_v4.lib.seeding import set_all_seeds
from experiments_v4.lib.vae_v3 import train_vae_type_aware

HIGH_RISK_THRESHOLD = 0.55
LOW_RISK_THRESHOLD = 0.45
K_NEIGHBORS = {"uci": 40, "nhanes_real": 80}
CALIB_FRAC = 0.15


@dataclass
class DatasetRunV4:
    dataset: str
    seed: int
    decoder: str
    feature_metadata: dict
    effort_weights: dict
    scaler: object
    feature_cols: List[str]
    vae: torch.nn.Module
    classifier: RiskClassifier
    Z_train: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    train_risk: np.ndarray
    low_risk_mask: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_risk: np.ndarray
    high_risk_test_idx: np.ndarray
    k_neighbors: int
    device: torch.device


def _loader_arrays(loader: DataLoader):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.vstack(xs), np.concatenate(ys).squeeze()


def run_dataset_seed_v4(
    dataset: str,
    seed: int,
    decoder: str = "v3",
    calib_frac: float = CALIB_FRAC,
    device: torch.device = torch.device("cpu"),
) -> DatasetRunV4:
    """Trains a seeded VAE + classifier on graph-train only, holding out a calibration split."""
    assert decoder in ("v2", "v3"), f"unknown decoder {decoder!r}"
    set_all_seeds(seed)
    feature_metadata, effort_weights, _ = get_dataset_config(dataset)

    # Identical outer split to v3 -> identical held-out test patients per seed.
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=dataset, batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    X_train_full, y_train_full = _loader_arrays(train_loader)
    X_graph, X_calib, y_graph, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=calib_frac, random_state=seed, stratify=y_train_full
    )

    graph_loader = DataLoader(EHRDataset(X_graph, y_graph), batch_size=64, shuffle=True)

    classifier = train_blackbox_model(graph_loader, input_dim=input_dim, epochs=20, device=device)
    if decoder == "v3":
        vae = train_vae_type_aware(
            graph_loader, input_dim=input_dim, feature_cols=feature_cols, feature_metadata=feature_metadata,
            latent_dim=4, epochs=25, age_loss_weight=8.0, device=device,
        )
    else:
        vae = train_vae_original(graph_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=device)
    vae.eval()
    classifier.eval()

    with torch.no_grad():
        Z_train = vae.encode(torch.tensor(X_graph, dtype=torch.float32, device=device))[0].cpu().numpy()
        train_risk = classifier(torch.tensor(X_graph, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    low_risk_mask = train_risk < LOW_RISK_THRESHOLD

    X_test, y_test = _loader_arrays(test_loader)
    with torch.no_grad():
        test_risk = classifier(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    high_risk_test_idx = np.where(test_risk > HIGH_RISK_THRESHOLD)[0]

    return DatasetRunV4(
        dataset=dataset, seed=seed, decoder=decoder, feature_metadata=feature_metadata,
        effort_weights=effort_weights, scaler=scaler, feature_cols=feature_cols,
        vae=vae, classifier=classifier,
        Z_train=Z_train, X_train=X_graph, y_train=y_graph, train_risk=train_risk, low_risk_mask=low_risk_mask,
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test, test_risk=test_risk, high_risk_test_idx=high_risk_test_idx,
        k_neighbors=K_NEIGHBORS[dataset], device=device,
    )
