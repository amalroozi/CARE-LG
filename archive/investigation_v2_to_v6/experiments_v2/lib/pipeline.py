"""
Per-(dataset, seed) model training + encoding pipeline, reusing the existing
src/ training functions verbatim, but with explicit seeding (REPO_MAP D4).
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from configs.dataset_config import get_dataset_config
from src.data.loader import get_dataloaders
from src.blackbox_model import RiskClassifier, train_blackbox_model
from src.vae.model import TabularVAE, train_vae
from experiments_v2.lib.seeding import set_all_seeds

HIGH_RISK_THRESHOLD = 0.55
LOW_RISK_THRESHOLD = 0.45
K_NEIGHBORS = {"uci": 40, "nhanes": 80}


@dataclass
class DatasetRun:
    dataset: str
    seed: int
    feature_metadata: dict
    effort_weights: dict
    scaler: object
    feature_cols: List[str]
    vae: TabularVAE
    classifier: RiskClassifier
    Z_train: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    train_risk: np.ndarray
    low_risk_mask: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_risk: np.ndarray
    high_risk_test_idx: np.ndarray
    k_neighbors: int
    device: torch.device


def run_dataset_seed(dataset: str, seed: int, device: torch.device = torch.device("cpu")) -> DatasetRun:
    """Trains fresh VAE + classifier for (dataset, seed) and encodes train/test sets."""
    set_all_seeds(seed)
    feature_metadata, effort_weights, _ = get_dataset_config(dataset)

    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=dataset, batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
    # epochs=25 for both datasets, matching benchmarks/run_benchmarks.py::run_all_benchmarks exactly
    vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=device)
    vae.eval()
    classifier.eval()

    all_z, all_x, all_y = [], [], []
    with torch.no_grad():
        for Xb, yb in train_loader:
            mu, _ = vae.encode(Xb.to(device))
            all_z.append(mu.cpu().numpy())
            all_x.append(Xb.numpy())
            all_y.append(yb.numpy())
    Z_train = np.vstack(all_z)
    X_train = np.vstack(all_x)
    y_train = np.concatenate(all_y).squeeze()

    with torch.no_grad():
        train_risk = classifier(torch.tensor(X_train, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    low_risk_mask = train_risk < LOW_RISK_THRESHOLD

    all_xt, all_yt = [], []
    for Xb, yb in test_loader:
        all_xt.append(Xb.numpy())
        all_yt.append(yb.numpy())
    X_test = np.vstack(all_xt)
    y_test = np.concatenate(all_yt).squeeze()

    with torch.no_grad():
        test_risk = classifier(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    high_risk_test_idx = np.where(test_risk > HIGH_RISK_THRESHOLD)[0]

    return DatasetRun(
        dataset=dataset, seed=seed, feature_metadata=feature_metadata, effort_weights=effort_weights,
        scaler=scaler, feature_cols=feature_cols, vae=vae, classifier=classifier,
        Z_train=Z_train, X_train=X_train, y_train=y_train, train_risk=train_risk, low_risk_mask=low_risk_mask,
        X_test=X_test, y_test=y_test, test_risk=test_risk, high_risk_test_idx=high_risk_test_idx,
        k_neighbors=K_NEIGHBORS[dataset], device=device,
    )
