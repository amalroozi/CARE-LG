"""
Verification script for CARE-LG Day 1 data loading, blackbox classifier, and tabular VAE on data/uci_heart.csv.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from configs.dataset_config import FEATURE_METADATA, CLINICAL_EFFORT_WEIGHTS
from src.data.loader import get_dataloaders
from src.blackbox_model import train_blackbox_model, get_device
from src.vae.model import train_vae


def test_day1():
    print("==================================================")
    print("CARE-LG Day 1 Verification (uci_heart.csv)")
    print("==================================================")

    # 1. Verify Dataset Configuration
    print("\n1. Verifying Feature Metadata & Clinical Effort Weights...")
    print(f"   Target: {FEATURE_METADATA['target']}")
    print(f"   Continuous Mutable: {FEATURE_METADATA['continuous_mutable']}")
    print(f"   Categorical Mutable: {FEATURE_METADATA['categorical_mutable']}")
    print(f"   Non-decreasing: {FEATURE_METADATA['non_decreasing']}")
    print(f"   Immutable: {FEATURE_METADATA['immutable']}")
    print(f"   Effort Weights Count: {len(CLINICAL_EFFORT_WEIGHTS)}")
    assert len(CLINICAL_EFFORT_WEIGHTS) == len(FEATURE_METADATA['continuous_mutable']) + len(FEATURE_METADATA['categorical_mutable'])

    # 2. Verify DataLoaders on data/uci_heart.csv
    print("\n2. Loading Data from data/uci_heart.csv & DataLoaders...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_path="data/uci_heart.csv", batch_size=64, seed=42)
    input_dim = len(feature_cols)
    print(f"   Feature columns ({input_dim}): {feature_cols}")
    print(f"   Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    X_sample, y_sample = next(iter(train_loader))
    print(f"   Sample batch X shape: {X_sample.shape}, y shape: {y_sample.shape}")
    assert X_sample.shape[1] == input_dim

    # 3. Verify Black-Box Classifier Training
    device = get_device()
    print(f"\n3. Training Black-Box Risk Classifier (20 epochs) on {device}...")
    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
    with torch.no_grad():
        test_preds = classifier(X_sample.to(device))
    print(f"   Classifier output shape: {test_preds.shape}, sample prediction range: [{test_preds.min().item():.4f}, {test_preds.max().item():.4f}]")
    assert test_preds.shape == (X_sample.shape[0], 1)

    # 4. Verify Tabular VAE Training
    print(f"\n4. Training Tabular VAE (10 epochs) on {device}...")
    vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=10, device=device)
    with torch.no_grad():
        recon_X, mu, logvar = vae(X_sample.to(device))
    print(f"   VAE Recon shape: {recon_X.shape}, Latent mu shape: {mu.shape}, logvar shape: {logvar.shape}")
    assert recon_X.shape == X_sample.shape
    assert mu.shape == (X_sample.shape[0], 4)
    assert logvar.shape == (X_sample.shape[0], 4)

    print("\n==================================================")
    print("SUCCESS: Day 1 Data Loading & Model Verification Passed!")
    print("==================================================")


if __name__ == "__main__":
    test_day1()
