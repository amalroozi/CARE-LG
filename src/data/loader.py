"""
Data pipeline and loader for CARE-LG research framework.
Supports multi-dataset loading (UCI Heart & NHANES Cohort).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.dataset_config import get_dataset_config, FEATURE_METADATA


class EHRDataset(Dataset):
    """
    PyTorch Dataset for tabular EHR records.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_synthetic_ehr(num_samples=3000, seed=42):
    """
    Generates synthetic tabular cardiovascular EHR records matching UCI metadata schema.
    """
    np.random.seed(seed)

    age = np.random.normal(54, 10, num_samples).clip(30, 80).astype(float)
    sex = np.random.binomial(1, 0.6, num_samples).astype(float)
    fasting_blood_sugar = np.random.binomial(1, 0.2, num_samples).astype(float)

    resting_bp = np.random.normal(130, 18, num_samples).clip(90, 200).astype(float)
    cholesterol = np.random.normal(240, 45, num_samples).clip(120, 500).astype(float)
    max_heart_rate = np.random.normal(150, 22, num_samples).clip(70, 210).astype(float)
    oldpeak = np.random.exponential(0.8, num_samples).clip(0, 6.2).astype(float)

    exercise_angina = np.random.binomial(1, 0.3, num_samples).astype(float)
    slope = np.random.choice([0, 1, 2], size=num_samples, p=[0.3, 0.5, 0.2]).astype(float)

    logit = (
        0.04 * (age - 50)
        + 0.5 * sex
        + 0.5 * fasting_blood_sugar
        + 0.02 * (resting_bp - 120)
        + 0.01 * (cholesterol - 200)
        - 0.03 * (max_heart_rate - 150)
        + 0.8 * oldpeak
        + 0.7 * exercise_angina
        + 0.4 * slope
        - 1.5
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    heart_disease_risk = (np.random.rand(num_samples) < prob).astype(int)

    data = {
        'age': age,
        'sex': sex,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'max_heart_rate': max_heart_rate,
        'oldpeak': oldpeak,
        'exercise_angina': exercise_angina,
        'slope': slope,
        'heart_disease_risk': heart_disease_risk
    }

    return pd.DataFrame(data)


def load_dataset(filepath="data/uci_heart.csv", seed=42):
    """
    Loads dataset from CSV file if available, else generates synthetic EHR records.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        print(f"Dataset path {filepath} not found. Generating synthetic EHR records...")
        df = generate_synthetic_ehr(num_samples=3000, seed=seed)
    return df


def get_dataloaders(df=None, dataset_name="uci", dataset_path=None, batch_size=64, seed=42):
    """
    Preprocesses EHR dataset (UCI or NHANES), scales continuous features, and builds PyTorch DataLoaders.

    Returns:
        train_loader (DataLoader): PyTorch DataLoader for training set.
        test_loader (DataLoader): PyTorch DataLoader for test set.
        scaler (StandardScaler): Fitted StandardScaler object for continuous features.
        feature_cols (list): List of feature column names.
    """
    feature_metadata, _, default_path = get_dataset_config(dataset_name)

    if dataset_path is None:
        dataset_path = default_path

    if df is None:
        df = load_dataset(filepath=dataset_path, seed=seed)

    target_col = feature_metadata['target']
    feature_cols = [c for c in df.columns if c != target_col]

    # Continuous features to scale
    continuous_cols = feature_metadata['continuous_mutable'] + feature_metadata['non_decreasing']

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Scale continuous features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    train_dataset = EHRDataset(X_train_scaled.values, y_train)
    test_dataset = EHRDataset(X_test_scaled.values, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, feature_cols
