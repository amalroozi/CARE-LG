"""Phase 3: 80/20 stratified split, scaler fit on train only, unscaled (recorded) copies kept.

Scaled space follows src/data/loader.py: continuous (+ordinal) mutable and non-decreasing columns are
standardised (scaler fit on train), binary/categorical columns stay raw. Recorded (unscaled) values are
kept from the adapter directly -- never reconstructed from the scaled copy -- so every rule check on a
real patient uses the value actually measured (CARE-LG B2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiment_final.adapters.base import DatasetBundle
from experiment_final.src.knowledge_layer import Schema


def legacy_feature_metadata(b: DatasetBundle) -> dict:
    """Adapter meta -> CARE-LG FEATURE_METADATA dict used by src/vae/type_aware.py (P3/P5)."""
    fm = {"continuous_mutable": [], "categorical_mutable": [], "non_decreasing": [], "immutable": [],
          "directional_reduce": [], "directional_increase": [], "target": "y"}
    for f, m in b.feature_meta.items():
        if m["class"] == "immutable":
            fm["immutable"].append(f)
        elif m["class"] == "non_decreasing":
            fm["non_decreasing"].append(f)
        elif m["type"] in ("continuous", "ordinal"):
            fm["continuous_mutable"].append(f)
        else:
            fm["categorical_mutable"].append(f)
        if m["direction"] == "reduce":
            fm["directional_reduce"].append(f)
        elif m["direction"] == "increase":
            fm["directional_increase"].append(f)
    return fm


@dataclass
class Split:
    """One (dataset, seed) split with scaled + recorded copies (P3)."""
    seed: int
    features: List[str]
    tr: np.ndarray            # positions into the bundle
    te: np.ndarray
    fit: np.ndarray           # positions into tr used to fit the risk models
    val: np.ndarray           # positions into tr used for model selection / calibration
    Xtr_raw: np.ndarray
    Xte_raw: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray
    scaler: StandardScaler
    cont_idx: List[int]
    legacy_meta: dict

    def scale(self, X_raw: np.ndarray) -> np.ndarray:
        """Recorded units -> model space (float32), as src/data/loader.py (P3)."""
        X = np.array(X_raw, dtype=np.float64, copy=True)
        X = np.atleast_2d(X)
        X[:, self.cont_idx] = (X[:, self.cont_idx] - self.scaler.mean_) / self.scaler.scale_
        return X.astype(np.float32)

    def unscale(self, X_s: np.ndarray) -> np.ndarray:
        """Model space -> clinical units (float64); only for synthetic points (P3)."""
        X = np.atleast_2d(np.array(X_s, dtype=np.float64, copy=True))
        X[:, self.cont_idx] = X[:, self.cont_idx] * self.scaler.scale_ + self.scaler.mean_
        return X

    @property
    def Xtr(self) -> np.ndarray:
        return self.scale(self.Xtr_raw)

    @property
    def Xte(self) -> np.ndarray:
        return self.scale(self.Xte_raw)


def make_split(b: DatasetBundle, seed: int, test_size: float = 0.2, val_frac: float = 0.2) -> Split:
    """Stratified 80/20 split (random_state=seed) + inner stratified fit/val split of train (P3)."""
    pos = np.arange(len(b.X))
    y = b.y.to_numpy()
    tr, te = train_test_split(pos, test_size=test_size, random_state=seed, stratify=y)
    fit, val = train_test_split(np.arange(len(tr)), test_size=val_frac, random_state=seed, stratify=y[tr])
    lm = legacy_feature_metadata(b)
    feats = list(b.X.columns)
    cont = lm["continuous_mutable"] + lm["non_decreasing"]          # src/data/loader.py order
    cont_idx = [feats.index(c) for c in cont]
    X = b.X.to_numpy(np.float64)
    sc = StandardScaler().fit(X[tr][:, cont_idx])
    return Split(seed, feats, tr, te, fit, val, X[tr], X[te], y[tr], y[te], sc, cont_idx, lm)


def mad_vector(X_raw: np.ndarray) -> np.ndarray:
    """Per-feature MAD on recorded train values; zero-MAD (binary) features fall back to the SD, then 1 (P9/P14)."""
    med = np.median(X_raw, axis=0)
    mad = np.median(np.abs(X_raw - med), axis=0)
    sd = X_raw.std(axis=0)
    return np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
