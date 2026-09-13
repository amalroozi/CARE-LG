"""
Phase B: type-aware VAE decoder, addressing the RQ1 root cause identified in
experiments_v2/FINDINGS.md (age MAE ~6yr, binary-sex MAE ~0.48 under the
original undifferentiated-continuous decoder).

Design (per feature, auto-detected from the training data + FEATURE_METADATA):
  - BINARY columns (immutables/categorical_mutable with <=2 distinct raw
    values, e.g. sex, fasting_blood_sugar, exercise_angina): a dedicated
    sigmoid head, trained with BCE loss, evaluated by decoded classification
    accuracy (round to nearest class), not MAE.
  - MULTI-CATEGORY columns (categorical_mutable with >2 distinct values,
    e.g. UCI's `slope` in {1,2,3}): a dedicated softmax head over the
    observed categories, trained with cross-entropy on the class index,
    evaluated by decoded classification accuracy. The value exposed through
    `decode()` (for compatibility with every existing downstream function,
    which expects one continuous scalar per column) is the SOFT EXPECTATION
    under the softmax distribution, sum_k class_value_k * P(k) -- fully
    differentiable in z (unlike a hard argmax), which matters because the
    Riemannian metric's Jacobian requires the decoder to be differentiable
    everywhere it is evaluated.
  - The NON-DECREASING column (age): a dedicated linear regression head,
    identical in form to a normal continuous head, but weighted higher in
    the reconstruction loss (age_loss_weight, default 8x a normal continuous
    feature's weight) given how much constraint logic depends on Δage.
  - All other CONTINUOUS_MUTABLE columns: a standard linear regression head,
    MSE loss, weight 1x.

`decode(z)` always returns a single (batch, input_dim) tensor in the SAME
column order/scale as the original TabularVAE.decode(), so it is a drop-in
replacement everywhere in experiments_v2/v3 code that calls vae.decode(z)
(build_edge_table, riemannian_batch, recourse decoding, etc.) -- no other
file needs to change to use this new model.

This is NEW code, added alongside (not instead of) src/vae/model.py, which
remains completely untouched -- consistent with the v2 precedent of never
editing src/, benchmarks/, scripts/, configs/, or data/.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class ColumnTypeInfo:
    binary_cols: List[str] = field(default_factory=list)
    categorical_cols: Dict[str, List[float]] = field(default_factory=dict)  # col -> sorted unique raw values
    age_col: str = None
    continuous_cols: List[str] = field(default_factory=list)


def detect_column_types(X_train: np.ndarray, feature_cols: List[str], feature_metadata: dict) -> ColumnTypeInfo:
    """Auto-detects per-column type from training data + the existing FEATURE_METADATA schema."""
    info = ColumnTypeInfo()
    candidate_cat_cols = feature_metadata.get("immutable", []) + feature_metadata.get("categorical_mutable", [])
    non_decreasing = feature_metadata.get("non_decreasing", [])
    continuous_mutable = feature_metadata.get("continuous_mutable", [])

    for col in feature_cols:
        idx = feature_cols.index(col)
        vals = X_train[:, idx]
        uniq = np.unique(np.round(vals, 6))
        if col in candidate_cat_cols:
            if len(uniq) <= 2:
                info.binary_cols.append(col)
            else:
                info.categorical_cols[col] = sorted(float(v) for v in uniq)
        elif col in non_decreasing:
            info.age_col = col
        elif col in continuous_mutable:
            info.continuous_cols.append(col)
        else:
            # any column not explicitly classified falls back to continuous (fail-safe,
            # matches validate_exhaustive_column_classification's guarantee that every
            # column IS classified somewhere in FEATURE_METADATA)
            info.continuous_cols.append(col)
    return info


class TabularVAETypeAware(nn.Module):
    """Type-aware TabularVAE: same encoder as the original, type-specialized decoder heads."""

    def __init__(self, input_dim: int, feature_cols: List[str], feature_metadata: dict,
                 column_types: ColumnTypeInfo, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_cols = feature_cols
        self.col_idx = {c: i for i, c in enumerate(feature_cols)}
        self.types = column_types

        # Encoder: identical architecture to src/vae/model.py::TabularVAE
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder shared trunk: identical shape to the original (latent -> 32 -> 64)
        self.decoder_fc1 = nn.Linear(latent_dim, 32)
        self.decoder_fc2 = nn.Linear(32, 64)

        # Type-specialized output heads
        self.binary_heads = nn.ModuleDict({c: nn.Linear(64, 1) for c in column_types.binary_cols})
        self.categorical_heads = nn.ModuleDict({
            c: nn.Linear(64, len(vals)) for c, vals in column_types.categorical_cols.items()
        })
        self.categorical_values = {
            c: torch.tensor(vals, dtype=torch.float32) for c, vals in column_types.categorical_cols.items()
        }
        cont_cols = ([column_types.age_col] if column_types.age_col else []) + column_types.continuous_cols
        self.continuous_heads = nn.ModuleDict({c: nn.Linear(64, 1) for c in cont_cols})

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decoder_trunk(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        return F.relu(self.decoder_fc2(h1))

    def decode(self, z):
        """
        Returns (batch, input_dim) soft reconstruction, same contract as the
        original TabularVAE. Built with OUT-OF-PLACE ops only (per-column
        tensors assembled via torch.stack) rather than in-place indexed
        assignment into a pre-allocated tensor, because the latter breaks
        under torch.func.vmap (used by riemannian_batch.py for the batched
        Jacobian): vmap needs every intermediate to be a function of its
        batched inputs, and an in-place write into a freshly-allocated,
        non-batched `zeros` tensor is not.
        """
        was_unbatched = z.dim() == 1
        h2 = z.unsqueeze(0) if was_unbatched else z
        h = self._decoder_trunk(h2)

        cols = [None] * self.input_dim
        for c, head in self.binary_heads.items():
            cols[self.col_idx[c]] = torch.sigmoid(head(h)).squeeze(-1)
        for c, head in self.categorical_heads.items():
            probs = F.softmax(head(h), dim=-1)
            vals = self.categorical_values[c].to(h.device)
            cols[self.col_idx[c]] = (probs * vals.unsqueeze(0)).sum(dim=-1)
        for c, head in self.continuous_heads.items():
            cols[self.col_idx[c]] = head(h).squeeze(-1)

        assert all(c is not None for c in cols), "every column must be covered by exactly one decoder head"
        out = torch.stack(cols, dim=-1)
        return out.squeeze(0) if was_unbatched else out

    def decode_with_logits(self, z):
        """Training-only: returns raw per-type outputs (sigmoid probs, softmax logits, continuous values)."""
        h = self._decoder_trunk(z)
        binary_probs = {c: torch.sigmoid(head(h)).squeeze(-1) for c, head in self.binary_heads.items()}
        categorical_logits = {c: head(h) for c, head in self.categorical_heads.items()}
        continuous_vals = {c: head(h).squeeze(-1) for c, head in self.continuous_heads.items()}
        return binary_probs, categorical_logits, continuous_vals

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function_type_aware(vae: TabularVAETypeAware, x: torch.Tensor, mu, logvar, z,
                              beta: float = 1.0, age_loss_weight: float = 8.0):
    """
    Type-aware reconstruction loss (sum-reduced, matching the original's
    reduction='sum' convention) + KL divergence. Binary columns use BCE,
    categorical columns use cross-entropy on the class index, continuous
    columns use MSE, age uses MSE weighted by age_loss_weight.
    """
    binary_probs, categorical_logits, continuous_vals = vae.decode_with_logits(z)
    recon_loss = 0.0

    for c, prob in binary_probs.items():
        target = x[:, vae.col_idx[c]]
        recon_loss = recon_loss + F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), target, reduction="sum")

    for c, logits in categorical_logits.items():
        target_vals = x[:, vae.col_idx[c]]
        class_values = vae.categorical_values[c].to(x.device)
        # map raw value -> class index by nearest match (robust to float noise)
        target_idx = torch.argmin((target_vals.unsqueeze(1) - class_values.unsqueeze(0)).abs(), dim=1)
        recon_loss = recon_loss + F.cross_entropy(logits, target_idx, reduction="sum")

    for c, val in continuous_vals.items():
        target = x[:, vae.col_idx[c]]
        weight = age_loss_weight if c == vae.types.age_col else 1.0
        recon_loss = recon_loss + weight * F.mse_loss(val, target, reduction="sum")

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def train_vae_type_aware(train_loader, input_dim: int, feature_cols: List[str], feature_metadata: dict,
                          latent_dim: int = 4, epochs: int = 25, beta: float = 1.0, lr: float = 1e-3,
                          age_loss_weight: float = 8.0, device=None) -> TabularVAETypeAware:
    """Trains TabularVAETypeAware. Same epoch/lr/beta defaults as the original train_vae for a fair comparison."""
    if device is None:
        device = torch.device("cpu")

    all_x = []
    for Xb, _ in train_loader:
        all_x.append(Xb.numpy())
    X_train_np = np.vstack(all_x)
    column_types = detect_column_types(X_train_np, feature_cols, feature_metadata)

    vae = TabularVAETypeAware(input_dim, feature_cols, feature_metadata, column_types, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(epochs):
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            mu, logvar = vae.encode(X_batch)
            z = vae.reparameterize(mu, logvar)
            loss = loss_function_type_aware(vae, X_batch, mu, logvar, z, beta=beta, age_loss_weight=age_loss_weight)
            loss.backward()
            optimizer.step()

    vae.eval()
    return vae
