"""
Exact-Guarantee Decoder (EGD) — Phase 1.

Replaces DSR's statistical calibration+repair (experiments_v4) with a decoder whose
OUTPUT LAYER makes constraint violations structurally impossible, per feature type:

  - IMMUTABLE (sex, fasting_blood_sugar): masked. The decoder does not produce this
    dimension; every decoded value copies the SOURCE state's value through unchanged.
    Zero reconstruction error is possible because there is no reconstruction -- the
    output IS the input for this column.
  - NON-DECREASING (age): a signed residual through a strictly-nonnegative activation,
    x_target[k] = x_source[k] + softplus(net_k(z_target)). softplus > 0 everywhere, so
    the sum is >= x_source[k] for ANY network weights -- the guarantee holds before any
    training happens, and no training procedure can break it.
  - DIRECTIONAL_REDUCE (must not increase: BP, cholesterol, etc.):
    x_target[k] = x_source[k] - softplus(net_k(z_target)) <= x_source[k], by the same
    argument, mirrored.
  - DIRECTIONAL_INCREASE (must not decrease: e.g. UCI's max_heart_rate): same
    construction as non-decreasing, x_target[k] = x_source[k] + softplus(net_k(z_target)).
  - FREE (anything not covered above): decoded normally, no restriction. (Neither UCI
    nor real NHANES's schema has any column landing here in practice -- every mutable
    feature carries a directional or non-decreasing constraint -- but the code path
    exists for completeness/robustness on a future schema.)

**Consequence, stated exactly**: because the residual/copy-through construction operates
in the SAME representation the feature is stored in (z-score standardized for
continuous_mutable/non_decreasing columns; raw integer units for immutable/
categorical_mutable columns), and StandardScaler's affine transform has a strictly
positive scale, the SIGN of a delta is identical in scaled and unscaled (clinical) units.
Enforcing "scaled delta >= 0" or "<= 0" therefore enforces the true clinical-unit
direction exactly -- with ZERO tolerance, not the original codebase's small epsilon
noise allowance (e.g. cholesterol's original 1.0 mg/dL slack, age's 0.1yr slack). EGD's
guarantee is consequently STRICTER than what experiments_v2-v4 required, which is a
reasonable, simpler, and defensible choice, not an oversight -- see METHOD.md.

**Sequential decoding — the structural requirement this brief calls out explicitly**:
decode_step needs the SOURCE state's real values, so a path's hop k+1 must be decoded
relative to hop k's OWN decoded output, not independently per-node the way the original
TabularVAE / type-aware decoder worked. `decode_path` implements this chain. A direct
consequence, worth stating plainly: a graph "node" no longer has one canonical decoded
value -- its value depends on which path reached it. This is handled explicitly in
`egd_graph.py` (graph search uses latent-space distance only; the decoded trajectory,
and the risk check that is NOT architecturally guaranteed, is computed AFTER a candidate
path is found, per query) -- see that module's docstring.

**What is NOT guaranteed by this architecture, stated once, precisely**: risk reduction.
Nothing here constrains classifier(x_target) relative to classifier(x_source) --  that
is checked post-hoc against the real classifier, exactly as the brief specifies, and can
fail (the search must handle that case, not assume it away).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EGDColumnPlan:
    immutable_cols: List[str] = field(default_factory=list)
    add_residual_cols: List[str] = field(default_factory=list)     # non_decreasing + directional_increase
    sub_residual_cols: List[str] = field(default_factory=list)     # directional_reduce
    free_cols: List[str] = field(default_factory=list)


def build_column_plan(feature_cols: List[str], feature_metadata: dict) -> EGDColumnPlan:
    """Pure rule-based classification from FEATURE_METADATA -- no data inspection needed,
    unlike vae_v3's type-aware decoder (which had to look at the data to find binary vs.
    multi-category columns). EGD's categories come directly from the existing constraint
    schema, so every column's treatment is deducible from configs/dataset_config.py alone."""
    plan = EGDColumnPlan()
    immutable = set(feature_metadata.get("immutable", []))
    non_decreasing = set(feature_metadata.get("non_decreasing", []))
    dir_reduce = set(feature_metadata.get("directional_reduce", []))
    dir_increase = set(feature_metadata.get("directional_increase", []))

    for col in feature_cols:
        if col in immutable:
            plan.immutable_cols.append(col)
        elif col in non_decreasing or col in dir_increase:
            plan.add_residual_cols.append(col)
        elif col in dir_reduce:
            plan.sub_residual_cols.append(col)
        else:
            plan.free_cols.append(col)
    return plan


class ExactGuaranteeDecoder(nn.Module):
    """
    Encoder identical in shape to TabularVAE/TabularVAETypeAware (Input -> 64 -> 32 ->
    (mu, logvar), latent_dim=4). Decoder is the type-per-feature construction above,
    requiring an explicit source state at decode time (`decode_step`) rather than a
    context-free `decode(z)`.
    """

    def __init__(self, input_dim: int, feature_cols: List[str], feature_metadata: dict, latent_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_cols = feature_cols
        self.col_idx = {c: i for i, c in enumerate(feature_cols)}
        self.plan = build_column_plan(feature_cols, feature_metadata)

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, 32)
        self.decoder_fc2 = nn.Linear(32, 64)

        # One scalar head per non-immutable column (immutables need no head at all --
        # their output is the identity function of the source, with no learnable part).
        heads = {}
        for col in self.plan.add_residual_cols + self.plan.sub_residual_cols + self.plan.free_cols:
            heads[col] = nn.Linear(64, 1)
        self.heads = nn.ModuleDict(heads)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _trunk(self, z):
        return F.relu(self.decoder_fc2(F.relu(self.decoder_fc1(z))))

    def decode_step(self, z_target: torch.Tensor, x_source: torch.Tensor) -> torch.Tensor:
        """
        One hop: z_target (batch, d) -> x_target (batch, D), guaranteed relative to
        x_source (batch, D) -- both must be batched (unsqueeze(0) single points first).
        """
        h = self._trunk(z_target)
        cols = [None] * self.input_dim

        for c in self.plan.immutable_cols:
            cols[self.col_idx[c]] = x_source[:, self.col_idx[c]]
        for c in self.plan.add_residual_cols:
            cols[self.col_idx[c]] = x_source[:, self.col_idx[c]] + F.softplus(self.heads[c](h).squeeze(-1))
        for c in self.plan.sub_residual_cols:
            cols[self.col_idx[c]] = x_source[:, self.col_idx[c]] - F.softplus(self.heads[c](h).squeeze(-1))
        for c in self.plan.free_cols:
            cols[self.col_idx[c]] = self.heads[c](h).squeeze(-1)

        assert all(c is not None for c in cols), "every column must be covered by exactly one EGD rule"
        return torch.stack(cols, dim=-1)

    def decode_path(self, z_path: torch.Tensor, x0: torch.Tensor) -> List[torch.Tensor]:
        """
        Sequential decode along a path. z_path: (T, d) latent codes for hops 1..T.
        x0: (D,) the TRUE source state (the query patient's real recorded values, or a
        real graph node's real recorded values, for a path segment that starts there).
        Returns a list of T decoded (D,) tensors, hop 1 first; x0 itself is NOT included
        (the caller already has it).
        """
        states = []
        src = x0.unsqueeze(0)
        for t in range(z_path.shape[0]):
            tgt = self.decode_step(z_path[t].unsqueeze(0), src)
            states.append(tgt.squeeze(0))
            src = tgt
        return states

    def forward(self, x_target, x_source):
        """
        Training pass with an EXPLICIT (possibly different) source: encodes the
        TARGET, decodes conditioned on the given source. See egd_loss's docstring for
        why source != target is the training regime actually used, not source==target.
        """
        mu, logvar = self.encode(x_target)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode_step(z, x_source)
        return x_hat, mu, logvar


def egd_loss(egd: ExactGuaranteeDecoder, x_target: torch.Tensor, x_hat: torch.Tensor, mu, logvar, beta: float = 1.0):
    """
    Reconstruction loss against the TARGET (whatever the source was). Immutable columns
    are EXCLUDED (they reconstruct exactly by construction, contribute no gradient
    signal, and would otherwise contribute exactly-zero loss anyway regardless of
    source). All other columns use MSE (matching the original TabularVAE's
    reduction='sum' convention).
    """
    trainable_cols = egd.plan.add_residual_cols + egd.plan.sub_residual_cols + egd.plan.free_cols
    idx = [egd.col_idx[c] for c in trainable_cols]
    recon_loss = F.mse_loss(x_hat[:, idx], x_target[:, idx], reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def _build_knn_pair_index(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """For each row i, returns a randomly-chosen index among its k nearest neighbors
    in raw feature space. Available as an alternative training-pair regime (see
    train_egd's docstring for why it was tried, and why it is NOT the default)."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, indices = nbrs.kneighbors(X)
    rng = np.random.default_rng(seed)
    choice = rng.integers(1, indices.shape[1], size=len(X))   # skip col 0 (self)
    return indices[np.arange(len(X)), choice]


def train_egd(train_loader, input_dim: int, feature_cols: List[str], feature_metadata: dict,
              latent_dim: int = 4, epochs: int = 25, beta: float = 1.0, lr: float = 1e-3,
              pairing: str = "random", knn_pair_k: int = 15, seed: int = 0,
              device: torch.device = torch.device("cpu")) -> ExactGuaranteeDecoder:
    """
    Trains ExactGuaranteeDecoder on (source, target) PAIRS. `pairing` selects the
    regime; three were tried and the choice is empirical, documented here rather than
    asserted:

    - 'self' (source==target==x): a first version of this function. Only ever shows
      the residual heads decode_step(encode(x), x_source=x) -- the network learns the
      residual for "source and target are literally the same patient" (near-zero) and
      generalizes poorly to real graph edges, which always connect DIFFERENT patients.
      Result: 0/10 successes on BOTH cohorts (seed-0 smoke test).

    - 'knn' (pair with a raw-feature-space k-nearest-neighbor, matching what a real
      graph edge typically connects): matches the deployment distribution, but gives
      residual TARGETS too small to bridge the gap a high-risk-to-low-risk recourse
      query actually needs, even accumulated over several hops. Result: UCI 2-3/15-20,
      real NHANES 0/15-20 across several k values tested (15, 40, 100).

    - 'random' (DEFAULT; shuffle the batch, pair x_i with an unrelated x_perm[i]):
      fixes 'self''s near-zero collapse (UCI: 13/15 successes, the best result found)
      but has a real, diagnosed weakness on real NHANES: since x_source, x_target are
      independent under random pairing, Var(x_target - x_source | z_target) is
      dominated by Var(x_source) -- noise unrelated to what z_target can predict --
      which empirically pushed learned residuals back toward near-zero on real NHANES
      specifically (mean |delta| ~0.04-0.05 std units, falling further with MORE
      training, not less -- verified this is a converged optimum, not underfitting, by
      testing 100 epochs and a deeper per-feature head; neither changed the outcome).
      Real NHANES's poor result under EVERY pairing regime tested is reported as a
      genuine, diagnosed limitation in FINDINGS.md -- not a training bug still to be
      fixed -- and is mechanistically the same 4-dimensional-latent-bottleneck
      difficulty `experiments_v3` Phase B already found for the ordinary (non-EGD)
      decoder on this cohort's continuous features.

    'random' is kept as the default because it is the best-performing regime actually
    found, on the one cohort where any regime succeeds well.
    """
    assert pairing in ("self", "knn", "random")
    all_x = np.vstack([xb.numpy() for xb, _ in train_loader])
    n = len(all_x)
    X_all = torch.tensor(all_x, dtype=torch.float32, device=device)

    if pairing == "knn":
        pair_idx = _build_knn_pair_index(all_x, k=knn_pair_k, seed=seed)
        X_paired_fixed = torch.tensor(all_x[pair_idx], dtype=torch.float32, device=device)

    egd = ExactGuaranteeDecoder(input_dim, feature_cols, feature_metadata, latent_dim).to(device)
    optimizer = torch.optim.Adam(egd.parameters(), lr=lr)

    egd.train()
    batch_size = 64
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            x_source = X_all[idx]
            if pairing == "self":
                x_target = x_source
            elif pairing == "knn":
                x_target = X_paired_fixed[idx]
            else:  # random: shuffle WITHIN this batch
                shuf = torch.randperm(len(idx), device=device)
                x_target = x_source[shuf]

            optimizer.zero_grad()
            x_hat, mu, logvar = egd(x_target, x_source)
            loss = egd_loss(egd, x_target, x_hat, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

    egd.eval()
    return egd
