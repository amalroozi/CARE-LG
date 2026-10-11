"""Phase 12: FACE, DiCE, Growing Spheres, PACE-style (reuse src/baselines, scripts/run_pace_reimpl.py), C-CHVAE + Wachter (CARLA or faithful reimpl), reachable-set recourse verification (Kothari et al. 2024) on a linear model.

CARLA (installed) fails to import without TensorFlow and the authors' `reachml` 0.2.0a1 pins
numpy 2.1 / pandas 2.2 + a SCIP solver (the shared venv has pandas 3.0), so C-CHVAE, Wachter and the
reachable-set verifier are faithful reimplementations; deviations are listed per function and in
DECISIONS.md. Every method receives the SAME per-patient target tau = min(age-bracket tau_p, r0) and is
judged by the SAME oracle; gradient methods use the MLP (a surrogate when the oracle is XGBoost).
Each function returns {"X": (m, F) recorded-unit candidates or empty, "latency_s": float}.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.baselines.dice import run_dice_diverse
from src.baselines.face import build_face_graph, run_face_query
from src.baselines.growing_spheres import run_growing_spheres

DEV = torch.device("cpu")


def _empty(t0):
    return {"X": np.zeros((0, 0)), "latency_s": time.perf_counter() - t0}


def face(ctx, x0s, tmask) -> dict:
    """FACE (Poyiadzi et al. AIES 2020) via src/baselines/face.py; endpoint is a recorded train patient (P12)."""
    t0 = time.perf_counter()
    ok, path, _, _ = run_face_query(x0s, ctx["Xtr_s"], ctx["face_graph"], tmask)
    if not ok:
        return _empty(t0)
    return {"X": ctx["Xtr_raw"][[path[-1]]], "latency_s": time.perf_counter() - t0, "n_hops": len(path) - 1}


def dice(ctx, x0s, tau, seed) -> dict:
    """DiCE (Mothilal et al. FAT* 2020) via src/baselines/dice.py, k=4, paper lambdas (P12)."""
    t0 = time.perf_counter()
    ok, cfs, _ = run_dice_diverse(x0s, ctx["grad_clf"], DEV, ctx["legacy_meta"], ctx["features"], ctx["Xtr_s"],
                                  k=4, target_threshold=tau, seed=seed)
    if not ok:
        return _empty(t0)
    return {"X": ctx["split"].unscale(np.vstack(cfs)), "latency_s": time.perf_counter() - t0}


def growing_spheres(ctx, x0s, tau, seed) -> dict:
    """Growing Spheres (Laugel et al. 2018) via src/baselines/growing_spheres.py; no actionability by design (P12)."""
    t0 = time.perf_counter()
    ok, x, _ = run_growing_spheres(x0s, ctx["oracle"].torch_callable(), DEV, target_threshold=tau, seed=seed)
    if not ok:
        return _empty(t0)
    return {"X": ctx["split"].unscale(x[None, :]), "latency_s": time.perf_counter() - t0}


PACE_MAGNITUDES = (0.25, 0.5, 0.75, 1.0, 1.5)   # scripts/run_pace_reimpl.py


def pace_style(ctx, x0s, tau) -> dict:
    """PACE-style incremental-budget generate-and-test (copied from scripts/run_pace_reimpl.py, oracle + per-patient tau) (P12)."""
    t0 = time.perf_counter()
    meta, feats = ctx["legacy_meta"], ctx["features"]
    adm = {f: -1 for f in meta["directional_reduce"] if f in meta["continuous_mutable"]}
    adm.update({f: 1 for f in meta["directional_increase"] if f in meta["continuous_mutable"]})
    std = {f: float(np.std(ctx["Xtr_s"][:, feats.index(f)])) for f in adm}
    names = list(adm)
    for b in range(1, len(names) + 1):
        cands = []
        for sub in itertools.combinations(names, b):
            for mags in itertools.product(PACE_MAGNITUDES, repeat=b):
                x = x0s.copy()
                for f, m in zip(sub, mags):
                    x[feats.index(f)] += adm[f] * m * std[f]
                cands.append(x)
        C = np.vstack(cands)
        p = ctx["oracle"].predict(C)
        hit = np.where(p < tau)[0]
        if len(hit):
            return {"X": ctx["split"].unscale(C[hit[0]][None, :]), "latency_s": time.perf_counter() - t0}
    return _empty(t0)


def wachter(ctx, x0s, tau, seed, lambdas=(0.1, 1.0, 10.0, 100.0), steps=200, lr=0.05) -> dict:
    """Wachter, Mittelstadt & Russell (Harv. JOLT 2018): min lambda*(f(x')-y')^2 + MAD-weighted L1, lambda
    increased until valid. Deviations: y'=0; immutables + age frozen (as DiCE here); fixed lambda schedule (P12)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    feats, meta = ctx["features"], ctx["legacy_meta"]
    free = [i for i, f in enumerate(feats) if f in meta["continuous_mutable"] + meta["categorical_mutable"]]
    mad = torch.tensor(ctx["mad_s"], dtype=torch.float32)
    x0 = torch.tensor(x0s, dtype=torch.float32)
    mask = torch.zeros(len(feats)); mask[free] = 1.0
    best = None
    for lam in lambdas:
        c = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            p = ctx["grad_clf"](c.unsqueeze(0)).squeeze()
            loss = lam * (p - 0.0) ** 2 + ((c - x0).abs() / mad * mask).sum()
            loss.backward()
            c.grad = c.grad * mask
            opt.step()
        xc = c.detach().numpy()
        if ctx["oracle"].predict(xc[None, :])[0] < tau:
            best = xc
            break
    if best is None:
        return _empty(t0)
    return {"X": ctx["split"].unscale(best[None, :]), "latency_s": time.perf_counter() - t0}


def cchvae(ctx, x0s, tau, seed, eta=0.1, n_samples=500, max_layers=30) -> dict:
    """C-CHVAE (Pawelczyk, Broelemann & Kasneci, WWW 2020): growing-sphere search in the latent space of an
    autoencoder, decoding candidates and keeping immutable attributes at the query's values. Deviations:
    the CARE-LG type-aware VAE (unconditional) replaces the conditional HVAE; conditioning on immutables
    (incl. age) is done by clamping them after decoding; first layer with a valid point, closest in L1 (P12)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    vae, feats, meta = ctx["vae"], ctx["features"], ctx["legacy_meta"]
    fixed = [feats.index(f) for f in meta["immutable"] + meta["non_decreasing"]]
    with torch.no_grad():
        z0 = vae.encode(torch.tensor(x0s[None, :]))[0].numpy()[0]
    d = len(z0)
    lo = 0.0
    for _ in range(max_layers):
        hi = lo + eta
        dirs = rng.standard_normal((n_samples, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        r = rng.uniform(lo ** d, hi ** d, n_samples) ** (1.0 / d)
        Z = z0 + dirs * r[:, None]
        with torch.no_grad():
            Xs = vae.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
        Xs[:, fixed] = x0s[fixed]
        p = ctx["oracle"].predict(Xs)
        ok = np.where(p < tau)[0]
        if len(ok):
            j = ok[np.argmin(np.abs(Xs[ok] - x0s).sum(1))]
            return {"X": ctx["split"].unscale(Xs[j][None, :]), "latency_s": time.perf_counter() - t0}
        lo = hi
    return _empty(t0)


# ------------------------------------------------------------------ reachable-set verification

def fit_linear(ctx) -> LogisticRegression:
    """Linear classifier for the reachable-set head-to-head (Kothari et al. 2024 audit linear models) (P12)."""
    s = ctx["split"]
    return LogisticRegression(max_iter=2000).fit(s.Xtr[s.fit], s.ytr[s.fit])


def action_grid(schema, rs, x0_raw) -> list:
    """Discrete reachable values per feature: immutables/age fixed; directional features step by the
    min meaningful change in the admissible direction up to max_plausible_change, within phys bounds;
    binary/ordinal move one integer level at a time (P12)."""
    grid = []
    for f in schema.features:
        j, kd, x = schema.idx(f), schema.kind[f], float(x0_raw[schema.idx(f)])
        if kd not in ("reduce", "increase"):
            grid.append(np.array([x])); continue
        sgn = -1 if kd == "reduce" else 1
        lo, hi = schema.phys[f]
        if schema.ftype[f] != "continuous":
            vals = np.arange(x, (lo - 0.5) if sgn < 0 else (hi + 0.5), sgn * 1.0)
        else:
            r = rs.rules[f]
            step = max(r.meaningful, 1e-6)
            mc = (r.maxchg / 1000.0) if r.maxchg is not None else 4 * step
            vals = x + sgn * step * np.arange(0, int(np.floor(mc / step + 1e-9)) + 1)
            vals = vals[(vals >= min(lo, x)) & (vals <= max(hi, x))]
        grid.append(np.unique(np.r_[x, vals]))
    return grid


def reachable_set_verify(ctx, lr: LogisticRegression, x0_raw, rs, tau_p) -> dict:
    """Enumerate the reachable set R(x0) and verify recourse under the linear model (P12).

    precluded = no point of R(x0) has LR probability < min(tau_p, LR risk of x0). Exact on the grid
    (exhaustive enumeration, as the paper's ReachableSetEnumerator does for discrete action sets).
    Also returns the min-L1 (MAD) reachable point that the LR accepts, so it can be scored by the oracle.
    """
    t0 = time.perf_counter()
    grid = action_grid(ctx["schema"], rs, x0_raw)
    size = int(np.prod([len(g) for g in grid]))
    R = np.array(np.meshgrid(*grid, indexing="ij")).reshape(len(grid), -1).T
    p = lr.predict_proba(ctx["split"].scale(R))[:, 1]
    r0 = float(lr.predict_proba(ctx["split"].scale(x0_raw[None, :]))[0, 1])
    tau = min(tau_p, r0)
    ok = np.where(p < tau)[0]
    rec = {"rs_size": size, "rs_fixed_point": size == 1, "rs_precluded": len(ok) == 0, "rs_lr_risk": r0,
           "rs_min_lr_risk": float(p.min()), "latency_s": time.perf_counter() - t0}
    if len(ok):
        cost = (np.abs(R[ok] - x0_raw) / ctx["mad_raw"]).sum(1)
        rec["X"] = R[ok[np.argmin(cost)]][None, :]
    else:
        rec["X"] = np.zeros((0, 0))
    return rec
