"""Phase 9: ontology-guided synthetic recourse (reuse archive/experiments_caren/src/v2 genetic search + guidance); Riemannian natural-gradient variant (scripts/run_stage2_synthetic_search.py); 3 diverse CFs; Tier 3 = certified no recourse.

Adapted (copied, not imported: the CARE-N code is bound to eICU state rows and APACHE strata) from
archive/experiments_caren/src/v2/generator.py. Search space = the patient's mutable features in RECORDED units.
HARD tier enforced by construction: immutables and age frozen (synthetic recourse is "now", as in the
CARE-LG stage-2 search), directional features one-way (reduce: [phys_min, x0]; increase: [x0, phys_max]),
discrete features snapped to their integer grid. Loss (per candidate x'):
  w_valid * max(0, logit f(x') - logit tau)        validity hinge (tau = min(age-bracket tau_p, r0))
  + w_prox * MAD-L1 / F + w_sparse * L0 / F
  + w_rule * soft-rule penalty / F                  (patient-specific targets + magnitude, MAD-scaled)
  + w_plaus * mean z-distance to 5 nearest train patients
  + w_contra * change reachable only via contraindicated classes
Symbolic guidance (CARE-N v2 step 7): features moved by an allowed (indicated, non-contraindicated)
intervention get priority and a preferred direction toward the patient-specific target.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.func import jacrev

from experiment_final.src.knowledge_layer import SOFT_KINDS, hard_violates, si, soft_degrees

EPS = 1e-9


@dataclass
class GenConfig:
    """Loss weights and search settings (CARE-N v2 defaults); ablations flip one field (P9/P13)."""
    w_valid: float = 5.0
    w_prox: float = 1.0
    w_sparse: float = 0.5
    w_rule: float = 1.0
    w_plaus: float = 0.3
    w_div: float = 0.5
    w_contra: float = 3.0
    guidance: bool = True
    hard_soft: bool = False
    k: int = 3
    pop: int = 100
    gens: int = 40
    seed: int = 0


def intervention_map(k, schema) -> Dict[Tuple[str, str], List[str]]:
    """(dataset feature, '+'/'-') -> classes acting on its concept in that direction (KG acts_on) (P9/P10)."""
    by_concept = {}
    for f in schema.features:
        by_concept.setdefault(schema.concept[f], []).append(f)
    out: Dict[Tuple[str, str], List[str]] = {}
    for r in k.interventions.itertuples():
        for c, d in zip(r.acts_on_features.split(";"), r.expected_direction.split(";")):
            for f in by_concept.get(c, []):
                out.setdefault((f, d), []).append(r.intervention_class)
    return out


@dataclass
class Guidance:
    weight: np.ndarray
    pref_dir: np.ndarray
    contra_only: np.ndarray


def guidance_for(rs, schema, k, x0_raw, M: List[int], allowed: List[str], contra: List[str], on: bool) -> Guidance:
    """Step-7 guidance: priority, preferred direction, contraindicated-only moves (P9)."""
    F = len(M)
    w, pref, co = np.ones(F), np.zeros(F), np.zeros((F, 2), bool)
    if not on:
        return Guidance(w, pref, co)
    imap = intervention_map(k, schema)
    for a, j in enumerate(M):
        f = schema.features[j]
        dirn = "-" if schema.kind[f] == "reduce" else "+"
        good = [c for c in imap.get((f, dirn), []) if c in allowed]
        bad = [c for c in imap.get((f, dirn), []) if c in contra]
        co[a, 0 if dirn == "-" else 1] = bool(bad) and not good
        r = rs.rules[f]; v = int(si(x0_raw[j]))
        outside = (r.hi is not None and v > r.hi) if dirn == "-" else (r.lo is not None and v < r.lo)
        w[a] = 1.0 if good else (0.1 if co[a].any() else 0.25)
        if good and not outside:
            w[a] = 0.5
        pref[a] = -1 if dirn == "-" else 1
    return Guidance(w, pref, co)


class Problem:
    """Everything the loss needs for one query (P9)."""

    def __init__(self, x0_raw, r0, tau, schema, rs, k, oracle, split, knn, zmu, zsd, mad_full, allowed, contra, cfg: GenConfig):
        self.schema, self.rs, self.k, self.oracle, self.split, self.cfg = schema, rs, k, oracle, split, cfg
        self.x0 = np.asarray(x0_raw, float)
        self.r0, self.tau = r0, min(tau, r0)
        self.M = [schema.idx(f) for f in schema.mutable]
        self.F = len(self.M)
        self.lo = np.empty(self.F); self.hi = np.empty(self.F)
        for a, j in enumerate(self.M):
            f = schema.features[j]; pmin, pmax = schema.phys[f]
            if schema.kind[f] == "reduce":
                self.lo[a], self.hi[a] = min(pmin, self.x0[j]), self.x0[j]
            elif schema.kind[f] == "increase":
                self.lo[a], self.hi[a] = self.x0[j], max(pmax, self.x0[j])
            else:
                self.lo[a], self.hi[a] = pmin, pmax
        self.discrete = np.array([schema.ftype[schema.features[j]] != "continuous" for j in self.M])
        self.mad = mad_full[self.M]
        self.mad_full = mad_full
        self.knn, self.zmu, self.zsd = knn, zmu, zsd
        self.xm0 = self.x0[self.M]
        self.logit_tau = np.log(self.tau + EPS) - np.log(1 - self.tau + EPS)
        self.guide = guidance_for(rs, schema, k, self.x0, self.M, allowed, contra, cfg.guidance)
        tgt = []
        for j in self.M:
            r = rs.rules[schema.features[j]]
            lo = r.lo / 1000 if r.lo is not None else None
            hi = r.hi / 1000 if r.hi is not None else None
            tgt.append(hi if (hi is not None and self.x0[j] > hi) else (lo if (lo is not None and self.x0[j] < lo) else np.nan))
        self.tgt = np.array(tgt, float)

    def project(self, Xm: np.ndarray) -> np.ndarray:
        """Hard tier by construction: box + discrete grid (P9)."""
        Xm = np.clip(Xm, self.lo, self.hi)
        Xm[:, self.discrete] = np.clip(np.round(Xm[:, self.discrete]), self.lo[self.discrete], self.hi[self.discrete])
        return Xm

    def full(self, Xm: np.ndarray) -> np.ndarray:
        X = np.tile(self.x0, (len(Xm), 1))
        X[:, self.M] = Xm
        return X

    def risk(self, Xm: np.ndarray) -> np.ndarray:
        return self.oracle.predict(self.split.scale(self.full(Xm)))

    def terms(self, Xm: np.ndarray) -> Dict[str, np.ndarray]:
        X = self.full(Xm)
        p = self.oracle.predict(self.split.scale(X))
        lg = np.log(p + EPS) - np.log(1 - p + EPS)
        D = Xm - self.xm0[None, :]
        deg = soft_degrees(self.rs, self.schema, self.x0, X)
        rule = sum((d[:, self.M] / 1000.0) / self.mad[None, :] for d in deg.values()).sum(1)
        dk, _ = self.knn.kneighbors((X - self.zmu) / self.zsd)
        contra = np.zeros(len(Xm))
        if self.cfg.guidance:
            neg = self.guide.contra_only[:, 0][None, :] & (D < -EPS)
            pos = self.guide.contra_only[:, 1][None, :] & (D > EPS)
            contra = (np.abs(D) / self.mad[None, :] * (neg | pos)).sum(1)
        any_soft = np.any(np.stack([d[:, self.M].any(1) for d in deg.values()]), axis=0)
        return {"p": p, "valid": np.maximum(lg - self.logit_tau, 0), "prox": (np.abs(D) / self.mad).sum(1) / self.F,
                "l0": (np.abs(D) > EPS).sum(1) / self.F, "rule": rule / self.F, "plaus": dk.mean(1),
                "contra": contra, "any_soft": any_soft}

    def loss(self, T: Dict[str, np.ndarray]) -> np.ndarray:
        c = self.cfg
        L = (c.w_valid * T["valid"] + c.w_prox * T["prox"] + c.w_sparse * T["l0"] + c.w_rule * T["rule"]
             + c.w_plaus * T["plaus"] + c.w_contra * T["contra"])
        return L + 1e6 * T["any_soft"] if c.hard_soft else L

    def is_valid(self, T) -> np.ndarray:
        v = T["p"] < self.tau
        return v & ~T["any_soft"] if self.cfg.hard_soft else v


def _init_pop(P: Problem, rng) -> np.ndarray:
    X = np.tile(P.xm0, (P.cfg.pop, 1))
    w = P.guide.weight / P.guide.weight.sum()
    for i in range(P.cfg.pop):
        m = rng.integers(1, min(4, P.F) + 1)
        for a in rng.choice(P.F, m, replace=False, p=w):
            if P.cfg.guidance and not np.isnan(P.tgt[a]) and rng.random() < 0.5:
                X[i, a] = P.xm0[a] + rng.uniform(0.5, 1.1) * (P.tgt[a] - P.xm0[a])
            else:
                s = P.guide.pref_dir[a] if (P.cfg.guidance and P.guide.pref_dir[a] != 0 and rng.random() < 0.8) else rng.choice([-1, 1])
                X[i, a] = P.xm0[a] + s * rng.uniform(0.3, 2.0) * P.mad[a]
    return P.project(X)


def _mutate(P: Problem, X: np.ndarray, rng) -> np.ndarray:
    pm = 0.3 * (P.guide.weight / P.guide.weight.max())
    Mk = rng.random(X.shape) < pm[None, :]
    step = np.abs(rng.normal(0, 0.6, X.shape)) * P.mad[None, :]
    sign = rng.choice([-1, 1], X.shape)
    if P.cfg.guidance:
        pd_ = np.tile(P.guide.pref_dir, (len(X), 1))
        sign = np.where((pd_ != 0) & (rng.random(X.shape) < 0.8), pd_, sign)
    X = np.where(Mk, X + sign * step, X)
    reset = (rng.random(X.shape) < 0.15) & (np.abs(X - P.xm0[None, :]) > EPS)
    return P.project(np.where(reset, P.xm0[None, :], X))


def genetic_search(P: Problem, rng):
    """Elitist GA with tournament selection (CARE-N v2) -> archive, losses, terms (P9)."""
    X = _init_pop(P, rng)
    AX, AT = [], []
    for _ in range(P.cfg.gens):
        T = P.terms(X); L = P.loss(T)
        AX.append(X); AT.append(T)
        elite = X[np.argsort(L)[:8]]
        a = rng.integers(0, len(X), (len(X), 3))
        parents = X[a[np.arange(len(X)), np.argmin(L[a], axis=1)]]
        mates = parents[rng.permutation(len(X))]
        kids = np.where((rng.random(len(X)) < 0.7)[:, None] & (rng.random(X.shape) < 0.5), mates, parents)
        X = np.vstack([elite, _mutate(P, kids, rng)[: len(X) - len(elite)]])
    AX = np.vstack(AX)
    AT = {k: np.concatenate([t[k] for t in AT]) for k in AT[0]}
    _, u = np.unique(np.round(AX, 6), axis=0, return_index=True)
    AT = {k: v[u] for k, v in AT.items()}
    return AX[u], P.loss(AT), AT


def select_diverse(P: Problem, X, L, valid, k) -> List[int]:
    """Best valid first, then argmax(-loss + w_div * min MAD-L1 distance) (CARE-N v2) (P9)."""
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return list(np.argsort(L)[:k])
    idx = idx[np.argsort(L[idx])][:300]
    chosen = [idx[0]]
    while len(chosen) < min(k, len(idx)):
        rest = np.array([i for i in idx if i not in chosen])
        dmin = np.min(np.stack([np.abs(X[rest] - X[c]).dot(1 / P.mad) for c in chosen]), axis=0)
        chosen.append(int(rest[np.argmax(-L[rest] + P.cfg.w_div * dmin)]))
    return chosen


def generate_ga(P: Problem, rng) -> dict:
    """k diverse counterfactuals via guided genetic search (P9)."""
    t0 = time.perf_counter()
    X, L, T = genetic_search(P, rng)
    valid = P.is_valid(T)
    sel = select_diverse(P, X, L, valid, P.cfg.k)
    return {"Xm": X[sel], "valid": valid[sel], "p": T["p"][sel], "latency_s": time.perf_counter() - t0,
            "min_risk_found": float(T["p"].min())}


def ceiling_search(P: Problem, rng) -> dict:
    """Validity-only search, 2x budget, no guidance (archive/experiments_caren/src/v2/ceiling_v2.py) (P8/P9)."""
    Q = Problem.__new__(Problem); Q.__dict__.update(P.__dict__)
    Q.cfg = replace(P.cfg, w_valid=100, w_prox=0, w_sparse=0, w_rule=0, w_plaus=0, w_div=0, w_contra=0,
                    guidance=False, hard_soft=False, pop=200, gens=80)
    Q.guide = guidance_for(P.rs, P.schema, P.k, P.x0, P.M, [], [], False)
    X, L, T = genetic_search(Q, rng)
    return {"ceiling_min_risk": float(T["p"].min()), "ceiling_reachable": bool((T["p"] < P.tau).any())}


def generate_ng(P: Problem, vae, torch_clf, steps: int = 50, lr: float = 0.12, reg: float = 1e-2) -> dict:
    """Riemannian natural-gradient latent traversal with hard projection (scripts/run_stage2_synthetic_search.py; Pegios et al. 2024) (P9).

    Gradient through `torch_clf` (the MLP; a surrogate when the oracle is XGBoost), validity by the oracle.
    """
    t0 = time.perf_counter()
    xs0 = torch.tensor(P.split.scale(P.x0[None, :])[0])
    with torch.no_grad():
        z = vae.encode(xs0.unsqueeze(0))[0].squeeze(0)

    def dec(zz):
        return vae.decode(zz.unsqueeze(0)).squeeze(0)

    best_p, best = float(P.oracle.predict(xs0.numpy()[None, :])[0]), P.xm0.copy()
    d = z.shape[0]
    for _ in range(steps):
        zr = z.clone().requires_grad_(True)
        risk = torch_clf(dec(zr).unsqueeze(0)).squeeze()
        g, = torch.autograd.grad(risk, zr)
        with torch.no_grad():
            J = jacrev(dec)(z)
            G = J.T @ J + reg * torch.eye(d)
            z_next = z - lr * torch.linalg.solve(G, g)
            x_raw = P.split.unscale(dec(z_next).numpy()[None, :])[0]
        xm = P.project(x_raw[P.M][None, :])
        p = float(P.risk(xm)[0])
        if p < best_p:
            best_p, best = p, xm[0]
        with torch.no_grad():
            z = vae.encode(torch.tensor(P.split.scale(P.full(xm))))[0].squeeze(0)
        if best_p < P.tau:
            break
    return {"Xm": best[None, :], "valid": np.array([best_p < P.tau]), "p": np.array([best_p]),
            "latency_s": time.perf_counter() - t0}


def describe(P: Problem, Xm: np.ndarray, lof) -> List[dict]:
    """Per-CF record: validity, changes, soft rules with degree, plausibility, hard-rule re-check (P9/P14)."""
    X = P.full(Xm)
    T = P.terms(Xm)
    deg = soft_degrees(P.rs, P.schema, P.x0, X)
    n_checks = sum(int(P.rs.rules[P.schema.features[j]].hi is not None or P.rs.rules[P.schema.features[j]].lo is not None)
                   + int(P.rs.rules[P.schema.features[j]].maxchg is not None) for j in P.M)
    lofv = -lof.score_samples((X - P.zmu) / P.zsd)
    hv = hard_violates(P.schema, P.x0, X, horizon=False)
    out = []
    for i in range(len(Xm)):
        D = Xm[i] - P.xm0
        viol = [(kd, P.schema.features[j], int(deg[kd][i, j])) for kd in SOFT_KINDS for j in P.M if deg[kd][i, j] > 0]
        out.append({"risk_after": float(T["p"][i]), "valid": bool(T["p"][i] < P.tau),
                    "changed": [P.schema.features[P.M[a]] for a in np.where(np.abs(D) > EPS)[0]],
                    "l0": int((np.abs(D) > EPS).sum()), "prox_mad_l1": float((np.abs(D) / P.mad).sum()),
                    "soft_violations": viol, "soft_checks": n_checks,
                    "soft_adherence": 1 - len(viol) / max(n_checks, 1),
                    "plaus_knn_z": float(T["plaus"][i]), "plaus_lof": float(lofv[i]),
                    "hard_violation": bool(hv[i]), "x": X[i].tolist()})
    return out
