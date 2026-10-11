"""Phase 4: MLP + XGBoost risk models, calibration, gate audit vs true risk, subgroup AUC, SHAP.

MLP = the existing CARE-LG RiskClassifier trained with src/classifier/blackbox_model.py
(train_blackbox_model, 20 epochs, unchanged). XGBoost = fixed hyper-parameters (no tuning on val, so
the val-AUC comparison is fair). Both are fit on the inner `fit` part of train; the recourse oracle is
the one with the higher validation AUC (config: report both). Thresholds 0.55 / 0.45 on raw probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from src.classifier.blackbox_model import train_blackbox_model
from src.data.loader import EHRDataset
from src.seeding import set_all_seeds


@dataclass
class Oracle:
    """Uniform predict interface over the two model families (P4)."""
    name: str
    model: object
    torch_module: Optional[torch.nn.Module] = None

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.atleast_2d(np.asarray(Xs, dtype=np.float32))
        if self.name == "mlp":
            with torch.no_grad():
                return self.torch_module(torch.from_numpy(Xs)).numpy().reshape(-1).astype(np.float64)
        return self.model.predict_proba(Xs)[:, 1].astype(np.float64)

    def torch_callable(self):
        """Callable on torch tensors (returns (n,1)); used by Growing Spheres / C-CHVAE (P12)."""
        def f(t):
            return torch.tensor(self.predict(t.detach().cpu().numpy()), dtype=torch.float32).unsqueeze(-1)
        return f


def fit_mlp(Xs: np.ndarray, y: np.ndarray, seed: int) -> Oracle:
    """Existing CARE-LG MLP (src/classifier/blackbox_model.py), seeded (P4)."""
    set_all_seeds(seed)
    dl = DataLoader(EHRDataset(Xs, y), batch_size=64, shuffle=True)
    m = train_blackbox_model(dl, input_dim=Xs.shape[1], epochs=20, device=torch.device("cpu"))
    m.eval()
    return Oracle("mlp", m, m)


def fit_xgb(Xs: np.ndarray, y: np.ndarray, seed: int) -> Oracle:
    """XGBoost with fixed hyper-parameters (P4)."""
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, n_jobs=1, eval_metric="logloss")
    m.fit(Xs, y)
    return Oracle("xgboost", m)


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table (P4)."""
    q = pd.qcut(p, q=n_bins, duplicates="drop", labels=False)
    return pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def evaluate_models(models: Dict[str, Oracle], split, bundle, hi: float, lo: float) -> dict:
    """AUC/Brier on val+test, calibration, subgroup AUC, gate vs true risk (P4)."""
    Xv, yv = split.Xtr[split.val], split.ytr[split.val]
    Xt, yt = split.Xte, split.yte
    out = {"models": {}}
    for name, m in models.items():
        pv, pt = m.predict(Xv), m.predict(Xt)
        sub = {}
        for g, s in bundle.subgroups.items():
            st = s.to_numpy()[split.te]
            sub[g] = {str(v): {"n": int((st == v).sum()), "auc": _auc(yt[st == v], pt[st == v])} for v in np.unique(st)}
        cal = calibration_bins(yt, pt)
        out["models"][name] = {"val_auc": _auc(yv, pv), "test_auc": _auc(yt, pt), "val_brier": float(brier_score_loss(yv, pv)),
                               "test_brier": float(brier_score_loss(yt, pt)), "subgroup_auc": sub,
                               "test_accuracy": float(np.mean((pt >= 0.5) == (yt == 1))), "val_accuracy": float(np.mean((pv >= 0.5) == (yv == 1))),
                               "calibration": cal.to_dict(orient="list"),
                               "n_high_test": int((pt > hi).sum()), "n_low_test": int((pt < lo).sum())}
        if "ascvd_10yr_risk_pct" in bundle.context:
            true = bundle.context["ascvd_10yr_risk_pct"].to_numpy()[split.te]
            gate = pt > hi
            out["models"][name]["gate_vs_true_risk"] = {
                "spearman_pred_vs_pce": float(spearmanr(pt, true).statistic),
                "pct_gate_high_with_pce_ge_7_5": float(100 * np.mean(true[gate] >= 7.5)) if gate.any() else float("nan"),
                "pct_gate_low_with_pce_lt_7_5": float(100 * np.mean(true[pt < lo] < 7.5)) if (pt < lo).any() else float("nan"),
                "median_pce_gate_high": float(np.median(true[gate])) if gate.any() else float("nan")}
        if "mort_cvd" in bundle.context:
            mc = bundle.context["mort_cvd"].to_numpy()[split.te]
            ma = bundle.context["mort_all"].to_numpy()[split.te]
            ok = ~np.isnan(mc)
            pce = bundle.context["ascvd_10yr_risk_pct"].to_numpy()[split.te]
            out["models"][name]["mortality_validation"] = {
                "n_linked_test": int(ok.sum()), "n_cvd_deaths": int(np.nansum(mc)), "n_all_deaths": int(np.nansum(ma)),
                "auc_pred_vs_cvd_death": _auc(mc[ok], pt[ok]), "auc_pce_vs_cvd_death": _auc(mc[ok], pce[ok]),
                "auc_pred_vs_all_death": _auc(ma[ok], pt[ok]),
                "cvd_death_rate_gate_high": float(np.nanmean(mc[ok & (pt > hi)])) if (ok & (pt > hi)).any() else float("nan"),
                "cvd_death_rate_gate_low": float(np.nanmean(mc[ok & (pt < lo)])) if (ok & (pt < lo)).any() else float("nan")}
    out["oracle"] = max(models, key=lambda n: out["models"][n]["val_auc"])
    return out


def global_shap(oracle: Oracle, split, n_explain: int = 200, seed: int = 0) -> pd.DataFrame:
    """Mean |SHAP| per feature on test patients (TreeSHAP for XGBoost, KernelSHAP for the MLP) (P4/P11)."""
    import shap
    rng = np.random.default_rng(seed)
    Xt = split.Xte[rng.choice(len(split.Xte), min(n_explain, len(split.Xte)), replace=False)]
    if oracle.name == "xgboost":
        sv = shap.TreeExplainer(oracle.model).shap_values(Xt)
    else:
        bg = shap.kmeans(split.Xtr, 20)
        sv = shap.KernelExplainer(oracle.predict, bg).shap_values(Xt, nsamples=100, silent=True)
    sv = np.asarray(sv).reshape(len(Xt), -1)
    return pd.DataFrame({"feature": split.features, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)


def patient_shap(oracle: Oracle, split, xs: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Per-patient SHAP (risk attribution) for explanations (P11)."""
    import shap
    if oracle.name == "xgboost":
        sv = shap.TreeExplainer(oracle.model).shap_values(np.atleast_2d(xs))
    else:
        bg = shap.kmeans(split.Xtr, 20)
        np.random.seed(seed)
        sv = shap.KernelExplainer(oracle.predict, bg).shap_values(np.atleast_2d(xs), nsamples=150, silent=True)
    return dict(zip(split.features, np.asarray(sv).reshape(-1).astype(float)))


def age_bracket_thresholds(ages_train: np.ndarray, train_risk: np.ndarray, n_q: int = 4, pct: float = 25):
    """Age-stratified 'low risk for your age' cut-offs (scripts/run_age_stratified_experiment.py logic) (P7)."""
    edges = np.percentile(ages_train, np.linspace(0, 100, n_q + 1))
    edges[0] -= 1e-6
    br = np.digitize(ages_train, edges[1:-1])
    thr = {b: float(np.percentile(train_risk[br == b], pct)) for b in range(n_q) if (br == b).any()}
    return edges, thr


def patient_threshold(age: float, edges: np.ndarray, thr: Dict[int, float]) -> float:
    """Bracket threshold for one age (clamped to the outer brackets) (P7)."""
    b = int(np.clip(np.digitize([age], edges[1:-1])[0], 0, len(edges) - 2))
    return thr.get(b, thr[min(thr, key=lambda k: abs(k - b))])
