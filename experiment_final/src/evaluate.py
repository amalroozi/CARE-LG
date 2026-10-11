"""Phase 14: metrics, stats (bootstrap CI, Wilcoxon+Holm), sensitivity, robustness to retraining, fairness tier-split, failure cases.

`aggregate(run_dir)` reads every <dataset>[__variant]/seed<k>/ folder of one run and writes CSV tables to
results/tables/ (never overwriting: new_path). Conventions: per-seed means are summarised as
mean +/- SD over the 5 seeds; 95% CIs are percentile bootstraps over patients pooled across seeds
(2000 resamples); paired tests are Wilcoxon signed-rank on patient-level differences vs the tiered
method with Holm correction within each (dataset, metric) family.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon
from statsmodels.stats.multitest import multipletests

from experiment_final.src.common import TABLES, load_config, new_path

OURS = "care_lg_tiered"
RNG = np.random.default_rng(0)


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    parts = []
    for d in sorted(run_dir.glob("*/seed*")):
        f = d / name
        if f.exists() and f.stat().st_size > 1:
            try:
                parts.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _jsonl(run_dir: Path, name: str, variant: bool = False) -> list:
    out = []
    for d in sorted(run_dir.glob("*/seed*")):
        if ("__" in d.parent.name) != variant:
            continue
        f = d / name
        if f.exists():
            out += [json.loads(l) for l in open(f)]
    return out


def boot_ci(x, n: int = 2000):
    """Percentile bootstrap 95% CI of the mean (P14)."""
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = RNG.choice(x, (n, len(x))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def seed_summary(df: pd.DataFrame, keys, metrics, scale: Dict[str, float] = None) -> pd.DataFrame:
    """mean/SD over seeds of per-seed means + pooled bootstrap CI (P14)."""
    scale = scale or {}
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,)))
        r["n_patient_rows"] = len(g)
        for m in metrics:
            if m not in g:
                continue
            s = scale.get(m, 1.0)
            v = pd.to_numeric(g[m], errors="coerce").astype(float)
            per = v.groupby(g["seed"]).mean() * s
            lo, hi = boot_ci(v * s)
            r.update({f"{m}_mean": per.mean(), f"{m}_sd": per.std(), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str, store: dict) -> None:
    p = new_path(TABLES / name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    store[name] = {"path": str(p.relative_to(TABLES.parents[1])), "df": df}


# ------------------------------------------------------------------ tables

def model_table(run_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(run_dir.glob("*/seed*")):
        mm = json.loads((d / "model_metrics.json").read_text())
        su = json.loads((d / "setup.json").read_text())
        for name, m in mm["models"].items():
            g = m.get("gate_vs_true_risk", {})
            rows.append({"dataset": d.parent.name, "seed": int(d.name[4:]), "model": name, "is_oracle": name == mm["oracle"],
                         "val_auc": m["val_auc"], "test_auc": m["test_auc"], "val_brier": m["val_brier"], "test_brier": m["test_brier"],
                         "test_accuracy": m.get("test_accuracy", np.nan), "val_accuracy": m.get("val_accuracy", np.nan),
                         "n_high_test": m["n_high_test"], "linear_test_auc": su["linear_test_auc"], **g})
    return pd.DataFrame(rows)


def subgroup_auc_table(run_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(run_dir.glob("*/seed*")):
        mm = json.loads((d / "model_metrics.json").read_text())
        m = mm["models"][mm["oracle"]]
        for g, vals in m["subgroup_auc"].items():
            for v, s in vals.items():
                rows.append({"dataset": d.parent.name, "seed": int(d.name[4:]), "oracle": mm["oracle"], "subgroup": g, "value": v, **s})
    df = pd.DataFrame(rows)
    return df.groupby(["dataset", "subgroup", "value"]).agg(n_mean=("n", "mean"), auc_mean=("auc", "mean"), auc_sd=("auc", "std")).reset_index()


def headline(arms: pd.DataFrame) -> pd.DataFrame:
    """Main-arm coverage, tier split and guarantees per dataset (P14)."""
    m = arms[arms.arm == "main"].copy()
    for t in (1, 2, 3):
        m[f"tier{t}"] = (m.tier == t).astype(float)
    m["covered_tier12"] = (m.tier <= 2).astype(float)
    m["card_covered_f"] = m.card_covered.astype(float)
    m["tier1_real_violation"] = (m.tier == 1) & (m.real_violation_any_node.fillna(False).astype(bool) | m.real_violation_endpoint.fillna(False).astype(bool))
    m["contra_card"] = (m.contraindicated_recs > 0).astype(float)
    # escalation metrics are defined over cards with >= 1 recommendation (uncovered cards are counted in card_covered)
    m["esc_ok"] = m.escalation_appropriate.map({True: 1.0, False: 0.0}).where(m.n_recs > 0)
    m["over"] = m.overtreatment.map({True: 1.0, False: 0.0}).where(m.n_recs > 0)
    m["global_low"] = m.reaches_global_low.astype(float)
    s = seed_summary(m, ["dataset"], ["tier1", "tier2", "tier3", "covered_tier12", "card_covered_f", "esc_ok", "over",
                                      "actionability", "risk_drop", "global_low", "n_recs"],
                     {k: 100 for k in ["tier1", "tier2", "tier3", "covered_tier12", "card_covered_f", "esc_ok", "over", "global_low"]})
    tot = m.groupby("dataset").agg(n_high_risk_total=("tier", "size"), tier1_real_violations=("tier1_real_violation", "sum"),
                                   contraindicated_cards=("contra_card", "sum"), contraindicated_recs=("contraindicated_recs", "sum")).reset_index()
    return s.merge(tot, on="dataset")


def methods_table(methods: pd.DataFrame) -> pd.DataFrame:
    """All methods incl. baselines: validity, violations, proximity, plausibility, ... (P12/P14)."""
    m = methods[methods.variant.fillna("") == ""].copy()
    m["valid_f"] = m.valid.astype(float)
    m["valid_and_admissible"] = (m.valid.astype(bool) & ~m.hard_violation.fillna(True).astype(bool)).astype(float)
    m["hard_violation_f"] = m.hard_violation.map({True: 1.0, False: 0.0})
    for c in ["prox_mad_l1", "l0", "plaus_lof", "plaus_knn_z", "soft_violations", "diversity", "actionability", "hard_violation_f"]:
        m.loc[~m.valid.astype(bool), c] = np.nan
    return seed_summary(m, ["dataset", "method"], ["valid_f", "valid_and_admissible", "hard_violation_f", "prox_mad_l1", "l0", "plaus_lof",
                                                   "plaus_knn_z", "soft_violations", "diversity", "actionability", "latency_s"],
                        {"valid_f": 100, "valid_and_admissible": 100, "hard_violation_f": 100})


def paired_tests(methods: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank vs the tiered method per (dataset, metric), Holm-corrected (P14)."""
    m = methods[methods.variant.fillna("") == ""].copy()
    m["valid_and_admissible"] = (m.valid.astype(bool) & ~m.hard_violation.fillna(True).astype(bool)).astype(float)
    rows = []
    for ds, g in m.groupby("dataset"):
        piv = {met: g.pivot_table(index=["seed", "test_pos"], columns="method", values=met, aggfunc="first")
               for met in ["valid_and_admissible", "prox_mad_l1", "plaus_lof", "soft_violations"]}
        for met, P in piv.items():
            fam = []
            for meth in P.columns:
                if meth == OURS:
                    continue
                d = P[[OURS, meth]].dropna()
                if met != "valid_and_admissible":
                    ok = g.pivot_table(index=["seed", "test_pos"], columns="method", values="valid_and_admissible", aggfunc="first")
                    keep = ok.index[(ok[OURS] == 1) & (ok[meth] == 1)]
                    d = d.loc[d.index.intersection(keep)]
                diff = (d[OURS] - d[meth]).to_numpy()
                p = 1.0 if len(diff) == 0 or np.allclose(diff, 0) else float(wilcoxon(diff, zero_method="wilcox").pvalue)
                fam.append({"dataset": ds, "metric": met, "method": meth, "n_pairs": len(diff),
                            "median_diff_ours_minus_method": float(np.median(diff)) if len(diff) else np.nan,
                            "mean_diff_ours_minus_method": float(np.mean(diff)) if len(diff) else np.nan, "p_raw": p})
            if fam:
                adj = multipletests([f["p_raw"] for f in fam], method="holm")[1]
                for f, a in zip(fam, adj):
                    f["p_holm"] = float(a)
                rows += fam
    return pd.DataFrame(rows)


def ablation_table(arms: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Arms incl. ontology-as-features variant; 'not applicable' rows on no-concept datasets (P13)."""
    a = arms.copy()
    a.loc[a.variant.fillna("") == "ontology_as_features", "arm"] = "ontology_as_features"
    a["dataset"] = a["dataset"].astype(str)
    a["covered"] = (a.tier <= 2).astype(float)
    a["tier1_f"] = (a.tier == 1).astype(float)
    a["contra_card"] = (a.contraindicated_recs > 0).astype(float)
    a["esc_ok"] = a.escalation_appropriate.map({True: 1.0, False: 0.0}).where(a.n_recs > 0)
    a["over"] = a.overtreatment.map({True: 1.0, False: 0.0}).where(a.n_recs > 0)
    a["real_viol"] = ((a.tier == 1) & (a.real_violation_endpoint.fillna(False).astype(bool) | a.real_violation_any_node.fillna(False).astype(bool))).astype(float)
    s = seed_summary(a, ["dataset", "arm"], ["covered", "tier1_f", "contra_card", "esc_ok", "over", "actionability",
                                             "soft_violations_true_rules", "real_viol", "risk_drop"],
                     {k: 100 for k in ["covered", "tier1_f", "contra_card", "esc_ok", "over", "real_viol"]})
    tot = a.groupby(["dataset", "arm"]).agg(contraindicated_recs_total=("contraindicated_recs", "sum"),
                                           real_violation_paths_total=("real_viol", "sum")).reset_index()
    s = s.merge(tot, on=["dataset", "arm"])
    s["status"] = "run"
    na = []
    for ds in sorted(set(a.dataset)):
        has = bool(_HAS_ONTO.get(ds, False))
        for ab in cfg["ablations"]:
            if ab in ("no_ontology", "no_inheritance", "ontology_as_features") and not has:
                na.append({"dataset": ds, "arm": ab, "status": "not applicable (no concepts in this dataset)"})
    return pd.concat([s, pd.DataFrame(na)], ignore_index=True)


_HAS_ONTO: Dict[str, bool] = {}


def certification_table(patients: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 failures: certified infeasible, horizon-only, 2-hop, ceiling categories; SRA identity check (P8)."""
    p = patients[patients.variant.fillna("") == ""].copy()
    p["identity_ok"] = (p.tier1_success.astype(bool) == (p.tier1_one_hop_targets > 0))
    f = p[~p.tier1_success.astype(bool)]
    rows = []
    for ds, g in f.groupby("dataset"):
        pg = p[p.dataset == ds]
        rows.append({"dataset": ds, "n_high_risk": len(pg), "n_tier1_fail": len(g),
                     "pct_certified_infeasible_real": 100 * g.certified_infeasible_real.astype(bool).mean(),
                     "pct_blocked_only_by_age_horizon": 100 * g.blocked_only_by_horizon.astype(bool).mean(),
                     "pct_two_hop_chain_hopwise": 100 * g.two_hop_chain_exists.astype(float).mean(),
                     "pct_age_ceiling": 100 * (g.ceiling_category == "age_ceiling").mean(),
                     "pct_data_scarcity": 100 * (g.ceiling_category == "data_scarcity").mean(),
                     "n_tier3": int((g.tier == 3).sum()),
                     "pct_tier3_model_ceiling": 100 * (g[g.tier == 3].ceiling_reachable == False).mean() if (g.tier == 3).any() else np.nan,  # noqa: E712
                     "sra_identity_violations": int((~pg.identity_ok).sum()),
                     "tier1_certificates_failed": int((pg.cert_ok == False).sum())})  # noqa: E712
    return pd.DataFrame(rows)


def blocking_table(certs: list) -> pd.DataFrame:
    rows = []
    for c in certs:
        for rule, fr in (c.get("blocking_frequency") or {}).items():
            rows.append({"dataset": c["dataset"], "rule": rule, "share_of_targets_blocked": fr})
        cm = c.get("closest_miss") or {}
        for kd, f in cm.get("blocking", []):
            rows.append({"dataset": c["dataset"], "rule": f"closest_miss:{kd}:{f}", "share_of_targets_blocked": np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    a = df[~df.rule.str.startswith("closest_miss")].groupby(["dataset", "rule"]).share_of_targets_blocked.mean().rename("mean_share_of_targets_blocked")
    b = df[df.rule.str.startswith("closest_miss")].groupby(["dataset", "rule"]).size().rename("n_closest_miss")
    out = pd.concat([a, b], axis=1).reset_index()
    out["n_failures"] = out.dataset.map(pd.Series([c["dataset"] for c in certs]).value_counts())
    return out.sort_values(["dataset", "mean_share_of_targets_blocked"], ascending=[True, False])


def fairness_table(arms: pd.DataFrame) -> pd.DataFrame:
    """Tier split by subgroup with chi-square test of independence (patients pooled over seeds) (P14, C4)."""
    m = arms[(arms.arm == "main") & (arms.variant.fillna("") == "")]
    rows = []
    for ds, g in m.groupby("dataset"):
        for col in [c for c in g.columns if c.startswith("sub_")]:
            if g[col].isna().all():
                continue
            ct = pd.crosstab(g[col], g.tier)
            p = float(chi2_contingency(ct)[1]) if ct.shape[0] > 1 and ct.shape[1] > 1 else np.nan
            for v, r in ct.iterrows():
                n = int(r.sum())
                rows.append({"dataset": ds, "subgroup": col[4:], "value": v, "n": n,
                             **{f"tier{t}_pct": 100 * r.get(t, 0) / n for t in (1, 2, 3)}, "chi2_p": p})
    return pd.DataFrame(rows)


def reachable_table(methods: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head on infeasibility detection: LR reachable-set verifier vs CARE-LG tiers (P12)."""
    r = methods[(methods.method == "reachable_set_lr") & (methods.variant.fillna("") == "")][["dataset", "seed", "test_pos", "rs_precluded", "rs_size", "valid"]]
    p = patients[patients.variant.fillna("") == ""][["dataset", "seed", "test_pos", "tier", "tier1_success"]]
    j = r.merge(p, on=["dataset", "seed", "test_pos"])
    rows = []
    for ds, g in j.groupby("dataset"):
        pre = g.rs_precluded.astype(bool)
        rows.append({"dataset": ds, "n": len(g), "pct_lr_precluded": 100 * pre.mean(),
                     "pct_carelg_tier1_infeasible": 100 * (~g.tier1_success.astype(bool)).mean(),
                     "pct_carelg_tier3": 100 * (g.tier == 3).mean(),
                     "pct_lr_precluded_but_carelg_tier1": 100 * (pre & (g.tier == 1)).mean(),
                     "pct_lr_precluded_but_carelg_tier2": 100 * (pre & (g.tier == 2)).mean(),
                     "pct_lr_feasible_but_carelg_tier3": 100 * (~pre & (g.tier == 3)).mean(),
                     "pct_lr_cf_valid_under_oracle": 100 * g.valid.astype(bool).mean(),
                     "median_reachable_set_size": float(g.rs_size.median())})
    return pd.DataFrame(rows)


def aggregate(run_dir: Path) -> dict:
    """Build every P13/P14 table from one run directory (P14)."""
    cfg = load_config()
    store: dict = {}
    for d in run_dir.glob("*/seed*"):
        su = json.loads((d / "setup.json").read_text())
        if not su["variant"]:
            _HAS_ONTO[su["dataset"]] = su["has_ontology"]
    patients, arms, methods = _read(run_dir, "patients.csv"), _read(run_dir, "arms.csv"), _read(run_dir, "methods.csv")
    for df in (patients, arms, methods):
        df["variant"] = df["variant"].fillna("") if "variant" in df else ""
    mt = model_table(run_dir)
    save(mt, "model_metrics_per_seed.csv", store)
    save(mt.groupby(["dataset", "model"]).agg(val_auc=("val_auc", "mean"), test_auc=("test_auc", "mean"), test_auc_sd=("test_auc", "std"),
                                              test_brier=("test_brier", "mean"), times_oracle=("is_oracle", "sum"),
                                              linear_test_auc=("linear_test_auc", "mean")).reset_index(), "model_metrics_summary.csv", store)
    save(subgroup_auc_table(run_dir), "model_subgroup_auc.csv", store)
    main_arms = arms[arms.variant == ""]
    save(headline(main_arms), "headline_tiered.csv", store)
    save(methods_table(methods), "methods_comparison.csv", store)
    save(paired_tests(methods), "methods_wilcoxon_holm.csv", store)
    save(ablation_table(arms, cfg), "ablations.csv", store)
    save(certification_table(patients), "certification.csv", store)
    certs = _jsonl(run_dir, "certify.jsonl")
    save(blocking_table(certs), "blocking_rules.csv", store)
    save(fairness_table(arms), "fairness_tier_split.csv", store)
    save(reachable_table(methods, patients), "reachable_set_head_to_head.csv", store)
    rob = []
    for d in sorted(run_dir.glob("*/seed*")):
        f = d / "robustness.csv"
        if f.exists() and f.stat().st_size > 1:
            r = pd.read_csv(f)
            rob.append({"dataset": d.parent.name, "seed": int(d.name[4:]), "n_targets": len(r),
                        **{f"still_valid_pct_tier{t}": 100 * r[r.tier == t].still_valid.mean() if (r.tier == t).any() else np.nan for t in (1, 2)}})
    rob = pd.DataFrame(rob)
    save(rob.groupby("dataset").agg(**{c: (c, "mean") for c in rob.columns if c.startswith("still")}, n_seeds=("seed", "size")).reset_index(),
         "robustness_retraining.csv", store)
    for name in ("sensitivity_thresholds.csv", "sensitivity_ga_weights.csv", "sensitivity_k.csv"):
        parts = []
        for d in sorted(run_dir.glob("*/seed*")):
            if (d / name).exists():
                parts.append(pd.read_csv(d / name).assign(dataset=d.parent.name, seed=int(d.name[4:])))
        if parts:
            df = pd.concat(parts)
            if name == "sensitivity_ga_weights.csv":
                df = df.groupby(["dataset", "param", "value"]).agg(valid_pct=("valid", lambda v: 100 * v.mean()), prox_mad_l1=("prox_mad_l1", "mean"),
                                                                  soft_adherence=("soft_adherence", "mean"), plaus_lof=("plaus_lof", "mean"), l0=("l0", "mean")).reset_index()
            elif name == "sensitivity_k.csv":
                df = df.groupby(["dataset", "k"]).agg(success_pct=("success", lambda v: 100 * v.mean()), mean_hops=("n_hops", "mean"),
                                                      mean_cost=("cost", "mean"), mean_risk_drop=("risk_drop", "mean")).reset_index()
            save(df, name, store)
    # tier-2 variants & card-level escalation detail
    t2 = methods[(methods.variant == "") & methods.method.str.startswith("tier2")].copy()
    t2["valid_f"] = t2.valid.astype(float)
    save(seed_summary(t2, ["dataset", "method"], ["valid_f", "prox_mad_l1", "l0", "plaus_lof", "soft_violations", "latency_s"], {"valid_f": 100}),
         "tier2_variants.csv", store)
    p = patients[patients.variant == ""]
    save(p.groupby("dataset").agg(n=("tier", "size"), tier1_paths=("tier1_success", "sum"),
                                  real_violation_any_node=("real_violation_any_node", lambda v: int(v.fillna(False).astype(bool).sum())),
                                  real_violation_endpoint=("real_violation_endpoint", lambda v: int(v.fillna(False).astype(bool).sum())),
                                  certificates_ok=("cert_ok", lambda v: int((v == True).sum())),  # noqa: E712
                                  mean_hops=("n_hops", "mean")).reset_index(), "tier1_guarantees.csv", store)
    tops = arms[(arms.arm.isin(["main", "no_escalation"])) & (arms.variant == "") & (arms.top_class.fillna("") != "")]
    save(tops.groupby(["dataset", "arm", "top_step", "top_class"]).size().rename("n").reset_index().sort_values(["dataset", "arm", "n"], ascending=[True, True, False]),
         "top_recommendations.csv", store)
    cohort = []
    for d in sorted(run_dir.glob("*/seed0")):
        su = json.loads((d / "setup.json").read_text())
        cohort.append({"dataset": d.parent.name, "label": su["label_definition"], "flow": json.dumps(su["cohort_flow"]),
                       "n_train": su["n_train"], "n_test": su["n_test"], "pct_edges_pruned": su["pct_edges_pruned"], "k": su["k"]})
    save(pd.DataFrame(cohort), "cohort_flow.csv", store)
    store["_frames"] = {"patients": patients, "arms": arms, "methods": methods, "certs": certs}
    return store
