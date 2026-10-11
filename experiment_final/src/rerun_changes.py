"""MD files/FULL_RERUN_CHANGES.md: every NHANES 1999-2023 number that changed between report_v3 (results/tables, 1,000-patient sample per seed,
300-patient cap for ontology-as-features) and the full rerun (results/tables, all high-risk test patients). Real numbers only: both sides
are read from CSVs. Bootstrap CI bounds go to a separate appendix. Wachter is excluded (removed from the report).

  .venv/bin/python -m experiment_final.src.rerun_changes
"""
from __future__ import annotations

import pandas as pd

from experiment_final.src.common import OLD_TABLES, RESULTS
from experiment_final.src.build_results import T4, T_V4, VIEW

OLD_T = OLD_TABLES
DS = ["nhanes_pooled", "nhanes_pooled__ontology_as_features"]
KEYS = {"model_metrics_summary.csv": ["dataset", "model"], "headline_tiered.csv": ["dataset"], "methods_comparison.csv": ["dataset", "method"],
        "ablations.csv": ["dataset", "arm"], "certification.csv": ["dataset"], "tier1_guarantees.csv": ["dataset"],
        "tier2_variants.csv": ["dataset", "method"], "fairness_tier_split.csv": ["dataset", "subgroup", "value"],
        "robustness_retraining.csv": ["dataset"], "reachable_set_head_to_head.csv": ["dataset"],
        "sensitivity_thresholds.csv": ["dataset", "high", "low", "age_stratified"], "sensitivity_k.csv": ["dataset", "k"],
        "sensitivity_ga_weights.csv": ["dataset", "param", "value"]}
TITLE = {"model_metrics_summary.csv": "Classifier (AUC, Brier)", "headline_tiered.csv": "Coverage, Tier 1/2/3, risk drop, guarantees",
         "methods_comparison.csv": "Baselines and CARE-LG (all methods)", "ablations.csv": "Ablation arms", "certification.csv": "Tier-3 certification",
         "tier1_guarantees.csv": "Tier-1 guarantees", "tier2_variants.csv": "Tier-2 variants", "fairness_tier_split.csv": "Fairness table",
         "robustness_retraining.csv": "Robustness to oracle retraining", "reachable_set_head_to_head.csv": "Reachable-set head-to-head",
         "sensitivity_thresholds.csv": "Threshold sensitivity", "sensitivity_k.csv": "k sensitivity", "sensitivity_ga_weights.csv": "Genetic-search weight sensitivity"}


def fmt(v) -> str:
    if pd.isna(v):
        return "n/a"
    if isinstance(v, (int, float)) and abs(v - round(v)) < 1e-9 and abs(v) >= 100:
        return f"{int(round(v)):,}"
    return f"{v:.4g}" if abs(v) < 1 else f"{v:.2f}"


def diff_one(name: str, k: list) -> pd.DataFrame:
    if not (OLD_T / name).exists() or not (T4 / name).exists():
        return pd.DataFrame()
    o, n = pd.read_csv(OLD_T / name), pd.read_csv(T4 / name)
    for d in (o, n):
        if "method" in d:
            d.drop(d[d.method == "wachter"].index, inplace=True)
    o, n = o[o.dataset.isin(DS)], n[n.dataset.isin(DS)]
    m = o.merge(n, on=k, how="outer", suffixes=("_old", "_new"))
    num = [c for c in o.columns if c not in k and c in n.columns and pd.api.types.is_numeric_dtype(o[c])]
    rows = []
    for _, r in m.iterrows():
        for c in num:
            a, b = r.get(f"{c}_old"), r.get(f"{c}_new")
            if pd.isna(a) and pd.isna(b):
                continue
            if not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) < 1e-9:
                continue
            rows.append({"table": name, **{kk: r[kk] for kk in k}, "metric": c, "old": a, "new": b, "ci": "_ci_" in c})
    return pd.DataFrame(rows)


def classifier_rows() -> pd.DataFrame:
    """Test accuracy (old: accuracy_backfill_v3runs.csv; new: per-seed model_metrics) and oracle AUC, mean over 5 seeds."""
    acc = pd.read_csv(T_V4 / "accuracy_backfill_v3runs.csv"); new = pd.read_csv(T4 / "model_metrics_per_seed.csv")
    rows = []
    for d in DS:
        for is_o in (True, False):
            og = acc[(acc.dataset == d) & (acc.is_oracle == is_o)]; ng = new[(new.dataset == d) & (new.is_oracle == is_o)]
            if len(og) and len(ng):
                tag = "oracle" if is_o else "non-oracle model"
                rows.append({"table": "classifier", "dataset": d, "model": tag, "metric": "test_accuracy (mean of 5 seeds)", "old": og.test_accuracy.mean(), "new": ng.test_accuracy.mean(), "ci": False})
                rows.append({"table": "classifier", "dataset": d, "model": tag, "metric": "test_auc (mean of 5 seeds)", "old": og.test_auc.mean(), "new": ng.test_auc.mean(), "ci": False})
    return pd.DataFrame(rows)


def learning_curve_row() -> pd.DataFrame:
    o = pd.read_csv(OLD_T / "pooled_learning_curve.csv"); n = pd.read_csv(T4 / "pooled_learning_curve.csv")
    full = int(o.n_patients.max())
    a, b = o[o.n_patients == full].tier1_coverage_pct, n[n.n_patients == full].tier1_coverage_pct
    return pd.DataFrame([{"table": "pooled_learning_curve", "dataset": "nhanes_pooled", "n_patients": full, "metric": "tier1_coverage_pct (full-data point, mean of 5 seeds)",
                          "old": a.mean(), "new": b.mean(), "ci": False},
                         {"table": "pooled_learning_curve", "dataset": "nhanes_pooled", "n_patients": full, "metric": "tier1_coverage_pct (full-data point, SD over seeds)",
                          "old": a.std(), "new": b.std(), "ci": False}])


def md_table(df: pd.DataFrame, keycols: list) -> str:
    lines = ["| " + " | ".join(keycols + ["metric", "old (report_v3)", "new (full rerun)", "change"]) + " |", "|" + "---|" * (len(keycols) + 4)]
    for r in df.itertuples():
        ch = "" if pd.isna(r.old) or pd.isna(r.new) else f"{r.new - r.old:+,.4f}".rstrip("0").rstrip(".")
        lines.append("| " + " | ".join([str(getattr(r, c)) for c in keycols] + [r.metric, fmt(r.old), fmt(r.new), ch]) + " |")
    return "\n".join(lines)


def flags() -> list:
    out = []
    h = pd.read_csv(T4 / "headline_tiered.csv").set_index("dataset")
    for d in DS:
        if d in h.index:
            v, c = int(h.loc[d, "tier1_real_violations"]), int(h.loc[d, "contraindicated_cards"])
            out.append((d, "Tier-1 real-value violations (headline)", v)); out.append((d, "contraindicated cards (headline)", c))
    g = pd.read_csv(T4 / "tier1_guarantees.csv")
    for r in g[g.dataset.isin(DS)].itertuples():
        out.append((r.dataset, f"Tier-1 paths with a real-value violation at any node / endpoint (of {int(r.tier1_paths):,} paths)", f"{int(r.real_violation_any_node)} / {int(r.real_violation_endpoint)}"))
    a = pd.read_csv(T4 / "ablations.csv"); a = a[a.dataset.isin(DS) & (a.status == "run")]
    for r in a.itertuples():
        out.append((f"{r.dataset} / arm {r.arm}", "contraindicated recommendations / Tier-1 paths with a real violation", f"{int(r.contraindicated_recs_total)} / {int(r.real_violation_paths_total)}"))
    return out


def build() -> str:
    parts, allrows = [], []
    for name, k in KEYS.items():
        d = diff_one(name, k)
        if len(d):
            allrows.append(d)
    cl = classifier_rows(); lc = learning_curve_row()
    allrows += [cl, lc]
    allr = pd.concat(allrows, ignore_index=True)
    allr.to_csv(T4 / "full_rerun_changes.csv", index=False)
    main_r = allr[~allr.ci]
    hdr = ("# Full NHANES 1999-2023 rerun: what changed from report_v3\n\n"
           "Old = `archive/experiment_final_prev/v3/tables/` (report_v3: 1,000 randomly sampled high-risk test patients per seed; ontology-as-features capped at 300). "
           "New = `results/tables/` from `results/run_full_nhanes/` (all high-risk test patients per seed, no cap, 5 seeds, all ablation arms). "
           "Wachter excluded (removed from the report). Kaggle CVD-8k, NHANES 2017-2020 and UCI Heart were not rerun and are unchanged. "
           "Only cells that differ are listed; machine-readable copy: `results/tables/full_rerun_changes.csv`.\n\n")
    n_pat = []
    for sd in range(5):
        p = pd.read_csv(VIEW / "nhanes_pooled" / f"seed{sd}" / "patients.csv"); n_pat.append(len(p))
    hdr += f"High-risk test patients evaluated per seed (new): {', '.join(f'{x:,}' for x in n_pat)} (total {sum(n_pat):,}; report_v3: 5,000).\n\n"
    fl = flags()
    nonzero = [f for f in fl if str(f[2]) not in ("0", "0 / 0")]
    sec = "## Flags: Tier-1 violations and contraindicated cards\n\n"
    main_bad = [f for f in nonzero if ("arm " not in f[0] or f[0].endswith(("arm main", "arm tier1_only", "arm soft_as_hard", "arm no_escalation"))) ]
    sec += ("**Non-zero values on arms that must be zero (main method and ordinary variants): " + ("NONE." if not main_bad else "SEE BELOW, STOP AND REVIEW.") + "**\n\n")
    sec += "| scope | quantity | value |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in fl) + "\n\n"
    sec += ("Non-zero values on the `no_sra`, `no_inheritance`, `no_ontology`, `ontology_as_features` ablation arms are the intended cost of removing the component, not failures of the full method.\n\n")
    body = hdr + sec
    for name, k in KEYS.items():
        d = main_r[main_r.table == name]
        if len(d):
            body += f"## {TITLE[name]} (`{name}`)\n\n{md_table(d, k)}\n\n"
    for tbl, title, kc in (("classifier", "Classifier: accuracy and AUC", ["dataset", "model"]), ("pooled_learning_curve", "Learning curve: full-data point", ["dataset", "n_patients"])):
        d = main_r[main_r.table == tbl]
        if len(d):
            body += f"## {title}\n\n{md_table(d, kc)}\n\n"
    ci = allr[allr.ci]
    body += (f"## Appendix: bootstrap confidence-interval bounds that changed ({len(ci)} cells)\n\nListed in `results/tables/full_rerun_changes.csv` (column `ci` = True); "
             "CI bounds change with the data and with the shared bootstrap RNG stream.\n")
    out = RESULTS.parent / "MD files" / "FULL_RERUN_CHANGES.md"
    out.write_text(body)
    return str(out)


if __name__ == "__main__":
    print(build())
