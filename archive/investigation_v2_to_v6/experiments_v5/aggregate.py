"""
Aggregates experiments_v5/results/{dataset}_per_patient.csv into table1_{dataset}.csv
(mean +/- std over seeds), plus a wide Phase-4 comparison table that pulls in
experiments_v3's DiCE/FACE/REVISE numbers and experiments_v4's DSR-full numbers
(both CLEARLY LABELED as reused, not rerun this session -- see CHANGES.md), and
paired Wilcoxon tests (EGD-riemannian vs EGD-euclidean; EGD vs old, wherever both
have enough paired successes to test).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = REPO_ROOT / "experiments_v5" / "results"
V3_RESULTS = REPO_ROOT / "experiments_v3" / "results"
V4_RESULTS = REPO_ROOT / "experiments_v4" / "results"

CELLS = ["old_latent_pruning", "egd_riemannian", "egd_euclidean"]


def build_table1(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{dataset}_per_patient.csv")
    rows = []
    for cell in CELLS:
        sub = df[df["cell"] == cell]
        per_seed_succ = sub.groupby("seed")["success"].mean()
        ok = sub[sub["success"] == True]
        agg = {}
        for metric in ["decoded_cvr_endpoint", "kde", "effort", "latency_sec"]:
            per_seed = ok.groupby("seed")[metric].mean() if len(ok) else pd.Series(dtype=float)
            agg[f"{metric}_mean"] = per_seed.mean() if len(per_seed) else np.nan
            agg[f"{metric}_std"] = per_seed.std() if len(per_seed) else np.nan
        rows.append({
            "cell": cell,
            "success_rate_mean": per_seed_succ.mean() * 100, "success_rate_std": per_seed_succ.std() * 100,
            "decoded_cvr_mean": agg["decoded_cvr_endpoint_mean"] * 100 if pd.notna(agg["decoded_cvr_endpoint_mean"]) else np.nan,
            "decoded_cvr_std": agg["decoded_cvr_endpoint_std"] * 100 if pd.notna(agg["decoded_cvr_endpoint_std"]) else np.nan,
            "kde_mean": agg["kde_mean"], "kde_std": agg["kde_std"],
            "effort_mean": agg["effort_mean"], "effort_std": agg["effort_std"],
            "latency_mean": agg["latency_sec_mean"], "latency_std": agg["latency_sec_std"],
            "n_seeds": sub["seed"].nunique(),
        })
    return pd.DataFrame(rows)


def paired_test(dataset, label_a, label_b, metric, sub_a, sub_b):
    a = sub_a[sub_a["success"] == True][["seed", "patient_idx", metric]].rename(columns={metric: "a"})
    b = sub_b[sub_b["success"] == True][["seed", "patient_idx", metric]].rename(columns={metric: "b"})
    m = a.merge(b, on=["seed", "patient_idx"]).dropna()
    m["a"] = m["a"].astype(float)
    m["b"] = m["b"].astype(float)
    if len(m) < 2 or (m["a"] == m["b"]).all():
        return {"dataset": dataset, "cell_a": label_a, "cell_b": label_b, "metric": metric,
               "n_pairs": len(m), "statistic": np.nan, "pvalue": np.nan, "median_diff": np.nan}
    stat, p = wilcoxon(m["a"], m["b"])
    return {"dataset": dataset, "cell_a": label_a, "cell_b": label_b, "metric": metric,
           "n_pairs": len(m), "statistic": stat, "pvalue": p, "median_diff": (m["a"] - m["b"]).median()}


def build_wide_comparison(dataset: str) -> pd.DataFrame:
    """Phase 4: EGD (this session) + old (this session) + v4's DSR-full best point + v3's DiCE/FACE/REVISE."""
    t1 = build_table1(dataset).set_index("cell")
    rows = []

    def fmt(m, s, digits=1, pct=False):
        if pd.isna(m):
            return "--"
        suf = "%" if pct else ""
        return f"{m:.{digits}f}{suf}" if pd.isna(s) else f"{m:.{digits}f}±{s:.{digits}f}{suf}"

    for cell, label in [("egd_riemannian", "EGD (this session)"), ("egd_euclidean", "EGD-euclidean (this session)"),
                       ("old_latent_pruning", "Old latent-pruning (this session, controlled rerun)")]:
        if cell not in t1.index:
            continue
        r = t1.loc[cell]
        rows.append({"Method": label, "Source": "v5 (this session)", "Success (%)": fmt(r["success_rate_mean"], r["success_rate_std"]),
                    "Decoded CVR (%)": fmt(r["decoded_cvr_mean"], r["decoded_cvr_std"]),
                    "KDE": fmt(r["kde_mean"], r["kde_std"], 3), "Effort": fmt(r["effort_mean"], r["effort_std"], 2),
                    "Latency (s)": fmt(r["latency_mean"], r["latency_std"], 4)})

    # v4 DSR-full's BEST operating point (from its own tau sweep), reused verbatim, labeled
    sweep_path = V4_RESULTS / f"tau_sweep_{dataset}.csv"
    if sweep_path.exists():
        sw = pd.read_csv(sweep_path)
        nz = sw[sw["success_rate_mean"] > 0].sort_values("success_rate_mean", ascending=False)
        if len(nz):
            b = nz.iloc[0]
            rows.append({"Method": f"DSR-full (v4, best q={b['tau_quantile']:.2f})", "Source": "REUSED from experiments_v4, NOT rerun",
                        "Success (%)": f"{b['success_rate_mean']:.1f}±{b['success_rate_std']:.1f}",
                        "Decoded CVR (%)": f"{b['decoded_cvr_mean']:.1f}" if pd.notna(b['decoded_cvr_mean']) else "--",
                        "KDE": "--", "Effort": "--", "Latency (s)": "--"})
        else:
            rows.append({"Method": "DSR-full (v4)", "Source": "REUSED from experiments_v4, NOT rerun",
                        "Success (%)": "0.0 (no viable tau)", "Decoded CVR (%)": "--", "KDE": "--", "Effort": "--", "Latency (s)": "--"})

    # v3's DiCE / FACE / REVISE, reused verbatim, labeled
    v3_path = V3_RESULTS / f"table1_final_{dataset}.csv"
    if v3_path.exists():
        v3 = pd.read_csv(v3_path)
        for method in ["DiCE", "FACE", "REVISE (genuinely implemented, this session)"]:
            m = v3[v3["Method"] == method]
            if len(m):
                r = m.iloc[0]
                rows.append({"Method": method.replace(" (genuinely implemented, this session)", " (v3)"),
                            "Source": "REUSED from experiments_v3, NOT rerun",
                            "Success (%)": r["Success (%)"], "Decoded CVR (%)": r["Decoded CVR (%)"],
                            "KDE": r["KDE"], "Effort": r["Effort"], "Latency (s)": r["Latency (s)"]})

    return pd.DataFrame(rows)


def main():
    all_table1, all_paired, all_wide = [], [], []
    for dataset in ["uci", "nhanes_real"]:
        path = RESULTS_DIR / f"{dataset}_per_patient.csv"
        if not path.exists():
            print(f"[aggregate] skipping {dataset}: not found")
            continue
        t1 = build_table1(dataset)
        t1.insert(0, "dataset", dataset)
        all_table1.append(t1)
        t1.to_csv(RESULTS_DIR / f"table1_{dataset}.csv", index=False)
        print(f"\n=== Table 1 ({dataset}) ===")
        print(t1.round(3).to_string(index=False))

        df = pd.read_csv(path)
        egd_r = df[df["cell"] == "egd_riemannian"]
        egd_e = df[df["cell"] == "egd_euclidean"]
        old = df[df["cell"] == "old_latent_pruning"]
        for metric in ["kde", "effort", "decoded_cvr_endpoint"]:
            all_paired.append(paired_test(dataset, "egd_riemannian", "egd_euclidean", metric, egd_r, egd_e))
        for metric in ["kde", "effort"]:
            all_paired.append(paired_test(dataset, "egd_riemannian", "old_latent_pruning", metric, egd_r, old))

        wide = build_wide_comparison(dataset)
        wide.insert(0, "dataset", dataset)
        all_wide.append(wide)
        wide.to_csv(RESULTS_DIR / f"table1_wide_{dataset}.csv", index=False)
        print(f"\n=== Wide comparison ({dataset}) ===")
        print(wide.to_string(index=False))

    if all_paired:
        pd.DataFrame(all_paired).to_csv(RESULTS_DIR / "paired_tests.csv", index=False)
        print("\n=== Paired tests ===")
        print(pd.DataFrame(all_paired).round(6).to_string(index=False))

    print("\n[aggregate] wrote table1_*.csv, table1_wide_*.csv, paired_tests.csv")


if __name__ == "__main__":
    main()
