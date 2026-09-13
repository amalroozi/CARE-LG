"""
Aggregates experiments_v2/results/{dataset}_per_patient.csv and
{dataset}_baselines.csv into:
  - experiments_v2/results/table1_{dataset}.csv (mean+-std over seeds)
  - experiments_v2/results/paired_tests.csv (Wilcoxon signed-rank, RQ1/RQ2)
  - experiments_v2/results/rq_summary.json (headline numbers for FINDINGS.md)

Also pulls 3 concrete example paths with decoded-space violations for the
report (Phase 3 "report prominently... with 3 concrete example paths").
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = REPO_ROOT / "experiments_v2" / "results"

MAIN_CELLS = [
    ("riemannian", "hard", None), ("riemannian", "soft", 1.0), ("riemannian", "soft", 10.0),
    ("euclidean", "hard", None), ("euclidean", "soft", 1.0), ("euclidean", "none", None),
]


def cell_label(metric, constraints, lam):
    return f"{metric}_{constraints}_lam{lam}" if constraints == "soft" else f"{metric}_{constraints}"


def load(dataset):
    df = pd.read_csv(RESULTS_DIR / f"{dataset}_per_patient.csv")
    df["lambda"] = df["lambda"].astype(object)
    base_path = RESULTS_DIR / f"{dataset}_baselines.csv"
    dfb = pd.read_csv(base_path) if base_path.exists() else None
    return df, dfb


def per_seed_cell_stats(df):
    """Per (cell, seed): success rate, decoded CVR rates, mean KDE/effort/latency among successes."""
    rows = []
    for (cell, seed), g in df.groupby(["cell", "seed"]):
        succ = g[g["success"] == True]
        rows.append({
            "cell": cell, "seed": seed, "n_patients": len(g), "n_success": len(succ),
            "success_rate": 100.0 * len(succ) / len(g) if len(g) else np.nan,
            "decoded_cvr_hop_rate": 100.0 * succ["decoded_cvr_hop"].mean() if len(succ) else np.nan,
            "decoded_cvr_endpoint_rate": 100.0 * succ["decoded_cvr_endpoint"].mean() if len(succ) else np.nan,
            "kde_mean": succ["kde"].mean() if len(succ) else np.nan,
            "effort_mean": succ["effort"].mean() if len(succ) else np.nan,
            "latency_mean": g["latency_sec"].mean(),
        })
    return pd.DataFrame(rows)


def aggregate_table(df_seedcell):
    agg = df_seedcell.groupby("cell").agg(
        success_rate_mean=("success_rate", "mean"), success_rate_std=("success_rate", "std"),
        decoded_cvr_hop_mean=("decoded_cvr_hop_rate", "mean"), decoded_cvr_hop_std=("decoded_cvr_hop_rate", "std"),
        decoded_cvr_endpoint_mean=("decoded_cvr_endpoint_rate", "mean"), decoded_cvr_endpoint_std=("decoded_cvr_endpoint_rate", "std"),
        kde_mean=("kde_mean", "mean"), kde_std=("kde_mean", "std"),
        effort_mean=("effort_mean", "mean"), effort_std=("effort_mean", "std"),
        latency_mean=("latency_mean", "mean"), latency_std=("latency_mean", "std"),
    ).reset_index()
    return agg


def baseline_table(dfb):
    if dfb is None:
        return pd.DataFrame()
    rows = []
    for method, g in dfb.groupby("method"):
        per_seed = g.groupby("seed").agg(
            success_rate=("success", lambda s: 100.0 * s.mean()),
        )
        succ = g[g["success"] == True]
        per_seed_cvr = g.groupby("seed").apply(lambda s: 100.0 * s.loc[s["success"] == True, "cvr_hard"].mean() if (s["success"] == True).any() else np.nan)
        per_seed_cvr_full = g.groupby("seed").apply(lambda s: 100.0 * s.loc[s["success"] == True, "cvr_full"].mean() if (s["success"] == True).any() else np.nan)
        per_seed_kde = g.groupby("seed").apply(lambda s: s.loc[s["success"] == True, "kde"].mean() if (s["success"] == True).any() else np.nan)
        per_seed_effort = g.groupby("seed").apply(lambda s: s.loc[s["success"] == True, "effort"].mean() if (s["success"] == True).any() else np.nan)
        per_seed_lat = g.groupby("seed")["latency_sec"].mean()
        rows.append({
            "cell": method,
            "success_rate_mean": per_seed["success_rate"].mean(), "success_rate_std": per_seed["success_rate"].std(),
            "cvr_hard_mean": per_seed_cvr.mean(), "cvr_hard_std": per_seed_cvr.std(),
            "decoded_cvr_endpoint_mean": per_seed_cvr_full.mean(), "decoded_cvr_endpoint_std": per_seed_cvr_full.std(),
            "kde_mean": per_seed_kde.mean(), "kde_std": per_seed_kde.std(),
            "effort_mean": per_seed_effort.mean(), "effort_std": per_seed_effort.std(),
            "latency_mean": per_seed_lat.mean(), "latency_std": per_seed_lat.std(),
        })
    return pd.DataFrame(rows)


def paired_test(df, cell_a, cell_b, metric):
    """Paired Wilcoxon on `metric` for patients where BOTH cell_a and cell_b succeeded, paired by (seed, patient_idx)."""
    a = df[(df["cell"] == cell_a) & (df["success"] == True)][["seed", "patient_idx", metric]].rename(columns={metric: "a"})
    b = df[(df["cell"] == cell_b) & (df["success"] == True)][["seed", "patient_idx", metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on=["seed", "patient_idx"])
    merged = merged.dropna()
    n = len(merged)
    if n < 2 or np.allclose(merged["a"], merged["b"]):
        return {"cell_a": cell_a, "cell_b": cell_b, "metric": metric, "n_pairs": n, "statistic": np.nan, "pvalue": np.nan, "median_diff": (merged["a"] - merged["b"]).median() if n else np.nan}
    stat, p = wilcoxon(merged["a"], merged["b"])
    return {"cell_a": cell_a, "cell_b": cell_b, "metric": metric, "n_pairs": n, "statistic": float(stat), "pvalue": float(p),
            "median_diff": float((merged["a"] - merged["b"]).median())}


def find_example_violations(df, dataset, cell="riemannian_hard", n=3):
    """3 concrete example paths with decoded-space violations, for the report."""
    sub = df[(df["cell"] == cell) & (df["success"] == True) & (df["decoded_cvr_hop"] == True)]
    examples = []
    for _, row in sub.head(n).iterrows():
        examples.append({
            "dataset": dataset, "seed": int(row["seed"]), "patient_idx": int(row["patient_idx"]),
            "path": row["path"], "decoded_violation_categories": row["decoded_violation_categories"],
            "decoded_cvr_hop": bool(row["decoded_cvr_hop"]), "decoded_cvr_endpoint": bool(row["decoded_cvr_endpoint"]),
        })
    return examples


def main():
    summary = {}
    all_paired = []
    for dataset in ["uci", "nhanes"]:
        df, dfb = load(dataset)
        seedcell = per_seed_cell_stats(df)
        table = aggregate_table(seedcell)
        base_table = baseline_table(dfb)
        combined = pd.concat([table, base_table], ignore_index=True, sort=False)
        combined.to_csv(RESULTS_DIR / f"table1_{dataset}.csv", index=False)
        seedcell.to_csv(RESULTS_DIR / f"seedcell_{dataset}.csv", index=False)

        # RQ2: R+hard vs E+hard, KDE + effort
        p1 = paired_test(df, "riemannian_hard", "euclidean_hard", "kde")
        p2 = paired_test(df, "riemannian_hard", "euclidean_hard", "effort")
        # RQ1 supporting: R+hard vs R+soft(lam1), KDE + effort (context for the CVR contrast)
        p3 = paired_test(df, "riemannian_hard", "riemannian_soft_lam1.0", "kde")
        p4 = paired_test(df, "riemannian_hard", "riemannian_soft_lam1.0", "effort")
        for p in (p1, p2, p3, p4):
            p["dataset"] = dataset
            all_paired.append(p)

        examples = find_example_violations(df, dataset)

        summary[dataset] = {
            "table1": combined.to_dict(orient="records"),
            "paired_tests": [p1, p2, p3, p4],
            "decoded_cvr_examples": examples,
        }
        print(f"=== {dataset.upper()} Table 1 ===")
        print(combined.round(4).to_string(index=False))
        print()

    pd.DataFrame(all_paired).to_csv(RESULTS_DIR / "paired_tests.csv", index=False)
    with open(RESULTS_DIR / "rq_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Wrote table1_{uci,nhanes}.csv, paired_tests.csv, rq_summary.json")


if __name__ == "__main__":
    main()
