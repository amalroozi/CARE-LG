"""
Aggregates experiments_v4/results/{dataset}_per_patient.csv into:
  - table1_{dataset}.csv           (mean +/- std per cell, over seeds)
  - tau_sweep_{dataset}.csv         (E2: success/decoded-CVR by tau quantile)
  - paired_tests.csv                (Wilcoxon: old vs dsr_full, on decoded CVR and effort)
  - per_constraint_breakdown.csv    (E1 col: which constraint type breaks, by cell)
  - decoder_dependence.csv          (E5: v3 vs v2 decoder, main cells)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = REPO_ROOT / "experiments_v4" / "results"

MAIN_CELLS = ["old_latent_pruning", "dsr_c1", "dsr_c1c2", "dsr_full", "dsr_full_euclidean", "dsr_bands_no_identity"]
DATASETS = ["uci", "nhanes_real"]


def load_main(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{dataset}_per_patient.csv")
    return df[df["tau_level"] == 0.95]  # main cells always ran at the operating point


def load_sweep(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{dataset}_per_patient.csv")
    return df[df["cell"] == "dsr_full"]  # includes every tau_level tested


def build_table1(dataset: str) -> pd.DataFrame:
    df = load_main(dataset)
    rows = []
    for cell in MAIN_CELLS:
        sub = df[df["cell"] == cell]
        per_seed = sub.groupby("seed").agg(
            success_rate=("success", "mean"),
            n_pruned=("n_edges_pruned", "first"),
            n_total=("n_edges_total", "first"),
            build_s=("graph_build_seconds", "first"),
        )
        ok = sub[sub["success"] == True]
        per_seed_cvr = ok.groupby("seed")["decoded_cvr_endpoint"].mean() if len(ok) else pd.Series(dtype=float)
        per_seed_kde = ok.groupby("seed")["kde"].mean() if len(ok) else pd.Series(dtype=float)
        per_seed_effort = ok.groupby("seed")["effort"].mean() if len(ok) else pd.Series(dtype=float)
        per_seed_lat = ok.groupby("seed")["latency_sec"].mean() if len(ok) else pd.Series(dtype=float)

        rows.append({
            "cell": cell,
            "success_rate_mean": per_seed["success_rate"].mean() * 100, "success_rate_std": per_seed["success_rate"].std() * 100,
            "decoded_cvr_mean": per_seed_cvr.mean() * 100 if len(per_seed_cvr) else np.nan,
            "decoded_cvr_std": per_seed_cvr.std() * 100 if len(per_seed_cvr) else np.nan,
            "kde_mean": per_seed_kde.mean() if len(per_seed_kde) else np.nan,
            "kde_std": per_seed_kde.std() if len(per_seed_kde) else np.nan,
            "effort_mean": per_seed_effort.mean() if len(per_seed_effort) else np.nan,
            "effort_std": per_seed_effort.std() if len(per_seed_effort) else np.nan,
            "latency_mean": per_seed_lat.mean() if len(per_seed_lat) else np.nan,
            "latency_std": per_seed_lat.std() if len(per_seed_lat) else np.nan,
            "pct_edges_pruned_mean": 100 * (per_seed["n_pruned"] / per_seed["n_total"]).mean(),
            "graph_build_seconds_mean": per_seed["build_s"].mean(),
            "n_seeds": len(per_seed),
        })
    return pd.DataFrame(rows)


# NOTE: viol_immutable/non_decreasing/age_horizon/directional (used below) come
# from the HOP-WISE decoded check (every consecutive step), OR'd across all
# hops -- NOT the same computation as decoded_cvr_endpoint (source-vs-terminal
# only, check_step_horizon=False) that Table 1 headlines. A path can pass every
# individual hop check yet still fail the endpoint check (small per-hop drift
# compounding over 2+ hops), so this breakdown is a genuinely different,
# complementary diagnostic -- reported as such in FINDINGS.md/REPORT.pdf, not
# conflated with the headline number.
def build_constraint_breakdown(dataset: str) -> pd.DataFrame:
    df = load_main(dataset)
    rows = []
    for cell in MAIN_CELLS:
        sub = df[(df["cell"] == cell) & (df["success"] == True)]
        if len(sub) == 0:
            rows.append({"cell": cell, "n": 0})
            continue
        rows.append({
            "cell": cell, "n": len(sub),
            "pct_immutable": 100 * sub["viol_immutable"].mean(),
            "pct_non_decreasing": 100 * sub["viol_non_decreasing"].mean(),
            "pct_age_horizon": 100 * sub["viol_age_horizon"].mean(),
            "pct_directional": 100 * sub["viol_directional"].mean(),
        })
    return pd.DataFrame(rows)


def build_tau_sweep(dataset: str) -> pd.DataFrame:
    df = load_sweep(dataset)
    rows = []
    for q in sorted(df["tau_level"].unique()):
        sub = df[df["tau_level"] == q]
        per_seed_succ = sub.groupby("seed")["success"].mean()
        ok = sub[sub["success"] == True]
        per_seed_cvr = ok.groupby("seed")["decoded_cvr_endpoint"].mean() if len(ok) else pd.Series(dtype=float)
        rows.append({
            "tau_quantile": q,
            "success_rate_mean": per_seed_succ.mean() * 100, "success_rate_std": per_seed_succ.std() * 100,
            "decoded_cvr_mean": per_seed_cvr.mean() * 100 if len(per_seed_cvr) else np.nan,
            "decoded_cvr_std": per_seed_cvr.std() * 100 if len(per_seed_cvr) else np.nan,
            "n_seeds": sub["seed"].nunique(),
        })
    return pd.DataFrame(rows)


def paired_test(dataset: str, label_a: str, label_b: str, metric: str,
                sub_a: pd.DataFrame, sub_b: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank on `metric`, paired by (seed, patient_idx), successes only on both sides."""
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


def best_dsr_full_tau(dataset: str) -> float:
    """The lowest-CVR-collapse tau quantile with the most dsr_full successes, for a paired test with real power."""
    sw = load_sweep(dataset)
    counts = sw[sw["success"] == True].groupby("tau_level").size()
    return float(counts.idxmax()) if len(counts) else 0.95


def build_decoder_dependence(dataset: str) -> pd.DataFrame:
    v3 = load_main(dataset)
    v3 = v3[v3["cell"].isin(["old_latent_pruning", "dsr_c1", "dsr_full"])].copy()
    v3["decoder"] = "v3"
    v2_path = RESULTS_DIR / f"{dataset}_per_patient_v2decoder.csv"
    if not v2_path.exists():
        return pd.DataFrame()
    v2 = pd.read_csv(v2_path)
    v2 = v2[(v2["tau_level"] == 0.95) & (v2["cell"].isin(["old_latent_pruning", "dsr_c1", "dsr_full"]))].copy()
    v2["decoder"] = "v2"
    combined = pd.concat([v3, v2], ignore_index=True)

    rows = []
    for decoder in ["v2", "v3"]:
        for cell in ["old_latent_pruning", "dsr_c1", "dsr_full"]:
            sub = combined[(combined["decoder"] == decoder) & (combined["cell"] == cell)]
            per_seed_succ = sub.groupby("seed")["success"].mean()
            ok = sub[sub["success"] == True]
            per_seed_cvr = ok.groupby("seed")["decoded_cvr_endpoint"].mean() if len(ok) else pd.Series(dtype=float)
            rows.append({
                "decoder": decoder, "cell": cell,
                "success_rate_mean": per_seed_succ.mean() * 100 if len(per_seed_succ) else np.nan,
                "decoded_cvr_mean": per_seed_cvr.mean() * 100 if len(per_seed_cvr) else np.nan,
                "decoded_cvr_std": per_seed_cvr.std() * 100 if len(per_seed_cvr) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    all_table1, all_breakdown, all_sweep, all_paired, all_decoder = [], [], [], [], []
    for dataset in DATASETS:
        path = RESULTS_DIR / f"{dataset}_per_patient.csv"
        if not path.exists():
            print(f"[aggregate] skipping {dataset}: {path} not found")
            continue

        t1 = build_table1(dataset)
        t1.insert(0, "dataset", dataset)
        all_table1.append(t1)
        t1.to_csv(RESULTS_DIR / f"table1_{dataset}.csv", index=False)
        print(f"\n=== Table 1 ({dataset}) ===")
        print(t1.round(3).to_string(index=False))

        cb = build_constraint_breakdown(dataset)
        cb.insert(0, "dataset", dataset)
        all_breakdown.append(cb)

        sw = build_tau_sweep(dataset)
        sw.insert(0, "dataset", dataset)
        all_sweep.append(sw)
        sw.to_csv(RESULTS_DIR / f"tau_sweep_{dataset}.csv", index=False)
        print(f"\n=== Tau sweep ({dataset}) ===")
        print(sw.round(3).to_string(index=False))

        # old vs dsr_full AT THE OPERATING POINT (q=0.95): almost certainly 0 pairs,
        # since dsr_full has near-zero successes there (see Table 1) -- reported
        # explicitly as a real, meaningful result (there is nothing to pair because
        # the method abstains almost everywhere at the nominal 95% level), not
        # hidden as a NaN.
        df_main = load_main(dataset)
        old_main = df_main[df_main["cell"] == "old_latent_pruning"]
        dsrfull_main = df_main[df_main["cell"] == "dsr_full"]
        for metric in ["decoded_cvr_endpoint", "effort", "kde"]:
            all_paired.append(paired_test(dataset, "old_latent_pruning@q0.95", "dsr_full@q0.95", metric,
                                          old_main, dsrfull_main))

        # old vs dsr_full AT DSR-FULL's BEST-POWERED OPERATING POINT from the sweep
        # (the lowest tau with the most successes) -- this is the meaningful
        # comparison for "does repair, where it actually succeeds, help."
        best_q = best_dsr_full_tau(dataset)
        dsrfull_best = load_sweep(dataset)
        dsrfull_best = dsrfull_best[dsrfull_best["tau_level"] == best_q]
        for metric in ["decoded_cvr_endpoint", "effort", "kde"]:
            all_paired.append(paired_test(dataset, "old_latent_pruning@q0.95", f"dsr_full@q{best_q}", metric,
                                          old_main, dsrfull_best))

        # E4 (Riemannian vs Euclidean under DSR) at q=0.95: both collapse to ~0
        # successes there, so this is reported as a real 0-pairs result at the
        # nominal operating point.
        for metric in ["effort", "kde"]:
            all_paired.append(paired_test(dataset, "dsr_full@q0.95", "dsr_full_euclidean@q0.95", metric,
                                          dsrfull_main, df_main[df_main["cell"] == "dsr_full_euclidean"]))

        # E4 at dsr_full's best-powered tau (run_e4_best_tau.py's output, if present).
        eucl_best_path = RESULTS_DIR / f"{dataset}_e4_euclidean_best_tau.csv"
        if eucl_best_path.exists():
            eucl_best = pd.read_csv(eucl_best_path)
            eq = float(eucl_best["tau_level"].iloc[0]) if len(eucl_best) else best_q
            for metric in ["effort", "kde", "decoded_cvr_endpoint"]:
                all_paired.append(paired_test(dataset, f"dsr_full@q{best_q}", f"dsr_full_euclidean@q{eq}", metric,
                                              dsrfull_best, eucl_best))

        dd = build_decoder_dependence(dataset)
        if len(dd):
            dd.insert(0, "dataset", dataset)
            all_decoder.append(dd)

    if all_breakdown:
        pd.concat(all_breakdown, ignore_index=True).to_csv(RESULTS_DIR / "constraint_breakdown.csv", index=False)
    if all_paired:
        pd.DataFrame(all_paired).to_csv(RESULTS_DIR / "paired_tests.csv", index=False)
        print("\n=== Paired tests (old vs dsr_full; dsr_full vs dsr_full_euclidean) ===")
        print(pd.DataFrame(all_paired).round(5).to_string(index=False))
    if all_decoder:
        pd.concat(all_decoder, ignore_index=True).to_csv(RESULTS_DIR / "decoder_dependence.csv", index=False)
        print("\n=== Decoder dependence (E5) ===")
        print(pd.concat(all_decoder, ignore_index=True).round(2).to_string(index=False))

    print("\n[aggregate] wrote table1_*.csv, tau_sweep_*.csv, constraint_breakdown.csv, paired_tests.csv, decoder_dependence.csv")


if __name__ == "__main__":
    main()
