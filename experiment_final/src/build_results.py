"""Final results build: NHANES 1999-2023 from the full rerun (results/run_full_nhanes, all high-risk test patients, 5 seeds), every other
dataset (Kaggle CVD-8k, NHANES 2017-2020, UCI) unchanged from the report_v3 run view. Outputs are NEW files only:
results/run_view/ (symlinks), results/tables/, results/figures/, results/report_final.html.
Wachter is excluded everywhere (user request). Dataset display names follow the user's naming.

  .venv/bin/python -m experiment_final.src.build_results [aggregate|figures|all]
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_final.src import evaluate as E
from experiment_final.src import visualize as V
from experiment_final.src.common import OLD_RUNS, OLD_TABLES, OLD_TABLES_V4, RESULTS

OLD = OLD_RUNS / "combined_20261004_215313"
NEW = RESULTS / "run_full_nhanes"
VIEW = RESULTS / "run_view"
T4, F4 = RESULTS / "tables", RESULTS / "figures"
T_V4 = OLD_TABLES_V4   # holds accuracy_backfill_v3runs.csv (report_v3 classifier accuracy)
ICD_FULL = "International Classification of Diseases, Tenth Revision, Clinical Modification (ICD-10-CM)"


CVD_LABEL = "Kaggle CVD-8k (8,000-patient stratified subsample of the Kaggle Cardiovascular Disease dataset, Ulianova 2019)"


def labels() -> dict:
    return {"nhanes_pooled": "NHANES 1999-2023", "nhanes": "NHANES 2017-2020", "uci": "UCI Heart", "cardio": CVD_LABEL}



# ------------------------------------------------------------------ view + aggregation

def build_view() -> Path:
    """Symlink view: NHANES 1999-2023 (+ ontology-as-features) from the full rerun, everything else from the report_v3 runs."""
    for ds in ["nhanes_pooled", "nhanes_pooled__ontology_as_features", "cardio", "nhanes", "nhanes__ontology_as_features", "uci"]:
        for s in range(5):
            d = ((NEW if ds.startswith("nhanes_pooled") else OLD) / ds / f"seed{s}").resolve()
            if not (d / "DONE").exists():
                raise SystemExit(f"not finished: {d}")
            tgt = VIEW / ds / f"seed{s}"
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if not tgt.exists():
                tgt.symlink_to(d)
    return VIEW


def aggregate() -> dict:
    """evaluate.aggregate into tables_v4 with Wachter removed before any statistic (incl. Holm families)."""
    build_view()
    E.TABLES = T4
    orig_read = E._read

    def read(run_dir, name):
        df = orig_read(run_dir, name)
        return df[df.method != "wachter"].reset_index(drop=True) if name == "methods.csv" and not df.empty else df
    E._read = read
    return E.aggregate(VIEW)


def relabel_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset ids -> display names (for the human-facing tables written next to the id-keyed ones)."""
    L = labels()
    d = df.copy()
    if "dataset" in d:
        d["dataset"] = d["dataset"].map(lambda x: L.get(x, L.get(str(x).split("__")[0], x) + (" (ontology as features)" if "__" in str(x) else "")))
    if "method" in d:
        d["method"] = d["method"].map(lambda m: V.METHOD_LABEL.get(m, m))
    return d


def write_labelled(store: dict) -> None:
    out = T4 / "labelled"
    out.mkdir(parents=True, exist_ok=True)
    for name, v in store.items():
        if not name.startswith("_") and isinstance(v, dict) and "df" in v:
            relabel_csv(v["df"]).to_csv(out / name, index=False)


# ------------------------------------------------------------------ old -> new diff

# ------------------------------------------------------------------ figures

def _save(fig, name, store):
    F4.mkdir(parents=True, exist_ok=True)
    p = F4 / f"{name}.svg"
    if p.exists():
        p = V.new_path(p)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=170, facecolor="white")
    V.plt.close(fig)
    store[name] = str(p)


def fig_architecture(store):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt = V.plt
    fig, ax = plt.subplots(figsize=(13.8, 5.4)); ax.axis("off"); ax.set_xlim(0, 13.8); ax.set_ylim(-0.75, 4.5)

    def box(x, y, w, h, title, body, color, kind=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=color, lw=1.6))
        ax.text(x + 0.1, y + h - 0.12, title, fontsize=8.6, weight="bold", color=V.INK, va="top")
        ax.text(x + 0.1, y + h - 0.48, "\n".join(textwrap.wrap(body, int(w * 9.4))), fontsize=8.6, color=V.INK, weight="bold", va="top")
        if kind:
            ax.text(x + w - 0.08, y + 0.08, kind, fontsize=7.2, weight="bold", color="white", ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.18", fc="#c2410c", ec="none"))

    def arrow(a, b, label="", big=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22 if big else 13, color=V.INK2, lw=2.6 if big else 1.3))
        if label and a[1] == b[1]:   # horizontal arrow: label above it
            ax.text((a[0] + b[0]) / 2, a[1] + 0.07, label, fontsize=9, weight="bold", color=V.INK, ha="center", va="bottom")
        elif label:                  # vertical arrow: label to its right
            ax.text(a[0] + 0.08, (a[1] + b[1]) / 2, label, fontsize=9, weight="bold", color=V.INK, ha="left", va="center")

    box(0.1, 2.85, 2.4, 1.55, "1. KNOWLEDGE LAYER", "ICD-10-CM ontology: a rule on a code passes down to its sub-codes. Hard, soft and filter rules.", V.MUTED)
    box(0.1, 0.95, 2.4, 1.65, "2. MODEL + LATENT GRAPH", "MLP or XGBoost; VAE; k-NN graph", V.MUTED)
    box(3.05, 2.85, 2.7, 1.55, "3. TIER 1: CERTIFIED REAL", "Dijkstra on the rule-pruned graph; clingo certificate", V.TIER_C[1], "NEURO-SYMBOLIC")
    box(3.05, 0.95, 2.7, 1.65, "4. TIER 2: GUIDED SYNTHETIC", "Genetic search for a new, valid, plausible profile. Ontology guides the start; hard rules always hold.", V.TIER_C[2], "NEURO-SYMBOLIC")
    box(3.05, -0.7, 2.7, 0.9, "5. TIER 3: CERTIFIED NONE", "Exhaustive proof; reports the closest miss", V.TIER_C[3])
    box(6.2, 1.4, 2.3, 2.0, "6. TREATMENT FILTER", "Meaningful changes; ontology removes unsafe drugs", V.INK2)
    box(8.9, 1.4, 2.5, 2.0, "7. RECOMMENDATION CARD", "Plan in steps: lifestyle, then 1 drug, then 2. Expected change per step from published effect sizes; review rule per step.", V.INK2)
    box(11.8, 1.4, 1.9, 2.0, "8. EXPLANATION", "Template text, no LLM. Every rule, ICD-10-CM code, number and citation is a tag checked by trace().", V.INK2)
    arrow((2.5, 3.75), (3.05, 3.75), "gate"); arrow((2.5, 1.75), (3.05, 1.75)); arrow((1.3, 2.85), (1.3, 2.6), "prune")
    arrow((4.4, 2.85), (4.4, 2.6), "none admissible"); arrow((4.4, 0.95), (4.4, 0.25), "no valid CF", big=True)
    arrow((5.75, 3.55), (6.2, 3.1)); arrow((5.75, 1.8), (6.2, 2.4)); arrow((5.75, -0.2), (6.2, 1.7)); arrow((8.5, 2.4), (8.9, 2.4)); arrow((11.4, 2.4), (11.8, 2.4))
    ax.set_title("CARE-LG Architecture", loc="left", fontsize=15, weight="bold", color=V.INK, pad=26)
    ax.text(0.0, 4.55, "A learned risk model and graph, constrained by medical rules, find the safest path to lower risk and turn it into a checked treatment plan.", fontsize=10, color=V.INK, va="bottom")
    _save(fig, "fig01_architecture", store)


def fig_coverage(store, head):
    plt = V.plt
    ds = [d for d in ["nhanes_pooled", "nhanes", "uci", "cardio"] if d in set(head.dataset)]
    L = labels()
    fig, ax = plt.subplots(figsize=(9.5, 0.9 + 0.75 * len(ds)))
    for r, d in enumerate(ds[::-1]):
        h = head[head.dataset == d].iloc[0]; left = 0
        for t in (1, 2, 3):
            v = h[f"tier{t}_mean"]
            ax.barh(r, v, left=left, color=V.TIER_C[t], height=0.6, edgecolor="white", linewidth=2)
            if v >= 6:
                ax.text(left + v / 2, r, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8, weight="bold")
            left += v
        ax.text(101, r, f"n={int(h['n_high_risk_total']):,}", va="center", fontsize=7.5, color=V.INK2)
    ax.set_yticks(range(len(ds))); ax.set_yticklabels(["\n".join(textwrap.wrap(L[d], 46)) for d in ds[::-1]], fontsize=7.5)
    ax.set_xlim(0, 112); ax.set_xlabel("% of high-risk test patients (mean over 5 seeds)"); ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=V.TIER_C[t]) for t in (1, 2, 3)], labels=[V.TIER_L[t] for t in (1, 2, 3)],
              ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=8)
    ax.set_title("Coverage: high-risk patients by tier", loc="left", fontsize=11, color=V.INK)
    _save(fig, "fig02_coverage_tiers", store)


def fig_methods(store, meth):
    plt = V.plt
    m = meth[~meth.method.str.startswith(("tier1_", "tier2_"))]
    ds = [d for d in ["nhanes_pooled", "nhanes", "uci", "cardio"] if d in set(m.dataset)]
    order = m[m.dataset == ds[0]].sort_values("valid_and_admissible_mean", ascending=False).method.tolist()
    fig, axes = plt.subplots(1, len(ds), figsize=(4.4 * len(ds), 3.9)); fig.subplots_adjust(wspace=0.6)
    for ax, d in zip(axes, ds):
        g = m[m.dataset == d].set_index("method").reindex(order); y = np.arange(len(g))
        ax.barh(y - 0.19, g.valid_f_mean, height=0.36, color=V.MUTED, label="valid")
        ax.barh(y + 0.19, g.valid_and_admissible_mean, height=0.36, color=V.OURS_C, label="valid AND admissible")
        ax.set_yticks(y); ax.set_yticklabels([V.METHOD_LABEL.get(x, x) for x in g.index], fontsize=7.5); ax.invert_yaxis()
        ax.set_xlim(0, 105); ax.grid(axis="y", visible=False); ax.set_xlabel("% of high-risk patients")
        ax.set_title("\n".join(textwrap.wrap(labels()[d], 40)), loc="left", fontsize=8.5, color=V.INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    fig.suptitle("Valid vs valid-and-admissible counterfactuals, all methods", x=0.01, ha="left", fontsize=11, color=V.INK, y=1.04)
    _save(fig, "fig03_valid_vs_admissible", store)


def fig_ablation_bars(store, abl, t2):
    plt = V.plt
    a = abl[(abl.dataset == "nhanes_pooled") & (abl.status == "run")].set_index("arm")
    main = a.loc["main", "covered_mean"]
    arms = [("soft_as_hard", "soft rules as hard vetoes", f"0 soft-rule violations; Tier 1 only {a.loc['soft_as_hard', 'tier1_f_mean']:.1f}%"),
            ("tier1_only", "Tier 1 only (no synthetic tier)", "no Tier 2 fallback"),
            ("ontology_as_features", "ontology as model features", f"Tier 1 falls to {a.loc['ontology_as_features', 'tier1_f_mean']:.1f}%; {int(a.loc['ontology_as_features', 'contraindicated_recs_total'])} contraindicated recommendations"),
            ("no_inheritance", "no inheritance / no ontology", f"{int(a.loc['no_inheritance', 'contraindicated_recs_total'])} / {int(a.loc['no_ontology', 'contraindicated_recs_total'])} contraindicated recommendations"),
            ("no_guidance", "no ontology guidance in Tier 2", f"coverage {a.loc['no_guidance', 'covered_mean'] - main:+.1f} points"),
            ("euclidean", "Euclidean instead of Riemannian graph", f"coverage {a.loc['euclidean', 'covered_mean'] - main:+.1f} points (no change)"),
            ("gradient", "Riemannian gradient instead of genetic search (Tier 2)", None)]
    g = t2[t2.dataset == "nhanes_pooled"].set_index("method")["valid_f_mean"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for i, (k, lab, cost) in enumerate(arms):
        v = g["tier2_riemannian_ng"] if k == "gradient" else a.loc[k, "covered_mean"]
        cost = f"valid counterfactuals for these patients (guided genetic search: {g['tier2_guided_ga']:.1f}%)" if k == "gradient" else cost
        ax.barh(i, v, color=V.MUTED, height=0.6)
        ax.text(v + 1, i, f"{v:.1f}%  |  {cost}", va="center", fontsize=7.5, color=V.INK2)
    ax.axvline(main, color=V.OURS_C, lw=1.5, ls="--"); ax.text(main, -0.75, f"full method {main:.1f}%", color=V.OURS_C, fontsize=7.5, ha="center")
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([x[1] for x in arms], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 185); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("high-risk patients with admissible recourse, Tier 1 + 2 (%) (last bar: valid Tier-2 search results); text = what the ablation costs")
    ax.set_title("Ablations, NHANES 1999-2023", loc="left", fontsize=11, color=V.INK)
    _save(fig, "fig04_ablations", store)


def fig_escalation(store, arms_df):
    plt = V.plt
    a = arms_df[(arms_df.dataset == "nhanes_pooled") & (arms_df.variant.fillna("") == "") & arms_df.arm.isin(["main", "no_escalation"])
                & (arms_df.n_recs > 0)]
    kind = lambda s: {"lifestyle": "lifestyle", "single_drug": "one drug", "combination": "combination"}[s]  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    cols = {"lifestyle": V.TIER_C[1], "one drug": V.TIER_C[2], "combination": "#4a3aa7"}
    for i, (arm, lab) in enumerate([("main", "with escalation"), ("no_escalation", "without escalation")]):
        g = a[a.arm == arm].top_step.map(kind).value_counts(normalize=True) * 100
        left = 0
        for k in ["lifestyle", "one drug", "combination"]:
            v = g.get(k, 0.0)
            ax.barh(i, v, left=left, color=cols[k], height=0.6, edgecolor="white", linewidth=1.5, label=k if i == 0 else None)
            if v > 6:
                ax.text(left + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8, weight="bold")
            left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["with escalation", "without escalation"]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False); ax.set_xlabel("% of recommendation cards: type of the first recommendation")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.45))
    ax.set_title("First recommendation, NHANES 1999-2023", loc="left", fontsize=10, color=V.INK)
    _save(fig, "fig05_escalation", store)


def learning_curve() -> pd.DataFrame:
    """report_v3 learning curve (2k-16k subsamples, unchanged) with the full-data point (32,786) replaced by the full rerun's Tier-1 coverage
    (main arm, all high-risk test patients, same exhaustive Tier-1 quantity), one value per seed. Written to tables/pooled_learning_curve.csv."""
    lc = pd.read_csv(OLD_TABLES / "pooled_learning_curve.csv")
    n_all = int(lc.n_patients.max())
    rows = []
    for sd in range(5):
        a = pd.read_csv(VIEW / "nhanes_pooled" / f"seed{sd}" / "arms.csv"); a = a[a.arm == "main"]
        rows.append({"n_patients": n_all, "seed": sd, "oracle": json.loads((VIEW / "nhanes_pooled" / f"seed{sd}" / "setup.json").read_text())["oracle"],
                     "test_auc": np.nan, "n_high": len(a), "tier1_coverage_pct": 100 * float((a.tier == 1).mean())})
    out = pd.concat([lc[lc.n_patients < n_all], pd.DataFrame(rows)], ignore_index=True)
    T4.mkdir(parents=True, exist_ok=True)
    out.to_csv(T4 / "pooled_learning_curve.csv", index=False)
    return out


def fig_learning_curve(store):
    plt = V.plt
    lc = learning_curve().groupby("n_patients").tier1_coverage_pct.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(lc.n_patients, lc["mean"], yerr=lc["std"], marker="o", ms=5, color=V.TIER_C[1], lw=2, capsize=3, label="exhaustive Tier-1 check on subsamples")
    for x, y in zip(lc.n_patients, lc["mean"]):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=V.INK2)
    ax.set_xscale("log"); ax.set_xticks(lc.n_patients); ax.set_xticklabels([f"{v:,}" for v in lc.n_patients], fontsize=7.5); ax.minorticks_off()
    ax.set_xlabel("patients in cohort (log scale)"); ax.set_ylabel("Tier-1 coverage (%), mean ± SD over 5 seeds")
    ax.set_title("Learning curve, NHANES 1999-2023", loc="left", fontsize=10, color=V.INK)
    _save(fig, "fig06_learning_curve", store)


def _copy_pair(src: Path, name: str) -> None:
    """Copy the SVG + PNG written by visualize.save into figures under `name`."""
    F4.mkdir(parents=True, exist_ok=True)
    for suf in (".svg", ".png"):
        (F4 / f"{name}{suf}").write_bytes(src.with_suffix(suf).read_bytes())


def fig_plaus(store, meth):
    m = meth[~meth.method.isin(["wachter"])]
    old = dict(V.DS_LABEL)
    V.DS_LABEL.update({k: "\n".join(textwrap.wrap(v, 40)) for k, v in labels().items()})
    tmp, old_dir = {}, V.FIGURES
    with tempfile.TemporaryDirectory() as td:             # V.save writes next to results/: keep it out of the repo
        V.FIGURES = Path(td)
        V.plaus_validity(m, tmp)
        V.FIGURES = old_dir
        V.DS_LABEL.clear(); V.DS_LABEL.update(old)
        _copy_pair(Path(tmp["fig07_plausibility_vs_validity"]), "fig07_plausibility_vs_validity")
    store["fig07_plausibility_vs_validity"] = str(F4 / "fig07_plausibility_vs_validity.svg")


def repro_figures(store):
    """KM and baseline failure-type figures from a verified reproduction of the NEW seed-2 run (all high-risk test patients).
    reproduce() aborts unless the oracle, edge count, % pruned, brackets, patient risk and every baseline's per-patient valid / violation flag match the stored run."""
    from experiment_final.src import new_figures as N
    if not N.CACHE.exists():
        N.reproduce()
    names = {"fig15_km_cvd_survival": "fig08_km", "fig16_baseline_failure_types": "fig09_failure_types"}
    N.save = lambda fig, name, st: _save(fig, names[name], st)
    N.fig_km(store); N.fig_failures(store)


def all_figures(store_tables: dict) -> dict:
    store = {}
    T = {k: v["df"] for k, v in store_tables.items() if not k.startswith("_")}
    fig_architecture(store)
    fig_coverage(store, T["headline_tiered.csv"])
    fig_methods(store, T["methods_comparison.csv"])
    fig_ablation_bars(store, T["ablations.csv"], T["tier2_variants.csv"])
    fig_escalation(store, store_tables["_frames"]["arms"])
    fig_learning_curve(store)
    fig_plaus(store, T["methods_comparison.csv"])
    repro_figures(store)
    return store


def main(stage: str):
    if stage in ("aggregate", "all"):
        st = aggregate()
        write_labelled(st)
        import pickle
        (T4 / "_store.pkl").write_bytes(pickle.dumps(st))
    if stage in ("figures", "all"):
        import pickle
        st = pickle.loads((T4 / "_store.pkl").read_bytes())
        print(all_figures(st))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
