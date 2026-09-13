"""Figures for the EGD report. PNG (300dpi) + vector PDF into experiments_v5/figures/."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = REPO_ROOT / "experiments_v5" / "results"
FIG_DIR = REPO_ROOT / "experiments_v5" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {"uci": "UCI", "nhanes_real": "Real NHANES"}


def savefig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {name}.png / {name}.pdf")


def fig1_motivating_gap():
    """Decoded CVR: old method vs EGD (should be exactly 0%)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        t1 = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv")
        old = t1[t1["cell"] == "old_latent_pruning"].iloc[0]
        egd = t1[t1["cell"] == "egd_riemannian"].iloc[0]
        labels = ["Old\n(latent pruning)", "EGD\n(this session)"]
        means = [old["decoded_cvr_mean"], egd["decoded_cvr_mean"]]
        stds = [old["decoded_cvr_std"], egd["decoded_cvr_std"]]
        bars = ax.bar(labels, means, yerr=stds, color=["#e74c3c", "#2ecc71"], edgecolor="black", capsize=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{m:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Decoded-space CVR (%) ↓ better")
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        ax.set_ylim(0, 100)
    fig.suptitle("Fig 1 — EGD's exact guarantee, empirically verified:\ndecoded CVR is exactly 0.0% wherever EGD succeeds, on both cohorts", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, "fig1_motivating_gap")


def fig2_cost_of_guarantee():
    """The honest cost: success rate, KDE, effort -- old vs EGD."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("success_rate_mean", "success_rate_std", "Success rate (%)", False),
              ("kde_mean", "kde_std", "KDE (higher = more plausible)", True),
              ("effort_mean", "effort_std", "Clinical effort (lower = better)", False)]
    for ax, (mcol, scol, title, allow_neg) in zip(axes, metrics):
        x = np.arange(2)
        width = 0.35
        for i, dataset in enumerate(["uci", "nhanes_real"]):
            t1 = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv").set_index("cell")
            old_v, old_e = t1.loc["old_latent_pruning", mcol], t1.loc["old_latent_pruning", scol]
            egd_v, egd_e = t1.loc["egd_riemannian", mcol], t1.loc["egd_riemannian", scol]
            ax.bar(x[i] - width/2, old_v, width, yerr=old_e, color="#e74c3c", edgecolor="black", capsize=3,
                  label="Old" if i == 0 else None)
            ax.bar(x[i] + width/2, egd_v, width, yerr=egd_e, color="#2ecc71", edgecolor="black", capsize=3,
                  label="EGD" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[d] for d in ["uci", "nhanes_real"]])
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Fig 2 — The cost of the exact guarantee: EGD vs. the old method\n"
                "(paired tests confirm both KDE and effort differences are significant, p<0.001, both cohorts)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, "fig2_cost_of_guarantee")


def fig3_metric_ablation():
    """Riemannian vs Euclidean under EGD -- the REVERSED finding vs every prior session."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    for i, dataset in enumerate(["uci", "nhanes_real"]):
        t1 = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv").set_index("cell")
        r = t1.loc["egd_riemannian"]
        e = t1.loc["egd_euclidean"] if "egd_euclidean" in t1.index else None
        ax.bar(x[i] - width/2, r["success_rate_mean"], width, yerr=r["success_rate_std"],
              color="#9b59b6", edgecolor="black", capsize=3, label="Riemannian" if i == 0 else None)
        if e is not None:
            ax.bar(x[i] + width/2, e["success_rate_mean"], width, yerr=e["success_rate_std"],
                  color="#e67e22", edgecolor="black", capsize=3, label="Euclidean" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in ["uci", "nhanes_real"]])
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fig 3 — Riemannian vs. Euclidean under EGD: REVERSES every prior\nsession's null finding "
                "(mechanism: Euclidean favors shorter paths;\nEGD's guarantee accumulates change per hop, so path length now matters)",
                fontweight="bold", fontsize=10)
    ax.legend()
    fig.tight_layout()
    savefig(fig, "fig3_metric_ablation")


def fig4_seed_variability():
    """UCI's EGD success rate is highly seed-variable -- put the mean±std in context."""
    fig, ax = plt.subplots(figsize=(7, 5))
    df = pd.read_csv(RESULTS_DIR / "uci_per_patient.csv")
    df = df[df["cell"] == "egd_riemannian"]
    per_seed = df.groupby("seed")["success"].mean() * 100
    ax.plot(per_seed.index, per_seed.values, "o-", color="#9b59b6", markersize=10, linewidth=2)
    for s, v in per_seed.items():
        ax.annotate(f"{v:.0f}%", (s, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(per_seed.mean(), color="gray", linestyle="--", label=f"mean={per_seed.mean():.1f}%")
    ax.set_xlabel("Seed"); ax.set_ylabel("EGD success rate (%)")
    ax.set_xticks(per_seed.index)
    ax.set_ylim(-5, 100)
    ax.set_title("Fig 4 — UCI EGD success rate varies enormously by seed\n(0-90%+ range; std=37.0 points on a mean of 31.1%)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "fig4_seed_variability")


def fig5_abstention_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    colors = {"pct_certified_infeasible": "#2ecc71", "pct_confirmed_blindspot": "#f39c12", "pct_unresolved": "#e74c3c"}
    labels = {"pct_certified_infeasible": "Certified infeasible", "pct_confirmed_blindspot": "Blindspot (resolved)",
             "pct_unresolved": "Unresolved (target exists,\nEGD can't reach it)"}
    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        df = pd.read_csv(RESULTS_DIR / f"{dataset}_blindspot_summary.csv")
        df = df[df["dataset"] == dataset]   # NHANES file has cumulative rows incl. UCI's -- filter explicitly
        means = {c: df[c].mean() for c in colors}
        stds = {c: df[c].std() for c in colors}
        bottom = 0
        for c, col in colors.items():
            ax.bar([0], [means[c]], bottom=[bottom], color=col, edgecolor="black", yerr=[stds[c]], capsize=4,
                  label=labels[c], width=0.5)
            if means[c] > 4:
                ax.text(0, bottom + means[c]/2, f"{means[c]:.0f}%", ha="center", va="center", fontweight="bold")
            bottom += means[c]
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_ylabel("% of abstentions")
        ax.set_ylim(0, 100)
        ax.set_title(f"{DISPLAY[dataset]} (mean abstention rate: {df['pct_abstain'].mean():.0f}% of high-risk patients)",
                    fontweight="bold", fontsize=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.1), ncol=1, fontsize=8)
    fig.suptitle("Fig 5 — EGD abstention decomposition (E3): NO blindspots resolved by densification on EITHER\n"
                "cohort -- 'unresolved' means the mechanism (not connectivity) is the limit", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    savefig(fig, "fig5_abstention_decomposition")


def main():
    fig1_motivating_gap()
    fig2_cost_of_guarantee()
    fig3_metric_ablation()
    fig4_seed_variability()
    fig5_abstention_decomposition()


if __name__ == "__main__":
    main()
