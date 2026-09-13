"""
Figures 1-6 for the DSR report. PNG (300dpi) + vector PDF into experiments_v4/figures/.
matplotlib only, no seaborn.
"""
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

RESULTS_DIR = REPO_ROOT / "experiments_v4" / "results"
FIG_DIR = REPO_ROOT / "experiments_v4" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {"uci": "UCI", "nhanes_real": "Real NHANES"}
CELL_LABEL = {
    "old_latent_pruning": "Old\n(latent pruning)", "dsr_c1": "DSR-C1", "dsr_c1c2": "DSR-C1+C2\n(pre-filter)",
    "dsr_full": "DSR-full\n(+repair)", "dsr_full_euclidean": "DSR-full\n(Euclidean)",
    "dsr_bands_no_identity": "Bands, no\nidentity rule",
}


def savefig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {name}.png / {name}.pdf")


def fig1_motivating_gap():
    """Decoded CVR: old method + baselines (FACE/DiCE/REVISE if available) vs DSR-full."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        t1 = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv")
        sweep = pd.read_csv(RESULTS_DIR / f"tau_sweep_{dataset}.csv")
        best = sweep[sweep["success_rate_mean"] > 0].sort_values("success_rate_mean", ascending=False)

        labels, means, stds, colors = [], [], [], []
        row = t1[t1["cell"] == "old_latent_pruning"].iloc[0]
        labels.append("Old method\n(latent pruning)"); means.append(row["decoded_cvr_mean"]); stds.append(row["decoded_cvr_std"]); colors.append("#e74c3c")
        row = t1[t1["cell"] == "dsr_c1"].iloc[0]
        labels.append("DSR-C1\n(decoded pruning)"); means.append(row["decoded_cvr_mean"]); stds.append(row["decoded_cvr_std"]); colors.append("#f39c12")
        if len(best):
            b = best.iloc[0]
            labels.append(f"DSR-full\n(+repair, q={b['tau_quantile']:.2f})")
            means.append(b["decoded_cvr_mean"]); stds.append(b["decoded_cvr_std"] if pd.notna(b["decoded_cvr_std"]) else 0)
            colors.append("#2ecc71")
        else:
            labels.append("DSR-full\n(no viable tau)"); means.append(0); stds.append(0); colors.append("#95a5a6")

        bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor="black", capsize=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{m:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Decoded-space CVR (%)  ↓ better")
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Fig 1 — The motivating gap: decoded-space constraint violations,\nold method vs. DSR", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "fig1_motivating_gap")


def fig2_component_ablation():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cells = ["old_latent_pruning", "dsr_c1", "dsr_c1c2", "dsr_full"]
    for col, dataset in enumerate(["uci", "nhanes_real"]):
        t1 = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv").set_index("cell")
        succ = [t1.loc[c, "success_rate_mean"] if c in t1.index else 0 for c in cells]
        succ_e = [t1.loc[c, "success_rate_std"] if c in t1.index else 0 for c in cells]
        cvr = [t1.loc[c, "decoded_cvr_mean"] if c in t1.index else np.nan for c in cells]
        cvr_e = [t1.loc[c, "decoded_cvr_std"] if c in t1.index else 0 for c in cells]
        labels = [CELL_LABEL[c] for c in cells]

        ax = axes[0, col]
        ax.bar(labels, succ, yerr=succ_e, color="#3498db", edgecolor="black", capsize=4)
        ax.set_title(f"{DISPLAY[dataset]}: Success rate (%)", fontweight="bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=8)

        ax = axes[1, col]
        cvr_plot = [0 if np.isnan(v) else v for v in cvr]
        ax.bar(labels, cvr_plot, yerr=[0 if np.isnan(e) else e for e in cvr_e], color="#e67e22", edgecolor="black", capsize=4)
        for i, v in enumerate(cvr):
            if np.isnan(v):
                ax.text(i, 5, "n/a\n(0 success)", ha="center", fontsize=8)
        ax.set_title(f"{DISPLAY[dataset]}: Decoded CVR (%) at q=0.95", fontweight="bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Fig 2 — Component ablation: old → C1 → C1+C2 (pre-filter) → full (repair)\n"
                  "at the q=0.95 operating point (mean ± std over 5 seeds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "fig2_component_ablation")


def fig3_tradeoff_curve():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset, color in zip(axes, ["uci", "nhanes_real"], ["#3498db", "#e74c3c"]):
        sw = pd.read_csv(RESULTS_DIR / f"tau_sweep_{dataset}.csv").sort_values("tau_quantile")
        ax2 = ax.twinx()
        l1 = ax.errorbar(sw["tau_quantile"], sw["success_rate_mean"], yerr=sw["success_rate_std"],
                         marker="o", color=color, label="Success rate (%)", capsize=3)
        cvr_plot = sw["decoded_cvr_mean"].fillna(0)
        l2 = ax2.plot(sw["tau_quantile"], cvr_plot, marker="s", color="black", linestyle="--", label="Decoded CVR (%) [0 where 0 successes]")
        ax.set_xscale("log")
        ax.set_xlabel("τ quantile (calibration confidence level, log scale)")
        ax.set_ylabel("Success rate (%)", color=color)
        ax2.set_ylabel("Decoded CVR (%) among successes")
        ax.set_ylim(-2, 105)
        ax2.set_ylim(-2, 105)
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        lines = [l1] + l2
        ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    fig.suptitle("Fig 3 — The safety/success trade-off curve (E2)\n"
                  "Tighter τ (higher confidence) → fewer edges survive → success falls", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, "fig3_tradeoff_curve")


def fig4_constraint_type_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cb_all = pd.read_csv(RESULTS_DIR / "constraint_breakdown.csv")
    cells = ["old_latent_pruning", "dsr_c1"]
    cats = ["pct_immutable", "pct_non_decreasing", "pct_age_horizon", "pct_directional"]
    cat_labels = ["Immutable\n(sex)", "Non-decreasing\n(age)", "Age horizon\n(>3yr/step)", "Directional\n(BP/chol/etc.)"]
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db"]

    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        cb = cb_all[cb_all["dataset"] == dataset].set_index("cell")
        x = np.arange(len(cells))
        width = 0.2
        for i, (cat, lab, col) in enumerate(zip(cats, cat_labels, colors)):
            vals = [cb.loc[c, cat] if c in cb.index and cat in cb.columns and pd.notna(cb.loc[c, cat]) else 0 for c in cells]
            ax.bar(x + i * width - 1.5 * width, vals, width, label=lab, color=col, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[c] for c in cells])
        ax.set_ylabel("% of successful paths with this\nviolation type (hop-wise check)")
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Fig 4 — Which constraint type breaks, by method (hop-wise decoded check)\n"
                  "Note: this is per-HOP, not the same computation as Table 1's endpoint decoded-CVR", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    savefig(fig, "fig4_constraint_breakdown")


def fig5_abstention_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    colors = {"pct_certified_infeasible": "#2ecc71", "pct_confirmed_blindspot": "#f39c12", "pct_unresolved": "#e74c3c"}
    cat_labels = {"pct_certified_infeasible": "Certified\ninfeasible", "pct_confirmed_blindspot": "Blindspot\n(densification-resolved)",
                 "pct_unresolved": "Unresolved\n(target exists, method fails)"}

    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        rows = []
        for cell in ["dsr_c1", "dsr_full"]:
            path = RESULTS_DIR / f"{dataset}_{cell}_blindspot_summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            rows.append({"cell": cell, **{c: df[c].mean() for c in colors}, **{c + "_std": df[c].std() for c in colors}})
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            continue
        summ = pd.DataFrame(rows).set_index("cell")
        x = np.arange(len(summ))
        bottom = np.zeros(len(summ))
        for cat, col in colors.items():
            vals = summ[cat].values
            errs = summ[cat + "_std"].values
            ax.bar(x, vals, bottom=bottom, color=col, edgecolor="black", label=cat_labels[cat], yerr=errs, capsize=3)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 4:
                    ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold")
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[c] for c in summ.index])
        ax.set_ylabel("% of abstentions")
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        ax.set_ylim(0, 100)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=3, fontsize=8)
    fig.suptitle("Fig 5 — Abstention decomposition under DSR (E3)\n"
                  "DSR-full introduces a new failure mode: repair-driven, densification-proof 'unresolved' abstentions", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, "fig5_abstention_decomposition")


def fig6_decoder_dependence():
    dd = pd.read_csv(RESULTS_DIR / "decoder_dependence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cells = ["old_latent_pruning", "dsr_c1"]
    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        sub = dd[dd["dataset"] == dataset]
        x = np.arange(len(cells))
        width = 0.35
        for i, decoder in enumerate(["v2", "v3"]):
            s = sub[sub["decoder"] == decoder].set_index("cell")
            vals = [s.loc[c, "decoded_cvr_mean"] if c in s.index else np.nan for c in cells]
            errs = [s.loc[c, "decoded_cvr_std"] if c in s.index and pd.notna(s.loc[c, "decoded_cvr_std"]) else 0 for c in cells]
            color = "#95a5a6" if decoder == "v2" else "#2ecc71"
            ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, label=f"{decoder} decoder", color=color, edgecolor="black", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[c] for c in cells])
        ax.set_ylabel("Decoded CVR (%) ↓ better")
        ax.set_title(DISPLAY[dataset], fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    fig.suptitle("Fig 6 — Decoder dependence (E5): does DSR-C1 need a good decoder,\n"
                  "or does it help regardless?", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    savefig(fig, "fig6_decoder_dependence")


def main():
    fig1_motivating_gap()
    fig2_component_ablation()
    fig3_tradeoff_curve()
    fig4_constraint_type_breakdown()
    fig5_abstention_decomposition()
    fig6_decoder_dependence()


if __name__ == "__main__":
    main()
