"""Figures for paper/main_final.tex, written as NEW files in results/figures/ (nothing overwritten):
fig_ablation_grid (2 x 3 panels, one metric per ablation), fig_blocking_rules and a title-free
copy of the architecture figure, plus the ICD-10-CM inheritance figure. All numbers read from results/tables.

  .venv/bin/python -m experiment_final.src.final_figs
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment_final.src import build_results  # noqa: E402
from experiment_final.src.build_results import F4, T4  # noqa: E402

BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#999999", "#222222"      # Okabe-Ito


def _out(fig, name: str) -> None:
    F4.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(F4 / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    fig.savefig(F4 / f"{name}.png", bbox_inches="tight", dpi=200, facecolor="white"); plt.close(fig)


def _bars(ax, labels, vals, colors, fmt="{:.0f}", top=None, rot=0):
    x = np.arange(len(vals)); ax.bar(x, vals, color=colors, width=0.62)
    top = top or max(vals) * 1.25 or 1
    for xi, v in zip(x, vals):
        ax.text(xi, v + top * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=rot); ax.set_ylim(0, top)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, labelsize=7.5)


def ablation_grid() -> None:
    """2 x 3 grid, one metric per ablation question. Pooled NHANES 1999-2023 (15,002 high-risk patients, 5 seeds) except panel (f)."""
    a = pd.read_csv(T4 / "ablations.csv"); a = a[(a.dataset == "nhanes_pooled") & (a.status == "run")].set_index("arm")
    t2 = pd.read_csv(T4 / "tier2_variants.csv"); t2 = t2[t2.dataset == "nhanes_pooled"].set_index("method")
    th = pd.read_csv(T4 / "sensitivity_thresholds.csv"); th = th[th.dataset == "nhanes_pooled"]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.7), gridspec_kw={"hspace": 0.75, "wspace": 0.6, "width_ratios": [1, 1.5, 1]})
    ttl = lambda ax, s: ax.set_title(s, loc="left", fontsize=8, color=INK, fontweight="bold")  # noqa: E731

    ax = axs[0, 0]; ttl(ax, "(a) rule layer")
    _bars(ax, ["full", "no\nontol.", "ontol. as\nfeatures"], [a.loc[k, "contraindicated_recs_total"] for k in ("main", "no_ontology", "ontology_as_features")], [BLUE, GREY, GREY], top=105)
    ax.set_ylabel("contraindicated cards", fontsize=7.5)

    sub = axs[0, 1].get_subplotspec().subgridspec(1, 2, wspace=0.9); axs[0, 1].remove()
    for k, (spec, col, lab, fmt, top) in enumerate(((sub[0], "covered_mean", "coverage (%)", "{:.1f}", 125),
                                                     (sub[1], "soft_violations_true_rules_mean", "violations\nper card", "{:.2f}", 3.6))):
        ax = fig.add_subplot(spec); _bars(ax, ["full", "vetoes"], [a.loc["main", col], a.loc["soft_as_hard", col]], [BLUE, GREY], fmt=fmt, top=top)
        ax.set_ylabel(lab, fontsize=7.5)
        if k == 0:
            ax.set_title("(b) soft rules as vetoes", loc="left", fontsize=8, color=INK, fontweight="bold")

    ax = axs[0, 2]; ttl(ax, "(c) source-relative admissibility")
    _bars(ax, ["full", "SRA off"], [a.loc["main", "real_violation_paths_total"], a.loc["no_sra", "real_violation_paths_total"]], [BLUE, GREY], top=135)
    ax.set_ylabel("Tier-1 paths with\nreal-value violation", fontsize=7.5)

    ax = axs[1, 0]; ttl(ax, "(d) escalation order")
    xs = np.arange(2); wd = 0.36
    for off, k, lab, col in ((-wd / 2, "main", "escalation on", BLUE), (wd / 2, "no_escalation", "off", GREY)):
        vals = [a.loc[k, "esc_ok_mean"], a.loc[k, "over_mean"]]
        ax.bar(xs + off, vals, wd, color=col, label=lab)
        for xi, vi in zip(xs + off, vals):
            ax.text(xi, vi + 1.5, f"{vi:.0f}" if vi in (0, 100) else f"{vi:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(["appropriate\nfirst rec.", "over-\ntreatment"], fontsize=7.5); ax.set_ylim(0, 145); ax.set_ylabel("% of cards", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2, handlelength=1, columnspacing=0.8, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, labelsize=7.5)

    ax = axs[1, 1]; ttl(ax, "(e) Tier-2 search")
    ks = ["tier2_guided_ga", "tier2_unguided_ga", "tier2_riemannian_ng"]
    _bars(ax, ["guided\nGA", "unguided\nGA", "Riemann.\ngrad."], [t2.loc[k, "valid_f_mean"] for k in ks], [BLUE, GREY, GREY], fmt="{:.1f}", top=122)
    ax.set_ylabel("valid under oracle (%)", fontsize=7.5)

    ax = axs[1, 2]; ttl(ax, "(f) target definition (seed 0)")
    th = th.sort_values("high")
    for flag, lab, col, mk in ((True, "age-stratified", BLUE, "o"), (False, "global", GREY, "s")):
        d = th[th.age_stratified == flag]
        ax.plot(range(len(d)), d.tier1_coverage_pct, color=col, marker=mk, ms=4, lw=1.4, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h:.2f}/\n{l:.2f}" for h, l in th[th.age_stratified].loc[:, ["high", "low"]].itertuples(index=False)], fontsize=7)
    ax.set_xlabel("high / low threshold", fontsize=7.5, labelpad=1); ax.set_ylabel("Tier 1 coverage (%)", fontsize=7.5); ax.set_ylim(45, 70)
    ax.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.2, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, labelsize=7.5)
    _out(fig, "fig_ablation_grid")


def blocking() -> None:
    """Which rule removes how many candidate targets, and which rule is the closest miss for the Tier-1 failures (pooled NHANES, all seeds)."""
    from experiment_final.src.calg import rule_words
    b = pd.read_csv(T4 / "blocking_rules.csv"); b = b[b.dataset == "nhanes_pooled"]
    def words(rule):
        k, f = rule.split(":")
        return rule_words(k, f)
    s = b[b.mean_share_of_targets_blocked.notna()].sort_values("mean_share_of_targets_blocked")
    m = b[b.n_closest_miss.notna()].assign(rule=lambda d: d.rule.str.replace("closest_miss:", "", regex=False)).sort_values("n_closest_miss")
    n_fail = int(b.n_failures.iloc[0])
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"wspace": 1.05})
    ax1.barh([words(r) for r in s.rule], 100 * s.mean_share_of_targets_blocked, color=BLUE, height=0.62)
    for yi, v in enumerate(100 * s.mean_share_of_targets_blocked):
        ax1.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=7.5)
    ax1.set_xlim(0, 118); ax1.set_xlabel("% of low-risk targets blocked (mean over Tier-1 failures)", fontsize=7.5); ax1.set_title("(a) how often a rule blocks a target", loc="left", fontsize=8, fontweight="bold")
    ax2.barh([words(r) for r in m.rule], m.n_closest_miss, color=ORANGE, height=0.62)
    for yi, v in enumerate(m.n_closest_miss):
        ax2.text(v + 30, yi, f"{int(v):,}", va="center", fontsize=7.5)
    ax2.set_xlim(0, m.n_closest_miss.max() * 1.18); ax2.set_xlabel(f"Tier-1 failures ({n_fail:,}) by their closest-miss rule", fontsize=7.5)
    ax2.set_title("(b) the rule that blocks the nearest target", loc="left", fontsize=8, fontweight="bold")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, labelsize=7.5)
    _out(fig, "fig_blocking_rules")


def notitle() -> None:
    """Architecture figure without the in-image title (the paper caption carries the title)."""
    store = {}
    orig_title = matplotlib.axes.Axes.set_title
    matplotlib.axes.Axes.set_title = lambda self, *a, **k: None
    saved = build_results._save
    build_results._save = lambda fig, name, s: (fig.savefig(F4 / f"{name}_notitle.svg", bbox_inches="tight", facecolor="white"),
                                     fig.savefig(F4 / f"{name}_notitle.pdf", bbox_inches="tight", facecolor="white"),
                                     fig.savefig(F4 / f"{name}_notitle.png", bbox_inches="tight", dpi=200, facecolor="white"), plt.close(fig))
    try:
        build_results.fig_architecture(store)
    finally:
        matplotlib.axes.Axes.set_title = orig_title; build_results._save = saved


def inheritance() -> None:
    """ICD-10-CM inheritance example (fig03_ontology_inheritance, used by the paper) from the current knowledge tables."""
    from experiment_final.src.knowledge_layer import load_knowledge
    from experiment_final.src import visualize as V
    old, store = V.FIGURES, {}
    with tempfile.TemporaryDirectory() as td:
        V.FIGURES = Path(td)
        try:
            V.inheritance(load_knowledge(), store)
        finally:
            V.FIGURES = old
        build_results._copy_pair(Path(store["fig03_ontology_inheritance"]), "fig03_ontology_inheritance")


if __name__ == "__main__":
    ablation_grid(); blocking(); notitle(); inheritance(); print("ok")
