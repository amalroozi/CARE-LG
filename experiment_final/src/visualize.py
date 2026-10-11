"""Phase 15: unified architecture, coverage Sankey, inheritance example, recommendation cards, ablation bars, plausibility-vs-validity scatter, fairness tier split, sensitivity curves.

Palette = dataviz reference instance (categorical slots in fixed order; tiers use slots 1-3; CARE-LG is
slot 1 against muted baselines with direct labels). Every figure is written as SVG + PNG.
"""
from __future__ import annotations

import textwrap
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from experiment_final.src.common import FIGURES, new_path  # noqa: E402

TIER_C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
TIER_L = {1: "Tier 1 certified real", 2: "Tier 2 guided synthetic", 3: "Tier 3 certified none"}
OURS_C, MUTED, INK, INK2, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"
DS_LABEL = {"nhanes_pooled": "NHANES 1999-2023 (pooled)", "nhanes": "NHANES 2017-2020", "uci": "UCI Heart", "cardio": "Kaggle Cardio"}
METHOD_LABEL = {"care_lg_tiered": "CARE-LG (tiered)", "tier1_real_only": "Tier 1 only", "tier2_guided_ga": "Tier 2 guided GA",
                "tier2_unguided_ga": "Tier 2 unguided GA", "tier2_riemannian_ng": "Tier 2 Riemannian NG", "face": "FACE", "dice": "DiCE",
                "growing_spheres": "Growing Spheres", "pace_style": "PACE-style", "cchvae": "C-CHVAE", "wachter": "Wachter",
                "reachable_set_lr": "Reachable set (LR)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "svg.fonttype": "none", "figure.dpi": 110})


def save(fig, name: str, store: Dict[str, str]) -> None:
    """SVG + PNG, never overwriting (P15)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    p_svg = new_path(FIGURES / f"{name}.svg")
    fig.savefig(p_svg, bbox_inches="tight", facecolor="white")
    fig.savefig(p_svg.with_suffix(".png"), bbox_inches="tight", dpi=170, facecolor="white")
    plt.close(fig)
    store[name] = str(p_svg)


def _ds_order(ds) -> List[str]:
    return [d for d in ["nhanes_pooled", "nhanes", "uci", "cardio"] if d in set(ds)]


# ---------------------------------------------------------------- 1. architecture

# ---------------------------------------------------------------- 2. coverage

# ---------------------------------------------------------------- 3. inheritance

def inheritance(K, store, concepts=("E11", "J45.909", "N18.9")):
    onto = K.onto
    depth = max(len(onto.ancestors(c)) for c in concepts)
    fig, ax = plt.subplots(figsize=(10.5, 1.2 + 0.62 * depth)); ax.axis("off")
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.1, 0.3 + depth * 0.85 + 0.1)
    fr, ci = K.feature_rules, K.contraindications
    for col, c in enumerate(concepts):
        chain = onto.ancestors(c)
        x = 1.2 + col * 3.3
        for lvl, n in enumerate(chain):
            y = 0.3 + lvl * 0.85
            attached = list(fr[fr.applies_to_icd10_node == n].rule_id) + list(ci[ci.contraindicated_icd10_node == n].rule_id)
            ec = TIER_C[1] if n == c else (TIER_C[2] if attached and n != "ROOT" else MUTED)
            ax.add_patch(FancyBboxPatch((x - 0.95, y - 0.25), 1.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=ec, lw=1.5))
            ax.text(x, y + 0.06, n, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
            ax.text(x, y - 0.13, textwrap.shorten(onto.label.get(n, ""), 30), ha="center", va="center", fontsize=6, color=INK2)
            if attached and n != "ROOT":
                ax.text(x + 1.0, y, "\n".join(attached[:3]), fontsize=6.5, color=TIER_C[2], va="center")
            if lvl:
                ax.annotate("", xy=(x, y - 0.25), xytext=(x, y - 0.6), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.set_title("ICD-10-CM inheritance: patient-specific targets and contraindications\n"
                 "Patient concepts (blue) inherit every rule attached to an ancestor (orange): E11 -> F-SBP-DM at E08-E13; J45.909 -> C-BB-J45 at J45",
                 loc="left", fontsize=9.5, color=INK)
    save(fig, "fig03_ontology_inheritance", store)


# ---------------------------------------------------------------- 4. cards

# ---------------------------------------------------------------- 5. ablations

# ---------------------------------------------------------------- 6. plausibility vs validity

def plaus_validity(meth: pd.DataFrame, store):
    ds = _ds_order(meth.dataset)
    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 4.0))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        m = meth[(meth.dataset == d) & ~meth.method.str.startswith(("tier1_", "tier2_"))]
        for _, r in m.iterrows():
            ours = r.method == "care_lg_tiered"
            ax.scatter(r.valid_and_admissible_mean, r.plaus_lof_mean, s=60 if ours else 34, color=OURS_C if ours else MUTED,
                       edgecolor="white", linewidth=1.5, zorder=3 if ours else 2)
            ax.annotate(METHOD_LABEL.get(r.method, r.method), (r.valid_and_admissible_mean, r.plaus_lof_mean), fontsize=6.4,
                        color=INK if ours else INK2, xytext=(4, 3), textcoords="offset points", weight="bold" if ours else "normal")
        ax.set_title(DS_LABEL[d], loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("valid AND hard-rule admissible (% of high-risk)"); ax.set_xlim(-3, 105)
    axes[0].set_ylabel("LOF outlier score of CFs (lower = more plausible)")
    fig.suptitle("Plausibility vs validity, all methods (CARE-LG highlighted)", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, "fig07_plausibility_vs_validity", store)


# ---------------------------------------------------------------- 7. fairness

# ---------------------------------------------------------------- 8. sensitivity

