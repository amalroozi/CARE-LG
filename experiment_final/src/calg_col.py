"""fig_calg_col: the same figure as fig_calg (same verified cache, same patient) redrawn at IEEE one-column width (3.5 in)
so that all text is 9 pt AT PRINT SIZE. fig_calg.{svg,pdf} (6.5 in, for the HTML report) is left untouched."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiment_final.src.calg import CACHE, OUT  # noqa: E402
from experiment_final.src.calg_draw import GREY, HIGH, INK, LOW, PATH, primary  # noqa: E402

FS = 9


def draw() -> None:
    plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    d = np.load(CACHE); m = json.loads(CACHE.with_suffix(".json").read_text())
    P, adm, tgt = d["P"], d["adm"], d["is_target"]
    v = P[1:] - P[0]; r = np.linalg.norm(v, axis=1, keepdims=True)
    P = np.vstack([[0.0, 0.0], v / np.maximum(r, 1e-12) * np.sqrt(r / r.max())])      # layout only: square-root radial compression
    p0, Pn = P[0], P[1:]
    high = d["y_nodes"] == 1
    ti = int(np.where(tgt)[0][0]); n = len(Pn)
    chg = {c["feature"]: c for c in m["card_changes"]}
    x0 = d["x0"]; r0, rt = 100 * float(d["r0"]), 100 * m["target_risk"]
    fig = plt.figure(figsize=(3.5, 5.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.1, 1.3, 0.95], hspace=0.2, wspace=0.07, left=0.015, right=0.985, top=0.995, bottom=0.01)
    ax = fig.add_subplot(gs[0, :])
    lo, hi = P.min(0), P.max(0); pad = (hi - lo) * 0.08
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1] * 1.5, hi[1] + pad[1])

    def base(a, edges, s):
        for j in range(n):
            if adm[j] and edges in ("all", "kept"):
                a.plot([p0[0], Pn[j, 0]], [p0[1], Pn[j, 1]], color=GREY, lw=0.7, zorder=1)
            elif not adm[j] and edges == "all":
                a.plot([p0[0], Pn[j, 0]], [p0[1], Pn[j, 1]], color=HIGH, lw=0.6, ls=(0, (3, 2)), alpha=0.35, zorder=1)
        a.scatter(Pn[~tgt & high, 0], Pn[~tgt & high, 1], s=s, color=HIGH, edgecolor="white", lw=0.3, zorder=3)
        a.scatter(Pn[~tgt & ~high, 0], Pn[~tgt & ~high, 1], s=s, color=LOW, edgecolor="white", lw=0.3, zorder=3)
        a.scatter(*Pn[ti], s=s * 4, color=LOW, edgecolor=INK, lw=1.2, zorder=5)
        a.scatter(*p0, marker="*", s=s * 9, color=INK, edgecolor="white", lw=0.4, zorder=6)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#bbbbbb")

    base(ax, "all", 14)
    seen = {}
    for j in np.argsort(np.linalg.norm(Pn - p0, axis=1)):
        if adm[j]:
            continue
        rr = primary(m["blocked_rules"][j])
        if rr not in seen and len(seen) < 3:
            seen[rr] = int(j)
    for k, (rr, j) in enumerate(seen.items()):
        ax.annotate(f"removed: {rr}", Pn[j], xytext=(0.99, 0.925 - 0.088 * k), textcoords="axes fraction", fontsize=FS, color=HIGH, ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=HIGH, lw=0.5), zorder=7,
                    arrowprops=dict(arrowstyle="-", color=HIGH, lw=0.6, alpha=0.8, shrinkA=0, shrinkB=1))
    ax.annotate("", xy=Pn[ti], xytext=p0, arrowprops=dict(arrowstyle="-|>", color=PATH, lw=2.6, shrinkA=7, shrinkB=7, mutation_scale=13), zorder=4)
    mid = (p0 + Pn[ti]) / 2
    ax.annotate(f"SBP {chg['systolic_bp']['from']:g}→{chg['systolic_bp']['to']:g} mmHg\nchol. {chg['cholesterol']['from']:g}→{chg['cholesterol']['to']:g} mg/dL\n"
                f"HbA1c {chg['glycemic_hba1c']['from']:g}→{chg['glycemic_hba1c']['to']:g} %\nrisk {r0:.0f}% → {rt:.0f}%", mid, xytext=(0.99, 0.015), textcoords="axes fraction",
                fontsize=FS, color=PATH, ha="right", va="bottom", weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PATH, lw=0.7), zorder=8,
                arrowprops=dict(arrowstyle="-", color=PATH, lw=0.7))
    ax.annotate(f"Patient (risk {r0:.0f}%)\nSBP {x0[2]:g}, chol. {x0[4]:g}", p0, xytext=(0.01, 0.93), textcoords="axes fraction", fontsize=FS, ha="left", va="top", weight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK, lw=0.7), arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=5))
    ax.annotate(f"Real low-risk patient\n#{m['target_pid']} (risk {rt:.0f}%)", Pn[ti], xytext=(0.01, 0.30), textcoords="axes fraction", fontsize=FS, ha="left", va="top", weight="bold", color=PATH,
                zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PATH, lw=0.7), arrowprops=dict(arrowstyle="-", color=PATH, lw=0.7))
    leg = [(Line2D([], [], marker="o", ls="", color=HIGH, ms=6), Line2D([], [], marker="o", ls="", color=LOW, ms=6)), Line2D([], [], color=GREY, lw=1.2),
           Line2D([], [], color=HIGH, lw=1.2, ls=(0, (3, 2)), alpha=0.6), Line2D([], [], color=PATH, lw=3)]
    names = ["Each dot is a real patient\n(red = high risk, green = low)", "Lines = similar patients", "Red dashed = blocked by a\nclinical rule", "Green arrow = recommended\nchange"]
    axl = fig.add_subplot(gs[1, :]); axl.axis("off")
    axl.legend(handles=leg, labels=names, handler_map={tuple: HandlerTuple(ndivide=2, pad=0.4)}, loc="center", ncol=1, frameon=True, fancybox=False, edgecolor="#bbbbbb",
               fontsize=FS, handlelength=2.0, borderpad=0.35, labelspacing=0.25)
    nk, nr = int(adm.sum()), int((~adm).sum())
    for i, (title, mode) in enumerate([(f"(a) all {n}", "all"), (f"(b) {nk} kept", "kept"), ("(c) path", "path")]):
        a = fig.add_subplot(gs[2, i]); a.set_xlim(ax.get_xlim()); a.set_ylim(ax.get_ylim())
        if mode == "path":
            a.scatter(Pn[:, 0], Pn[:, 1], s=4, color="#d0d0d0", zorder=2)
            a.annotate("", xy=Pn[ti], xytext=p0, arrowprops=dict(arrowstyle="-|>", color=PATH, lw=2.0, shrinkA=3, shrinkB=3), zorder=4)
            a.scatter(*Pn[ti], s=40, color=LOW, edgecolor=INK, lw=1.0, zorder=5); a.scatter(*p0, marker="*", s=70, color=INK, zorder=6)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#bbbbbb")
        else:
            base(a, mode, 4)
            if mode == "kept":
                a.scatter(Pn[~adm, 0], Pn[~adm, 1], s=4, color="#d0d0d0", zorder=4)
        a.set_title(title, fontsize=FS, loc="left", pad=2)
    fig.savefig(OUT / "fig_calg_col.pdf", facecolor="white"); plt.close(fig)
    print("wrote", OUT / "fig_calg_col.pdf")


if __name__ == "__main__":
    draw()
