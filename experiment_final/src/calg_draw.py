"""Render fig_calg (SVG + PDF) from the cache written by calg.compute. 6.5 in wide (one text column), all text >= 9 pt."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiment_final.src.calg import CACHE, NAME, OUT, UNIT  # noqa: E402

HIGH, LOW, PATH, INK, GREY = "#D55E00", "#009E73", "#00694C", "#111111", "#8a8a8a"   # Okabe-Ito vermillion / bluish green
PRIORITY = ["sex cannot change", "age cannot decrease", "age gap too large"]          # label the rule that best explains a removal
FS = 9
NAME2 = {"cholesterol": "chol."}


def primary(rules: str) -> str:
    rs = rules.split(";")
    for p in PRIORITY:
        if p in rs:
            return p
    return rs[0]


def draw() -> None:
    plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
    d = np.load(CACHE); m = json.loads(CACHE.with_suffix(".json").read_text())
    P, adm, risk, tgt = d["P"], d["adm"], d["risk"], d["is_target"]
    v = P[1:] - P[0]; r = np.linalg.norm(v, axis=1, keepdims=True)
    P = np.vstack([[0.0, 0.0], v / np.maximum(r, 1e-12) * np.sqrt(r / r.max())])      # layout only: square-root radial compression around the patient
    p0, Pn = P[0], P[1:]
    high = d["y_nodes"] == 1                                    # real PCE label: red = 10-year risk >= 7.5%
    ti = int(np.where(tgt)[0][0]); n = len(Pn)
    chg = {c["feature"]: c for c in m["card_changes"]}
    lab = lambda f: f"{NAME2.get(f, NAME[f])} {chg[f]['from']:g}→{chg[f]['to']:g} {UNIT[f]}"  # noqa: E731
    change_txt = "\n".join([lab("systolic_bp"), lab("cholesterol") + ", " + lab("glycemic_hba1c")])
    x0 = d["x0"]
    pat_txt = f"Patient (risk {100 * float(d['r0']):.0f}%)\nSBP {x0[2]:g}, cholesterol {x0[4]:g}, HbA1c {x0[6]:g}"
    tgt_txt = f"Real low-risk patient #{m['target_pid']}\n(risk {100 * m['target_risk']:.0f}%)"

    fig = plt.figure(figsize=(6.5, 8.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[4.3, 0.62, 1.5], hspace=0.2, wspace=0.08, left=0.02, right=0.98, top=0.955, bottom=0.015)
    ax = fig.add_subplot(gs[0, :])
    lo, hi = P.min(0), P.max(0); pad = (hi - lo) * np.array([0.10, 0.10])
    ax.set_xlim(lo[0] - pad[0] * 0.6, hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1] * 3.3, hi[1] + pad[1])

    def base(a, edges="all", s=22):
        for j in range(n):
            if adm[j] and edges in ("all", "kept"):
                a.plot([p0[0], Pn[j, 0]], [p0[1], Pn[j, 1]], color=GREY, lw=0.7, zorder=1)
            elif not adm[j] and edges == "all":
                a.plot([p0[0], Pn[j, 0]], [p0[1], Pn[j, 1]], color=HIGH, lw=0.7, ls=(0, (3, 2)), alpha=0.32 if a is ax else 0.4, zorder=1)
        a.scatter(Pn[~tgt & high, 0], Pn[~tgt & high, 1], s=s, color=HIGH, edgecolor="white", lw=0.4, zorder=3)
        a.scatter(Pn[~tgt & ~high, 0], Pn[~tgt & ~high, 1], s=s, color=LOW, edgecolor="white", lw=0.4, zorder=3)
        a.scatter(*Pn[ti], s=s * 4.5, color=LOW, edgecolor=INK, lw=1.4, zorder=5)
        a.scatter(*p0, marker="*", s=s * 12, color=INK, edgecolor="white", lw=0.5, zorder=6)
        a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_color("#bbbbbb")

    base(ax)
    # one label per rule: the nearest removed edge whose primary rule is new
    seen = {}
    for j in np.argsort(np.linalg.norm(Pn - p0, axis=1)):
        if adm[j]:
            continue
        r = primary(m["blocked_rules"][j])
        if r not in seen and len(seen) < 3:
            seen[r] = int(j)
    for k, (r, j) in enumerate(seen.items()):
        ax.annotate(f"removed: {r}", Pn[j], xycoords="data", xytext=(0.985, 0.95 - 0.085 * k), textcoords="axes fraction", fontsize=FS, color=HIGH, ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=HIGH, lw=0.6), zorder=7,
                    arrowprops=dict(arrowstyle="-", color=HIGH, lw=0.7, alpha=0.8, shrinkA=0, shrinkB=2))
    ax.annotate("", xy=Pn[ti], xytext=p0, arrowprops=dict(arrowstyle="-|>", color=PATH, lw=3.2, shrinkA=9, shrinkB=9, mutation_scale=18), zorder=4)
    mid = (p0 + Pn[ti]) / 2
    ax.annotate(f"{change_txt}\nrisk {100 * float(d['r0']):.0f}% → {100 * m['target_risk']:.0f}%", mid, xytext=(0.70, 0.115), textcoords="axes fraction", fontsize=FS, color=PATH,
                ha="center", weight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PATH, lw=0.8), zorder=8,
                arrowprops=dict(arrowstyle="-", color=PATH, lw=0.8))
    ax.annotate(pat_txt, p0, xytext=(0.02, 0.93), textcoords="axes fraction", fontsize=FS, ha="left", va="center", weight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK, lw=0.8), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=6))
    ax.annotate(tgt_txt, Pn[ti], xytext=(0.03, 0.30), textcoords="axes fraction", fontsize=FS, ha="left", weight="bold", color=PATH, zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PATH, lw=0.8), arrowprops=dict(arrowstyle="-", color=PATH, lw=0.8))
    ax.text(0.01, 0.012, "layout: 2-D PCA of the VAE latent space (distances\nsquare-root compressed); axes carry no meaning", transform=ax.transAxes, fontsize=FS, color="#555555", va="bottom")
    fig.text(0.02, 0.975, f"CALG for one patient: the {n - 1} nearest training patients and the certified target", fontsize=FS + 1, weight="bold", va="center")

    # legend in plain words
    leg = [(Line2D([], [], marker="o", ls="", color=HIGH, ms=7), Line2D([], [], marker="o", ls="", color=LOW, ms=7)),
           Line2D([], [], color=GREY, lw=1.2),
           Line2D([], [], color=HIGH, lw=1.2, ls=(0, (3, 2)), alpha=0.6),
           Line2D([], [], color=PATH, lw=3)]
    names = ["Each dot is a real patient (red = high risk, PCE \u2265 7.5%; green = low risk)", "Lines = similar patients",
             "Red dashed = blocked by a clinical rule", "Green arrow = recommended change"]
    axl = fig.add_subplot(gs[1, :]); axl.axis("off")
    from matplotlib.legend_handler import HandlerTuple
    axl.legend(handles=leg, labels=names, handler_map={tuple: HandlerTuple(ndivide=2, pad=0.4)}, loc="center", ncol=1, frameon=True, fancybox=False, edgecolor="#bbbbbb", fontsize=FS, handlelength=2.2, borderpad=0.5, labelspacing=0.3)

    # three-panel strip: what the rule layer removes
    nk, nr = int(adm.sum()), int((~adm).sum())
    for i, (title, mode) in enumerate([(f"(a) all {n} neighbours", "all"), (f"(b) rules keep {nk}, remove {nr}", "kept"), ("(c) chosen path", "path")]):
        a = fig.add_subplot(gs[2, i])
        a.set_xlim(ax.get_xlim()); a.set_ylim(ax.get_ylim())
        if mode == "path":
            a.scatter(Pn[:, 0], Pn[:, 1], s=7, color="#d0d0d0", zorder=2)
            a.annotate("", xy=Pn[ti], xytext=p0, arrowprops=dict(arrowstyle="-|>", color=PATH, lw=2.6, shrinkA=5, shrinkB=5), zorder=4)
            a.scatter(*Pn[ti], s=70, color=LOW, edgecolor=INK, lw=1.2, zorder=5); a.scatter(*p0, marker="*", s=130, color=INK, zorder=6)
            a.set_xticks([]); a.set_yticks([])
            for sp_ in a.spines.values():
                sp_.set_color("#bbbbbb")
        elif mode == "all":
            base(a, "all", s=7)
        else:
            base(a, "kept", s=7)
            a.scatter(Pn[~adm, 0], Pn[~adm, 1], s=7, color="#d0d0d0", zorder=4)   # removed neighbours greyed out
        a.set_title(title, fontsize=FS, loc="left", pad=3)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"fig_calg.{ext}", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "fig_calg.svg")
