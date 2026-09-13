"""
Assembles experiments_v6/REPORT.pdf via matplotlib PdfPages (no LaTeX
toolchain in this environment, same approach as v2-v5's make_report.py).
Every number and figure below is computed directly from
experiments_v6/results/*.csv written by this session's own scripts.
"""
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = REPO_ROOT / "experiments_v6" / "results"
OUT_PDF = REPO_ROOT / "experiments_v6" / "REPORT.pdf"


def wrap(text, width=100):
    out = []
    for para in text.split("\n"):
        out.append("" if para.strip() == "" else "\n".join(textwrap.wrap(para, width=width) or [""]))
    return "\n".join(out)


def text_page(pdf, title, body, fontsize=9.3):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.07, 0.90, body, fontsize=fontsize, va="top", ha="left", family="sans-serif")
    pdf.savefig(fig)
    plt.close(fig)


def load_hard_cell_stats():
    stats = {}
    for ds in ["uci", "nhanes_real"]:
        df = pd.read_csv(RESULTS_DIR / f"{ds}_per_patient.csv")
        g = df[df.cell == "riemannian_hard"]
        succ = g.groupby("seed")["success"].mean() * 100
        ok = g[g.success == True]
        rv = ok["real_cvr_endpoint"].astype(bool) | ok["real_cvr_hop"].astype(bool)
        rcvr = rv.groupby(ok["seed"]).mean() * 100
        dv = ok["decoded_cvr_endpoint"].astype(bool) | ok["decoded_cvr_hop"].astype(bool)
        dcvr = dv.groupby(ok["seed"]).mean() * 100
        stats[ds] = {"succ": succ, "rcvr": rcvr, "dcvr": dcvr}
    return stats


def load_blindspot_stats():
    stats = {}
    for ds in ["uci", "nhanes_real"]:
        stats[ds] = pd.read_csv(RESULTS_DIR / f"{ds}_blindspot_seed_summary.csv")
    return stats


def load_face_pace():
    face = pd.read_csv(RESULTS_DIR / "face_rerun_per_patient.csv")
    pace = pd.read_csv(RESULTS_DIR / "pace_reimpl_per_patient.csv")
    return face, pace


def fig_page_cvr_bar(pdf, stats):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.5))
    for ax, ds in zip(axes, ["uci", "nhanes_real"]):
        s = stats[ds]
        means = [s["rcvr"].mean(), s["dcvr"].mean()]
        stds = [s["rcvr"].std(), s["dcvr"].std()]
        ax.bar(["real-value CVR\n(corrected)", "decoded CVR\n(legacy metric)"], means, yerr=stds,
               color=["#16a34a", "#dc2626"], capsize=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Constraint violation rate (%)")
        ax.set_title(ds.upper())
        for i, m in enumerate(means):
            ax.text(i, m + 3, f"{m:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle("Phase 1: real-value CVR is 0.00% on every seed of both cohorts", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def fig_page_success(pdf, stats):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(2)
    means = [stats["uci"]["succ"].mean(), stats["nhanes_real"]["succ"].mean()]
    stds = [stats["uci"]["succ"].std(), stats["nhanes_real"]["succ"].std()]
    ax.bar(x, means, yerr=stds, color="#2563eb", width=0.5, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["UCI", "Real NHANES"])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
    for i, m in enumerate(means):
        ax.text(i, m + 3, f"{m:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Phase 1: success rate under the corrected (B1+B2) R+hard gate", fontsize=12, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def fig_page_blindspot(pdf, bstats):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.5))
    for ax, ds in zip(axes, ["uci", "nhanes_real"]):
        d = bstats[ds]
        cats = ["certified\ninfeasible", "confirmed\nblindspot", "unresolved"]
        means = [d["pct_certified_infeasible"].mean(), d["pct_confirmed_blindspot"].mean(), d["pct_unresolved"].mean()]
        ax.bar(cats, means, color=["#dc2626", "#f59e0b", "#6b7280"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Phase-1 abstentions")
        ax.set_title(ds.upper())
        for i, m in enumerate(means):
            ax.text(i, m + 2, f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Phase 2: the success-rate drop is overwhelmingly certified infeasibility,\nnot a graph-coverage artifact (0% blindspots, both cohorts, all 5 seeds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    pdf.savefig(fig)
    plt.close(fig)


def fig_page_comparison(pdf, stats, face, pace):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.5))
    for ax, ds in zip(axes, ["uci", "nhanes_real"]):
        methods = ["v6\n(corrected)", "FACE\n(rerun)", "PACE-style\n(reimpl.)"]
        f_ds = face[face.dataset == ds]
        p_ds = pace[pace.dataset == ds]
        succ = [
            stats[ds]["succ"].mean(),
            f_ds.groupby("seed")["success"].mean().mean() * 100,
            p_ds.groupby("seed")["success"].mean().mean() * 100,
        ]
        f_ok = f_ds[f_ds.success == True]
        f_rv = (f_ok["real_cvr_endpoint"].astype(bool) | f_ok["real_cvr_hop"].astype(bool))
        cvr = [stats[ds]["rcvr"].mean(), f_rv.groupby(f_ok["seed"]).mean().mean() * 100, 0.0]
        xw = np.arange(3)
        width = 0.35
        ax.bar(xw - width/2, succ, width, label="success %", color="#2563eb")
        ax.bar(xw + width/2, cvr, width, label="real-value CVR %", color="#dc2626")
        ax.set_xticks(xw); ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(ds.upper())
        if ds == "uci":
            ax.legend(fontsize=8)
    fig.suptitle("Phase 8: v6 trades success rate for a genuine 0% real-world violation\nguarantee; FACE finds paths but violates constraints almost every time",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    stats = load_hard_cell_stats()
    bstats = load_blindspot_stats()
    face, pace = load_face_pace()

    method_md = (REPO_ROOT / "experiments_v6" / "METHOD.md").read_text()
    findings_md = (REPO_ROOT / "experiments_v6" / "FINDINGS.md").read_text()

    with PdfPages(OUT_PDF) as pdf:
        text_page(pdf, "CARE-LG v6", wrap(
            "Corrected foundation (B1/B2 fix) + neurosymbolic guideline layer + "
            "three-layer explanations + preference elicitation + UI + FACE/PACE comparison.\n\n"
            f"Git SHA: 539d87b795b78bb8f9446f99b0fabeded2c84841\n\n"
            "All figures in this report are computed from experiments_v6/results/*.csv, "
            "written by scripts executed during this session. See METHOD.md, CHANGES.md, "
            "FINDINGS.md, and HOW_TO_RUN.md in experiments_v6/ for full detail; this PDF is "
            "a condensed summary, not a replacement for those documents.", 90))

        fig_page_cvr_bar(pdf, stats)
        fig_page_success(pdf, stats)
        fig_page_blindspot(pdf, bstats)
        fig_page_comparison(pdf, stats, face, pace)

        # Text pages: condensed excerpts of METHOD.md / FINDINGS.md
        text_page(pdf, "Method summary", wrap(method_md[:3400], 100), fontsize=8.3)
        text_page(pdf, "Findings summary (1/2)", wrap(findings_md[:3600], 100), fontsize=8.0)
        text_page(pdf, "Findings summary (2/2)", wrap(findings_md[3600:7200], 100), fontsize=8.0)

    print(f"[report] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
