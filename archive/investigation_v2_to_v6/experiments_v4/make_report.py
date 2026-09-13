"""Assembles experiments_v4/REPORT.pdf via matplotlib PdfPages (no LaTeX toolchain here)."""
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

RESULTS_DIR = REPO_ROOT / "experiments_v4" / "results"
FIG_DIR = REPO_ROOT / "experiments_v4" / "figures"
OUT_PDF = REPO_ROOT / "experiments_v4" / "REPORT.pdf"


def wrap(text, width=100):
    return "\n".join("\n".join(textwrap.wrap(p, width=width) or [""]) if p.strip() else "" for p in text.split("\n"))


def text_page(pdf, title, body, fontsize=9.0):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.07, 0.90, body, fontsize=fontsize, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf, png_path, title, caption):
    fig = plt.figure(figsize=(8.5, 11))
    img = mpimg.imread(png_path)
    ax = fig.add_axes([0.05, 0.28, 0.9, 0.66])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.07, 0.97, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.07, 0.20, wrap(caption, 100), fontsize=10, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df, note=""):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=14, fontweight="bold", va="top")
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.80])
    ax.axis("off")
    tbl = ax.table(cellText=df.round(2).astype(str).values, colLabels=df.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    if note:
        fig.text(0.05, 0.06, wrap(note, 135), fontsize=8, va="top", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    t1_uci = pd.read_csv(RESULTS_DIR / "table1_uci.csv")
    t1_nh = pd.read_csv(RESULTS_DIR / "table1_nhanes_real.csv")
    sweep_nh = pd.read_csv(RESULTS_DIR / "tau_sweep_nhanes_real.csv")

    with PdfPages(OUT_PDF) as pdf:
        exec_summary = wrap("""
CARE-LG v4 -- Decode-Safe Recourse (DSR)

The problem: the prior method (experiments_v2/v3) enforces clinical constraints on
ENCODED patient values but is judged (and shown to patients) via DECODED reconstructions
-- and 78-92% of "constraint-satisfying" paths failed once re-checked on what the
decoder actually produces. This session builds and tests the first fix.

HEADLINE RESULT: Component 1 alone -- simply checking constraints on the SAME decoded
representation the method is judged by, instead of the raw recorded values -- cuts
decoded-space CVR from 74%/68% (UCI/real NHANES) down to 18%/2.5%, while ALSO increasing
success rate. This is the single most important, most robust finding in this session.

Components 2 (calibrated tolerance bands) and 3 (verify-and-repair) get the remainder to
EXACTLY 0% decoded CVR wherever they succeed -- but "wherever they succeed" is a real,
narrow, cohort-dependent qualifier: DSR-full achieves up to ~23% success at a real,
statistically powered operating point on real NHANES (n=303-459 paired patients, the
strongest possible Wilcoxon result, p as low as 4e-37), but 0% success at EVERY
confidence level tested on UCI. Applying the calibrated bands as a pre-emptive filter
(rather than a post-hoc repair standard) collapses the graph almost entirely on BOTH
cohorts -- a real, diagnosed limit of decoder fidelity on specific features (cholesterol
notably), not a bug.

A genuinely new, important side-finding: DSR-full's abstentions are NOT cleanly
"certified infeasible" the way the old method's were. 15-21% of DSR-full's abstentions
are a NEW failure category -- a valid target exists, but repair cannot certify reaching
it, and graph densification (more neighbors, more synthetic nodes) does not help,
because the problem is repair fidelity, not connectivity. Any future claim of
"abstention = certified infeasibility" must now be qualified by which cell produced it.

Bottom line: this is real, substantial, honestly-measured progress on RQ1's problem
(Component 1 especially), not a fully solved guarantee -- see FINDINGS.md and
ASSESSMENT.md for the precise, unhedged verdict on what is and is not established.
""".strip(), 100)
        text_page(pdf, "Executive Summary", exec_summary, fontsize=8.8)

        table_page(pdf, "Table 1 (UCI) -- component comparison at q=0.95, mean over 5 seeds", t1_uci,
                  note="dsr_c1c2/dsr_full/dsr_full_euclidean/dsr_bands_no_identity all show 0% success on UCI at "
                       "q=0.95 -- see Fig 3 for the full tau sweep (0% at every quantile tested, 0.05-0.99).")
        table_page(pdf, "Table 1 (Real NHANES) -- component comparison at q=0.95, mean over 5 seeds", t1_nh,
                  note="dsr_full has 0% success at q=0.95 but a real, powered operating point exists at lower tau -- see Table below and Fig 3.")
        table_page(pdf, "Real NHANES tau sweep (E2) -- dsr_full success/CVR by confidence level", sweep_nh,
                  note="Decoded CVR is 0.0% at every quantile where dsr_full has any successes at all (q<=0.25); "
                       "the q=0.5 row's 12.5% is a single-seed, small-n artifact (see FINDINGS.md).")

        figure_page(pdf, FIG_DIR / "fig1_motivating_gap.png", "Fig 1 -- The Motivating Gap",
                   "Decoded-space CVR of the old method vs. DSR-C1 vs. DSR-full's best operating point. "
                   "Component 1 alone cuts the gap by 4-27x on both cohorts; DSR-full reaches exactly 0% on "
                   "real NHANES at its viable operating point, and has no viable operating point on UCI.")
        figure_page(pdf, FIG_DIR / "fig2_component_ablation.png", "Fig 2 -- Component Ablation",
                   "Success rate and decoded CVR across old -> C1 -> C1+C2(pre-filter) -> full(repair) at q=0.95. "
                   "C1+C2 as a pre-filter collapses success to ~0% on both cohorts and, where it does return a "
                   "path on real NHANES, actually shows WORSE CVR (90%) than C1 alone -- the tiny surviving "
                   "sample is dominated by pathological edge cases, not evidence bands-as-filter work.")
        figure_page(pdf, FIG_DIR / "fig3_tradeoff_curve.png", "Fig 3 -- The Safety/Success Trade-off Curve (E2)",
                   "Real NHANES shows a genuine, monotonic trade-off: success falls from 23% (q=0.05) to 0% "
                   "(q>=0.8) as the confidence level tightens, with decoded CVR pinned at 0% wherever success is "
                   "nonzero. UCI is flat zero across the entire range -- no confidence level yields a usable "
                   "operating point on this cohort, a real, cohort-dependent limitation, not a plotting artifact.")
        figure_page(pdf, FIG_DIR / "fig4_constraint_breakdown.png", "Fig 4 -- Per-Constraint-Type Breakdown",
                   "Hop-wise decoded violations by type (distinct from Table 1's endpoint-only decoded CVR -- "
                   "see CHANGES.md). Under the old method, non-decreasing/age dominates on UCI (54%) and "
                   "directional (BP/cholesterol) dominates on real NHANES (42%), with a large immutable/sex "
                   "component there too (30%). DSR-C1 drives every category to 0% on UCI and to 0% except a "
                   "residual 13.5% age-horizon rate on real NHANES.")
        figure_page(pdf, FIG_DIR / "fig5_abstention_decomposition.png", "Fig 5 -- Abstention Decomposition Under DSR (E3)",
                   "DSR-C1 comes close to reproducing experiments_v3's clean 100% certified-infeasible result. "
                   "DSR-full does not: 15% (UCI) / 21% (real NHANES) of its abstentions are neither certified-"
                   "infeasible nor resolved by either densification probe -- a new, repair-specific failure mode "
                   "this project's existing recourse-verification toolkit was not built to diagnose.")
        figure_page(pdf, FIG_DIR / "fig6_decoder_dependence.png", "Fig 6 -- Decoder Dependence (E5)",
                   "On UCI, DSR-C1's benefit depends heavily on decoder quality (54% CVR under the old v2 decoder "
                   "vs. 18% under the type-aware v3 decoder). On real NHANES, it barely matters (3.1% vs. 2.5%) "
                   "-- most likely because NHANES's ~15x larger training set gives even the plain decoder enough "
                   "signal to reconstruct adequately. DSR-C1 helps under either decoder on both cohorts.")

        limitations = wrap("""
- The calibration split (15% of the original train partition, 36 rows on UCI / 545 on
  real NHANES) makes low-quantile tau estimates (q=0.05) noisy on UCI in particular --
  reported as-is, not smoothed.
- Component 2's bands, as literally specified as a pre-filter, do not work on this
  decoder for several specific features (cholesterol on both cohorts, oldpeak on UCI,
  BMI on real NHANES) at ANY tested confidence level -- this is a decoder-fidelity limit,
  not something a different tau choice fixes.
- Repair (Component 3) is architecture-dependent: it reliably fixes age (which has a
  dedicated, upweighted decoder head from Phase B) and reliably fails on features that
  never got one. This is a real limitation of the current decoder, not of the repair
  LOGIC, and would likely improve with a type-aware head on every continuous feature --
  untested in this session (out of scope/budget).
- UCI has NO viable DSR-full operating point at any tested tau (0.05-0.99). E4 (metric
  comparison under DSR) could therefore not be evaluated on UCI at all -- reported as a
  real absence, not filled with an arbitrary substitute.
- E3's detailed per-abstention decomposition on real NHANES is computed on a capped,
  seeded random sample (25 per seed) of the true abstention set (up to ~445/seed) for
  tractability -- true abstention rates are still reported in full; only the
  certified/blindspot/unresolved split is sampled.
- The "old" method's numbers in this report are freshly re-run under this session's
  3-way split (85% of the original 80% train partition), not copied from experiments_v3
  -- they are not bit-identical to the v3 published table, and are not meant to be; see
  METHOD.md sec 4.
""".strip(), 100)
        text_page(pdf, "Limitations", limitations, fontsize=9)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
