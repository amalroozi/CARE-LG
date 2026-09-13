"""Assembles experiments_v5/REPORT.pdf via matplotlib PdfPages."""
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

RESULTS_DIR = REPO_ROOT / "experiments_v5" / "results"
FIG_DIR = REPO_ROOT / "experiments_v5" / "figures"
OUT_PDF = REPO_ROOT / "experiments_v5" / "REPORT.pdf"


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
    fig.text(0.05, 0.95, title, fontsize=13, fontweight="bold", va="top")
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.78])
    ax.axis("off")
    tbl = ax.table(cellText=df.astype(str).values, colLabels=df.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.4)
    if note:
        fig.text(0.05, 0.06, wrap(note, 135), fontsize=8, va="top", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    t1_uci = pd.read_csv(RESULTS_DIR / "table1_uci.csv").round(3)
    t1_nh = pd.read_csv(RESULTS_DIR / "table1_nhanes_real.csv").round(3)
    wide_uci = pd.read_csv(RESULTS_DIR / "table1_wide_uci.csv")
    wide_nh = pd.read_csv(RESULTS_DIR / "table1_wide_nhanes_real.csv")

    with PdfPages(OUT_PDF) as pdf:
        exec_summary = wrap("""
CARE-LG v5 -- Exact-Guarantee Decoder (EGD)

The premise: v4's DSR (calibrated bands + repair) partially fixed decoded-space
constraint violations but is a STATISTICAL patch -- it collapses on features whose
decoder error exceeds any workable tolerance. This session replaces statistics with
architecture: a decoder whose output layer makes immutability and monotonicity
violations STRUCTURALLY IMPOSSIBLE, regardless of training.

PHASE 0 FINDING (checked before building anything): exact/architectural constraint
guarantees are NOT novel -- CounterFlowNet (2602.17244) already achieves this, via
discrete action masking, in a completely different (non-VAE) architecture. It also does
NOT distinguish true infeasibility from method failure -- the ONE thing this project's
certified-infeasibility machinery already does well. The novelty claim was narrowed
accordingly BEFORE Phase 1 began (NOVELTY_CLAIM.md): exact guarantees IN a VAE-latent-
graph architecture, PAIRED WITH certified-infeasibility analysis.

HEADLINE RESULT: the guarantee is real and unconditional -- decoded-space CVR is
EXACTLY 0.0% for every successful EGD path, verified on the full training set and every
search-returned path, both cohorts, zero exceptions. But this is bought at a SEVERE,
honestly-measured cost: success rate is wildly seed-variable on UCI (0-90%+ range) and
collapses to near-zero on real NHANES (1.0% vs. the old method's 45.5%); where EGD DOES
succeed, its paths are significantly LESS plausible and require ~50x MORE clinical
effort than the old method's (p<0.001, both differences, both cohorts). A genuine,
mechanistically-explained new finding: the Riemannian-vs-Euclidean metric choice, null
in every prior session, now matters a great deal (31% vs 0% success on UCI) because
EGD's guarantee accumulates change per hop and Euclidean systematically picks shorter
paths. Certified infeasibility also does NOT survive cleanly: 40-42% of EGD's
abstentions are targets that genuinely exist but the mechanism cannot reach, not
provably nonexistent ones -- and densification never resolves a single one, confirming
this is a mechanism limit, not a connectivity one.

Bottom line: EGD proves its narrow technical claim (architecturally exact guarantees are
achievable here) but is not a practical improvement over the old method or v4's DSR --
see FINDINGS.md and ASSESSMENT.md for the full, unsoftened verdict.
""".strip(), 100)
        text_page(pdf, "Executive Summary", exec_summary, fontsize=8.6)

        table_page(pdf, "Table 1 (UCI) -- EGD vs. old, this session's own controlled rerun", t1_uci)
        table_page(pdf, "Table 1 (Real NHANES) -- EGD vs. old, this session's own controlled rerun", t1_nh)
        table_page(pdf, "Wide comparison (UCI) -- includes v3/v4 numbers, clearly labeled as reused", wide_uci,
                  note="Rows marked 'REUSED from experiments_vN, NOT rerun' are NOT paired with this session's "
                       "patients/seeds and are context only, not a controlled comparison.")
        table_page(pdf, "Wide comparison (Real NHANES)", wide_nh,
                  note="Rows marked 'REUSED...' are context only, not a controlled comparison.")

        figure_page(pdf, FIG_DIR / "fig1_motivating_gap.png", "Fig 1 -- The Exact Guarantee, Verified",
                   "EGD's decoded CVR is exactly 0.0% wherever it succeeds, on both cohorts -- the one claim in "
                   "this session that is unconditionally confirmed, with zero exceptions across the full 5-seed grid.")
        figure_page(pdf, FIG_DIR / "fig2_cost_of_guarantee.png", "Fig 2 -- The Cost of the Guarantee",
                   "EGD vs. the old method: success rate falls (dramatically on real NHANES), plausibility (KDE) "
                   "is significantly worse, and clinical effort is significantly higher -- all paired tests "
                   "p<0.001 on both cohorts. This is a real, severe, honestly-measured trade-off.")
        figure_page(pdf, FIG_DIR / "fig3_metric_ablation.png", "Fig 3 -- Riemannian vs. Euclidean: A Reversal",
                   "Every prior session found no meaningful metric difference. Under EGD, Euclidean is nearly "
                   "unusable (0% UCI success) because it systematically selects shorter paths, and EGD's "
                   "guarantee-preserving construction accumulates change per hop -- shorter paths structurally "
                   "cannot accumulate enough change. Confirmed directly by inspecting path lengths per candidate.")
        figure_page(pdf, FIG_DIR / "fig4_seed_variability.png", "Fig 4 -- UCI Seed Variability",
                   "EGD's UCI success rate swings from near-0% to over 90% depending on seed -- the widest "
                   "seed-to-seed variation measured anywhere in this project's five-session history.")
        figure_page(pdf, FIG_DIR / "fig5_abstention_decomposition.png", "Fig 5 -- Abstention Decomposition",
                   "Zero blindspots resolved by densification on either cohort, at any seed -- confirming "
                   "'unresolved' abstentions (40-42%) are a mechanism limit (residual magnitude), not a graph "
                   "connectivity problem more neighbors or synthetic nodes could fix.")

        limitations = wrap("""
- Real NHANES's near-total EGD failure (1.0% success) means most of this report's real-
  NHANES findings describe FAILURE MODES, not a working system -- stated plainly, not
  minimized.
- Training-regime selection (self / k-NN / random pairing) was empirical, not derived
  from theory; a fourth regime (e.g. a curriculum mixing near and far pairs) was not
  tried and might do better, particularly on real NHANES -- an open, unexplored
  direction, not evidence the problem is unsolvable.
- UCI's extreme seed variance (std=37 points on a mean of 31%) means any single-seed
  UCI success-rate claim for EGD is nearly meaningless; only the 5-seed mean±std should
  ever be cited.
- CounterFlowNet was read closely (Phase 0) but not reimplemented as a controlled
  baseline -- any comparison to it in this report is qualitative, from their paper's
  own numbers on different (financial) datasets, not a head-to-head rerun.
- The wide comparison table's DiCE/FACE/REVISE/DSR-full rows are reused from prior
  sessions under a DIFFERENT train/calibration split protocol in DSR's case -- they are
  context, not validly paired with this session's own EGD/old numbers.
- Real NHANES's blindspot decomposition is computed on a 60-patient random sample per
  seed (true abstention rate ~97-100%, far more than 60) -- the TRUE rate is exact and
  reported in full; only the certified/blindspot/unresolved split is a sample estimate.
""".strip(), 100)
        text_page(pdf, "Limitations", limitations, fontsize=9)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
