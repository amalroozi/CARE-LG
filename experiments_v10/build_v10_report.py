"""
Builds experiments_v10/v10_report.html: the full CARE-LG architecture run
top-to-bottom on the third dataset (Kaggle Cardiovascular Disease, see
experiments_v10/data_cardio.py). Structure: 5 real patient end-to-end
walkthroughs first, then headline numbers, then comparison charts/tables
against UCI Heart and Real NHANES (this project's other two datasets).
Auto-opens.

Usage:
  .venv/bin/python -m experiments_v10.build_v10_report
"""
import json
import re
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from experiments_v10.run_v10_cardio import run_dataset_seed_v10, N_SUBSAMPLE, K_NEIGHBORS
from experiments_v10.single_patient_v10 import run_single_patient_v10
from src.reporting.report_html import build_success_report, build_abstain_report, CSS as PATIENT_CSS

V10_DIR = REPO_ROOT / "experiments_v10" / "results"
TABLES_DIR = REPO_ROOT / "results" / "tables"
_SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'
N_PATIENT_SAMPLES = 5
REPORT_SEED = 0


def bar_svg(labels, series, width=560, height=240, fmt="{:.1f}%", ymax=100, colors=None):
    names = list(series.keys())
    colors = colors or ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]
    n = len(labels)
    group_w = (width - 60) / n
    bar_w = group_w / (len(names) + 0.6)
    legend_h = 22
    bars = []
    for gi, lab in enumerate(labels):
        gx = 40 + gi * group_w
        for si, name in enumerate(names):
            v = series[name][gi]
            h = (v / ymax) * (height - 55) if ymax else 0
            x = gx + si * bar_w + 6
            y = legend_h + height - 35 - h
            c = colors[si % len(colors)]
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" height="{h:.1f}" fill="{c}" rx="2"/>')
            bars.append(f'<text x="{x+(bar_w-4)/2:.1f}" y="{y-4:.1f}" font-size="9.5" text-anchor="middle" fill="#111">{fmt.format(v)}</text>')
        bars.append(f'<text x="{gx+group_w/2:.1f}" y="{legend_h+height-16:.1f}" font-size="10.5" text-anchor="middle" fill="#555">{lab}</text>')
    legend = []
    lx = 10
    for si, name in enumerate(names):
        legend.append(f'<rect x="{lx}" y="4" width="10" height="10" rx="2" fill="{colors[si%len(colors)]}"/>')
        legend.append(f'<text x="{lx+14}" y="13" font-size="10.5" fill="#444">{name}</text>')
        lx += 16 + 6.3 * len(name)
    total_h = height + legend_h
    return (f'<svg {_SVG_NS} viewBox="0 0 {width} {total_h}" width="100%" height="{total_h}">'
            f'<rect width="{width}" height="{total_h}" fill="white"/>'
            f'{"".join(legend)}'
            f'<line x1="35" y1="{legend_h+height-35}" x2="{width-10}" y2="{legend_h+height-35}" stroke="#ccc"/>'
            f'{"".join(bars)}</svg>')


def _scope_css(css: str, scope: str) -> str:
    scoped_rules = []
    for block in css.split("}"):
        block = block.strip()
        if not block:
            continue
        if ":root" in block:
            scoped_rules.append(block + "}")
            continue
        sel, _, props = block.partition("{")
        selectors = [s.strip() for s in sel.split(",")]
        scoped_sel = ", ".join(f"{scope} {s}" if s != "*" else f"{scope} *" for s in selectors)
        scoped_rules.append(f"{scoped_sel} {{{props}}}")
    return "\n".join(scoped_rules)


def build_five_patient_samples_section():
    """5 real high-risk cardio test patients, run end-to-end through the
    actual v10 pipeline. One DatasetRunV10 trained once and reused."""
    run = run_dataset_seed_v10(REPORT_SEED)
    high_risk_idx = [int(i) for i in run.high_risk_test_idx][:N_PATIENT_SAMPLES]

    cards = []
    for rank, idx in enumerate(high_risk_idx, start=1):
        r = run_single_patient_v10(run, idx, "no_preference")
        body = build_success_report(r) if r["success"] else build_abstain_report(r)
        body = re.sub(r"<title>.*?</title>\s*<style>.*?</style>", "", body, flags=re.DOTALL)
        body = re.sub(r"<h1>Recourse Report.*?</h1>", "", body, count=1)
        cards.append(f'<div class="patient-report" style="margin-bottom:26px;">'
                      f'<h3 style="margin:0 0 6px;">Patient sample {rank} of {N_PATIENT_SAMPLES} '
                      f'&mdash; cardio test #{idx}</h3>{body}</div>')

    scoped_css = _scope_css(PATIENT_CSS, ".patient-report")
    section = f"""
    <section class="headline">
      <h1>5 Sample Patients &mdash; Risk, Counterfactual, and Recommendations</h1>
      <p class="note">5 real high-risk patients from the cardio test set (seed {REPORT_SEED}), each run
      end-to-end through the real v10 pipeline: recorded values, model-predicted risk, the counterfactual
      explanation (real recorded change to a real admissible target patient, or the closest-miss semifactual
      explanation if none exists), and named clinical recommendations tied to guideline citations. Nothing
      below is templated or invented.</p>
      <style>{scoped_css}</style>
      {"".join(cards)}
    </section>"""
    return section, run


def main():
    v10_summary = json.loads((V10_DIR / "v10_summary.json").read_text())
    v10_per_seed = pd.read_csv(V10_DIR / "v10_summary_per_seed.csv")
    age_df = pd.read_csv(TABLES_DIR / "age_stratified_per_seed.csv")
    stage2_summary = json.loads((TABLES_DIR / "stage2_summary.json").read_text())

    patient_section, _run = build_five_patient_samples_section()

    datasets = ["uci", "nhanes_real", "cardio"]
    ds_labels = ["UCI Heart", "Real NHANES", "Cardio (v10)"]

    stage1_success = {
        "uci": age_df[(age_df.dataset == "uci") & (age_df.label == "age_stratified")].success_pct.mean(),
        "nhanes_real": age_df[(age_df.dataset == "nhanes_real") & (age_df.label == "age_stratified")].success_pct.mean(),
        "cardio": v10_summary["stage1_success_pct_mean"],
    }
    combined_success = {
        "uci": stage2_summary["uci"]["combined_success_pct_mean"],
        "nhanes_real": stage2_summary["nhanes_real"]["combined_success_pct_mean"],
        "cardio": v10_summary["combined_success_pct_mean"],
    }
    real_cvr = {
        "uci": 0.0, "nhanes_real": 0.0,  # architectural 0.00% for Stage1+2, both datasets (see results/final_report.html)
        "cardio": v10_summary["stage1_real_cvr_pct_mean"],
    }
    n_train = {"uci": 238, "nhanes_real": 3632, "cardio": int(N_SUBSAMPLE * 0.8)}
    n_high_risk = {
        "uci": age_df[(age_df.dataset == "uci") & (age_df.label == "age_stratified")].n_high_risk.mean(),
        "nhanes_real": age_df[(age_df.dataset == "nhanes_real") & (age_df.label == "age_stratified")].n_high_risk.mean(),
        "cardio": v10_summary["n_high_risk_mean"],
    }

    success_chart = bar_svg(ds_labels, {
        "Stage 1 (real search) only": [stage1_success[d] for d in datasets],
        "+ Stage 2 (synthetic fallback)": [combined_success[d] for d in datasets],
    }, colors=["#94a3b8", "#2563eb"])

    cvr_chart = bar_svg(ds_labels, {
        "Real-value CVR (Stage 1 successes)": [real_cvr[d] for d in datasets],
    }, colors=["#dc2626"], ymax=max(5.0, max(real_cvr.values()) * 1.3))

    scale_rows = "".join(
        f"<tr><td>{dl}</td><td>{n_train[d]:,}</td><td>{n_high_risk[d]:.0f}</td></tr>"
        for d, dl in zip(datasets, ds_labels)
    )

    per_seed_rows = "".join(
        f"<tr><td>{r.seed}</td><td>{r.n_high_risk}</td><td>{r.stage1_success_pct:.1f}%</td>"
        f"<td>{r.stage1_real_cvr_pct:.2f}%</td><td>{r.n_certified_infeasible}</td>"
        f"<td>{r.stage2_success_pct_of_attempted:.1f}%</td><td>{r.combined_success_pct:.1f}%</td></tr>"
        for r in v10_per_seed.itertuples()
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>v10 &mdash; CARE-LG on the Cardio Dataset</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#f7f7f9; margin:0; padding:24px; color:#111; }}
.wrap {{ max-width: 960px; margin:0 auto; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; }}
section {{ background:#fff; border:1px solid #e2e2e6; border-radius:10px; padding:18px 22px; margin-bottom:18px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:8px 0; }}
th,td {{ border:1px solid #e2e2e6; padding:5px 8px; text-align:left; }}
th {{ background:rgba(37,99,235,.08); }}
.note {{ font-size:12px; color:#666; }}
.badge {{ background:#dcfce7; color:#166534; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:700; margin-left:8px;}}
.warn {{ background:#fef2f2; color:#991b1b; padding:10px 14px; border-radius:8px; font-size:12.5px; border-left:4px solid #dc2626; margin-top:10px; }}
section.headline h1 {{ font-size:20px; margin:0 0 6px; }}
</style></head><body><div class="wrap">
<h1>v10 &mdash; CARE-LG on a Third Dataset <span class="badge">real data, cardio, 5 seeds</span></h1>
<p class="note">The full CARE-LG architecture (type-aware VAE, Riemannian pullback-metric graph search,
B1/B2-fixed hard clinical-constraint gate, per-age-bracket target definition, certified-infeasibility
check, Stage 2 synthetic fallback) run top-to-bottom on the Kaggle Cardiovascular Disease dataset
(70,000 raw rows; {N_SUBSAMPLE:,}-row stratified subsample per seed for computational tractability &mdash;
disclosed scoping decision, see experiments_v10/data_cardio.py) instead of UCI Heart or Real NHANES.
Separate experiment tree (experiments_v10/), src/ untouched; Stage 1's target definition and Stage 2's
algorithm are reused unmodified from scripts/run_age_stratified_experiment.py and
scripts/run_stage2_synthetic_search.py. Stage 1's path search uses <code>find_path_sra</code>
(src/graph/source_relative_admissibility.py) &mdash; a first cardio run using the older
<code>find_path_augmented</code> came back with a nonzero real CVR (~1.5%), the same compositional
constraint-drift leak found and fixed earlier this project, now confirmed on a third, independent
dataset; this report reflects the SRA-fixed rerun.</p>
<div class="note" style="background:#eef2ff;border-left:3px solid #4338ca;padding:8px 12px;border-radius:0 8px 8px 0;margin-top:8px;">
Provenance caveat, disclosed not hidden: this dataset's origin is an uncredited Kaggle upload
(no cited clinical institution), and, like every CLINICAL_EFFORT_WEIGHTS dict in this project
(see configs/dataset_config.py's own note on 'uci'/'nhanes'), the cost weights in
experiments_v10/data_cardio.py are freshly invented, illustrative constants, not derived from a
published treatment-burden instrument.
</div>

{patient_section}

<section>
  <h2>Success rate vs. UCI Heart and Real NHANES</h2>
  {success_chart}
  <table><thead><tr><th>Dataset</th><th>Stage 1 only</th><th>+ Stage 2</th></tr></thead>
  <tbody>{"".join(f"<tr><td>{dl}</td><td>{stage1_success[d]:.1f}%</td><td><b>{combined_success[d]:.1f}%</b></td></tr>" for d, dl in zip(datasets, ds_labels))}</tbody></table>
</section>

<section>
  <h2>Real-value CVR</h2>
  {cvr_chart}
  <p class="note">0.00% on all three datasets, by construction, using the SRA-fixed search
  (<code>find_path_sra</code>). A first cardio run using the pre-fix <code>find_path_augmented</code>
  measured ~1.5% real CVR &mdash; the compositional constraint-drift leak (two admissible-looking hops
  compounding into a real endpoint violation neither hop's own check catches) reproducing on this third,
  independent, much larger dataset. That is useful evidence the vulnerability is dataset-general, not a
  UCI/NHANES quirk &mdash; and confirmation the fix generalizes too, since it closed it here as well.</p>
</section>

<section>
  <h2>Dataset scale</h2>
  <table><thead><tr><th>Dataset</th><th>Training patients</th><th>High-risk test patients (mean/seed)</th></tr></thead>
  <tbody>{scale_rows}</tbody></table>
  <p class="note">Cardio (v10)'s {N_SUBSAMPLE:,}-row-per-seed subsample is ~{N_SUBSAMPLE*0.8/3632:.1f}x
  Real NHANES's full training set and ~{N_SUBSAMPLE*0.8/238:.0f}x UCI's, drawn from a 70,000-row source
  (largest of the three by over an order of magnitude even before subsampling).</p>
</section>

<section>
  <h2>Per-seed detail (cardio, v10)</h2>
  <table><thead><tr><th>Seed</th><th>n high-risk</th><th>Stage 1 success</th><th>Stage 1 real CVR</th>
  <th>Certified infeasible</th><th>Stage 2 success (of attempted)</th><th>Combined</th></tr></thead>
  <tbody>{per_seed_rows}</tbody></table>
  <p class="note">k_neighbors={K_NEIGHBORS}, n_subsample={N_SUBSAMPLE:,}/seed. Stage 2 is attempted on only
  the certified-infeasible pool (n={v10_summary['n_certified_infeasible_mean']:.0f}/seed on average)
  &mdash; too small a sample for its own success-rate-of-attempted number
  ({v10_summary['stage2_success_pct_of_attempted_mean']:.1f}%&plusmn;{v10_summary['stage2_success_pct_of_attempted_std']:.1f})
  to be read as a stable finding either way; it moves the combined total by well under 1 percentage
  point regardless.</p>
</section>

<section>
  <h2>Honest read</h2>
  <p class="note"><b>The headline number is mostly a scale artifact, not a methodological win.</b>
  Stage 1 success here ({stage1_success['cardio']:.1f}%) is far higher than UCI
  ({stage1_success['uci']:.1f}%) or NHANES ({stage1_success['nhanes_real']:.1f}%) primarily because the
  low-risk training pool is much larger ({int(v10_summary['n_low_risk_train_mean']):,} vs. NHANES's
  ~909, UCI's ~60) &mdash; the same dynamic this project's own threshold-sensitivity sweep already showed:
  a denser target pool makes reaching SOME admissible low-risk patient trivially easier, independent of
  any change in search quality. This should not be read as "the model got dramatically better" on cardio;
  it is a property of the dataset's scale, stated plainly rather than left implicit. A secondary factor:
  cardio's features are almost entirely directional lifestyle factors (weight, BP, cholesterol/glucose
  category, smoking/alcohol/activity) with no age-horizon cap comparable to UCI/NHANES's continuous labs.</p>
  <p class="note">Two disclosed simplifications: <code>height</code> was dropped entirely (see
  experiments_v10/data_cardio.py, same category as v8 dropping age) rather than forced into the
  type-aware VAE's high-cardinality categorical decoder head, and cholesterol/glucose are 3-level
  ordinal categories in the source data, not continuous mg/dL values &mdash; weaker signal than UCI's
  continuous cholesterol, directionally usable regardless.</p>
</section>
</div></body></html>"""

    out_path = REPO_ROOT / "experiments_v10" / "v10_report.html"
    out_path.write_text(html)
    print(f"[v10-report] wrote {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
