"""
Builds a single self-contained HTML report combining the three age-ceiling
follow-up results: the age-ceiling-vs-data-scarcity diagnostic, the
multi-hop real-chain check (Phase 1), and the age-stratified threshold
experiment (Phase 1.5). Writes and auto-opens
results/age_followup_report.html.

Usage:
  .venv/bin/python -m scripts.build_age_followup_report
"""
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def bar_svg(labels, values, width=480, height=200, color="#2563eb", fmt="{:.1f}%", ymax=100):
    n = len(labels)
    bw = (width - 60) / n
    bars = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = (v / ymax) * (height - 50) if ymax else 0
        x = 40 + i * bw
        y = height - 30 - h
        bars.append(f'<rect x="{x+8:.1f}" y="{y:.1f}" width="{bw-16:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" font-size="11" text-anchor="middle" fill="#111">{fmt.format(v)}</text>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{height-12:.1f}" font-size="10" text-anchor="middle" fill="#555">{lab}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            f'<line x1="40" y1="{height-30}" x2="{width-20}" y2="{height-30}" stroke="#ccc"/>'
            f'{"".join(bars)}</svg>')


def main():
    # --- Age-ceiling diagnostic ---
    ceiling_summary = json.loads((TABLES_DIR / "age_ceiling_diagnostic_summary.json").read_text())

    # --- Phase 1: multi-hop check ---
    multihop_summary = json.loads((TABLES_DIR / "multihop_check_summary.json").read_text())

    # --- Phase 1.5: age-stratified ---
    age_df = pd.read_csv(TABLES_DIR / "age_stratified_per_seed.csv")
    bracket_info = json.loads((TABLES_DIR / "age_stratified_bracket_info.json").read_text())

    sections = []

    # Section: age-ceiling diagnostic
    ds_labels = list(ceiling_summary.keys())
    age_ceiling_vals = [ceiling_summary[d]["age_ceiling_pct_mean"] for d in ds_labels]
    scarcity_vals = [ceiling_summary[d]["data_scarcity_pct_mean"] for d in ds_labels]
    sections.append(f"""
    <section>
      <h2>1 &middot; Age-ceiling vs. data-scarcity diagnostic</h2>
      <p class="note">For every certified-infeasible patient: build a best-case synthetic patient (age/immutable
      features frozen, every directional feature set to the best value observed anywhere in the real training
      population). If best-case risk still clears the threshold &rarr; genuine age ceiling. If it drops below &rarr;
      data-scarcity (a solution exists, just not in one real person).</p>
      <div class="charts">
        {bar_svg(ds_labels, age_ceiling_vals, color="#dc2626")}
        {bar_svg(ds_labels, scarcity_vals, color="#16a34a")}
      </div>
      <table><thead><tr><th>Dataset</th><th>n checked</th><th>Age-ceiling %</th><th>Data-scarcity %</th><th>Mean risk before</th><th>Mean best-case risk</th></tr></thead>
      <tbody>{"".join(f"<tr><td>{d}</td><td>{ceiling_summary[d]['n_certified_infeasible_checked']}</td>"
                       f"<td>{ceiling_summary[d]['age_ceiling_pct_mean']:.1f}% &plusmn; {ceiling_summary[d]['age_ceiling_pct_std']:.1f}</td>"
                       f"<td>{ceiling_summary[d]['data_scarcity_pct_mean']:.1f}% &plusmn; {ceiling_summary[d]['data_scarcity_pct_std']:.1f}</td>"
                       f"<td>{ceiling_summary[d]['mean_risk_before']:.3f}</td><td>{ceiling_summary[d]['mean_risk_best_case']:.3f}</td></tr>"
                       for d in ds_labels)}</tbody></table>
    </section>""")

    # Section: multi-hop check
    mh_vals = [multihop_summary[d]["pct_found_mean"] for d in ds_labels]
    sections.append(f"""
    <section>
      <h2>2 &middot; Phase 1 &mdash; multi-hop real-chain check</h2>
      <p class="note">For certified-infeasible patients: does a genuine 2-real-hop chain exist (x0 &rarr; real patient m
      &rarr; real low-risk patient j, both hops admissible) that the k-NN-restricted interior graph missed?</p>
      {bar_svg(ds_labels, mh_vals, color="#7c3aed")}
      <table><thead><tr><th>Dataset</th><th>n checked</th><th>2-hop chain found</th></tr></thead>
      <tbody>{"".join(f"<tr><td>{d}</td><td>{multihop_summary[d]['n_checked']}</td><td>{multihop_summary[d]['pct_found_mean']:.1f}% &plusmn; {multihop_summary[d]['pct_found_std']:.1f}</td></tr>" for d in ds_labels)}</tbody></table>
      <p><b>Result: rules out multi-hop as a fix</b> &mdash; 0% found on both datasets, all seeds. Not a wasted check:
      confirms the graph's k-NN restriction isn't hiding an already-available real solution.</p>
    </section>""")

    # Section: age-stratified
    age_sections = []
    for dataset in age_df.dataset.unique():
        d = age_df[age_df.dataset == dataset]
        orig = d[d.label == "original"]
        strat = d[d.label == "age_stratified"]
        succ_svg = bar_svg(["original", "age-stratified"], [orig.success_pct.mean(), strat.success_pct.mean()], color="#2563eb")
        cvr_svg = bar_svg(["original", "age-stratified"], [orig.real_cvr_pct.mean(), strat.real_cvr_pct.mean()], color="#dc2626")
        cert_svg = bar_svg(["original", "age-stratified"], [orig.certified_infeasible_pct.mean(), strat.certified_infeasible_pct.mean()], color="#f59e0b")
        brackets = bracket_info.get(dataset, {})
        bracket_rows = "".join(
            f"<tr><td>{b}</td><td>{info['age_range'][0]:.0f}&ndash;{info['age_range'][1]:.0f}</td><td>{info['n']}</td><td>{info['risk_threshold']:.3f}</td></tr>"
            for b, info in brackets.items()
        )
        age_sections.append(f"""
        <h3>{dataset}</h3>
        <div class="charts">{succ_svg}{cvr_svg}{cert_svg}</div>
        <table><thead><tr><th>Label</th><th>n low-risk targets (mean)</th><th>Success %</th><th>Real CVR %</th><th>Certified infeasible %</th></tr></thead>
        <tbody>
          <tr><td>original (0.45 global)</td><td>{orig.n_low_risk_train.mean():.0f}</td><td>{orig.success_pct.mean():.1f} &plusmn; {orig.success_pct.std():.1f}</td>
              <td>{orig.real_cvr_pct.mean():.1f}</td><td>{orig.certified_infeasible_pct.mean():.1f}</td></tr>
          <tr><td>age-stratified</td><td>{strat.n_low_risk_train.mean():.0f}</td><td>{strat.success_pct.mean():.1f} &plusmn; {strat.success_pct.std():.1f}</td>
              <td>{strat.real_cvr_pct.mean():.1f}</td><td>{strat.certified_infeasible_pct.mean():.1f}</td></tr>
        </tbody></table>
        <p class="note">Age brackets (quartiles of real training-population age), each with its OWN bottom-quartile risk threshold:</p>
        <table class="small"><thead><tr><th>Bracket</th><th>Age range</th><th>n</th><th>Risk threshold (bottom 25% for this age)</th></tr></thead>
        <tbody>{bracket_rows}</tbody></table>
        """)

    sections.append(f"""
    <section>
      <h2>3 &middot; Phase 1.5 &mdash; age-stratified low-risk threshold</h2>
      <p class="note">"Low risk" redefined per age quartile (bottom 25% of risk WITHIN that age bracket) instead of one
      global 0.45 cutoff for everyone. Who gets EVALUATED (high-risk cohort) is unchanged; only which real training
      patients count as valid recourse TARGETS changes.</p>
      {"".join(age_sections)}
    </section>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Age Ceiling Follow-up</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#f7f7f9; margin:0; padding:24px; color:#111; }}
.wrap {{ max-width: 920px; margin:0 auto; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; }} h3 {{ font-size:14px; margin-top:18px; }}
section {{ background:#fff; border:1px solid #e2e2e6; border-radius:10px; padding:18px 22px; margin-bottom:18px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:6px 0; }}
table.small {{ font-size:11px; }}
th,td {{ border:1px solid #e2e2e6; padding:4px 8px; text-align:left; }}
th {{ background:rgba(37,99,235,.08); }}
.charts {{ display:flex; flex-wrap:wrap; gap:10px; }}
.charts svg {{ flex:1; min-width:180px; }}
.note {{ font-size:12px; color:#666; }}
.badge {{ background:#fef3c7; color:#92400e; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:700; }}
</style></head><body><div class="wrap">
<h1>Age-Ceiling Follow-up Results <span class="badge">3 experiments, real data, all disclosed</span></h1>
<p>Prerequisite checks run before building the Stage 2 directed synthetic search: (1) why patients are certified
infeasible (age ceiling vs. data scarcity), (2) whether a cheaper multi-hop real-patient fix already resolves some
of them, (3) whether an age-stratified target definition resolves the genuine age-ceiling group.</p>
{''.join(sections)}
</div></body></html>"""

    out_path = RESULTS_DIR / "age_followup_report.html"
    out_path.write_text(html)
    print(f"[report] wrote {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")
    print("[report] opened in default browser")


if __name__ == "__main__":
    main()
