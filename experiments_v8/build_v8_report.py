"""
Builds a self-contained HTML report for v8 (age dropped entirely as a model
input), comparing against the current best real-graph numbers (age-
stratified Stage 1 + Stage 2). Auto-opens.

Usage:
  .venv/bin/python -m experiments_v8.build_v8_report
"""
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

V8_DIR = REPO_ROOT / "experiments_v8" / "results"
TABLES_DIR = REPO_ROOT / "results" / "tables"


def bar_svg(labels, series, width=520, height=230, fmt="{:.1f}%", ymax=100, colors=None):
    names = list(series.keys())
    colors = colors or ["#2563eb", "#dc2626", "#16a34a"]
    n = len(labels)
    group_w = (width - 60) / n
    bar_w = group_w / (len(names) + 0.6)
    bars, legend = [], []
    for gi, lab in enumerate(labels):
        gx = 40 + gi * group_w
        for si, name in enumerate(names):
            v = series[name][gi]
            h = (v / ymax) * (height - 55) if ymax else 0
            x = gx + si * bar_w + 6
            y = height - 35 - h
            c = colors[si % len(colors)]
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" height="{h:.1f}" fill="{c}" rx="2"/>')
            bars.append(f'<text x="{x+(bar_w-4)/2:.1f}" y="{y-4:.1f}" font-size="9.5" text-anchor="middle" fill="#111">{fmt.format(v)}</text>')
        bars.append(f'<text x="{gx+group_w/2:.1f}" y="{height-16:.1f}" font-size="10.5" text-anchor="middle" fill="#555">{lab}</text>')
    for si, name in enumerate(names):
        legend.append(f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:11px;color:#444;">'
                       f'<span style="width:10px;height:10px;background:{colors[si%len(colors)]};display:inline-block;border-radius:2px;"></span>{name}</span>')
    svg = (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
           f'<line x1="35" y1="{height-35}" x2="{width-10}" y2="{height-35}" stroke="#ccc"/>'
           f'{"".join(bars)}</svg>')
    return f'<div class="legend">{"".join(legend)}</div>{svg}'


def main():
    v8 = pd.read_csv(V8_DIR / "v8_no_age_per_seed.csv")
    v8_summary = json.loads((V8_DIR / "v8_no_age_summary.json").read_text())
    stage2_summary = json.loads((TABLES_DIR / "stage2_summary.json").read_text())
    age_df = pd.read_csv(TABLES_DIR / "age_stratified_per_seed.csv")

    datasets = ["uci", "nhanes_real"]
    ds_labels = ["UCI Heart", "Real NHANES"]

    orig_success = [age_df[(age_df.dataset == d) & (age_df.label == "original")].success_pct.mean() for d in datasets]
    with_age_final = [stage2_summary[d]["combined_success_pct_mean"] for d in datasets]
    v8_success = [v8_summary[d]["success_pct_mean"] for d in datasets]
    v8_cvr = [v8_summary[d]["real_cvr_pct_mean"] for d in datasets]

    success_chart = bar_svg(ds_labels, {
        "Original (age input, global 0.45)": orig_success,
        "Age-stratified + Stage 2 (age frozen)": with_age_final,
        "v8: age dropped entirely": v8_success,
    }, colors=["#94a3b8", "#2563eb", "#16a34a"])

    cvr_chart = bar_svg(ds_labels, {"Real CVR %": v8_cvr}, colors=["#dc2626"], ymax=10)

    rows = "".join(
        f"<tr><td>{r.dataset}</td><td>{r.seed}</td><td>{r.n_high_risk}</td><td>{r.n_low_risk_train}</td>"
        f"<td>{r.success_pct:.1f}%</td><td>{r.real_cvr_pct:.2f}%</td><td>{r.certified_infeasible_pct if pd.notna(r.certified_infeasible_pct) else 'n/a (0 abstentions)'}</td></tr>"
        for r in v8.itertuples())

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>v8: Age Dropped Entirely</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#f7f7f9; margin:0; padding:24px; color:#111; }}
.wrap {{ max-width: 900px; margin:0 auto; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; }}
section {{ background:#fff; border:1px solid #e2e2e6; border-radius:10px; padding:18px 22px; margin-bottom:18px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:8px 0; }}
th,td {{ border:1px solid #e2e2e6; padding:4px 8px; text-align:left; }}
th {{ background:rgba(37,99,235,.08); }}
.note {{ font-size:12px; color:#666; }}
.badge {{ background:#dcfce7; color:#166534; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:700; margin-left:8px;}}
.warn {{ background:#fef3c7; color:#92400e; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:700; margin-left:8px;}}
</style></head><body><div class="wrap">
<h1>v8 &mdash; Age Dropped Entirely <span class="badge">real data, both datasets, 5 seeds</span></h1>
<p class="note">Age removed as a MODEL INPUT (not just frozen in the search): classifier and VAE retrained from
scratch on every feature except age. No age-related constraint exists any more because there is no age feature
left to constrain. Separate experiment tree (experiments_v8/), src/ untouched.</p>

<section>
  <h2>Success rate</h2>
  {success_chart}
  <table><thead><tr><th>Dataset</th><th>Original (with age)</th><th>Best with-age result (Stage1+Stage2)</th><th>v8 (age dropped)</th></tr></thead>
  <tbody>{"".join(f"<tr><td>{dl}</td><td>{o:.1f}%</td><td>{w:.1f}%</td><td><b>{v:.1f}%</b></td></tr>" for dl,o,w,v in zip(ds_labels, orig_success, with_age_final, v8_success))}</tbody></table>
</section>

<section>
  <h2>Real CVR <span class="warn">small nonzero on Real NHANES &mdash; disclosed below</span></h2>
  {cvr_chart}
  <p class="note">UCI: exactly 0.00% every seed. Real NHANES: 0.6-1.1% (not exactly 0, unlike every other result in
  this project). Root cause checked directly (seed 0): 2 patients out of ~379 whose ENDPOINT (start-to-finish,
  not any individual hop) directional delta exceeds the single-hop tolerance epsilon after a multi-hop path &mdash;
  each individual hop is admissible, but small same-direction moves compound across hops past the endpoint
  tolerance. This is a multi-hop cumulative-drift edge case, not a broken gate (every individual edge is still
  checked and passes); disclosed honestly rather than rounded to 0.</p>
</section>

<section>
  <h2>Per-seed detail</h2>
  <table><thead><tr><th>Dataset</th><th>Seed</th><th>n high-risk</th><th>n low-risk targets</th><th>Success</th><th>Real CVR</th><th>Certified infeasible (of abstentions)</th></tr></thead>
  <tbody>{rows}</tbody></table>
</section>

<section>
  <h2>Honest read</h2>
  <p class="note">Dropping age entirely produces by far the largest success-rate gain of any experiment this
  session (UCI ~38%&rarr;~96%, Real NHANES ~51%&rarr;~89%), because the single most frequently-blocking
  constraint (age monotonicity, ~97% of blocked candidates in the worked neurosymbolic example) and the
  single largest risk driver in the classifier are both removed at once. This is NOT a free win: it changes
  what the model IS &mdash; it can no longer explain or predict risk using age at all, which may not be
  clinically defensible on its own (age is one of the strongest, most literature-supported ASCVD risk factors).
  The near-zero-but-not-quite CVR on Real NHANES is the one wrinkle worth fixing before this becomes a
  headline number.</p>
</section>
</div></body></html>"""

    out_path = REPO_ROOT / "experiments_v8" / "v8_report.html"
    out_path.write_text(html)
    print(f"[v8-report] wrote {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
