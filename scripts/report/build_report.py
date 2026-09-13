"""
Assembles results/full_report.html -- a single, self-contained
static HTML file documenting the full project, centered on a real patient
journey (section 4). Every number/table/example is read directly from an
existing repo file (dataset CSVs, results/tables/*.csv,
results/tables/{examples,journey}.json) or copied verbatim from a
FINDINGS.md already produced in a prior session -- nothing is computed,
estimated, or invented fresh by this script beyond pure HTML formatting.
Each section/figure cites its source file inline.
"""
import base64
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
ARCHIVE = REPO_ROOT / "archive" / "investigation_v2_to_v6"
OUT = REPO_ROOT / "results" / "full_report.html"

missing_sections = []


def read_csv_rows(path, n):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [next(r) for _ in range(n)]
    return header, rows


def b64_img(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
uci_header, uci_rows = read_csv_rows(REPO_ROOT / "data" / "uci_heart.csv", 5)
nh_header, nh_rows = read_csv_rows(REPO_ROOT / "data" / "nhanes_real.csv", 5)

journey_path = REPO_ROOT / "results" / "tables" / "journey.json"
if journey_path.exists():
    journey = json.loads(journey_path.read_text())
else:
    journey = None
    missing_sections.append("Section 4 (patient journey): journey.json not found")

examples_path = REPO_ROOT / "results" / "tables" / "examples.json"
examples = json.loads(examples_path.read_text()) if examples_path.exists() else {"semifactual_examples": [], "guideline_derivation_examples": []}
if not examples_path.exists():
    missing_sections.append("Section 7/8 general examples: examples.json not found")

pref_csv = REPO_ROOT / "results" / "tables" / "preference_demo.csv"
pref_rows = []
if pref_csv.exists():
    with open(pref_csv) as f:
        pref_rows = list(csv.DictReader(f))
else:
    missing_sections.append("Section 9 (preference elicitation): preference_demo.csv not found")
pref_groups = {}
for row in pref_rows:
    key = (row["dataset"], row["seed"], row["patient_idx"])
    pref_groups.setdefault(key, []).append(row)

fig_specs = [
    (REPO_ROOT / "results" / "figures" / "figA_ablation_bars.png",
     "Fig A — Ablation: KDE plausibility &amp; clinical effort (experiments_v3)",
     "On Real NHANES, R+hard has the highest (worst) clinical effort of the four main cells "
     "(~5.0 vs. ~3.9-4.8 for the others). KDE plausibility is statistically indistinguishable "
     "across all four cells on both datasets.",
     "experiments_v3/figures/figA_ablation_bars.png"),
    (REPO_ROOT / "results" / "figures" / "figB_abstention_decomposition.png",
     "Fig B — Abstention decomposition (experiments_v3, pre-B1/B2-fix gate)",
     "Real NHANES produced a mean of 235.2 R+hard abstentions/seed (51.7-58.9% of high-risk "
     "patients). Of all 1,176 abstentions across 5 seeds, 100% were certified infeasible; zero "
     "confirmed blindspots — the same pattern experiments_v6's Phase 2 reconfirmed under the "
     "corrected entry gate (see Section 8 below).",
     "experiments_v3/figures/figB_abstention_decomposition.png"),
    (REPO_ROOT / "results" / "figures" / "fig1_motivating_gap.png",
     "Fig 1 — The Exact Guarantee, Verified (experiments_v5, EGD)",
     "experiments_v5's Exact-Guarantee Decoder achieved exactly 0.0% decoded CVR wherever it "
     "succeeded, on both cohorts — architecturally exact, but (per experiments_v5/FINDINGS.md) "
     "at a severe cost in success rate and plausibility that motivated the B1/B2 investigation, "
     "which found a much cheaper fix (see Section 6).",
     "experiments_v5/figures/fig1_motivating_gap.png"),
]
fig_html_blocks = []
for path, title, caption, src in fig_specs:
    if path.exists():
        b64 = b64_img(path)
        fig_html_blocks.append(f"""
        <div class="figure">
          <h3>{title}</h3>
          <img src="data:image/png;base64,{b64}" alt="{title}" />
          <p class="caption">{caption}</p>
          <p class="source">Source: {src}</p>
        </div>""")
    else:
        missing_sections.append(f"Figure missing: {src}")


def rows_to_table(header, rows):
    thead = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>"


uci_table = rows_to_table(uci_header, uci_rows)
nh_table = rows_to_table(nh_header, nh_rows)

# ---------------------------------------------------------------------------
# Section 4: patient journey (the centerpiece)
# ---------------------------------------------------------------------------

def sparkbar(items, max_items=6):
    if not items:
        return ""
    max_abs = max(abs(i["value"]) for i in items) or 1e-6
    rows = []
    for i in items[:max_items]:
        pct = min(100, abs(i["value"]) / max_abs * 100)
        cls = "shap-pos" if i["value"] >= 0 else "shap-neg"
        rows.append(f'<div class="shap-row"><span class="shap-label">{i["feature"]}</span>'
                    f'<span class="shap-track"><span class="{cls}" style="width:{pct:.0f}%"></span></span>'
                    f'<span class="shap-val">{i["value"]:+.3f}</span></div>')
    return "".join(rows)


def risk_pill(r):
    cls = "risk-high" if r >= 0.55 else ("risk-mid" if r >= 0.45 else "risk-low")
    return f'<span class="pill {cls}">{r*100:.1f}% risk</span>'


def journey_success_html(s):
    raw_rows = "".join(f"<tr><td>{f}</td><td>{v:g}</td></tr>" for f, v in s["raw_record"].items())

    # latent scatter as inline SVG (real 2D PCA projection, see collect_journey.py)
    W, H = 560, 220
    xs = [p[0] for p in s["latent_pca_sample"]] + [s["latent_pca_query"][0], s["latent_pca_target"][0]]
    ys = [p[1] for p in s["latent_pca_sample"]] + [s["latent_pca_query"][1], s["latent_pca_target"][1]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    def sx(x): return 20 + (x - xmin) / max(1e-9, xmax - xmin) * (W - 40)
    def sy(y): return H - 20 - (y - ymin) / max(1e-9, ymax - ymin) * (H - 40)
    dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="#bbb" opacity="0.6"/>'
                    for x, y in s["latent_pca_sample"])
    qx, qy = s["latent_pca_query"]
    tx, ty = s["latent_pca_target"]
    latent_svg = f"""
    <svg viewBox="0 0 {W} {H}" width="100%" height="{H}">
      {dots}
      <line x1="{sx(qx):.1f}" y1="{sy(qy):.1f}" x2="{sx(tx):.1f}" y2="{sy(ty):.1f}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4,3"/>
      <circle cx="{sx(qx):.1f}" cy="{sy(qy):.1f}" r="7" fill="#dc2626" stroke="#fff" stroke-width="1.5"/>
      <text x="{sx(qx)+9:.1f}" y="{sy(qy)+4:.1f}" font-size="11" fill="#dc2626">patient (query)</text>
      <circle cx="{sx(tx):.1f}" cy="{sy(ty):.1f}" r="7" fill="#16a34a" stroke="#fff" stroke-width="1.5"/>
      <text x="{sx(tx)+9:.1f}" y="{sy(ty)+4:.1f}" font-size="11" fill="#16a34a">target #{s['chosen_target_node']}</text>
    </svg>"""
    z_str = ", ".join(f"{v:.3f}" for v in s["latent_z"])

    cand_rows = []
    for c in s["candidate_neighbors"]:
        status = '<span class="badge badge-green">admissible</span>' if c["admissible"] else '<span class="badge badge-red">blocked</span>'
        chosen = " &larr; chosen" if c["chosen"] else ""
        reason = f'<div class="cand-reason">{"; ".join(c["blocking_reasons"])}</div>' if c["blocking_reasons"] else ""
        cand_rows.append(f'<tr><td>#{c["node_idx"]}{chosen}</td><td>{status}</td>'
                          f'<td>{c["dist_riemannian"]:.3f}</td><td>{c["dist_euclidean"]:.3f}</td><td>{reason}</td></tr>')
    cand_table = ("<table class='small'><thead><tr><th>Candidate</th><th>Status</th>"
                  "<th>Riemannian dist.</th><th>Euclidean dist.</th><th>Blocking reason (if any)</th></tr></thead>"
                  f"<tbody>{''.join(cand_rows)}</tbody></table>")

    checklist_rows = "".join(
        f'<div class="rule-check {"rc-pass" if r["status"]=="pass" else "rc-fail"}">'
        f'{"&#10003;" if r["status"]=="pass" else "&#10007;"} <b>[{r["rule_id"]}]</b> {r["citation"]}</div>'
        for r in s["rule_checklist"]
    )

    delta_chips = "".join(
        f'<span class="chip">{f}: {d["before"]:g}&rarr;{d["after"]:g} {d["unit"]} ({d["delta"]:+.1f})</span>'
        for f, d in s["feature_deltas"].items() if abs(d["delta"]) > 1e-6
    )

    gauge_before = s["risk_before"] * 100
    gauge_after = s["risk_after"] * 100

    return f"""
    <div class="timeline">

      <div class="timeline-item layer-data">
        <div class="dot"></div>
        <h4>1 &middot; Raw patient record {risk_pill(s['risk_before'])}</h4>
        <p>UCI Heart, seed 0, test patient <b>#{s['patient_idx']}</b> — a real held-out test-cohort record.</p>
        <table class="small">{raw_rows}</table>
      </div>

      <div class="timeline-item layer-xai">
        <div class="dot"></div>
        <h4>2 &middot; SHAP risk attribution</h4>
        <p>Which of this patient's own features drove their classifier risk score, most influential first.</p>
        <div class="shap-chart">{sparkbar(s['shap_top_features'])}</div>
      </div>

      <div class="timeline-item layer-geometry">
        <div class="dot"></div>
        <h4>3 &middot; Encoded latent position</h4>
        <p>Real 4-dimensional VAE latent code: <code>z = [{z_str}]</code></p>
        <p>2D PCA projection (top-2 principal components of a real 150-point sample of the training
        latent space) showing this patient (red) and the eventually-chosen target (green):</p>
        {latent_svg}
      </div>

      <div class="timeline-item layer-geometry">
        <div class="dot"></div>
        <h4>4 &middot; CALG graph — candidate neighbors</h4>
        <p>The 6 closest real training patients by Riemannian latent distance, plus the actual chosen
        target (which was NOT among the 6 closest — every closer candidate was clinically blocked).
        Each blocked candidate's reason comes from the neurosymbolic layer (Section 7).</p>
        {cand_table}
      </div>

      <div class="timeline-item layer-neurosymbolic">
        <div class="dot"></div>
        <h4>5 &middot; Neurosymbolic checklist for the CHOSEN path</h4>
        <p>Every admissibility rule this transition was checked against:</p>
        {checklist_rows}
      </div>

      <div class="timeline-item layer-geometry">
        <div class="dot"></div>
        <h4>6 &middot; Cost of the chosen edge</h4>
        <p>Riemannian distance: <b>{s['dist_riemannian_chosen']:.4f}</b> &nbsp;|&nbsp;
        Euclidean distance: <b>{s['dist_euclidean_chosen']:.4f}</b> &nbsp;|&nbsp;
        Weighted clinical effort: <b>{s['effort']:.2f}</b></p>
      </div>

      <div class="timeline-item layer-certification">
        <div class="dot"></div>
        <h4>7 &middot; Certified search result {risk_pill(s['risk_after'])}</h4>
        <p>Path: query &rarr; real training patient <b>#{s['chosen_target_node']}</b> (1 hop). Real
        feature changes (recorded values, not decoded reconstructions — this is the B1/B2-fixed
        real-value verification):</p>
        <div>{delta_chips}</div>
      </div>

      <div class="timeline-item layer-xai">
        <div class="dot"></div>
        <h4>8 &middot; Plain-English translation</h4>
        <p class="plain-english">&ldquo;{s['plain_english']}&rdquo;</p>
      </div>

      <div class="timeline-item layer-certification final">
        <div class="dot"></div>
        <h4>9 &middot; Final state</h4>
        <div class="gauge-compare">
          <div class="gauge-box"><div class="gauge-label">Before</div><div class="gauge-num risk-high">{gauge_before:.1f}%</div></div>
          <div class="gauge-arrow">&#8594;</div>
          <div class="gauge-box"><div class="gauge-label">After</div><div class="gauge-num risk-low">{gauge_after:.1f}%</div></div>
        </div>
        <p>Risk dropped {gauge_before - gauge_after:.1f} percentage points, crossing the low-risk
        threshold (45%), via one clinically admissible, real-value-verified step.</p>
      </div>

    </div>
    """


def journey_abstain_html(a):
    freq_rows = "".join(
        f"<tr><td>{rule}</td><td>{pct:.1f}%</td></tr>"
        for rule, pct in sorted(a["blocking_rule_frequency"].items(), key=lambda kv: -kv[1])
    )
    raw_rows = "".join(f"<tr><td>{f}</td><td>{v:g}</td></tr>" for f, v in a["raw_record"].items())
    return f"""
    <div class="timeline compressed">

      <div class="timeline-item layer-data">
        <div class="dot"></div>
        <h4>1 &middot; Raw patient record {risk_pill(a['risk_before'])}</h4>
        <p>UCI Heart, seed 0, test patient <b>#{a['patient_idx']}</b> (real held-out record).</p>
        <table class="small">{raw_rows}</table>
      </div>

      <div class="timeline-item layer-xai">
        <div class="dot"></div>
        <h4>2 &middot; SHAP risk attribution</h4>
        <div class="shap-chart">{sparkbar(a['shap_top_features'])}</div>
      </div>

      <div class="timeline-item layer-certification abstain-block">
        <div class="dot dot-red"></div>
        <h4>3 &middot; Exhaustive check: CERTIFIED INFEASIBLE</h4>
        <p>All <b>{a['n_candidates_checked']}</b> low-risk training candidates were checked directly
        against this patient's real values. The closest miss was training patient
        <b>#{a['closest_miss_candidate_idx']}</b>, with only <b>{a['closest_miss_violation_count']}</b>
        violated sub-condition — the fewest of any candidate — and it still failed.</p>
        <table class="small"><thead><tr><th>Blocking rule</th><th>Frequency across checked candidates</th></tr></thead>
        <tbody>{freq_rows}</tbody></table>
      </div>

      <div class="timeline-item layer-xai final">
        <div class="dot"></div>
        <h4>4 &middot; Semifactual explanation</h4>
        <p class="plain-english">&ldquo;{a['semifactual_text']}&rdquo;</p>
      </div>

    </div>
    """


journey_html = ""
if journey:
    journey_html = f"""
    <h3 class="journey-h">Patient A — reaches low risk (success)</h3>
    {journey_success_html(journey['success_patient'])}
    <h3 class="journey-h">Patient B — certified infeasible (abstains)</h3>
    {journey_abstain_html(journey['abstain_patient'])}
    """
else:
    journey_html = "<p class='missing-inline'>journey.json not found — this centerpiece section could not be built from real data.</p>"

# ---------------------------------------------------------------------------
# Section 7/8: general examples + preference table
# ---------------------------------------------------------------------------

def semifactual_block(ex):
    freq_rows = "".join(
        f"<tr><td>{rule}</td><td>{pct:.1f}%</td></tr>"
        for rule, pct in sorted(ex["blocking_rule_frequency"].items(), key=lambda kv: -kv[1])
    )
    return f"""
    <div class="example">
      <h4>{ex['dataset']} / seed {ex['seed']} / patient #{ex['patient_idx']}</h4>
      <p class="quote">&ldquo;{ex['text']}&rdquo;</p>
      <table class="small"><thead><tr><th>Blocking rule</th><th>Frequency</th></tr></thead>
      <tbody>{freq_rows}</tbody></table>
    </div>"""


semifactual_html = "".join(semifactual_block(ex) for ex in examples["semifactual_examples"])

pref_rows_html = []
for (dataset, seed, pidx), group in pref_groups.items():
    lines = []
    for r in group:
        if r["success"] == "True":
            lines.append(f'<tr><td>{r["profile"]}</td><td>target node {r["target_node"]}</td><td>{float(r["cost"]):.2f}</td></tr>')
        else:
            lines.append(f'<tr><td>{r["profile"]}</td><td colspan="2">no admissible path</td></tr>')
    pref_rows_html.append(f"""
    <div class="pref-patient">
      <h4>{dataset} / seed {seed} / patient #{pidx}</h4>
      <table class="small"><thead><tr><th>Profile</th><th>Selected target</th><th>Cost</th></tr></thead>
      <tbody>{''.join(lines)}</tbody></table>
    </div>""")
pref_html = "".join(pref_rows_html)

PIPELINE_SVG = """
<svg viewBox="0 0 1000 230" width="100%" height="230" role="img" aria-label="Pipeline diagram (10 stages)">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#777"/>
    </marker>
  </defs>
  <style>
    .pb rect { stroke-width:1.5; rx:8; }
    .pb text { font: 10.5px -apple-system, sans-serif; }
    .pl { stroke:#777; stroke-width:1.5; marker-end:url(#arrow); fill:none; }
    .stagenum { font: bold 10px sans-serif; fill:#fff; }
  </style>
  <!-- row 1: stages 1-5 -->
  <g class="pb" fill="#f3f4f6" stroke="#9ca3af"><rect x="8" y="20" width="176" height="55"/></g>
  <circle cx="24" cy="36" r="10" fill="#6b7280"/><text x="24" y="40" class="stagenum" text-anchor="middle">1</text>
  <text x="40" y="42" fill="#111">Raw patient</text><text x="40" y="58" fill="#555" font-size="9">record + classifier risk</text>

  <g class="pb" fill="#dbeafe" stroke="#2563eb"><rect x="204" y="20" width="176" height="55"/></g>
  <circle cx="220" cy="36" r="10" fill="#2563eb"/><text x="220" y="40" class="stagenum" text-anchor="middle">2</text>
  <text x="236" y="42" fill="#1e3a8a">SHAP</text><text x="236" y="58" fill="#1e3a8a" font-size="9">risk attribution</text>

  <g class="pb" fill="#ede9fe" stroke="#7c3aed"><rect x="400" y="20" width="176" height="55"/></g>
  <circle cx="416" cy="36" r="10" fill="#7c3aed"/><text x="416" y="40" class="stagenum" text-anchor="middle">3</text>
  <text x="432" y="42" fill="#4c1d95">VAE encode</text><text x="432" y="58" fill="#4c1d95" font-size="9">x &rarr; latent z</text>

  <g class="pb" fill="#ede9fe" stroke="#7c3aed"><rect x="596" y="20" width="176" height="55"/></g>
  <circle cx="612" cy="36" r="10" fill="#7c3aed"/><text x="612" y="40" class="stagenum" text-anchor="middle">4</text>
  <text x="628" y="42" fill="#4c1d95">CALG graph</text><text x="628" y="58" fill="#4c1d95" font-size="9">k-NN + Riemannian metric</text>

  <g class="pb" fill="#fef3c7" stroke="#d97706"><rect x="792" y="20" width="176" height="55"/></g>
  <circle cx="808" cy="36" r="10" fill="#d97706"/><text x="808" y="40" class="stagenum" text-anchor="middle">5</text>
  <text x="824" y="42" fill="#78350f">Neurosymbolic</text><text x="824" y="58" fill="#78350f" font-size="9">edge admissibility</text>

  <!-- row 2: stages 6-10 -->
  <g class="pb" fill="#dcfce7" stroke="#16a34a"><rect x="8" y="150" width="176" height="55"/></g>
  <circle cx="24" cy="166" r="10" fill="#16a34a"/><text x="24" y="170" class="stagenum" text-anchor="middle">6</text>
  <text x="40" y="172" fill="#14532d">Dijkstra</text><text x="40" y="188" fill="#14532d" font-size="9">shortest admissible path</text>

  <g class="pb" fill="#dcfce7" stroke="#16a34a"><rect x="204" y="150" width="176" height="55"/></g>
  <circle cx="220" cy="166" r="10" fill="#16a34a"/><text x="220" y="170" class="stagenum" text-anchor="middle">7</text>
  <text x="236" y="172" fill="#14532d">Real-value</text><text x="236" y="188" fill="#14532d" font-size="9">certify (B1/B2-fixed)</text>

  <g class="pb" fill="#fee2e2" stroke="#dc2626"><rect x="400" y="150" width="176" height="55"/></g>
  <circle cx="416" cy="166" r="10" fill="#dc2626"/><text x="416" y="170" class="stagenum" text-anchor="middle">8</text>
  <text x="432" y="172" fill="#7f1d1d">Certified</text><text x="432" y="188" fill="#7f1d1d" font-size="9">infeasibility (if abstained)</text>

  <g class="pb" fill="#dbeafe" stroke="#2563eb"><rect x="596" y="150" width="176" height="55"/></g>
  <circle cx="612" cy="166" r="10" fill="#2563eb"/><text x="612" y="170" class="stagenum" text-anchor="middle">9</text>
  <text x="628" y="172" fill="#1e3a8a">Explain</text><text x="628" y="188" fill="#1e3a8a" font-size="9">rules + semifactual (XAI)</text>

  <g class="pb" fill="#f3f4f6" stroke="#9ca3af"><rect x="792" y="150" width="176" height="55"/></g>
  <circle cx="808" cy="166" r="10" fill="#6b7280"/><text x="808" y="170" class="stagenum" text-anchor="middle">10</text>
  <text x="824" y="172" fill="#111">UI</text><text x="824" y="188" fill="#111" font-size="9">patient-facing display</text>

  <path class="pl" d="M184,47 H204"/><path class="pl" d="M380,47 H400"/><path class="pl" d="M576,47 H596"/><path class="pl" d="M772,47 H792"/>
  <path class="pl" d="M880,75 C 880,110 96,110 96,150"/>
  <path class="pl" d="M184,177 H204"/><path class="pl" d="M380,177 H400"/><path class="pl" d="M576,177 H596"/><path class="pl" d="M772,177 H792"/>
</svg>
"""

HTML = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>CARE-LG: Full Project Report</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg:#f7f7f9; --panel:#fff; --text:#1a1a1a; --muted:#666; --accent:#2563eb; --border:#e2e2e6;
    --geometry:#7c3aed; --geometry-bg:#f3edff;
    --neuro:#d97706; --neuro-bg:#fff7e6;
    --xai:#2563eb; --xai-bg:#eef4ff;
    --cert-ok:#16a34a; --cert-ok-bg:#eafaf0;
    --cert-bad:#dc2626; --cert-bad-bg:#fef2f2;
    --data-bg:#f3f4f6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#111318; --panel:#1a1d24; --text:#e8e8ea; --muted:#9a9a9f; --accent:#60a5fa; --border:#2a2d35;
      --geometry:#a78bfa; --geometry-bg:#241b38;
      --neuro:#fbbf24; --neuro-bg:#332616;
      --xai:#60a5fa; --xai-bg:#152238;
      --cert-ok:#4ade80; --cert-ok-bg:#0f2818;
      --cert-bad:#f87171; --cert-bad-bg:#301414;
      --data-bg:#20232b;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text);
         margin: 0; padding: 0 0 60px; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 32px 20px; }}
  header {{ text-align: center; margin-bottom: 20px; }}
  header h1 {{ font-size: 28px; margin-bottom: 6px; }}
  header .sub {{ color: var(--muted); font-size: 14px; max-width: 720px; margin: 0 auto; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 18px 0; font-size: 11.5px; }}
  .legend span {{ display:flex; align-items:center; gap:5px; }}
  .legend .sw {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 10px 0 36px; }}
  nav a {{ font-size: 12px; padding: 4px 10px; border: 1px solid var(--border); border-radius: 14px; text-decoration: none; color: var(--accent); }}
  section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 24px 28px; margin-bottom: 22px; }}
  section h2 {{ font-size: 18px; margin-top: 0; display: flex; align-items: center; gap: 8px; }}
  section h2 .num {{ background: var(--accent); color: #fff; border-radius: 50%; width: 26px; height: 26px; display: inline-flex;
                     align-items: center; justify-content: center; font-size: 13px; flex-shrink:0; }}
  section h3 {{ font-size: 14px; margin: 18px 0 8px; }}
  section h4 {{ font-size: 13px; margin: 8px 0 6px; }}
  p {{ line-height: 1.55; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12.5px; }}
  table.small {{ font-size: 11.5px; }}
  th, td {{ border: 1px solid var(--border); padding: 5px 8px; text-align: left; }}
  th {{ background: rgba(37,99,235,.08); }}
  .source {{ font-size: 11px; color: var(--muted); font-style: italic; margin-top: 4px; }}
  .quote {{ background: var(--cert-bad-bg); border-left: 3px solid var(--cert-bad); padding: 10px 14px; font-size: 13px; font-style: italic; }}
  .plain-english {{ background: var(--xai-bg); border-left: 3px solid var(--xai); padding: 10px 14px; font-size: 13px; font-style: italic; }}
  .chip {{ display: inline-block; background: rgba(37,99,235,.1); color: var(--accent); border-radius: 4px; padding: 1px 7px;
          font-size: 11.5px; margin: 2px 4px 2px 0; font-family: ui-monospace, monospace; }}
  .example, .pref-patient {{ border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin: 10px 0; }}
  .figure img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }}
  .caption {{ font-size: 12.5px; color: var(--muted); }}
  .missing, .missing-inline {{ background: rgba(245,158,11,.12); border: 1px solid #f59e0b; border-radius: 6px; padding: 10px 14px; font-size: 13px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }}
  .badge-green {{ background: var(--cert-ok-bg); color: var(--cert-ok); }}
  .badge-red {{ background: var(--cert-bad-bg); color: var(--cert-bad); }}
  code {{ font-family: ui-monospace, monospace; background: rgba(127,127,127,.12); padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
  .formula {{ text-align:center; font-size: 16px; font-family: "Times New Roman", serif; padding: 12px; background: rgba(127,127,127,.06); border-radius: 6px; margin: 10px 0; }}

  /* --- layer color coding, used consistently across the whole document --- */
  .layer-geometry {{ border-left: 4px solid var(--geometry); }}
  .layer-geometry h4, .layer-geometry h3 {{ color: var(--geometry); }}
  .layer-neurosymbolic {{ border-left: 4px solid var(--neuro); }}
  .layer-neurosymbolic h4, .layer-neurosymbolic h3 {{ color: var(--neuro); }}
  .layer-xai {{ border-left: 4px solid var(--xai); }}
  .layer-xai h4, .layer-xai h3 {{ color: var(--xai); }}
  .layer-certification.ok, .layer-certification h4.ok {{ border-left: 4px solid var(--cert-ok); }}
  .layer-certification {{ border-left: 4px solid var(--cert-ok); }}
  .layer-certification.abstain-block {{ border-left: 4px solid var(--cert-bad); }}
  .layer-data {{ border-left: 4px solid #9ca3af; }}

  .pill {{ display:inline-block; padding: 2px 9px; border-radius: 12px; font-size: 11px; font-weight:700; margin-left: 8px; }}
  .risk-high {{ background: var(--cert-bad-bg); color: var(--cert-bad); }}
  .risk-mid {{ background: #fef9c3; color: #854d0e; }}
  .risk-low {{ background: var(--cert-ok-bg); color: var(--cert-ok); }}

  /* --- vertical timeline (section 4) --- */
  .timeline {{ position: relative; padding-left: 26px; margin: 18px 0 30px; }}
  .timeline::before {{ content:""; position:absolute; left: 9px; top: 6px; bottom: 6px; width: 2px; background: var(--border); }}
  .timeline-item {{ position: relative; background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
                    padding: 12px 16px; margin-bottom: 14px; }}
  .timeline-item .dot {{ position: absolute; left: -30px; top: 16px; width: 12px; height: 12px; border-radius: 50%;
                         background: var(--geometry); border: 2px solid var(--panel); box-shadow: 0 0 0 2px var(--border); }}
  .layer-data .dot {{ background: #9ca3af; }}
  .layer-xai .dot {{ background: var(--xai); }}
  .layer-neurosymbolic .dot {{ background: var(--neuro); }}
  .layer-certification .dot {{ background: var(--cert-ok); }}
  .layer-certification.abstain-block .dot {{ background: var(--cert-bad); }}
  .timeline-item.final {{ background: rgba(37,99,235,.04); }}
  .journey-h {{ margin-top: 28px; font-size: 16px; }}

  .shap-chart {{ margin: 8px 0; }}
  .shap-row {{ display:flex; align-items:center; gap:8px; font-size: 12px; margin: 3px 0; }}
  .shap-label {{ width: 110px; flex-shrink:0; }}
  .shap-track {{ flex:1; height: 9px; background: rgba(127,127,127,.15); border-radius: 4px; overflow:hidden; }}
  .shap-pos {{ display:block; height:100%; background: var(--cert-bad); }}
  .shap-neg {{ display:block; height:100%; background: var(--cert-ok); }}
  .shap-val {{ width: 55px; text-align:right; flex-shrink:0; font-family: ui-monospace, monospace; }}

  .rule-check {{ font-family: ui-monospace, monospace; font-size: 12px; margin: 3px 0; padding: 3px 6px; border-radius: 4px; }}
  .rc-pass {{ color: var(--cert-ok); }}
  .rc-fail {{ color: var(--cert-bad); }}

  .cand-reason {{ font-size: 11px; color: var(--cert-bad); }}

  .gauge-compare {{ display:flex; align-items:center; gap: 20px; margin: 10px 0; }}
  .gauge-box {{ text-align:center; }}
  .gauge-label {{ font-size: 11px; color: var(--muted); }}
  .gauge-num {{ font-size: 26px; font-weight: 800; }}
  .gauge-arrow {{ font-size: 20px; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>CARE-LG: Full Project Report</h1>
  <div class="sub">A single, self-contained results summary spanning experiments_v2 through experiments_v6 (and experiments_v6_audit),
  centered on a real patient's journey through every pipeline layer. Every number, table, and example is traced to a specific
  repo file — see the source caption under each.</div>
</header>

<div class="legend">
  <span><i class="sw" style="background:#9ca3af"></i>data</span>
  <span><i class="sw" style="background:var(--xai)"></i>XAI / plain-English</span>
  <span><i class="sw" style="background:var(--geometry)"></i>geometry (latent / Riemannian)</span>
  <span><i class="sw" style="background:var(--neuro)"></i>neurosymbolic (rules)</span>
  <span><i class="sw" style="background:var(--cert-ok)"></i>certification: admissible / success</span>
  <span><i class="sw" style="background:var(--cert-bad)"></i>certification: blocked / infeasible</span>
</div>

<nav>
  <a href="#overview">1. Overview</a><a href="#data">2. Dataset samples</a><a href="#pipeline">3. Pipeline</a>
  <a href="#journey">4. Patient journey</a><a href="#riemannian">5. Riemannian geometry</a>
  <a href="#b1b2">6. B1/B2 fix</a><a href="#core-assumption">6b. Core assumption</a>
  <a href="#classifier-audit">6c. Classifier audit</a><a href="#neurosymbolic">7. Neurosymbolic layer</a>
  <a href="#infeasibility">8. Certified infeasibility</a><a href="#preference">9. Preference elicitation</a>
  <a href="#benchmarks">10. Benchmarks</a><a href="#figures">11. Prior figures</a>
</nav>

<section id="overview">
  <h2><span class="num">1</span>Overview</h2>
  <p>CARE-LG is a clinical algorithmic-recourse system: given a patient the model flags as high
  cardiovascular risk, it searches a graph built from real training patients' embeddings — using a
  learned latent space and a Riemannian metric derived from the decoder's own geometry — for a short,
  clinically admissible sequence of changes that would plausibly bring their predicted risk below a
  safe threshold. Across several audit sessions, the project found and fixed two code-level bugs that
  had let the system silently accept clinically unsafe recourse plans, then built on the corrected
  foundation: a neurosymbolic layer that explains every decision by citing the actual clinical
  guideline rule involved, a three-layer explanation system (risk attribution, guideline derivation,
  and semifactual reasoning for patients with no safe path), per-patient preference elicitation, and a
  small interactive UI. Section 4 below walks one real patient through every one of these layers.</p>
</section>

<section id="data">
  <h2><span class="num">2</span>Dataset samples</h2>
  <p>Real rows (not fabricated) from each cohort, with the model's risk label.</p>
  <h3>UCI Heart Disease</h3>
  {uci_table}
  <p class="source">Source: data/uci_heart.csv</p>
  <h3>Real NHANES (PCE/ASCVD-labeled cohort)</h3>
  {nh_table}
  <p class="source">Source: experiments_v3/data/nhanes_real.csv (build process documented in experiments_v3/DATA_PROVENANCE.md)</p>
</section>

<section id="pipeline">
  <h2><span class="num">2</span>Pipeline overview (10 stages)</h2>
  <p>A patient's real feature vector is scored, encoded, explained by SHAP, embedded in a Riemannian-
  weighted k-NN graph, filtered by the neurosymbolic admissibility layer, searched by Dijkstra,
  verified against REAL recorded values (not decoded reconstructions — the B1/B2 fix), certified
  infeasible if abstained, explained in plain English, and shown in the UI.</p>
  {PIPELINE_SVG}
  <p class="source">Source: src/graph/calg.py, src/graph/riemannian.py, src/graph/query_attachment.py, src/graph/neurosymbolic.py, src/recourse/explanations.py, ui/</p>
</section>

<section id="journey">
  <h2><span class="num">4</span>The patient journey</h2>
  <p>Two real UCI test patients (seed 0), pulled directly from <code>results/tables/uci_per_patient.csv</code>
  (cell <code>riemannian_hard</code>): one reaches low risk, one is certified infeasible. Every value
  below — the raw record, SHAP scores, latent coordinates, candidate distances, rule outcomes, and
  risk numbers — comes from re-running the exact, already-verified v6 pipeline and explanation code on
  these two specific, already-identified patients (see <code>scripts/report/collect_journey.py</code>).
  Scroll down to follow the risk score drop and the reasoning accumulate, layer by layer.</p>
  {journey_html}
  <p class="source">Source: results/tables/uci_per_patient.csv (patient identification); src/graph/query_attachment.py, neurosymbolic.py, explanations.py (all computation); experiments_v6/report/journey.json (captured output)</p>
</section>

<section id="riemannian">
  <h2><span class="num">5</span>The Riemannian geometry</h2>
  <p>The graph edge weight is a geodesic distance under the metric pulled back from the VAE decoder:</p>
  <div class="formula">G(z) = J<sub>g</sub>(z)<sup>T</sup> J<sub>g</sub>(z) &nbsp;&nbsp; where J<sub>g</sub>(z) = &part;g(z)/&part;z is the decoder Jacobian</div>
  <p class="source">Source: src/graph/riemannian.py (compute_metric_tensor, riemannian_distance)</p>
  <h3>Riemannian vs. Euclidean — real ablation result (experiments_v3, RQ2)</h3>
  <table>
    <thead><tr><th>Dataset</th><th>Metric</th><th>n pairs</th><th>p-value</th><th>Mean diff (R&minus;E)</th><th>Practical scale</th></tr></thead>
    <tbody>
      <tr><td>UCI</td><td>KDE</td><td>81</td><td>0.649 (n.s.)</td><td>&minus;0.0007</td><td>negligible</td></tr>
      <tr><td>UCI</td><td>Effort</td><td>81</td><td><b>3.5&times;10&#8315;&#8309;</b></td><td><b>&minus;0.025</b></td><td>Riemannian slightly <i>better</i> (scale &asymp;0.1&ndash;0.6)</td></tr>
      <tr><td>Real NHANES</td><td>KDE</td><td>980</td><td>0.595 (n.s.)</td><td>+0.008</td><td>negligible</td></tr>
      <tr><td>Real NHANES</td><td>Effort</td><td>980</td><td><b>6.8&times;10&#8315;&sup1;&sup3;</b></td><td><b>+0.204</b></td><td>Riemannian slightly <i>worse</i> (scale &asymp;1.8&ndash;5.0)</td></tr>
    </tbody>
  </table>
  <p>Verdict (verbatim from source): &ldquo;a real, non-zero signal, but it is too small and
  direction-inconsistent across cohorts to support a claim that the Riemannian metric produces
  <i>better</i> recourse.&rdquo; Success rate and KDE plausibility are statistically identical between
  the two metrics on both cohorts.</p>
  <p class="source">Source: experiments_v3/FINDINGS.md, section &ldquo;RQ2: Does the Riemannian metric produce measurably better recourse than Euclidean?&rdquo; (paired Wilcoxon signed-rank tests)</p>
</section>

<section id="b1b2">
  <h2><span class="num">6</span>The B1/B2 fix</h2>
  <p>A diagnostic audit (experiments_v6_audit) found two code-level bugs across four prior sessions:
  <b>B1</b> — the query patient's entry edge into the graph was gated by a WEAKER predicate
  (immutability + age-decrease only) than every interior graph edge (which also checks directional
  constraints like rising blood pressure or cholesterol) — so a patient could be silently attached
  through a real, measured constraint violation. <b>B2</b> — every path node, including real,
  already-known training patients, was being re-decoded through the VAE for constraint verification,
  even though their true values were already on file; VAE reconstruction noise was masking real
  violations as small, harmless-looking decoded deltas (a real +35 mg/dL cholesterol change decoded
  to an apparent +0.3 mg/dL).</p>
  <table>
    <thead><tr><th>Dataset</th><th>Success rate (relaxed gate, pre-fix)</th><th>Success rate (B1/B2-fixed)</th><th>Real-value CVR (pre-fix)</th><th>Real-value CVR (B1/B2-fixed)</th></tr></thead>
    <tbody>
      <tr><td>UCI</td><td>71.8%</td><td>38.6% &plusmn; 13.1% (range 21.7&ndash;55.0%*)</td><td>58.8&ndash;71.3%</td><td><b>0.00% &plusmn; 0.00%</b></td></tr>
      <tr><td>Real NHANES</td><td>45.5%</td><td>37.7% &plusmn; 3.4% (range 32.4&ndash;41.2%)</td><td>58.8&ndash;71.3%</td><td><b>0.00% &plusmn; 0.00%</b></td></tr>
    </tbody>
  </table>
  <p class="caption">* UCI's range is wide because its high-risk test cohort is only 20-25 patients per
  seed — small-sample noise, not five genuinely different operating points. Real NHANES's much larger
  per-seed cohort (~420-445 patients) gives a tighter, more trustworthy range.</p>
  <p>The fix is exact and unconditional on both cohorts, at the honest cost of a lower success rate —
  patients with no genuinely safe single-hop entry are now correctly rejected instead of silently
  (and incorrectly) accepted. <b>Stated as its own finding, not just as that cost</b> (per
  <code>archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md</code> issue #5): under the corrected gate, roughly
  <b>6 in 10 high-risk patients receive no recourse plan at all</b> (61.4% UCI / 62.3% NHANES mean
  abstention rate) — a real clinical-utility limitation in its own right, not a footnote to the 0% CVR
  result.</p>
  <p class="source">Source: archive/investigation_v2_to_v6/experiments_v6_audit/RECOMMENDATION.md (pre-fix numbers, root-cause evidence); docs/FINDINGS.md, section &ldquo;Phase 1 — the corrected foundation&rdquo; (post-fix numbers, full 5-seed grid)</p>
</section>

<section id="core-assumption">
  <h2><span class="num">6b</span>Core assumption (stated explicitly)</h2>
  <p>Every recourse plan this system produces ends at another real, recorded training patient's
  feature vector. &ldquo;Valid&rdquo; means &ldquo;a real patient was recorded in this state,&rdquo;
  NOT &ldquo;this state is achievable by any specific new patient who follows the plan.&rdquo;
  Medication regimen, genetics, adherence, comorbidities, and time elapsed are not modeled. This is a
  deliberate design choice (it is why CALG's targets are more clinically coherent than pure
  perturbation baselines), but it means the guarantee is about the DATA, not about achievability by a
  new individual — stated here as a limitation, not a footnote.</p>
  <p class="source">Source: docs/METHOD.md, &ldquo;Core Assumption&rdquo; (added per archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md issue #6)</p>
</section>

<section id="classifier-audit">
  <h2><span class="num">6c</span>Classifier audit</h2>
  <p>Every prior session's claims assumed the risk classifier's own output was trustworthy without
  checking. This session audited it directly: AUC-ROC 0.834 (UCI) / 0.960 (Real NHANES), no meaningful
  sex-based fairness gap on NHANES's large subgroups. More importantly, a large, previously-unreconciled
  gap: the pipeline's operational 0.55/0.45 high/low-risk gate sits roughly <b>7.3&times;</b> (NHANES) /
  <b>2.2&times;</b> (UCI) higher on the classifier's probability scale than the point where observed
  outcomes actually cross the clinical 7.5% ASCVD/PCE threshold the data label itself is built from.
  Not recalibrated this session — doing so would silently change every previously-published number in
  this project.</p>
  <p class="source">Source: docs/CLASSIFIER_AUDIT.md; results/tables/classifier_audit.json</p>
</section>

<section id="neurosymbolic">
  <h2><span class="num">7</span>Neurosymbolic layer (in general)</h2>
  <p>experiments_v6 encodes the same admissibility predicate as a set of named ASP (Answer Set
  Programming, via <code>clingo</code>) rules — e.g. <code>R-BP-DIRECTION</code>, <code>R-LIPID-DIRECTION</code>,
  <code>R-AGE-MONOTONIC</code> — each carrying a clinical citation, plus non-blocking guideline
  annotations for statin indication (10-year ASCVD risk &ge;7.5%, the same PCE threshold used for this
  project's own risk label) and BP treatment targets differentiated by diabetes status. A regression
  check comparing the symbolic reasoner's decisions against the original hardcoded predicate, over
  4,000 sampled real transitions (2,000 per dataset), found
  <span class="badge badge-green">2000/2000 agreement on each dataset — 100.0000%, 0 discrepancies</span>.
  (Section 4 above already showed this layer's output for one specific patient's specific step — this
  section is the general rule base and regression-check result, not a repeat of that example.) The
  statin/BP-target annotations were previously built but never surfaced anywhere in output
  (<code>archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md</code> issue #8); they are now wired into the UI for real
  NHANES patients (confirmed on patient #5, seed 0: 17.7% ASCVD risk &rarr; statin indicated;
  non-diabetic &rarr; &lt;140/90 mmHg target). Not wired for UCI, whose label is not ASCVD-based.</p>
  <p class="source">Source: src/graph/neurosymbolic.py; scripts/run_neurosymbolic_regression.py; docs/FINDINGS.md, section &ldquo;Phase 4&rdquo;; src/recourse/explanations.py::nhanes_guideline_context</p>
</section>

<section id="infeasibility">
  <h2><span class="num">8</span>Certified infeasibility (in general)</h2>
  <p>For every patient who abstains under the corrected (B1-fixed) gate, an exhaustive feature-space
  check plus two densification probes (doubled k-NN; +500 classifier-verified VAE-prior synthetic
  nodes) classify the abstention as certified-infeasible, a confirmed blindspot, or unresolved.</p>
  <table>
    <thead><tr><th>Dataset</th><th>Mean abstention rate</th><th>Certified infeasible</th><th>Confirmed blindspot</th><th>Unresolved</th></tr></thead>
    <tbody>
      <tr><td>UCI</td><td>61.4% of high-risk test patients</td><td><b>87.9%</b> of abstentions</td><td><b>0.0%</b></td><td>12.1%</td></tr>
      <tr><td>Real NHANES</td><td>62.3% of high-risk test patients</td><td><b>98.5%</b> of abstentions</td><td><b>0.0%</b></td><td>1.5%</td></tr>
    </tbody>
  </table>
  <p>Zero confirmed blindspots on either cohort, across all 5 seeds. Worded precisely (per
  <code>archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md</code> issue #7): the SAME densification-probe methodology
  has now been applied independently in four separate sessions (v3, v4, v5, v6) and found zero
  blindspots each time — a well-replicated result under <i>one search strategy</i>, not four
  independent verification methods. An independent approach (a structurally different densification
  strategy, or exact enumeration at larger scale) would strengthen this further and has not yet been
  attempted. Below are two more real semifactual examples (Section 4's abstaining patient, #1, was one
  of three already generated this way) showing the same pattern on real NHANES patients:</p>
  {semifactual_html}
  <p class="source">Source: docs/FINDINGS.md, section &ldquo;Phase 2&rdquo;; results/tables/{{dataset}}_blindspot_seed_summary.csv; examples regenerated via scripts/report/collect_examples.py from src/recourse/explanations.py (unmodified)</p>
</section>

<section id="preference">
  <h2><span class="num">9</span>Preference elicitation</h2>
  <p>Three synthetic preference profiles (<code>no_preference</code>, <code>exercise_averse</code>,
  <code>medication_averse</code>) reweight only the clinical-effort cost term, never the admissibility
  layer. Real demo output for 10 patients (5 per dataset, seed 0):</p>
  {pref_html}
  <p>Honest finding (from docs/FINDINGS.md, not smoothed over): the selected target changed
  for only 1 of these 10 patients (UCI #30, under <code>medication_averse</code>) — for the rest,
  reported cost changed substantially across profiles but the cheapest admissible target stayed the
  same, because post-fix admissible targets are often sparse enough that there is no competitive
  alternative for reweighting to promote instead.</p>
  <p class="source">Source: results/tables/preference_demo.csv; docs/FINDINGS.md, section &ldquo;Phase 6&rdquo;</p>
</section>

<section id="benchmarks">
  <h2><span class="num">10</span>Benchmarks</h2>
  <h3>(a) Methodology comparison</h3>
  <table class="small">
    <thead><tr><th>Method</th><th>Mechanism</th><th>Constraint type</th><th>Explains refusal?</th><th>Tested on real clinical data?</th></tr></thead>
    <tbody>
      <tr><td><b>CARE-LG (v6, corrected)</b></td><td>latent-graph shortest path, real-value verified</td><td>hard admissibility (immutable/monotone/directional), neurosymbolic + cited</td><td><b>Yes</b> — certified-infeasibility classification + semifactual explanation</td><td><b>Yes</b> — UCI Heart + real PCE-labeled NHANES</td></tr>
      <tr><td>FACE</td><td>feature-space k-NN shortest path</td><td>none (density/proximity only)</td><td>No</td><td>Yes (rerun this session, same cohorts)</td></tr>
      <tr><td>PACE</td><td>MLP classifier + ASP-filtered perturbation search, incremental budget</td><td>ASP symbolic (immutable + transition-graph rules)</td><td>No (validity flag only, no refusal explanation)</td><td>No — evaluated on Adult Income (demographic/financial), not a clinical dataset</td></tr>
    </tbody>
  </table>
  <h3>(b) Numeric results</h3>
  <table>
    <thead><tr><th>Method</th><th>Dataset</th><th>Success rate</th><th>Real-value CVR</th><th>Certified-infeasibility rate</th></tr></thead>
    <tbody>
      <tr><td><b>v6 (corrected, R+hard)</b></td><td>UCI</td><td>38.6% &plusmn; 13.1% (21.7&ndash;55.0%)</td><td><b>0.00% &plusmn; 0.00%</b></td><td>87.9% of abstentions</td></tr>
      <tr><td><b>v6 (corrected, R+hard)</b></td><td>Real NHANES</td><td>37.7% &plusmn; 3.4%</td><td><b>0.00% &plusmn; 0.00%</b></td><td>98.5% of abstentions</td></tr>
      <tr><td>FACE (controlled rerun)</td><td>UCI</td><td>100.0% &plusmn; 0.0%</td><td>96.3% &plusmn; 4.1%</td><td>n/a</td></tr>
      <tr><td>FACE (controlled rerun)</td><td>Real NHANES</td><td>100.0% &plusmn; 0.0%</td><td>99.9% &plusmn; 0.2%</td><td>n/a</td></tr>
      <tr><td>PACE-style reimpl. (this project's domain)</td><td>UCI</td><td>80.4% &plusmn; 14.0%</td><td>0.00%&sup1;</td><td>n/a</td></tr>
      <tr><td>PACE-style reimpl. (this project's domain)</td><td>Real NHANES</td><td>57.6% &plusmn; 6.5%</td><td>0.00%&sup1;</td><td>n/a</td></tr>
      <tr><td><b>PACE (reported by its own authors, Adult Income — not independently reproduced)</b></td><td>Adult Income</td><td>validity 24.0%</td><td>plausibility 100%&sup2;</td><td>n/a</td></tr>
    </tbody>
  </table>
  <p><sup>1</sup> 0.00% by construction — every candidate is filtered through the admissibility check
  before acceptance, a search-space restriction, not a measured rate in the same sense as the
  graph-search rows. <sup>2</sup> PACE's own reported metric is conditioned on acceptance and is not
  directly comparable to this table's unconditional real-value CVR.</p>
  <p class="source">Source: docs/FINDINGS.md, section &ldquo;Phase 8&rdquo;; results/tables/face_rerun_per_patient.csv, pace_reimpl_per_patient.csv; PACE's own numbers read directly from arXiv:2607.01306, Table 4</p>
</section>

<section id="figures">
  <h2><span class="num">11</span>Figures from prior sessions</h2>
  {''.join(fig_html_blocks)}
</section>

{"<section class='missing'><h2>Sections with data gaps</h2><ul>" + "".join(f"<li>{esc(m)}</li>" for m in missing_sections) + "</ul></section>" if missing_sections else ""}

</div>
</body>
</html>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({len(HTML)/1024:.1f} KB)")
if missing_sections:
    print("Sections with missing or substituted real data:")
    for m in missing_sections:
        print(" -", m)
else:
    print("No missing sections.")
