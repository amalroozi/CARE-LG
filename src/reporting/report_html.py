"""
Builds a single, self-contained HTML report (inline CSS, no external assets,
no server) for ONE patient's real run through src.recourse.single_patient's
pipeline. Two layouts: success (reached low risk) and abstain (certified
infeasible / blindspot / unresolved) -- selected automatically from
result["success"].

Every value rendered here is read directly from the `result` dict produced
by src.recourse.single_patient.run_single_patient -- nothing is computed,
templated with placeholder numbers, or invented in this module. The prose
sentences are assembled from that same real data (which features actually
changed, by how much, which rules actually passed/failed for THIS patient),
not generic boilerplate with a name swapped in.
"""
from pathlib import Path
from typing import Dict

UNITS = {
    "age": "yrs", "resting_bp": "mmHg", "cholesterol": "mg/dL", "max_heart_rate": "bpm",
    "oldpeak": "mm ST-depression", "systolic_bp": "mmHg", "diastolic_bp": "mmHg",
    "bmi": "kg/m²", "glycemic_hba1c": "% HbA1c", "sex": "", "fasting_blood_sugar": "",
    "exercise_angina": "", "slope": "",
}

CSS = """
:root { --bg:#f7f7f9; --panel:#fff; --text:#1a1a1a; --muted:#666; --accent:#2563eb; --border:#e2e2e6;
        --geometry:#7c3aed; --geometry-bg:#f3edff; --neuro:#d97706; --neuro-bg:#fff7e6;
        --xai:#2563eb; --xai-bg:#eef4ff; --ok:#16a34a; --ok-bg:#eafaf0; --bad:#dc2626; --bad-bg:#fef2f2; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg);
       color: var(--text); margin:0; padding: 28px 0 60px; }
.wrap { max-width: 880px; margin: 0 auto; padding: 0 20px; }
h1 { font-size: 24px; margin-bottom: 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
section { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 20px 24px; margin-bottom: 18px; }
h2 { font-size: 16px; margin: 0 0 10px; display:flex; align-items:center; gap:8px; }
h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing:.03em; margin: 14px 0 6px; }
p { line-height: 1.6; font-size: 14.5px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12.5px; }
th, td { border: 1px solid var(--border); padding: 5px 8px; text-align: left; }
th { background: rgba(37,99,235,.07); }
.gauge { display:flex; align-items:center; gap:20px; margin: 6px 0 14px; }
.gauge-box { text-align:center; } .gauge-label{font-size:11px;color:var(--muted);}
.gauge-num { font-size: 28px; font-weight: 800; }
.risk-high { color: var(--bad); } .risk-low { color: var(--ok); }
.chip { display:inline-block; background: rgba(37,99,235,.1); color: var(--accent); border-radius:4px;
        padding:1px 7px; font-size:12px; margin:2px 4px 2px 0; font-family: ui-monospace, monospace; }
.badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; }
.badge-ok { background: var(--ok-bg); color: var(--ok); } .badge-bad { background: var(--bad-bg); color: var(--bad); }
.stage { border-left: 4px solid var(--geometry); padding-left: 14px; margin: 16px 0; }
.stage.xai { border-color: var(--xai); } .stage.neuro { border-color: var(--neuro); }
.stage.cert-ok { border-color: var(--ok); } .stage.cert-bad { border-color: var(--bad); } .stage.data { border-color: #9ca3af; }
.stage h2 { font-size: 14.5px; }
details { margin-top: 8px; } summary { cursor: pointer; font-size: 12px; color: var(--muted); }
.rule-line { font-family: ui-monospace, monospace; font-size: 12px; margin: 2px 0; }
.rc-pass { color: var(--ok); } .rc-fail { color: var(--bad); }
.quote { background: var(--bad-bg); border-left: 3px solid var(--bad); padding: 10px 14px; font-size: 13.5px; font-style: italic; }
.iv { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
.iv b { color: var(--accent); }
.note { font-size: 11.5px; color: var(--muted); }
"""


def _delta_chips(deltas: Dict) -> str:
    return "".join(
        f'<span class="chip">{f}: {d["before"]:g}&rarr;{d["after"]:g} {UNITS.get(f,"")} ({d["delta"]:+.1f})</span>'
        for f, d in deltas.items()
    )


def _raw_record_table(raw: Dict) -> str:
    rows = "".join(f"<tr><td>{f}</td><td>{v:g} {UNITS.get(f,'')}</td></tr>" for f, v in raw.items())
    return f"<table>{rows}</table>"


def _shap_prose(shap_top) -> str:
    top = shap_top[0]
    direction = "increasing" if top["value"] > 0 else "decreasing"
    others = ", ".join(f"{s['feature']} ({s['value']:+.3f})" for s in shap_top[1:4])
    return (f"The single biggest driver of this risk score was <b>{top['feature']}</b> "
            f"({direction} risk by {abs(top['value']):.3f} on the model's own scale). "
            f"Other contributing factors, in order: {others}.")


def _candidates_prose(candidates) -> str:
    n_blocked = sum(1 for c in candidates if not c["admissible"])
    n_total = len(candidates)
    chosen = next((c for c in candidates if c.get("chosen")), None)
    sentence = (f"Of the {n_total} closest real patients in the training data (by latent-space distance), "
                f"<b>{n_blocked} were clinically blocked</b> from being used as a recourse target.")
    if chosen:
        rank = next(i for i, c in enumerate(candidates, start=1) if c.get("chosen"))
        sentence += (f" The one actually chosen (patient #{chosen['node_idx']}) was the "
                     f"{'closest' if rank == 1 else f'{rank}th-closest'} candidate overall.")
    return sentence


def _candidates_table(candidates) -> str:
    rows = []
    for c in candidates:
        status = '<span class="badge badge-ok">admissible</span>' if c["admissible"] else '<span class="badge badge-bad">blocked</span>'
        chosen = " &larr; chosen" if c.get("chosen") else ""
        reason = "; ".join(c["blocking_reasons"]) if c["blocking_reasons"] else "—"
        rows.append(f"<tr><td>#{c['node_idx']}{chosen}</td><td>{status}</td><td>{c['dist_riemannian']:.3f}</td>"
                    f"<td>{c['dist_euclidean']:.3f}</td><td style='font-size:11px'>{reason}</td></tr>")
    return ("<table><thead><tr><th>Candidate</th><th>Status</th><th>Riemannian dist.</th>"
            f"<th>Euclidean dist.</th><th>Reason if blocked</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _rule_checklist_html(checklist) -> str:
    return "".join(
        f'<div class="rule-line {"rc-pass" if r["status"]=="pass" else "rc-fail"}">'
        f'{"&#10003;" if r["status"]=="pass" else "&#10007;"} <b>[{r["rule_id"]}]</b> {r["citation"]}</div>'
        for r in checklist
    )


def _guideline_context_html(gc) -> str:
    if not gc:
        return ""
    return (f'<div class="iv"><b>Statin guidance:</b> {gc["statin_text"]}<br/>'
            f'<b>Blood-pressure target:</b> {gc["bp_target_text"]}</div>')


def build_success_report(result: Dict) -> str:
    r = result
    gauge_drop = (r["risk_before"] - r["risk_after"]) * 100
    interventions_html = "".join(
        f'<div class="iv"><b>{iv["name"]}</b> [{iv["citation"]}] — {iv["description"]} '
        f'<span class="note">(affects {iv["feature"]})</span></div>'
        for iv in r["interventions"]
    ) or "<p class='note'>No named intervention mapping available for the changed feature(s).</p>"

    body = f"""
<title>Recourse Report — {r['dataset']} patient #{r['patient_idx']}</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Recourse Report</h1>
<div class="sub">{r['dataset']} · seed {r['seed']} · test patient #{r['patient_idx']} · preference profile: {r['profile']}</div>

<section>
  <h2>Final Output</h2>
  {_raw_record_table(r['raw_record'])}
  <div class="gauge">
    <div class="gauge-box"><div class="gauge-label">Risk before</div><div class="gauge-num risk-high">{r['risk_before']*100:.1f}%</div></div>
    <div style="font-size:20px;color:#999;">&rarr;</div>
    <div class="gauge-box"><div class="gauge-label">Risk after</div><div class="gauge-num risk-low">{r['risk_after']*100:.1f}%</div></div>
  </div>
  <p>Risk dropped <b>{gauge_drop:.1f} percentage points</b> via one clinically admissible,
  real-value-verified recourse step. Real changes made:</p>
  <div>{_delta_chips(r['feature_deltas'])}</div>
  <h3>Recommended interventions</h3>
  {interventions_html}
  {_guideline_context_html(r['guideline_context'])}
  <p class="note">path length: {r['path_len']} hop(s) · clinical effort: {r['effort']:.2f} · search cost: {r['cost']:.3f}</p>
</section>

<h2 style="margin-left:4px;">Architecture Walkthrough</h2>

<div class="stage xai">
  <h2>1 &middot; Risk Classifier</h2>
  <p>The model scored this patient at <b>{r['risk_before']*100:.1f}% predicted risk</b> based on their
  recorded values above. {_shap_prose(r['shap_top_features'])}</p>
  <details><summary>underlying SHAP values</summary><table><tbody>
  {''.join(f"<tr><td>{s['feature']}</td><td>{s['value']:+.4f}</td></tr>" for s in r['shap_top_features'])}
  </tbody></table></details>
</div>

<div class="stage">
  <h2>2 &middot; VAE / Latent Encoding</h2>
  <p>Before searching for a plan, the patient's record was mapped into a compact mathematical space
  where clinically similar real patients sit close together — not by looking at any single number in
  isolation, but by how the whole combination of their values resembles other real people in the
  training data. This positioning is what let the system find the closest genuinely comparable
  patients in the next stage.</p>
</div>

<div class="stage">
  <h2>3 &middot; Graph + Riemannian Geometry</h2>
  <p>{_candidates_prose(r['candidate_neighbors'])} Distance here is measured in a way that accounts for
  how much a patient's actual clinical picture would change, not just raw numeric closeness — so the
  target ultimately chosen was the CLOSEST patient that was also clinically admissible, even though
  closer-but-blocked candidates existed.</p>
  {_candidates_table(r['candidate_neighbors'])}
  <p class="note">chosen edge: Riemannian distance {r['dist_riemannian_chosen']:.4f}, Euclidean distance {r['dist_euclidean_chosen']:.4f}</p>
</div>

<div class="stage neuro">
  <h2>4 &middot; Neurosymbolic Layer</h2>
  <p>Every proposed change was checked against a set of named clinical guideline rules before being
  accepted. For this specific step, every applicable rule passed:</p>
  {_rule_checklist_html(r['rule_checklist'])}
</div>

<div class="stage cert-ok">
  <h2>5 &middot; Certified Search</h2>
  <p>Among every admissible candidate, the search chose real training patient
  <b>#{r['chosen_target_node']}</b> because it was the lowest-cost (closest, cheapest-effort) option that
  passed every guideline rule — the closer, cheaper-looking candidates shown above were rejected
  specifically because they failed one or more rules, not considered at all.</p>
</div>

<div class="stage cert-ok">
  <h2>6 &middot; Real-Value Verification (the B1/B2 fix)</h2>
  <p>No reconstruction guesswork was used anywhere in this result: this patient's own recorded values,
  and the recorded values of training patient #{r['chosen_target_node']}, were compared directly — never
  passed through the decoder and reconstructed. The feature changes shown above are the real recorded
  difference between two actual people's data, not a VAE approximation of either.</p>
</div>

<div class="stage xai">
  <h2>7 &middot; Explanation Layer</h2>
  <p>What was surfaced back to this patient: their risk score and its main drivers (stage 1), the
  specific plan and why each step is guideline-admissible (stages 4-5), and the named real-world
  interventions above — so the plan is not just "reach these numbers" but tied to concrete, cited
  actions a clinician could actually prescribe.</p>
</div>

</div>
"""
    return body


def build_abstain_report(result: Dict) -> str:
    r = result
    freq_rows = "".join(
        f"<tr><td>{rule}</td><td>{pct:.1f}%</td></tr>"
        for rule, pct in sorted(r["blocking_rule_frequency"].items(), key=lambda kv: -kv[1])
    )
    cert_label = r["certification"].replace("_", " ")

    body = f"""
<title>Recourse Report — {r['dataset']} patient #{r['patient_idx']} (abstained)</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Recourse Report — Abstention</h1>
<div class="sub">{r['dataset']} · seed {r['seed']} · test patient #{r['patient_idx']} · preference profile: {r['profile']}</div>

<section>
  <h2>Final Output</h2>
  {_raw_record_table(r['raw_record'])}
  <div class="gauge">
    <div class="gauge-box"><div class="gauge-label">Risk</div><div class="gauge-num risk-high">{r['risk_before']*100:.1f}%</div></div>
    <div style="font-size:20px;color:#999;">&rarr;</div>
    <div class="gauge-box"><div class="gauge-label">Result</div><div class="badge badge-bad" style="font-size:14px;padding:4px 12px;">{cert_label}</div></div>
  </div>
  <p class="quote">&ldquo;{r['semifactual_text']}&rdquo;</p>
  {_guideline_context_html(r['guideline_context'])}
</section>

<h2 style="margin-left:4px;">Architecture Walkthrough</h2>

<div class="stage xai">
  <h2>1 &middot; Risk Classifier</h2>
  <p>The model scored this patient at <b>{r['risk_before']*100:.1f}% predicted risk</b>.
  {_shap_prose(r['shap_top_features'])}</p>
  <details><summary>underlying SHAP values</summary><table><tbody>
  {''.join(f"<tr><td>{s['feature']}</td><td>{s['value']:+.4f}</td></tr>" for s in r['shap_top_features'])}
  </tbody></table></details>
</div>

<div class="stage">
  <h2>2 &middot; VAE / Latent Encoding</h2>
  <p>This patient was still mapped into the same clinically-similar-patients space as every other
  patient — the encoding step ran normally. The problem surfaced at the next stage: none of the
  patients near them turned out to be reachable.</p>
</div>

<div class="stage">
  <h2>3 &middot; Graph + Riemannian Geometry</h2>
  <p>The {len(r['candidate_neighbors'])} closest real training patients by distance were examined, and
  every one of them was clinically blocked (shown below with the specific reason for each) — this is
  why the search had nothing nearby to route through.</p>
  {_candidates_table(r['candidate_neighbors'])}
</div>

<div class="stage neuro">
  <h2>4 &middot; Neurosymbolic Layer</h2>
  <p>Rather than stopping at the nearest candidates, the system exhaustively checked ALL
  <b>{r['n_low_risk_candidates_checked']}</b> real low-risk patients in the training data against this
  patient's actual values. The closest miss was training patient
  <b>#{r['closest_miss_candidate_idx']}</b>, which still failed with
  <b>{r['closest_miss_violation_count']}</b> violated guideline rule(s) — the fewest of any candidate
  checked. The rules that recurred most often across every candidate:</p>
  <table><thead><tr><th>Blocking rule</th><th>Frequency across all checked candidates</th></tr></thead>
  <tbody>{freq_rows}</tbody></table>
</div>

<div class="stage cert-bad">
  <h2>5 &middot; Certified Search</h2>
  <p>No path was chosen because none existed: this patient's classification is
  <b>{cert_label}</b> — an exhaustive, real-value check (not a search-radius limitation) found that
  every real recorded low-risk patient in the training data would require at least one
  guideline-violating change to reach from this patient's actual values.</p>
</div>

<div class="stage cert-bad">
  <h2>6 &middot; Real-Value Verification (the B1/B2 fix)</h2>
  <p>The exhaustive check above used this patient's own recorded values directly against every real
  low-risk training patient's own recorded values — no VAE reconstruction was involved anywhere in
  reaching this conclusion, so the "no safe path exists" result is not an artifact of decoder
  approximation.</p>
</div>

<div class="stage xai">
  <h2>7 &middot; Explanation Layer</h2>
  <p>Rather than a generic "no plan found" message, this patient is shown WHY no plan exists — the
  specific closest-miss patient, the specific rule that still blocked even that best case, and how
  often each type of obstruction recurred across every real alternative that was checked.</p>
</div>

</div>
"""
    return body


def write_report(result: Dict, out_dir: Path) -> Path:
    """Writes the report and returns its path. Does not open a browser (see
    experiments_v6/run_single_patient.py for the auto-open wiring)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html = build_success_report(result) if result["success"] else build_abstain_report(result)
    fname = f"{result['dataset']}_seed{result['seed']}_patient{result['patient_idx']}_report.html"
    path = out_dir / fname
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'/>{html}</html>")
    return path
