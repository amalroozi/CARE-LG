"""Condensed final report in the layout of ../results/final_report.html (PIPELINE P15).

Order: (1) Final Output for one real high-risk patient, (2) Architecture Walkthrough of how the pipeline
produced it, stage by stage, (3) Final Results -- one headline table, then numbered figures with a one-line
note and caption each. Every number appears once. Figures are inlined (self-contained page).

  .venv/bin/python -m experiment_final.src.report_final [--dataset nhanes_pooled --seed 2 --test-pos 139]
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_final import adapters
from experiment_final.src.common import FIGURES, RESULTS, RUNS, TABLES, load_config
from experiment_final.src.knowledge_layer import KNOWLEDGE, load_knowledge, schema_from_meta

DS = {"nhanes_pooled": "NHANES 1999&ndash;2023 (pooled)", "nhanes": "NHANES 2017&ndash;2020", "uci": "UCI Heart", "cardio": "Kaggle Cardio"}
ORDER = ["nhanes_pooled", "nhanes", "uci", "cardio"]
IV_NAME = {"dash_sodium": "DASH diet + sodium restriction", "weight_management": "Weight management", "exercise": "Aerobic exercise programme",
           "smoking_cessation": "Smoking cessation", "alcohol_moderation": "Alcohol moderation", "statin": "Statin",
           "acei_arb": "ACE inhibitor / ARB", "thiazide": "Thiazide diuretic", "ccb": "Calcium-channel blocker (dihydropyridine)",
           "beta_blocker_nonselective": "Non-selective beta-blocker", "metformin": "Metformin", "sglt2i": "SGLT2 inhibitor", "glp1ra": "GLP-1 receptor agonist"}
STEP = {"lifestyle": "lifestyle", "single_drug": "one drug", "combination": "combination"}

CSS = """
body{font-family:-apple-system,sans-serif;background:#f7f7f9;margin:0;padding:24px;color:#111}
.wrap{max-width:1100px;margin:0 auto}h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 8px}h3{font-size:13px;margin:14px 0 4px;color:#333}
section{background:#fff;border:1px solid #e2e2e6;border-radius:10px;padding:18px 22px;margin-bottom:18px}
section.headline{border:2px solid #16a34a;background:#f0fdf4}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #e2e2e6;padding:4px 8px;text-align:left}
th{background:rgba(37,99,235,.08)}p{line-height:1.55;font-size:14px}.note{font-size:12px;color:#666}
.imgcap{font-size:11px;color:#555;font-style:italic;margin:4px 2px 0}.fig svg{width:100%;height:auto;display:block}
.badge{background:#dcfce7;color:#166534;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:700;margin-left:8px}
.gauge{display:flex;align-items:center;gap:20px;margin:6px 0 14px}.gauge-box{text-align:center}.gauge-label{font-size:11px;color:#666}
.gauge-num{font-size:28px;font-weight:800}.risk-high{color:#dc2626}.risk-low{color:#16a34a}
.chip{display:inline-block;background:rgba(37,99,235,.1);color:#2563eb;border-radius:4px;padding:1px 7px;font-size:12px;margin:2px 4px 2px 0;font-family:ui-monospace,monospace}
.ivgrid{display:block;margin:6px 0}
.iv{display:flex;gap:16px;align-items:baseline;border:0;border-bottom:1px solid #ececf0;border-radius:0;padding:6px 4px;margin:0;font-size:13.5px;line-height:1.45}.iv>.ivh{flex:0 0 270px}.iv>div:not(.ivh){flex:1}.iv .ivh{font-size:15.5px;font-weight:700;color:#1d4ed8}
.iv .ivs{margin:0 0 2px;font-size:14px;color:#111}.iv .ivm{font-size:11.5px;color:#777}.iv .ivstep{display:inline-block;background:rgba(37,99,235,.1);color:#1d4ed8;border-radius:4px;padding:0 7px;font-size:11px;font-weight:600;margin-left:7px;vertical-align:2px}
.iv.off{background:#fafafa;color:#555;padding:5px 10px;font-size:11.5px;line-height:1.35}.iv.off .ivh{color:#dc2626;font-size:12.5px}.iv.off .ivs{font-size:11.5px;margin:2px 0 0;color:#555}
.ivgrid.offgrid .iv>.ivh{flex:0 0 270px}
.stage{border-left:4px solid #7c3aed;padding-left:14px;margin:16px 0}.stage.xai{border-color:#2563eb}.stage.neuro{border-color:#d97706}
.stage.ok{border-color:#16a34a}.stage.data{border-color:#9ca3af}.stage h2{font-size:14.5px}
.rule{font-family:ui-monospace,monospace;font-size:12px;margin:2px 0}.pass{color:#16a34a}
pre{white-space:pre-wrap;font-size:11.5px;background:#f7f7f9;border:1px solid #e2e2e6;border-radius:6px;padding:8px}
details summary{cursor:pointer;font-size:12px;color:#666}
"""


def svg(name: str) -> str:
    """Newest version of a figure, inlined (strip XML prolog)."""
    c = sorted(FIGURES.glob(f"{name}*.svg"), key=lambda p: p.stat().st_mtime)
    s = c[-1].read_text()
    return re.sub(r"<\?xml[^>]*>|<!DOCTYPE[^>]*>", "", s)


def fig(no: int, title: str, note: str, name: str, cap: str) -> str:
    return (f"<section><h2>Fig. {no} &middot; {title}</h2><p class='note'>{note}</p><div class='fig'>{svg(name)}</div>"
            f"<p class='imgcap'>{cap}</p></section>")


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def pct(v, d=1):
    return "&ndash;" if pd.isna(v) else f"{v:.{d}f}%"


# ------------------------------------------------------------------ patient

def load_patient(view: Path, ds: str, seed: int, pos: int) -> dict:
    d = view / ds / f"seed{seed}"
    card = next(c for c in map(json.loads, open(d / "cards.jsonl")) if c["test_pos"] == pos)
    ex = next(e for e in map(json.loads, open(d / "explanations.jsonl")) if e["test_pos"] == pos)
    p = pd.read_csv(d / "patients.csv").set_index("test_pos").loc[pos]
    return {"card": card, "expl": ex, "row": p, "setup": json.loads((d / "setup.json").read_text())}


def final_output(P: dict, units: dict) -> str:
    c, r, cd = P["card"], P["row"], P["card"]["card"]
    feats = P["setup"]["features"]
    base = "".join(f"<tr><td>{f}</td><td>{v:g} {units.get(f, '')}</td></tr>" for f, v in zip(feats, c["x0"]))
    chips = "".join(f"<span class='chip'>{h['feature']}: {h['from']:g}&rarr;{h['to']:g} ({h['delta']:+.1f})</span>" for h in cd["changes"])
    K = load_knowledge()
    sentence = pd.read_csv(KNOWLEDGE / "intervention_text.csv", dtype=str, keep_default_na=False).set_index("intervention_class")["action_sentence"]
    from experiment_final.src.narrate import NAME, plan_values
    pv = plan_values(cd, K, schema_from_meta(adapters.get(c["dataset"]).FEATURE_META, K), np.asarray(c["x0"], float), None)["effect"]
    ivs = ""
    for it in cd["recommendations"]:
        note = K.iv(it["class"])["notes"]
        eff = [f"{NAME.get(f, f)} by about {pv[(it['class'], f)]:g} {units.get(f, '')}".rstrip() for f in cd["changes"] and [x["feature"] for x in cd["changes"]]
               if (it["class"], f) in pv and pv[(it["class"], f)] > 0]
        sent = sentence[it["class"]] + (" Typically lowers " + ", ".join(eff[:-1]) + (" and " if len(eff) > 1 else "") + eff[-1] + "." if eff else "")
        ivs += (f"<div class='iv'><div class='ivh'>{it['rank']}. {IV_NAME.get(it['class'], it['class'])}<span class='ivstep'>{STEP[it['step']]}</span></div>"
                f"<div><div class='ivs'>{html.escape(sent)}</div>"
                f"<div class='ivm'>[{it['citation'].replace(';', ', ')}] &mdash; {html.escape(note)} (acts on {', '.join(it['features'])})</div></div></div>")
    seen, offs = set(), ""
    for e in cd["excluded"]:
        if (e["class"], e["reason"]) in seen or e["reason"] == "not_indicated":
            continue
        seen.add((e["class"], e["reason"]))
        why = {"already_taking": "already taking &mdash; not re-recommended (intensification not modelled)",
               "contraindicated": "contraindicated"}[e["reason"]]
        if e.get("derivation"):
            rid, _, node, conc = e["derivation"][0]
            why += f": rule <b>{rid}</b> attached at ICD-10-CM <b>{node}</b>, inherited by this patient's code <b>{conc}</b>"
        offs += f"<div class='iv off'><div class='ivh'>&times; {IV_NAME.get(e['class'], e['class'])}</div><div><div class='ivs'>{why}</div></div></div>"
    legacy = f"""<p>Risk falls by <b>{100 * r['risk_drop']:.1f} percentage points</b> by moving to the recorded values of a real lower-risk patient of the
same sex and age, reached in {int(r['n_hops'])} admissible step and verified against this patient's own recorded values. Changes:</p>
<div>{chips}</div>
<h3>Recommended interventions (escalation order: lifestyle &rarr; one drug &rarr; combination)</h3><div class='ivgrid'>{ivs}</div>
<h3 style='font-size:13px'>Removed for this patient</h3><div class='ivgrid offgrid'>{offs}</div>
<p class="note">Escalation: lifestyle alone does not cover the required blood-pressure fall, so drugs are added up to a
{STEP[cd['min_sufficient_tier']]}. Decision support only; clinician judgement required.</p>"""
    return f"""
<section class="headline"><h1>Recommendation &mdash; Getting This Patient to Low Risk</h1>
<p class="note">One real high-risk patient from pooled NHANES, worked end to end: the recommended plan first, then exactly how the
pipeline arrived at it. {DS[c['dataset']]} &middot; seed {c['seed']} &middot; test patient #{c['test_pos']} &middot; cycle {r['sub_cycle']}
&middot; ICD-10-CM concepts {', '.join(c['concepts'])} &middot; current medication classes {', '.join(c['current_meds']) or 'none'}</p>
<section><h2>Final Output <span class="badge">Tier 1 &middot; certified real-patient recourse</span></h2>
<table>{base}</table>
<div class="gauge"><div class="gauge-box"><div class="gauge-label">Predicted risk now</div><div class="gauge-num risk-high">{100 * r['r0']:.1f}%</div></div>
<div style="font-size:20px;color:#999">&rarr;</div><div class="gauge-box"><div class="gauge-label">Risk at the recourse target</div><div class="gauge-num risk-low">{100 * r['target_risk']:.1f}%</div></div></div>
{legacy}</section>
"""


def walkthrough(P: dict) -> str:
    c, r, ex, cert = P["card"], P["row"], P["expl"], P["card"]["certificate"]
    shap = sorted(ex["shap"].items(), key=lambda kv: -abs(kv[1]))
    vias = [v for v in cert["rule_vias"] if v[1] != "ROOT"]
    rules = "".join(f"<div class='rule'>{rid} at <b>{node}</b> &larr; patient code <b>{conc}</b></div>" for rid, node, conc in vias)
    blocks = "".join(f"<div class='rule'>{rid}: {cls} blocked at <b>{node}</b> &larr; patient code <b>{conc}</b></div>" for rid, cls, node, conc in cert["blocked_via"])
    passed = "".join(f"<div class='rule pass'>&#10003; {k}: {f}</div>" for k, f in cert["passed_checks_endpoint"])
    return f"""
<h2 style="margin-left:4px">Architecture Walkthrough</h2>
<section><div class="fig">{svg('fig01_architecture')}</div><p class="imgcap">The same pipeline runs for every dataset; only the adapter is dataset-specific.</p>

<div class="stage data"><h2>1 &middot; Dataset adapter &amp; ICD-10-CM concepts</h2>
<p>The NHANES adapter turned this participant's questionnaire answers and prescription reason codes into ICD-10-CM concepts
(<b>{', '.join(c['concepts'])}</b>) and matched drug names to intervention classes they already take (<b>{', '.join(c['current_meds'])}</b>).
Nothing downstream refers to NHANES by name.</p></div>

<div class="stage xai"><h2>2 &middot; Risk classifier</h2>
<p>The oracle (MLP vs XGBoost, chosen by validation AUC) scored the patient at <b>{100 * r['r0']:.1f}%</b>, above the 0.55 high-risk gate.
Largest SHAP drivers: {', '.join(f'{f} ({v:+.3f})' for f, v in shap[:3])}.</p></div>

<div class="stage neuro"><h2>3 &middot; Knowledge layer &mdash; patient-specific rules by inheritance</h2>
<p>Rules attached to ancestors in the ICD-10-CM hierarchy apply to every descendant code. Diabetes coded E11 inherits the
diabetes rules attached at E08&ndash;E13 (BP target &lt;130/80 instead of the generic &lt;140/90), and asthma coded J45.909 inherits the
beta-blocker contraindication attached at J45:</p>{rules}{blocks}
<details><summary>inheritance diagram</summary><div class="fig">{svg('fig03_ontology_inheritance')}</div></details></div>

<div class="stage"><h2>4 &middot; Latent space, graph and targets</h2>
<p>The record was encoded by the type-aware VAE and attached to every training patient in the Riemannian-weighted graph.
<b>{int(r['n_targets']):,}</b> training patients are lower-risk for this patient's age and lower than her own risk; only
<b>{int(r['tier1_one_hop_targets'])}</b> of them are reachable without changing sex, reversing age, or worsening any blood pressure,
cholesterol, BMI or HbA1c value (source-relative admissibility). Dijkstra chose the closest of these.</p></div>

<div class="stage ok"><h2>5 &middot; Certified search &amp; real-value verification</h2>
<p>Every node on the path was re-checked on recorded values against this patient's baseline, and the clingo program independently
re-derived the result (certificate: all hard rules pass):</p>{passed}</div>

<div class="stage neuro"><h2>6 &middot; Recommendation card</h2>
<p>Each change was mapped to the interventions that move it (acts_on). Contraindicated, non-indicated and already-taken classes were
removed, and the rest were escalated lifestyle &rarr; one drug &rarr; combination using guideline effect sizes. If no real target had been
admissible, the ontology-guided genetic search (Tier 2) would have produced a synthetic target; if that also failed, the patient would
receive a certified &ldquo;no recourse&rdquo; with the closest miss (Tier 3).</p></div>

<div class="stage xai"><h2>7 &middot; Explanation layer</h2>
<p>The four-part explanation shown to the clinician; every tag (rule, ICD node, intervention, citation, value, risk) was machine-checked
against the knowledge tables and computed numbers (trace failures: {len(ex['trace_failures'])}).</p>
<details><summary>explanation text</summary><pre>{html.escape(ex['text'])}</pre></details></div>
</section></section>
"""


# ------------------------------------------------------------------ results

def results() -> str:
    h = t("headline_tiered.csv").set_index("dataset")
    head = "".join(
        f"<tr><td>{DS[d]}</td><td>{int(h.loc[d, 'n_high_risk_total']):,}</td>"
        f"<td><b>{h.loc[d, 'covered_tier12_mean']:.1f}% &plusmn; {h.loc[d, 'covered_tier12_sd']:.1f}</b></td>"
        f"<td>{pct(h.loc[d, 'global_low_mean'])}</td><td>{int(h.loc[d, 'tier1_real_violations'])}</td><td>{int(h.loc[d, 'contraindicated_cards'])}</td></tr>"
        for d in ORDER)
    m = t("methods_comparison.csv")
    mp = m[m.dataset == "nhanes_pooled"].set_index("method")["valid_and_admissible_mean"]
    a = t("ablations.csv")
    ap = a[a.dataset == "nhanes_pooled"].set_index("arm")
    cert = t("certification.csv").set_index("dataset")
    rs = t("reachable_set_head_to_head.csv").set_index("dataset")
    rob = t("robustness_retraining.csv").set_index("dataset")
    cyc = t("pooled_cycle_held_out.csv").groupby("test_on")[["auc", "tier1_coverage_pct"]].mean()
    mort = t("pooled_mortality_validation.csv").set_index("outcome")
    lc = t("pooled_learning_curve.csv").groupby("n_patients")["tier1_coverage_pct"].mean()
    sup = "".join(
        f"<tr><td>{DS[d]}</td><td>{int(cert.loc[d, 'n_tier1_fail']):,} &rarr; {pct(cert.loc[d, 'pct_certified_infeasible_real'], 0)}</td>"
        f"<td>{pct(cert.loc[d, 'pct_tier3_model_ceiling'])}</td><td>{pct(rs.loc[d, 'pct_lr_precluded'])}</td>"
        f"<td>{pct(rs.loc[d, 'pct_lr_precluded_but_carelg_tier1'] + rs.loc[d, 'pct_lr_precluded_but_carelg_tier2'])}</td>"
        f"<td>{pct(rob.loc[d, 'still_valid_pct_tier1'])} / {pct(rob.loc[d, 'still_valid_pct_tier2'])}</td></tr>" for d in ORDER)
    return f"""
<section class="headline"><h1>Final Results <span class="badge">4 datasets &middot; 5 seeds &middot; 0 real-value violations &middot; 0 contraindicated cards</span></h1>
<p class="note">Share of high-risk test patients receiving admissible recourse (Tier 1 real or Tier 2 synthetic), mean &plusmn; SD over seeds.
Pooled NHANES evaluates 1,000 randomly sampled high-risk patients per seed. The tier split is in Fig. 1.</p>
<table><thead><tr><th>Dataset</th><th>High-risk patients</th><th>Recourse (Tier 1 + 2)</th><th>Target also below global 0.45</th>
<th>Tier-1 real-value violations</th><th>Contraindicated cards</th></tr></thead><tbody>{head}</tbody></table></section>
{fig(1, "Coverage by tier", "Tier 1 = certified path to a real lower-risk patient; Tier 2 = ontology-guided synthetic counterfactual, only after Tier 1 is proven impossible; Tier 3 = certified none.",
     "fig02_coverage_tiers", "Real-patient recourse covers most of the pooled cohort; the guided search closes most of the remaining gap.")}
{fig(2, "CARE-LG vs. FACE, DiCE, Growing Spheres, PACE-style, C-CHVAE, Wachter, reachable sets",
     "x: share of patients given a counterfactual that is both valid and breaks no hard clinical rule; y: LOF outlier score (lower = more like real patients). Same oracle, same per-patient goal for every method.",
     "fig07_plausibility_vs_validity",
     f"On pooled NHANES CARE-LG reaches {mp['care_lg_tiered']:.1f}% vs {mp['pace_style']:.1f}% for the best baseline (PACE-style); FACE, Growing Spheres, DiCE and C-CHVAE look valid but break hard rules (Wilcoxon + Holm p &lt; 1e-4 vs every baseline).")}
{fig(3, "Ablations (pooled NHANES)", "Each arm removes one component; contraindications and real-value violations are judged against the full rule set.",
     "fig06_ablations_nhanes_pooled",
     f"Without the ontology or its inheritance, {int(ap.loc['no_ontology', 'contraindicated_recs_total'])}/{int(ap.loc['no_inheritance', 'contraindicated_recs_total'])} contraindicated drugs get through; without SRA {int(ap.loc['no_sra', 'real_violation_paths_total'])} real paths violate a rule; without escalation half the cards jump to drugs; soft rules as vetoes cut coverage to {ap.loc['soft_as_hard', 'covered_mean']:.0f}%.")}
{fig(4, "Fairness of infeasibility (pooled NHANES)", "Tier split by sex, age band, race/ethnicity and NHANES cycle, patients pooled over seeds; chi-square tests in panel titles.",
     "fig08_fairness_nhanes_pooled", "Infeasibility falls on men (no woman ends in Tier 3), an effect of the age-stratified PCE target definition rather than of the search; cycles are balanced.")}
{fig(5, "Sensitivity and scale", "Left: thresholds &times; target definition (seed 0). Middle: GA loss weights vs soft-rule adherence. Right: learning curve on pooled NHANES (5 seeds).",
     "fig09_sensitivity",
     f"Tier-1 coverage grows with cohort size ({lc.iloc[0]:.0f}% at 2k &rarr; {lc.iloc[-1]:.0f}% at {int(lc.index[-1]):,} patients) and is stable across thresholds; a heavier rule weight trades proximity for guideline adherence.")}
<section><h2>Supporting &middot; certification, verification and validation</h2>
<table><thead><tr><th>Dataset</th><th>Tier-1 failures &rarr; certified infeasible over all real patients</th><th>Tier 3 that is a model ceiling</th>
<th>Linear reachable-set verifier says &ldquo;precluded&rdquo;</th><th>&hellip; but CARE-LG finds recourse</th><th>Still valid after oracle retraining (Tier 1 / Tier 2)</th></tr></thead><tbody>{sup}</tbody></table>
<p class="note">Pooled NHANES external checks: training on 1999&ndash;2016 and testing on 2017&ndash;2020 / 2021&ndash;2023 gives AUC {cyc.loc['2017-2020', 'auc']:.3f} / {cyc.loc['2021-2023', 'auc']:.3f}
with Tier-1 coverage {cyc.loc['2017-2020', 'tier1_coverage_pct']:.1f}% / {cyc.loc['2021-2023', 'tier1_coverage_pct']:.1f}%. The PCE label discriminates linked CVD death with AUC {mort.loc['mort_cvd', 'auc_pce_pct']:.3f}
({int(mort.loc['mort_cvd', 'events']):,} CVD deaths among {int(mort.loc['mort_cvd', 'n_linked']):,} participants, median follow-up {mort.loc['mort_cvd', 'median_followup_months']:.0f} months).
Full tables: <code>results/tables/</code>; narrative and limitations: <code>results/SUMMARY.md</code>; judgement calls: <code>results/DECISIONS.md</code>.</p></section>
"""


def build(ds: str = "nhanes_pooled", seed: int = 2, pos: int = 139) -> Path:
    view = sorted(RUNS.glob("combined_*"))[-1]
    P = load_patient(view, ds, seed, pos)
    units = {f: m["unit"] for f, m in adapters.get(ds).FEATURE_META.items()}
    units = {f: (u if u not in ("1=male",) else "(1 = male)") for f, u in units.items()}
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>"
            f"<title>CARE-LG Final Report</title><style>{CSS}</style></head><body><div class='wrap'>"
            f"{final_output(P, units)}{walkthrough(P)}{results()}</div></body></html>")
    out = RESULTS / "report_v2.html"   # user request: replace report_v2.html in place
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes_pooled"); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--test-pos", type=int, default=139)
    a = ap.parse_args()
    print(build(a.dataset, a.seed, a.test_pos))
