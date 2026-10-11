"""report_final.html: Example output (Tier 1 + Tier 2 cards in natural language) first, then the architecture
walkthrough, then the condensed final results. Run after `src/build_results.py all` and `src/calg.py`.

  .venv/bin/python -m experiment_final.src.build_report
"""
from __future__ import annotations

import html
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_final import adapters
from experiment_final.src.common import EXAMPLE_DATASET, EXAMPLE_POS, EXAMPLE_SEED, OLD_RUNS, OLD_TABLES_V4, RESULTS
from experiment_final.src.knowledge_layer import load_knowledge, schema_from_meta
from experiment_final.src.build_results import F4, ICD_FULL, T4, VIEW, labels
from experiment_final.src.visualize import METHOD_LABEL as V_LABEL


CSS = """
body{font-family:-apple-system,sans-serif;background:#f7f7f9;margin:0;padding:24px;color:#111}
.wrap{max-width:1100px;margin:0 auto}h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 8px}h3{font-size:13px;margin:14px 0 4px;color:#333}
section{background:#fff;border:1px solid #e2e2e6;border-radius:10px;padding:18px 22px;margin-bottom:18px}
section.headline{border:2px solid #16a34a;background:#f0fdf4}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #e2e2e6;padding:4px 8px;text-align:left}
th{background:rgba(37,99,235,.08)}p,li{line-height:1.55;font-size:14px}.note{font-size:12px;color:#666}
.imgcap{font-size:11.5px;color:#555;font-style:italic;margin:4px 2px 0}.fig svg,.fig img{width:100%;height:auto;display:block}
.badge{background:#dcfce7;color:#166534;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:700;margin-left:8px}
.card{border:1px solid #e2e2e6;border-left:5px solid #2a78d6;border-radius:8px;padding:12px 16px;margin:12px 0;background:#fff}
.card.t2{border-left-color:#eb6834}.card h3{margin-top:4px;color:#111;font-size:15px}
.why{color:#555;font-size:12.5px}.src{font-family:ui-monospace,monospace;font-size:11.5px;color:#7c3aed}
.no{color:#b91c1c}.stage{border-left:4px solid #9ca3af;padding-left:14px;margin:14px 0}.stage h2{font-size:14.5px}
pre{white-space:pre-wrap;font-size:11.5px;background:#f7f7f9;border:1px solid #e2e2e6;border-radius:6px;padding:8px}
details summary{cursor:pointer;font-size:12px;color:#666}
"""


def svg(name: str) -> str:
    p = F4 / f"{name}.svg"
    if not p.exists():
        return ""
    return re.sub(r"<\?xml[^>]*>|<!DOCTYPE[^>]*>", "", p.read_text())


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(T4 / name)


def load_example(RF) -> dict:
    """The example patient (common.EXAMPLE_*). The full rerun stores explanation text only for the first 60 patients processed; for any other patient
    the card and numbers come from cards.jsonl / patients.csv and the explanation text is omitted. For #139 the stored explanation of the report_v3 run
    is reused after checking that its card (changes, path, concepts, medications, risks) is identical."""
    import json as _j
    try:
        return RF.load_patient(VIEW, EXAMPLE_DATASET, EXAMPLE_SEED, EXAMPLE_POS)
    except StopIteration:
        pass
    d = VIEW / EXAMPLE_DATASET / f"seed{EXAMPLE_SEED}"
    P = {"card": next(c for c in map(_j.loads, open(d / "cards.jsonl")) if c["test_pos"] == EXAMPLE_POS),
         "row": pd.read_csv(d / "patients.csv").set_index("test_pos").loc[EXAMPLE_POS], "setup": _j.loads((d / "setup.json").read_text()), "expl": None}
    old_dir = OLD_RUNS / "combined_20261004_215313" / EXAMPLE_DATASET / f"seed{EXAMPLE_SEED}"
    oc = next((c for c in map(_j.loads, open(old_dir / "cards.jsonl")) if c["test_pos"] == EXAMPLE_POS), None)
    orow = pd.read_csv(old_dir / "patients.csv").set_index("test_pos").loc[EXAMPLE_POS]
    if oc and oc["card"] == P["card"]["card"] and oc["path"] == P["card"]["path"] and oc["concepts"] == P["card"]["concepts"] \
            and oc["current_meds"] == P["card"]["current_meds"] and abs(orow["r0"] - P["row"]["r0"]) < 1e-9 and abs(orow["target_risk"] - P["row"]["target_risk"]) < 1e-9:
        P["expl"] = next((e for e in map(_j.loads, open(old_dir / "explanations.jsonl")) if e["test_pos"] == EXAMPLE_POS), None)
    return P


def calg_caption() -> str:
    """Caption for Fig. 2, every number read from the figure's verified cache."""
    from experiment_final.src.calg import CACHE
    m = json.loads(CACHE.with_suffix(".json").read_text()); d = np.load(CACHE)
    n, k = len(d["adm"]), int(d["adm"].sum())
    return (f"The patient of the Example output (NHANES 1999-2023, seed {EXAMPLE_SEED}, test patient #{EXAMPLE_POS}, SEQN {m['pid']}), with the {n - 1} nearest training patients by Riemannian "
            f"distance in the VAE latent space plus the certified target (#{m['target_pid']}, which lies farther away). Each dot is a real patient; the layout is a 2-D PCA of the latents "
            f"(square-root distance compression, axes carry no meaning). Of the {n} similarity edges from the patient, the clinical rules remove {n - k} and keep {k}; "
            f"removed edges are labelled once per rule. Risk = model-predicted probability of PCE &ge; 7.5% ({100 * float(d['r0']):.0f}% now, {100 * m['target_risk']:.0f}% at the target); "
            f"dot colour = the real PCE label. Strip: (a) all neighbours, (b) after rule pruning, (c) the chosen path.")


# ------------------------------------------------------------------ example output

# ------------------------------------------------------------------ assemble

def build() -> Path:
    K = load_knowledge(); onto = K.onto
    cfg_b = adapters.get("nhanes_pooled")
    sch = schema_from_meta(cfg_b.FEATURE_META, K)
    units = {f: m["unit"] for f, m in cfg_b.FEATURE_META.items()}
    st = pickle.loads((T4 / "_store.pkl").read_bytes())
    L = labels()
    # Recommendation block (src/report_final.py): same patient as the CALG figure (see common.EXAMPLE_POS)
    from experiment_final.src import report_final as RF
    RF.DS["nhanes_pooled"] = "NHANES 1999-2023"
    P3 = load_example(RF)
    units3 = {f: (u if u != "1=male" else "(1 = male)") for f, u in units.items()}
    ex_html = RF.final_output(P3, units3) + "</section>"
    ex_html = ex_html.replace("One real high-risk patient from pooled NHANES", "One real high-risk patient from NHANES 1999-2023")
    ex_html = ex_html.replace("&middot; ICD-10-CM concepts", "&middot; " + ICD_FULL + " concepts", 1)
    ex_html = re.sub(r'<p class="note">One real high-risk patient.*?</p>', "", ex_html, count=1, flags=re.S)   # drop the write-up under the header
    ex_html = ex_html.replace("</h1>", f"</h1><div style='font-size:44px;font-weight:800;margin:6px 0 10px'>Patient #{EXAMPLE_POS}</div>", 1)
    expl_html = (f"<details><summary>explanation text generated for this patient (every tag machine-checked)</summary><pre>{html.escape(P3['expl']['text'])}</pre></details>"
                 if P3["expl"] else "")
    arch = (f"<section><h2>Fig. 1 &middot; CARELG Architecture</h2><div class='fig'>{svg('fig01_architecture')}</div>"
            f"<p class='imgcap'>Steps 1-8. Tier 1 is tried first; Tier 2 only if no real patient is admissible; Tier 3 certifies that none exists.</p>"
            f"{expl_html}</section>"
            f"<section><h2>Fig. 2 &middot; The same patient in the rule-pruned latent graph (CALG)</h2><div class='fig' style='max-width:700px;margin:auto'>{svg('fig_calg')}</div>"
            f"<p class='imgcap'>{calg_caption()}</p></section>")
    h = t("headline_tiered.csv").set_index("dataset")
    order = ["nhanes_pooled", "nhanes", "uci", "cardio"]
    head = "".join(f"<tr><td>{L[d]}</td><td>{int(h.loc[d, 'n_high_risk_total']):,}</td><td><b>{h.loc[d, 'covered_tier12_mean']:.1f} ± {h.loc[d, 'covered_tier12_sd']:.1f}%</b></td>"
                   f"<td>{h.loc[d, 'tier1_mean']:.1f} ± {h.loc[d, 'tier1_sd']:.1f}%</td><td>{int(h.loc[d, 'tier1_real_violations'])}</td><td>{int(h.loc[d, 'contraindicated_cards'])}</td></tr>" for d in order)
    mm = t("model_metrics_per_seed.csv")
    acc_bf = pd.read_csv(OLD_TABLES_V4 / "accuracy_backfill_v3runs.csv")
    clf = ""
    for d in order:
        g = mm[(mm.dataset == d) & mm.is_oracle]
        accs = g.test_accuracy if "test_accuracy" in g and g.test_accuracy.notna().all() else acc_bf[(acc_bf.dataset == d) & acc_bf.is_oracle & acc_bf.auc_reproduced].test_accuracy
        acc_txt = f"{accs.mean():.3f} ± {accs.std():.3f}" if len(accs) == 5 else "n/a"
        clf += (f"<tr><td>{L[d]}</td><td>{g.test_auc.mean():.3f} ± {g.test_auc.std():.3f}</td><td>{acc_txt}</td>"
                f"<td>{g.test_brier.mean():.3f}</td><td>{', '.join(f'{k}: {v}' for k, v in g.model.value_counts().items())}</td></tr>")
    km_cap = "Same participants, linked by SEQN to the National Death Index through 2019 (NHANES 1999-2016; 24,518 people, 1,329 CVD deaths)."

    def figsec(no, title, name, cap):
        no += 2   # Fig. 1 = architecture, Fig. 2 = CALG
        return f"<section><h2>Fig. {no} &middot; {title}</h2><div class='fig'>{svg(name)}</div><p class='imgcap'>{cap}</p></section>"

    ab = t("ablations.csv"); ab = ab[ab.dataset == "nhanes_pooled"].set_index("arm")
    mc = t("methods_comparison.csv"); mp = mc[mc.dataset == "nhanes_pooled"].set_index("method")["valid_and_admissible_mean"]
    best_base = mp.drop([m for m in mp.index if m.startswith(("tier", "care_lg"))]).sort_values().index[-1]
    res = (f"<section class='headline'><h1>Final results</h1><p><b>All results are mean ± SD over 5 random seeds.</b> "
           f"Every high-risk test patient is evaluated on every dataset.</p>"
           f"<table><thead><tr><th>Dataset</th><th>High-risk test patients (5 seeds)</th><th>Admissible recourse (Tier 1 + 2)</th><th>Tier 1 (certified real)</th>"
           f"<th>Tier-1 real-value violations</th><th>Contraindicated cards</th></tr></thead><tbody>{head}</tbody></table>"
           f"<h3>Risk classifier (recourse oracle; test set)</h3><table><thead><tr><th>Dataset</th><th>AUC</th><th>Accuracy (threshold 0.5)</th><th>Brier</th><th>Oracle chosen</th></tr></thead><tbody>{clf}</tbody></table></section>"
           + figsec(1, "Coverage by tier", "fig02_coverage_tiers", "Tier 1 = certified path to a real lower-risk patient; Tier 2 = guided synthetic target; Tier 3 = certified none.")
           + figsec(2, "Valid vs valid-and-admissible, all methods", "fig03_valid_vs_admissible",
                    f"On NHANES 1999-2023, CARE-LG reaches {mp['care_lg_tiered']:.1f}% valid-and-admissible vs {mp[best_base]:.1f}% for the best baseline ({V_LABEL.get(best_base, best_base)}).")
           + figsec(3, "Plausibility vs validity", "fig07_plausibility_vs_validity", "Lower LOF = closer to real patients.")
           + figsec(4, "Ablations, NHANES 1999-2023", "fig04_ablations", "Bars: admissible-recourse coverage with one component removed or swapped; text: what that costs. Pooled NHANES, 15,002 high-risk patients over 5 seeds.")
           + figsec(5, "Which rule blocks recourse", "fig_blocking_rules", "Pooled NHANES, Tier-1 failures; the closest miss is the rule that blocks the nearest low-risk target.")
           + figsec(6, "First recommendation with vs without escalation", "fig05_escalation", "Escalation keeps lifestyle first unless lifestyle cannot reach the required change.")
           + figsec(7, "Learning curve, NHANES 1999-2023", "fig06_learning_curve", "Tier-1 coverage grows with cohort size (stratified subsamples, 5 seeds).")
           + figsec(8, "Kaplan-Meier CVD-death survival", "fig08_km", km_cap)
           + figsec(9, "Baseline failure types", "fig09_failure_types", "Which hard rule the baselines' valid counterfactuals break (seed 2, all high-risk test patients)."))
    sk = t("sensitivity_k.csv"); sk = sk[sk.dataset == "nhanes_pooled"]
    arow = lambda k, lab: (f"<tr><td>{lab}</td><td>{ab.loc[k, 'covered_mean']:.1f} ± {ab.loc[k, 'covered_sd']:.1f}</td><td>{ab.loc[k, 'tier1_f_mean']:.1f}</td>"  # noqa: E731
                           f"<td>{int(ab.loc[k, 'contraindicated_recs_total'])}</td><td>{int(ab.loc[k, 'real_violation_paths_total'])}</td></tr>")
    res += ("<section><h2>Appendix &middot; Further ablation arms and k sensitivity (NHANES 1999-2023)</h2>"
            "<table><thead><tr><th>Arm</th><th>Tier 1+2 coverage (%)</th><th>Tier 1 (%)</th><th>Contraindicated cards</th><th>Tier-1 real-value violations</th></tr></thead><tbody>"
            + arow("main", "CARE-LG (full)") + arow("euclidean", "Euclidean instead of Riemannian graph") + arow("no_inheritance", "no ICD-10-CM inheritance")
            + arow("no_guidance", "no ontology guidance in Tier 2") + "</tbody></table>"
            "<table><thead><tr><th>k (neighbours)</th><th>Success (%)</th><th>Mean hops</th><th>Mean cost</th><th>Mean risk drop</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{int(r.k)}</td><td>{r.success_pct:.1f}</td><td>{r.mean_hops:.2f}</td><td>{r.mean_cost:.3f}</td><td>{r.mean_risk_drop:.3f}</td></tr>" for r in sk.itertuples())
            + "</tbody></table><p class='note'>Source: results/tables/ablations.csv and sensitivity_k.csv (k sensitivity on a small sample).</p></section>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>"
            f"<title>CARE-LG Final Report</title><style>{CSS}{RF.CSS}</style></head><body><div class='wrap'>{ex_html}{arch}{res}"
            f"<p class='note'>NHANES 1999-2023 from the full rerun (results/run_full_nhanes); other datasets from the report_v3 runs. Tables: results/tables/ (labelled copies in results/tables/labelled/).</p>"
            f"</div></body></html>")
    out = RESULTS / "report_final.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(build())
