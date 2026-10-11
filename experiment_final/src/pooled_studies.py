"""Pooled-NHANES studies (NHANES_POOLED.md; PIPELINE P1/P4/P14): learning curve, cycle-held-out
validation, with/without oscillometric cycles, label/model vs linked mortality, provenance document.

Tier-1 coverage here uses the SRA identity proven in src/certify.py (tier-1 success <=> some target in
T_p is admissible from x0 under the full hard tier), which the main runs verify patient by patient
(certification.csv: sra_identity_violations); no VAE/graph is needed to count coverage.

  .venv/bin/python -m experiment_final.src.pooled_studies
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from experiment_final import adapters
from experiment_final.src.common import RESULTS, TABLES, EXP_DIR, load_config, md_table, new_path
from experiment_final.src.knowledge_layer import hard_violates, load_knowledge, schema_from_meta
from experiment_final.src.model import age_bracket_thresholds, fit_mlp, fit_xgb, patient_threshold
from experiment_final.src.prep import make_split
from experiment_final.src.tier1 import target_mask

DS = "nhanes_pooled"


def _sub_bundle(b, idx):
    """Row subset of a bundle (positions) (P1)."""
    from copy import copy
    s = copy(b)
    s.X = b.X.iloc[idx].reset_index(drop=True); s.y = b.y.iloc[idx].reset_index(drop=True)
    s.patient_id = b.patient_id.iloc[idx].reset_index(drop=True); s.context = b.context.iloc[idx].reset_index(drop=True)
    s.concepts = [b.concepts[i] for i in idx]; s.current_meds = [b.current_meds[i] for i in idx]
    s.subgroups = {k: v.iloc[idx].reset_index(drop=True) for k, v in b.subgroups.items()}
    return s


def fit_oracle(sp, seed):
    """Both families, oracle = higher validation AUC (as src/model.py::evaluate_models) (P4)."""
    m = {"mlp": fit_mlp(sp.Xtr[sp.fit], sp.ytr[sp.fit], seed), "xgboost": fit_xgb(sp.Xtr[sp.fit], sp.ytr[sp.fit], seed)}
    auc = {n: roc_auc_score(sp.ytr[sp.val], o.predict(sp.Xtr[sp.val])) for n, o in m.items()}
    return m[max(auc, key=auc.get)], auc


def tier1_coverage(oracle, Xtr_raw, Xtr_s, Xte_raw, Xte_s, schema, hi=0.55) -> dict:
    """Share of high-risk test patients with an admissible low-risk-for-age real target (P7/P8)."""
    rtr, rte = oracle.predict(Xtr_s), oracle.predict(Xte_s)
    age_j = next(schema.idx(f) for f in schema.features if schema.kind[f] == "non_decreasing")
    edges, thr = age_bracket_thresholds(Xtr_raw[:, age_j], rtr)
    H = np.where(rte > hi)[0]
    ok = [bool((target_mask(rtr, patient_threshold(Xte_raw[i, age_j], edges, thr), rte[i])
                & ~hard_violates(schema, Xte_raw[i], Xtr_raw, horizon=True)).any()) for i in H]
    return {"n_high": int(len(H)), "tier1_coverage_pct": 100 * float(np.mean(ok)) if ok else np.nan}


def learning_curve(b, schema, seeds) -> pd.DataFrame:
    """Tier-1 coverage at 2k/4k/8k/16k/all patients, 5 seeds (stratified subsamples) (P14)."""
    rows = []
    for n in (2000, 4000, 8000, 16000, len(b.X)):
        for seed in seeds:
            idx = np.arange(len(b.X)) if n >= len(b.X) else train_test_split(np.arange(len(b.X)), train_size=n, random_state=seed, stratify=b.y)[0]
            s = _sub_bundle(b, np.sort(idx))
            sp = make_split(s, seed)
            orc, auc = fit_oracle(sp, seed)
            cov = tier1_coverage(orc, sp.Xtr_raw, sp.Xtr, sp.Xte_raw, sp.Xte, schema)
            rows.append({"n_patients": n, "seed": seed, "oracle": orc.name, "test_auc": roc_auc_score(sp.yte, orc.predict(sp.Xte)), **cov})
            print("learning_curve", rows[-1], flush=True)
    return pd.DataFrame(rows)


def _metrics(orc, Xs, y):
    p = orc.predict(Xs)
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}


def cycle_held_out(b, schema, seeds) -> pd.DataFrame:
    """Train on older cycles (1999-2016, auscultatory) -> test on 2017-2020 and 2021-2023 (P4/P14)."""
    cyc = b.context["cycle"].to_numpy()
    newest = ["2017-2020", "2021-2023"]
    old = np.where(~np.isin(cyc, newest))[0]
    rows = []
    for seed in seeds:
        s_old = _sub_bundle(b, old)
        sp = make_split(s_old, seed)
        orc, _ = fit_oracle(sp, seed)
        rows.append({"seed": seed, "test_on": "older cycles (random 20%)", "oracle": orc.name, **_metrics(orc, sp.Xte, sp.yte),
                     **tier1_coverage(orc, sp.Xtr_raw, sp.Xtr, sp.Xte_raw, sp.Xte, schema)})
        for c in newest + ["2017-2020+2021-2023"]:
            te = np.where(np.isin(cyc, c.split("+")))[0]
            Xte_raw = b.X.to_numpy(float)[te]
            rows.append({"seed": seed, "test_on": c, "oracle": orc.name, **_metrics(orc, sp.scale(Xte_raw), b.y.to_numpy()[te]),
                         **tier1_coverage(orc, sp.Xtr_raw, sp.Xtr, Xte_raw, sp.scale(Xte_raw), schema)})
        print("cycle_held_out", rows[-1], flush=True)
    return pd.DataFrame(rows)


def oscillometric_sensitivity(b, schema, seeds) -> pd.DataFrame:
    """Random-split model + tier-1 coverage with all cycles vs auscultatory-BP cycles only (P14)."""
    rows = []
    aus = np.where(b.context["bp_method"].to_numpy() == "auscultatory")[0]
    for label, idx in (("all cycles", np.arange(len(b.X))), ("auscultatory cycles only (1999-2016)", aus)):
        for seed in seeds:
            sp = make_split(_sub_bundle(b, idx), seed)
            orc, _ = fit_oracle(sp, seed)
            rows.append({"cohort": label, "seed": seed, "n": len(idx), "oracle": orc.name, **_metrics(orc, sp.Xte, sp.yte),
                         **tier1_coverage(orc, sp.Xtr_raw, sp.Xtr, sp.Xte_raw, sp.Xte, schema)})
    return pd.DataFrame(rows)


def mortality_validation(b) -> pd.DataFrame:
    """Label (PCE >= 7.5%) and PCE% vs linked CVD / all-cause death, 1999-2016 (P4)."""
    c = b.context
    ok = c["mort_linked"].to_numpy(bool)
    rows = []
    for out in ("mort_cvd", "mort_all"):
        y = c[out].to_numpy()[ok]
        rows.append({"outcome": out, "n_linked": int(ok.sum()), "events": int(np.nansum(y)),
                     "median_followup_months": float(np.nanmedian(c["followup_months"].to_numpy()[ok])),
                     "auc_pce_pct": roc_auc_score(y, c["ascvd_10yr_risk_pct"].to_numpy()[ok]),
                     "event_rate_label1": float(y[b.y.to_numpy()[ok] == 1].mean()), "event_rate_label0": float(y[b.y.to_numpy()[ok] == 0].mean())})
    return pd.DataFrame(rows)


def provenance_md(b) -> str:
    """results/DATA_PROVENANCE_pooled.md (P1)."""
    flow = pd.read_csv(EXP_DIR / "data" / "processed" / "nhanes_pooled_flow.csv")
    vm = pd.read_csv(EXP_DIR / "knowledge" / "nhanes_variable_map.csv")
    piv = flow.pivot_table(index="cycle", columns="step", values="n", aggfunc="first")
    order = ["DEMO participants", "present in DEMO, BMX, BP exam, TCHOL, HDL, GHB, BPQ, DIQ, SMQ", "ages 40-79 (PCE-valid)",
             "complete case on model features + HDL", "positive cholesterol/HDL/SBP (PCE log terms)", "final (PCE label computed)"]
    piv = piv[[o for o in order if o in piv.columns]]
    piv.loc["TOTAL"] = piv.sum()
    fin = flow[flow.step == "final (PCE label computed)"].set_index("cycle")[["n", "label_pos", "with_mortality_link", "deaths"]]
    fin.loc["TOTAL"] = fin.sum()
    vm_w = vm.pivot_table(index="canonical_name", columns="cycle", values="variable", aggfunc="first").fillna("-")
    ctx = b.context
    return "\n".join([
        "# DATA_PROVENANCE_pooled.md -- pooled NHANES 1999-2023 (CARE-LG primary dataset)", "",
        "Built by `adapters/nhanes_pooled.py` (download: `adapters/nhanes_pooled_download.py`, log in `data/raw/nhanes_pooled/download_log.csv`). "
        "All 11 continuous cycles 1999-2000 ... Aug 2021-2023 were downloaded from wwwn.cdc.gov (every component + its .htm codebook) and the 2019 public-use "
        "Linked Mortality Files from ftp.cdc.gov. NHANES III (optional) was not used. No cycle failed.", "",
        "## N per cycle at every filter", "", md_table(piv.astype(int)), "",
        "## Final cohort per cycle (label, mortality link)", "", md_table(fin.astype(int)), "",
        f"Pooled final N = {len(b.X)}; label prevalence (PCE >= 7.5%) = {100 * b.y.mean():.1f}%; mortality-linked = {int(ctx['mort_linked'].sum())}; "
        f"deaths = {int(np.nansum(ctx['mort_all']))}; CVD deaths (UCOD 001 heart disease or 005 cerebrovascular) = {int(np.nansum(ctx['mort_cvd']))}.", "",
        "## Harmonisation choices", "",
        "1. `cycle` column added; cycle-held-out validation trains on 1999-2016 and tests on 2017-2020 and 2021-2023 (`tables/pooled_cycle_held_out.csv`).",
        "2. Blood pressure: mean of available readings. Auscultatory BPXSY1-4/BPXDI1-4 (1999-2016); oscillometric BPXOSY1-3/BPXODI1-3 (2017-2020 `P_BPXO`, 2021-2023 `BPXO_L`). "
        "Diastolic readings of 0 (NHANES codes for an inaudible Korotkoff phase V) are treated as missing. Results with vs without oscillometric cycles: `tables/pooled_oscillometric_sensitivity.csv`.",
        "3. Ages 40-79 only (PCE-valid; also avoids the 80/85 age top-codes).",
        "4. Lab-method changes (total cholesterol, HDL, HbA1c assays changed across cycles) are not re-calibrated; noted as a limitation.",
        "5. Rx: drug names RXD240B (1999-2002) / RXDDRUG (2003-2020) -> intervention classes (interventions.csv patterns); reason codes RXDRSC1-3 only 2013-2020; "
        "2021-2023 `RXQ_RX_L` contains no drug names (only counts), so current_meds are empty there.",
        "6. Complete case on the 7 model features + HDL (PCE input); N dropped per cycle above. Non-model fields are not imputed.",
        "7. Unweighted analysis (methods paper); survey weights are not used.",
        "8. Treated-BP flag: BPQ050A (1999-2020); 2021-2023 has no BPQ050A, so BPQ150 ('Taking high blood pressure medication', same skip pattern after BPQ020/030) is used.",
        "9. PCE label computed with `archive/investigation_v2_to_v6/experiments_v3/build_real_nhanes.py::pce_10yr_risk` (imported unchanged). "
        "Race for PCE: RIDRETH1 = 4 (non-Hispanic Black) -> Black equation, all others -> White equation (same rule as docs/DATA_PROVENANCE.md). "
        "Check: the pooled 2017-2020 slice reproduces `data/nhanes_real_full.csv` exactly (same 4,539 SEQNs; every feature, flag, PCE% and label identical).",
        "10. Mortality: LMF 2019 linkage is available for 1999-2018 only; the pre-pandemic `P_` files (2017-March 2020) assign new SEQNs that do not match the 2017-2018 (J) LMF, "
        "so 2017-2020 and 2021-2023 participants have no linked outcome. Fields: eligstat, mortstat, ucod_leading, permth_exm (months from exam).",
        "11. Kidney: KIQ022 (2001+) / KIQ020 (1999-2000) 'weak/failing kidneys' -> N18.9; dialysis KIQ025 (2001+) -> N18.6 + Z99.2. COPD: MCQ160P (2017+) -> J44.9; "
        "before 2017 emphysema MCQ160G -> J43.9 and chronic bronchitis MCQ160K -> J42. Asthma 'still have' MCQ035 is absent in 1999-2000.", "",
        "## Variable map (canonical name x cycle; full table `knowledge/nhanes_variable_map.csv`)", "", md_table(vm_w), ""])


def main():
    cfg = load_config()
    seeds = cfg["seeds"]
    b = adapters.get(DS).load(cfg)
    schema = schema_from_meta(b.feature_meta, load_knowledge())
    out = {}
    p = new_path(RESULTS / "DATA_PROVENANCE_pooled.md"); p.write_text(provenance_md(b)); out["provenance"] = str(p)
    for name, fn in (("pooled_mortality_validation.csv", lambda: mortality_validation(b)),
                     ("pooled_cycle_held_out.csv", lambda: cycle_held_out(b, schema, seeds)),
                     ("pooled_oscillometric_sensitivity.csv", lambda: oscillometric_sensitivity(b, schema, seeds)),
                     ("pooled_learning_curve.csv", lambda: learning_curve(b, schema, seeds))):
        df = fn()
        p = new_path(TABLES / name); df.to_csv(p, index=False); out[name] = str(p)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
