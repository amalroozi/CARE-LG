"""
Phase A: builds a REAL NHANES cohort matching the existing synthetic schema
(age, sex, systolic_bp, diastolic_bp, cholesterol, bmi, glycemic_hba1c,
cvd_risk_flag), from the raw NHANES 2017-March 2020 pre-pandemic cycle
(the most recent complete pre-pandemic cycle -- 2019-2020 alone was
interrupted by COVID and is not released as a standalone cycle; CDC
combined it with 2017-2018 into this single "P_" prefixed cycle instead).

Source files (downloaded via curl from https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/,
saved in experiments_v3/data_raw/, real binary XPT verified -- see DATA_PROVENANCE.md):
  P_DEMO  (Demographics):        SEQN, RIAGENDR (sex), RIDAGEYR (age), RIDRETH3 (race)
  P_BPXO  (Blood Pressure, oscillometric): BPXOSY1-3, BPXODI1-3
  P_BMX   (Body Measures):       BMXBMI
  P_TCHOL (Total Cholesterol):   LBXTC
  P_HDL   (HDL Cholesterol):     LBDHDD
  P_GHB   (Glycohemoglobin):     LBXGH
  P_BPQ   (Blood Pressure Questionnaire): BPQ020 (told high BP), BPQ050A (on BP meds)
  P_DIQ   (Diabetes Questionnaire): DIQ010 (told diabetes)
  P_SMQ   (Smoking Questionnaire): SMQ020 (100+ cigs lifetime), SMQ040 (smokes now)

Label: 2013 ACC/AHA Pooled Cohort Equations (Goff et al. 2014, J Am Coll Cardiol
63(25 Pt B):2935-2959, PMC4700825) 10-year ASCVD risk, thresholded at the
guideline's >=7.5% "elevated risk" statin-initiation cutoff. Coefficients
verified against the open-source reference implementation at
https://github.com/cerner/ascvd-risk-calculator (archived; itself a direct
transcription of the published Goff et al. 2014 coefficient table) -- see
DATA_PROVENANCE.md for the full coefficient table and citation.

PCE is only validated for ages 40-79 with no prior ASCVD event; we restrict
the cohort to that age range (a real, documented consequence of choosing
this label -- see DATA_PROVENANCE.md) and do NOT additionally exclude
prevalent ASCVD cases (a simplification, also documented as a limitation).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(__file__).resolve().parent / "data_raw"
OUT_DIR = Path(__file__).resolve().parent / "data"

import numpy as np
import pandas as pd

PCE_COEF = {
    ("white", "female"): dict(
        s0=0.96652, mnxb=-29.1817,
        ln_age=-29.799, ln_age_sq=4.884, ln_tc=13.54, ln_age_tc=-3.114,
        ln_hdl=-13.578, ln_age_hdl=3.149, ln_sbp_tr=2.019, ln_age_sbp_tr=0.0,
        ln_sbp_nt=1.957, ln_age_sbp_nt=0.0, smoker=7.574, ln_age_smoker=-1.665, diabetes=0.661,
    ),
    ("black", "female"): dict(
        s0=0.95334, mnxb=86.6081,
        ln_age=17.1141, ln_age_sq=0.0, ln_tc=0.9396, ln_age_tc=0.0,
        ln_hdl=-18.9196, ln_age_hdl=4.4748, ln_sbp_tr=29.2907, ln_age_sbp_tr=-6.4321,
        ln_sbp_nt=27.8197, ln_age_sbp_nt=-6.0873, smoker=0.6908, ln_age_smoker=0.0, diabetes=0.8738,
    ),
    ("white", "male"): dict(
        s0=0.91436, mnxb=61.1816,
        ln_age=12.344, ln_age_sq=0.0, ln_tc=11.853, ln_age_tc=-2.664,
        ln_hdl=-7.99, ln_age_hdl=1.769, ln_sbp_tr=1.797, ln_age_sbp_tr=0.0,
        ln_sbp_nt=1.764, ln_age_sbp_nt=0.0, smoker=7.837, ln_age_smoker=-1.795, diabetes=0.658,
    ),
    ("black", "male"): dict(
        s0=0.89536, mnxb=19.5425,
        ln_age=2.469, ln_age_sq=0.0, ln_tc=0.302, ln_age_tc=0.0,
        ln_hdl=-0.307, ln_age_hdl=0.0, ln_sbp_tr=1.916, ln_age_sbp_tr=0.0,
        ln_sbp_nt=1.809, ln_age_sbp_nt=0.0, smoker=0.549, ln_age_smoker=0.0, diabetes=0.645,
    ),
}


def pce_10yr_risk(age, is_male, race_group, tc, hdl, sbp, treated_bp, smoker, diabetic):
    """Vectorized 2013 ACC/AHA Pooled Cohort Equations 10-year ASCVD risk (0-1 fraction).
    Computed group-wise (4 race x sex groups, each with its own coefficient set) then
    reassembled into the original row order.
    """
    age = np.asarray(age, dtype=float)
    tc = np.asarray(tc, dtype=float)
    hdl = np.asarray(hdl, dtype=float)
    sbp = np.asarray(sbp, dtype=float)
    treated_bp = np.asarray(treated_bp, dtype=bool)
    smoker = np.asarray(smoker, dtype=bool)
    diabetic = np.asarray(diabetic, dtype=bool)
    is_male = np.asarray(is_male, dtype=bool)
    race_group = np.asarray(race_group)

    risk = np.full(len(age), np.nan, dtype=float)
    for (rg, sex_key), c in PCE_COEF.items():
        mask = (race_group == rg) & (is_male == (sex_key == "male"))
        if not mask.any():
            continue
        ln_age = np.log(age[mask])
        ln_tc = np.log(tc[mask])
        ln_hdl = np.log(hdl[mask])
        ln_sbp = np.log(sbp[mask])
        tr = treated_bp[mask]
        sbp_tr = np.where(tr, ln_sbp, 0.0)
        sbp_nt = np.where(~tr, ln_sbp, 0.0)
        smoker_f = smoker[mask].astype(float)
        sum_terms = (
            c["ln_age"] * ln_age + c["ln_age_sq"] * ln_age ** 2 +
            c["ln_tc"] * ln_tc + c["ln_age_tc"] * ln_age * ln_tc +
            c["ln_hdl"] * ln_hdl + c["ln_age_hdl"] * ln_age * ln_hdl +
            c["ln_sbp_tr"] * sbp_tr + c["ln_age_sbp_tr"] * ln_age * sbp_tr +
            c["ln_sbp_nt"] * sbp_nt + c["ln_age_sbp_nt"] * ln_age * sbp_nt +
            c["smoker"] * smoker_f + c["ln_age_smoker"] * ln_age * smoker_f +
            c["diabetes"] * diabetic[mask].astype(float)
        )
        risk[mask] = 1.0 - c["s0"] ** np.exp(sum_terms - c["mnxb"])

    assert not np.isnan(risk).any(), "PCE risk not computed for some rows (unhandled race/sex group)"
    return risk


def load_component(name, cols):
    df = pd.read_sas(RAW_DIR / f"{name}.xpt", format="xport")
    return df[["SEQN"] + cols]


def main():
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("=== Phase A: Real NHANES cohort construction ===")

    demo = load_component("P_DEMO", ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "RIDSTATR"])
    bpxo = load_component("P_BPXO", ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"])
    bmx = load_component("P_BMX", ["BMXBMI"])
    tchol = load_component("P_TCHOL", ["LBXTC"])
    hdl = load_component("P_HDL", ["LBDHDD"])
    ghb = load_component("P_GHB", ["LBXGH"])
    bpq = load_component("P_BPQ", ["BPQ020", "BPQ050A"])
    diq = load_component("P_DIQ", ["DIQ010"])
    smq = load_component("P_SMQ", ["SMQ020", "SMQ040"])

    log(f"Component N (raw rows): DEMO={len(demo)} BPXO={len(bpxo)} BMX={len(bmx)} "
        f"TCHOL={len(tchol)} HDL={len(hdl)} GHB={len(ghb)} BPQ={len(bpq)} DIQ={len(diq)} SMQ={len(smq)}")

    df = demo
    for comp in [bpxo, bmx, tchol, hdl, ghb, bpq, diq, smq]:
        df = df.merge(comp, on="SEQN", how="inner")
    log(f"N after inner-join merge on SEQN across all 9 components: {len(df)}")

    # Interview+exam participants only (RIDSTATR==2: both interviewed and MEC examined)
    df = df[df["RIDSTATR"] == 2].copy()
    log(f"N after restricting to interviewed+examined respondents (RIDSTATR==2): {len(df)}")

    # Derived fields
    df["systolic_bp"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1, skipna=True)
    df["diastolic_bp"] = df[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1, skipna=True)
    df["cholesterol"] = df["LBXTC"]
    df["bmi"] = df["BMXBMI"]
    df["glycemic_hba1c"] = df["LBXGH"]
    df["age"] = df["RIDAGEYR"]
    df["sex"] = (df["RIAGENDR"] == 1).astype(float)  # 1=male, 0=female (matches existing schema convention)
    df["is_male"] = df["RIAGENDR"] == 1
    df["race_group"] = np.where(df["RIDRETH3"] == 4, "black", "white")  # AHA guidance: use White eqn for non-Black races
    df["treated_bp"] = (df["BPQ050A"] == 1)
    df["smoker"] = (df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))  # ever smoked 100+ AND currently smokes (every/some days)
    df["diabetic"] = (df["DIQ010"] == 1)

    # Missingness report on the model-relevant + PCE-input columns, BEFORE dropping
    check_cols = ["age", "sex", "systolic_bp", "diastolic_bp", "cholesterol", "bmi",
                  "glycemic_hba1c", "race_group", "treated_bp", "smoker", "diabetic"]
    log("\nMissingness per feature (after component merge, before age restriction / final dropna):")
    miss_rows = []
    for c in check_cols:
        n_miss = df[c].isna().sum()
        pct = 100.0 * n_miss / len(df)
        log(f"  {c:16s}: {n_miss:5d} missing / {len(df)} ({pct:.2f}%)")
        miss_rows.append({"feature": c, "n_missing": int(n_miss), "pct_missing": round(pct, 3), "n_total": len(df)})
    pd.DataFrame(miss_rows).to_csv(OUT_DIR / "missingness_report.csv", index=False)

    # Age restriction: PCE is validated for ages 40-79 only
    n_before_age = len(df)
    df = df[(df["age"] >= 40) & (df["age"] <= 79)].copy()
    log(f"\nN after restricting to PCE-valid age range [40,79]: {len(df)} (dropped {n_before_age-len(df)})")

    # Final complete-case restriction on all needed fields (exclusion, not imputation --
    # see DATA_PROVENANCE.md for why imputation was rejected)
    n_before_cc = len(df)
    df_cc = df.dropna(subset=check_cols).copy()
    log(f"N after complete-case exclusion on all model+label input fields: {len(df_cc)} (dropped {n_before_cc-len(df_cc)})")

    # Sanity: cholesterol/hdl/sbp must be > 0 for the log() terms in PCE
    valid_pos = (df_cc["cholesterol"] > 0) & (df_cc["LBDHDD"] > 0) & (df_cc["systolic_bp"] > 0)
    n_before_pos = len(df_cc)
    df_cc = df_cc[valid_pos].copy()
    log(f"N after excluding non-positive cholesterol/HDL/SBP (required for PCE log terms): {len(df_cc)} (dropped {n_before_pos-len(df_cc)})")

    # Compute PCE risk and threshold
    risk = pce_10yr_risk(
        age=df_cc["age"].values, is_male=df_cc["is_male"].values, race_group=df_cc["race_group"].values,
        tc=df_cc["cholesterol"].values, hdl=df_cc["LBDHDD"].values, sbp=df_cc["systolic_bp"].values,
        treated_bp=df_cc["treated_bp"].values, smoker=df_cc["smoker"].values, diabetic=df_cc["diabetic"].values,
    )
    df_cc["ascvd_10yr_risk_pct"] = risk * 100.0
    df_cc["cvd_risk_flag"] = (df_cc["ascvd_10yr_risk_pct"] >= 7.5).astype(int)

    log(f"\nASCVD 10-yr risk distribution: mean={df_cc['ascvd_10yr_risk_pct'].mean():.2f}%, "
        f"median={df_cc['ascvd_10yr_risk_pct'].median():.2f}%, "
        f"min={df_cc['ascvd_10yr_risk_pct'].min():.2f}%, max={df_cc['ascvd_10yr_risk_pct'].max():.2f}%")
    log(f"cvd_risk_flag positive rate (>=7.5% 10-yr ASCVD risk): "
        f"{df_cc['cvd_risk_flag'].mean()*100:.1f}% ({df_cc['cvd_risk_flag'].sum()}/{len(df_cc)})")

    final_cols = ["SEQN", "age", "sex", "systolic_bp", "diastolic_bp", "cholesterol", "bmi",
                  "glycemic_hba1c", "cvd_risk_flag", "ascvd_10yr_risk_pct", "race_group", "smoker", "diabetic", "treated_bp"]
    df_final = df_cc[final_cols].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_DIR / "nhanes_real_full.csv", index=False)

    # Model-facing CSV matching the EXACT existing synthetic schema column set (drop provenance-only cols)
    model_cols = ["age", "sex", "systolic_bp", "diastolic_bp", "cholesterol", "bmi", "glycemic_hba1c", "cvd_risk_flag"]
    df_final[model_cols].to_csv(OUT_DIR / "nhanes_real.csv", index=False)

    log(f"\nFinal N (real NHANES cohort, model-ready): {len(df_final)}")
    log(f"Wrote {OUT_DIR/'nhanes_real.csv'} (model-facing, schema-matched) and "
        f"{OUT_DIR/'nhanes_real_full.csv'} (full, with PCE provenance columns)")

    with open(OUT_DIR / "build_log.txt", "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
