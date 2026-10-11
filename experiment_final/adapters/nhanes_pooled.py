"""Adapter: pooled NHANES 1999-2023 -> DatasetBundle. Downloads per NHANES_POOLED.md (scripts allowed to fetch from wwwn.cdc.gov and ftp.cdc.gov), harmonises per cycle, adds `cycle`, real-outcome fields from linked mortality.

PIPELINE P1/P2. Steps (every choice is recorded in results/DATA_PROVENANCE_pooled.md):
 1. download (adapters/nhanes_pooled_download.py; skips files already present);
 2. per cycle: inner-join DEMO, BMX, BP exam, TCHOL, HDL, GHB, BPQ, DIQ, SMQ (the same components the
    2017-2020 cohort required, docs/DATA_PROVENANCE.md); left-join MCQ, KIQ, RXQ_RX and the LMF;
    variable names per cycle from knowledge/nhanes_variable_map.csv (verified against each file);
 3. BP = mean of available readings (auscultatory BPXSY1-4/BPXDI1-4 to 2015-16; oscillometric
    BPXOSY1-3/BPXODI1-3 from 2017); diastolic 0 readings treated as missing;
 4. ages 40-79; complete case on model features + PCE inputs; positive chol/HDL/SBP;
 5. label = PCE >= 7.5% computed with the SAME function as the 2017-2020 cohort
    (archive/.../experiments_v3/build_real_nhanes.py::pce_10yr_risk, imported, not copied);
 6. concepts (ICD-10-CM) and current intervention classes as in adapters/nhanes.py, per cycle
    availability (Rx names 1999-2020, Rx reason codes 2013-2020; none in 2021-23);
 7. linked mortality (2019 LMF; cycles 1999-2018): all-cause death, CVD death (UCOD 001 heart
    disease or 005 cerebrovascular), months of follow-up from the exam.
Unweighted analysis (methods paper). The harmonised cohort is cached in data/processed/.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiment_final.adapters.base import DatasetBundle, age_band, validate_bundle
from experiment_final.adapters.nhanes import FEATURE_META, FEATURES
from experiment_final.adapters.nhanes_pooled_download import CYCLES, RAW, candidates, download_all

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parent
HAS_ONTOLOGY = True
CACHE = EXP_DIR / "data" / "processed" / "nhanes_pooled_cohort.parquet"
RIDRETH1 = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 5: "Other/Multiracial"}

# canonical name -> list of candidate source variables (first present wins), per component
VARMAP = {
    "DEMO": {"age": ["RIDAGEYR"], "sex_raw": ["RIAGENDR"], "race1": ["RIDRETH1"], "pregnant": ["RIDEXPRG"], "exam_status": ["RIDSTATR"]},
    "BMX": {"bmi": ["BMXBMI"]},
    "BPX": {**{f"sbp{i}": [f"BPXSY{i}", f"BPXOSY{i}"] for i in range(1, 5)}, **{f"dbp{i}": [f"BPXDI{i}", f"BPXODI{i}"] for i in range(1, 5)}},
    "TCHOL": {"cholesterol": ["LBXTC"]},
    "HDL": {"hdl": ["LBDHDD", "LBDHDL", "LBXHDD"]},
    "GHB": {"glycemic_hba1c": ["LBXGH"]},
    "BPQ": {"told_hbp": ["BPQ020"], "bp_meds": ["BPQ050A", "BPQ150"], "told_chol": ["BPQ080"], "lipid_meds": ["BPQ100D", "BPQ101D"]},
    "DIQ": {"DIQ010": ["DIQ010"]},
    "SMQ": {"SMQ020": ["SMQ020"], "SMQ040": ["SMQ040"]},
    "MCQ": {"MCQ160B": ["MCQ160B"], "MCQ160C": ["MCQ160C"], "MCQ160E": ["MCQ160E"], "MCQ160F": ["MCQ160F"],
            "MCQ160G": ["MCQ160G"], "MCQ160K": ["MCQ160K"], "MCQ160P": ["MCQ160P"], "MCQ035": ["MCQ035"], "MCQ170L": ["MCQ170L"],
            "MCQ230A": ["MCQ230A"], "MCQ230B": ["MCQ230B"], "MCQ230C": ["MCQ230C"], "MCQ230D": ["MCQ230D"]},
    "KIQ": {"weak_kidneys": ["KIQ022", "KIQ020"], "dialysis": ["KIQ025"]},
}
REQUIRED = ["DEMO", "BMX", "BPX", "TCHOL", "HDL", "GHB", "BPQ", "DIQ", "SMQ"]
OPTIONAL = ["MCQ", "KIQ"]


def _pce():
    """Import the 2017-2020 cohort's PCE function unchanged (P1)."""
    p = REPO / "archive" / "investigation_v2_to_v6" / "experiments_v3" / "build_real_nhanes.py"
    spec = importlib.util.spec_from_file_location("build_real_nhanes", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.pce_10yr_risk


def _file(cycle: str, comp: str):
    year, suf, _ = CYCLES[cycle]
    for stem in candidates(comp, suf):
        f = RAW / cycle / f"{stem}.xpt"
        if f.exists():
            return f
    return None


def _component(cycle: str, comp: str, mapping_rows: list) -> pd.DataFrame:
    """Read one component and rename to canonical names; logs the variable map (P1)."""
    f = _file(cycle, comp)
    if f is None:
        return None
    d = pd.read_sas(f, format="xport", encoding="latin1")
    out = pd.DataFrame({"SEQN": d["SEQN"].astype(int)})
    for canon, cands in VARMAP[comp].items():
        src = next((c for c in cands if c in d.columns), None)
        mapping_rows.append({"cycle": cycle, "file": f.name, "variable": src or "", "canonical_name": canon, "present": src is not None})
        out[canon] = d[src].to_numpy() if src else np.nan
    return out.drop_duplicates("SEQN")


def _rx(cycle: str, iv: pd.DataFrame, mapping_rows: list) -> Tuple[Dict[int, set], Dict[int, set], dict]:
    """Rx drug names -> intervention classes; reason codes RXDRSC1-3 when present (P1/P2)."""
    f = _file(cycle, "RXQ_RX")
    info = {"rx_names": False, "rx_reasons": False}
    if f is None:
        return {}, {}, info
    d = pd.read_sas(f, format="xport", encoding="latin1")
    name_col = next((c for c in ("RXDDRUG", "RXD240B") if c in d.columns), None)
    mapping_rows.append({"cycle": cycle, "file": f.name, "variable": name_col or "", "canonical_name": "rx_drug_name", "present": name_col is not None})
    meds, codes = {}, {}
    if name_col:
        info["rx_names"] = True
        pats = {r.intervention_class: re.compile(r.nhanes_rx_match_pattern) for r in iv.itertuples()
                if isinstance(r.nhanes_rx_match_pattern, str) and r.nhanes_rx_match_pattern}
        for s, nm in zip(d["SEQN"].astype(int), d[name_col].astype(str).str.upper()):
            for cls, p in pats.items():
                if p.search(nm):
                    meds.setdefault(s, set()).add(cls)
    rc = [c for c in ("RXDRSC1", "RXDRSC2", "RXDRSC3") if c in d.columns]
    mapping_rows.append({"cycle": cycle, "file": f.name, "variable": ";".join(rc), "canonical_name": "rx_reason_codes", "present": bool(rc)})
    if rc:
        info["rx_reasons"] = True
        for c in rc:
            for s, v in zip(d["SEQN"].astype(int), d[c].astype(str)):
                v = v.strip()
                if v and v.lower() != "nan":
                    codes.setdefault(s, set()).add(v)
    return meds, codes, info


def _mortality(cycle: str) -> pd.DataFrame:
    """Public-use linked mortality file (2019 release) fixed-width parse (NCHS layout) (P1)."""
    mort = CYCLES[cycle][2]
    if not mort:
        return None
    f = RAW / cycle / f"NHANES_{mort[0]}_{mort[1]}_MORT_2019_PUBLIC.dat"
    if not f.exists():
        return None
    d = pd.read_fwf(f, colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (42, 45), (45, 48)], header=None,
                    names=["SEQN", "eligstat", "mortstat", "ucod_leading", "permth_int", "permth_exm"],
                    dtype=str)
    d["SEQN"] = d["SEQN"].astype(int)
    for c in ("eligstat", "mortstat", "permth_int", "permth_exm"):
        d[c] = pd.to_numeric(d[c].str.strip(), errors="coerce")
    d["ucod_leading"] = d["ucod_leading"].str.strip()
    return d


def harmonise_cycle(cycle: str, iv: pd.DataFrame, mapping_rows: list, flow: list) -> Tuple[pd.DataFrame, Dict, Dict]:
    """One cycle -> canonical rows + Rx dictionaries; appends N at every filter (P1)."""
    comps = {c: _component(cycle, c, mapping_rows) for c in REQUIRED + OPTIONAL}
    missing = [c for c in REQUIRED if comps[c] is None]
    if missing:
        flow.append({"cycle": cycle, "step": f"FAILED: missing required component(s) {missing}", "n": 0})
        return None, {}, {}
    df = comps["DEMO"]
    flow.append({"cycle": cycle, "step": "DEMO participants", "n": len(df)})
    for c in REQUIRED[1:]:
        df = df.merge(comps[c], on="SEQN", how="inner")
    flow.append({"cycle": cycle, "step": "present in DEMO, BMX, BP exam, TCHOL, HDL, GHB, BPQ, DIQ, SMQ", "n": len(df)})
    for c in OPTIONAL:
        df = df.merge(comps[c], on="SEQN", how="left") if comps[c] is not None else df.assign(**{k: np.nan for k in VARMAP[c]})
    df["cycle"] = cycle
    df["bp_method"] = "oscillometric" if CYCLES[cycle][1] in ("P_", "_L") else "auscultatory"
    sb = df[[f"sbp{i}" for i in range(1, 5)]].astype(float)
    db = df[[f"dbp{i}" for i in range(1, 5)]].astype(float).where(lambda x: x > 0)
    df["systolic_bp"], df["diastolic_bp"] = sb.mean(axis=1, skipna=True), db.mean(axis=1, skipna=True)
    df["sex"] = (df["sex_raw"] == 1).astype(float)
    df["race_group"] = np.where(df["race1"] == 4, "black", "white")
    df["treated_bp"] = df["bp_meds"] == 1
    df["smoker"] = (df["SMQ020"] == 1) & df["SMQ040"].isin([1, 2])
    df["diabetic"] = df["DIQ010"] == 1
    df = df[(df["age"] >= 40) & (df["age"] <= 79)]
    flow.append({"cycle": cycle, "step": "ages 40-79 (PCE-valid)", "n": len(df)})
    df = df.dropna(subset=FEATURES + ["hdl", "race1"])
    flow.append({"cycle": cycle, "step": "complete case on model features + HDL", "n": len(df)})
    df = df[(df["cholesterol"] > 0) & (df["hdl"] > 0) & (df["systolic_bp"] > 0)]
    flow.append({"cycle": cycle, "step": "positive cholesterol/HDL/SBP (PCE log terms)", "n": len(df)})
    meds, codes, info = _rx(cycle, iv, mapping_rows)
    mort = _mortality(cycle)
    if mort is not None:
        df = df.merge(mort, on="SEQN", how="left")
    else:
        for c in ("eligstat", "mortstat", "ucod_leading", "permth_int", "permth_exm"):
            df[c] = np.nan
    df["rx_names_available"], df["rx_reasons_available"] = info["rx_names"], info["rx_reasons"]
    return df.reset_index(drop=True), meds, codes


def concepts_for(r, codes: set, onto) -> List[str]:
    """Flags + questionnaire items + Rx reason codes -> ICD-10-CM nodes (knowledge/nhanes_condition_map.csv) (P2)."""
    from experiment_final.knowledge.ontology import resolve_code
    c = set()
    if r.diabetic: c.add("E11")
    if r.treated_bp or r.told_hbp == 1: c.add("I10")
    if r.told_chol == 1: c.add("E78.5")
    if r.smoker: c.add("Z72.0")
    if r.bmi >= 30: c.add("E66.9")
    if r.pregnant == 1: c.add("Z33.1")
    for var, code in (("MCQ160B", "I50.9"), ("MCQ160C", "I25.10"), ("MCQ160E", "I25.2"), ("MCQ160F", "Z86.73"),
                      ("MCQ160P", "J44.9"), ("MCQ160G", "J43.9"), ("MCQ160K", "J42"), ("MCQ035", "J45.909"),
                      ("MCQ170L", "K76.9"), ("weak_kidneys", "N18.9")):
        if getattr(r, var) == 1:
            c.add(code)
    if r.dialysis == 1:
        c.update({"N18.6", "Z99.2"})
    if any(getattr(r, v) == 37 for v in ("MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D")):
        c.add("C73")
    for raw in codes:
        n = resolve_code(onto, raw)
        if n:
            c.add(n)
    return sorted(c)


def build_cohort() -> Tuple[pd.DataFrame, list, list]:
    """Download (if needed), harmonise every cycle, compute the PCE label; cached (P1)."""
    from experiment_final.knowledge.ontology import get_ontology
    if not all((RAW / cyc).exists() and any((RAW / cyc).glob("*.xpt")) for cyc in CYCLES):
        download_all()
    iv = pd.read_csv(EXP_DIR / "knowledge" / "interventions.csv")
    onto = get_ontology()
    pce = _pce()
    mapping_rows, flow, parts = [], [], []
    for cyc in CYCLES:
        try:
            df, meds, codes = harmonise_cycle(cyc, iv, mapping_rows, flow)
        except Exception as e:  # noqa: BLE001 -- a failing cycle is logged and skipped (prompt rule)
            flow.append({"cycle": cyc, "step": f"FAILED: {type(e).__name__}: {e}", "n": 0})
            continue
        if df is None or df.empty:
            continue
        risk = pce(age=df["age"].values, is_male=df["sex"].values == 1, race_group=df["race_group"].values,
                   tc=df["cholesterol"].values, hdl=df["hdl"].values, sbp=df["systolic_bp"].values,
                   treated_bp=df["treated_bp"].values, smoker=df["smoker"].values, diabetic=df["diabetic"].values)
        df["ascvd_10yr_risk_pct"] = risk * 100.0
        df["cvd_risk_flag"] = (df["ascvd_10yr_risk_pct"] >= 7.5).astype(int)
        df["concepts"] = [";".join(concepts_for(r, codes.get(int(r.SEQN), set()), onto)) for r in df.itertuples(index=False)]
        df["current_meds"] = [";".join(sorted(meds.get(int(s), set()))) for s in df["SEQN"]]
        flow.append({"cycle": cyc, "step": "final (PCE label computed)", "n": len(df), "label_pos": int(df["cvd_risk_flag"].sum()),
                     "with_mortality_link": int((df["eligstat"] == 1).sum()), "deaths": int((df["mortstat"] == 1).sum())})
        parts.append(df)
    cohort = pd.concat(parts, ignore_index=True)
    keep = ["SEQN", "cycle", "bp_method", *FEATURES, "hdl", "race1", "race_group", "treated_bp", "smoker", "diabetic",
            "ascvd_10yr_risk_pct", "cvd_risk_flag", "concepts", "current_meds", "eligstat", "mortstat", "ucod_leading",
            "permth_exm", "rx_names_available", "rx_reasons_available", "lipid_meds"]
    cohort = cohort[keep]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(CACHE, index=False)
    pd.DataFrame(mapping_rows).drop_duplicates().to_csv(EXP_DIR / "knowledge" / "nhanes_variable_map.csv", index=False)
    pd.DataFrame(flow).to_csv(EXP_DIR / "data" / "processed" / "nhanes_pooled_flow.csv", index=False)
    return cohort, flow, mapping_rows


def load(config: dict) -> DatasetBundle:
    """Pooled NHANES bundle (P1). Optional config keys: pooled_cycles (list) / pooled_exclude_cycles."""
    cohort = pd.read_parquet(CACHE) if CACHE.exists() else build_cohort()[0]
    if config.get("pooled_exclude_cycles"):
        cohort = cohort[~cohort["cycle"].isin(config["pooled_exclude_cycles"])].reset_index(drop=True)
    flow_df = pd.read_csv(EXP_DIR / "data" / "processed" / "nhanes_pooled_flow.csv")
    X = cohort[FEATURES].astype(float).reset_index(drop=True)
    elig = cohort["eligstat"] == 1
    ctx = pd.DataFrame({
        "SEQN": cohort["SEQN"].astype(int), "cycle": cohort["cycle"], "bp_method": cohort["bp_method"],
        "ascvd_10yr_risk_pct": cohort["ascvd_10yr_risk_pct"],
        "race_pce": cohort["race_group"].map({"black": "Black", "white": "White/other"}),
        "race": cohort["race1"].map(RIDRETH1).fillna("Unknown"), "smoker": cohort["smoker"].astype(bool),
        "diabetic": cohort["diabetic"].astype(bool), "treated_bp": cohort["treated_bp"].astype(bool),
        "mort_linked": elig, "mort_all": np.where(elig, (cohort["mortstat"] == 1).astype(float), np.nan),
        "mort_cvd": np.where(elig, ((cohort["mortstat"] == 1) & cohort["ucod_leading"].isin(["001", "005"])).astype(float), np.nan),
        "followup_months": np.where(elig, cohort["permth_exm"], np.nan),
    })
    b = DatasetBundle(
        X=X, y=cohort["cvd_risk_flag"].astype(int).reset_index(drop=True), patient_id=ctx["SEQN"].copy(),
        feature_meta={k: dict(v) for k, v in FEATURE_META.items()}, context=ctx,
        concepts=[c.split(";") if c else [] for c in cohort["concepts"]],
        current_meds=[m.split(";") if m else [] for m in cohort["current_meds"]],
        subgroups={"sex": X["sex"].map({1.0: "Male", 0.0: "Female"}), "age_band": age_band(X["age"]),
                   "race": ctx["race"], "cycle": ctx["cycle"]},
        name="nhanes_pooled", has_ontology=True,
        cohort_flow=[{"step": f"{r.cycle}: {r.step}", "n": int(r.n)} for r in flow_df.itertuples()] + [{"step": "pooled final", "n": len(X)}],
        label_definition="PCE 10-year ASCVD risk >= 7.5% (Goff 2013), same function as the 2017-2020 cohort",
        citation="NCHS NHANES 1999-2023 continuous cycles + NCHS public-use Linked Mortality Files (2019)",
    )
    validate_bundle(b)
    return b


if __name__ == "__main__":
    c, flow, _ = build_cohort()
    print(pd.DataFrame(flow).to_string())
    print(c.groupby("cycle").agg(n=("SEQN", "size"), pos=("cvd_risk_flag", "mean"), dead=("mortstat", lambda v: (v == 1).sum())))
