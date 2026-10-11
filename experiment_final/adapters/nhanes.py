"""Adapter: nhanes -> DatasetBundle (see DATASET_ADAPTER.md). PIPELINE P1 + P2 inputs.

Cohort = ../data/nhanes_real_full.csv (NHANES 2017-Mar 2020, ages 40-79, complete PCE inputs), in the
SAME row order as ../data/nhanes_real.csv (asserted). Label = PCE 10-year ASCVD risk >= 7.5%
(cvd_risk_flag, built in archive/.../experiments_v3/build_real_nhanes.py).

Concepts (ICD-10-CM, knowledge/nhanes_condition_map.csv): existing flags (diabetic, treated_bp,
smoker), raw questionnaire XPTs already on disk (P_BPQ BPQ020/BPQ080, P_DEMO RIDEXPRG), derived
obesity (BMI >= 30), and -- if present in data/raw/nhanes_extra/ -- P_MCQ, P_KIQ_U and the
P_RXQ_RX reason-for-use codes RXDRSC1-3. Current medications = intervention classes whose
nhanes_rx_match_pattern (knowledge/interventions.csv) matches an RXDDRUG name.
Every merge is a LEFT merge on SEQN that preserves the cohort's row order.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiment_final.adapters.base import DatasetBundle, age_band, validate_bundle

EXP_DIR = Path(__file__).resolve().parents[1]
HAS_ONTOLOGY = True
REPO = EXP_DIR.parent

RIDRETH3 = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian", 7: "Other/Multiracial"}

FEATURE_META = {
    "age": dict(type="continuous", **{"class": "non_decreasing"}, direction="none", unit="years", phys_min=18, phys_max=100, concept="age"),
    "sex": dict(type="binary", **{"class": "immutable"}, direction="none", unit="1=male", phys_min=0, phys_max=1, concept="sex"),
    "systolic_bp": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=70, phys_max=250, concept="systolic_bp", noise_tol=1.0),
    "diastolic_bp": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mmHg", phys_min=40, phys_max=150, concept="diastolic_bp", noise_tol=1.0),
    "cholesterol": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="mg/dL", phys_min=70, phys_max=450, concept="total_cholesterol", noise_tol=1.0),
    "bmi": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="kg/m2", phys_min=14, phys_max=80, concept="bmi", noise_tol=0.1),
    "glycemic_hba1c": dict(type="continuous", **{"class": "mutable"}, direction="reduce", unit="%", phys_min=3.5, phys_max=17, concept="hba1c", noise_tol=0.05),
}
FEATURES = list(FEATURE_META)


def _xpt(path: Path, cols: List[str]) -> pd.DataFrame:
    d = pd.read_sas(path, format="xport", encoding="latin1")
    return d[[c for c in ["SEQN"] + cols if c in d.columns]]


def _left(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Row-order-preserving left merge on SEQN (asserted)."""
    out = base.merge(other.drop_duplicates("SEQN"), on="SEQN", how="left")
    assert (out["SEQN"].to_numpy() == base["SEQN"].to_numpy()).all()
    return out


def _rx_tables(extra: Path, seqn: pd.Series, iv: pd.DataFrame):
    """Per-SEQN reason codes and current intervention classes from P_RXQ_RX (P1)."""
    rx = pd.read_sas(extra / "P_RXQ_RX.xpt", format="xport", encoding="latin1")
    rx = rx[rx["SEQN"].isin(set(seqn))]
    codes: Dict[float, set] = {}
    for c in ("RXDRSC1", "RXDRSC2", "RXDRSC3"):
        for s, v in zip(rx["SEQN"], rx[c].astype(str)):
            if v.strip() and v.strip().lower() != "nan":
                codes.setdefault(s, set()).add(v.strip())
    meds: Dict[float, set] = {}
    pats = {r.intervention_class: re.compile(r.nhanes_rx_match_pattern) for r in iv.itertuples()
            if isinstance(r.nhanes_rx_match_pattern, str) and r.nhanes_rx_match_pattern}
    for s, name in zip(rx["SEQN"], rx["RXDDRUG"].astype(str).str.upper()):
        for cls, p in pats.items():
            if p.search(name):
                meds.setdefault(s, set()).add(cls)
    return codes, meds


def _concepts(df: pd.DataFrame, rx_codes: Dict[float, set], onto) -> List[List[str]]:
    """Flags + questionnaire items + Rx reason codes -> resolved ICD-10-CM nodes (P2)."""
    from experiment_final.knowledge.ontology import resolve_code
    out = []
    for r in df.itertuples(index=False):
        c = set()
        if r.diabetic: c.add("E11")
        if r.treated_bp or r.BPQ020 == 1: c.add("I10")
        if r.BPQ080 == 1: c.add("E78.5")
        if r.smoker: c.add("Z72.0")
        if r.bmi >= 30: c.add("E66.9")
        if r.RIDEXPRG == 1: c.add("Z33.1")
        for var, code in (("MCQ160B", "I50.9"), ("MCQ160C", "I25.10"), ("MCQ160E", "I25.2"), ("MCQ160F", "Z86.73"),
                          ("MCQ160P", "J44.9"), ("MCQ035", "J45.909"), ("MCQ170L", "K76.9"), ("KIQ022", "N18.9")):
            if getattr(r, var, np.nan) == 1:
                c.add(code)
        if getattr(r, "KIQ025", np.nan) == 1:
            c.update({"N18.6", "Z99.2"})
        if any(getattr(r, v, np.nan) == 37 for v in ("MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D")):
            c.add("C73")
        for raw in rx_codes.get(r.SEQN, ()):
            node = resolve_code(onto, raw)
            if node:
                c.add(node)
        out.append(sorted(c))
    return out


def load(config: dict) -> DatasetBundle:
    """NHANES bundle (P1). Missing extra XPTs degrade gracefully (fewer concepts, no meds)."""
    from experiment_final.knowledge.ontology import get_ontology
    paths = config["paths"]
    full = pd.read_csv(EXP_DIR / paths["nhanes_full_csv"])
    slim = pd.read_csv(REPO / "data" / "nhanes_real.csv")
    assert np.allclose(full[FEATURES].to_numpy(float), slim[FEATURES].to_numpy(float)), "row order differs from nhanes_real.csv"
    flow = [{"step": "nhanes_real_full.csv (ages 40-79, complete PCE inputs)", "n": len(full)}]
    raw = EXP_DIR / paths["nhanes_raw_xpt_dir"]
    df = _left(full, _xpt(raw / "P_BPQ.xpt", ["BPQ020", "BPQ080"]))
    df = _left(df, _xpt(raw / "P_DEMO.xpt", ["RIDEXPRG", "RIDRETH3"]))
    extra = EXP_DIR / paths["nhanes_extra_dir"]
    have = {f: (extra / f"{f}.xpt").exists() for f in ("P_RXQ_RX", "P_MCQ", "P_KIQ_U")}
    if have["P_MCQ"]:
        df = _left(df, _xpt(extra / "P_MCQ.xpt", ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ160P", "MCQ035",
                                                 "MCQ170L", "MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D"]))
    if have["P_KIQ_U"]:
        df = _left(df, _xpt(extra / "P_KIQ_U.xpt", ["KIQ022", "KIQ025"]))
    iv = pd.read_csv(EXP_DIR / "knowledge" / "interventions.csv")
    rx_codes, meds = _rx_tables(extra, df["SEQN"], iv) if have["P_RXQ_RX"] else ({}, {})
    onto = get_ontology(paths.get("icd10cm_table_index_zip"))
    concepts = _concepts(df, rx_codes, onto)
    current = [sorted(meds.get(s, set())) for s in df["SEQN"]]
    flow.append({"step": "extra XPTs present", "n": len(df), **{k: bool(v) for k, v in have.items()}})
    flow.append({"step": "patients with >=1 ICD-10-CM concept", "n": int(sum(bool(c) for c in concepts))})
    flow.append({"step": "patients with >=1 current intervention-class medication", "n": int(sum(bool(m) for m in current))})
    race = df["RIDRETH3"].map(RIDRETH3).fillna("Unknown")
    context = pd.DataFrame({
        "SEQN": df["SEQN"].astype(int), "ascvd_10yr_risk_pct": df["ascvd_10yr_risk_pct"],
        "race_pce": df["race_group"].map({"black": "Black", "white": "White/other"}), "race": race,
        "smoker": df["smoker"].astype(bool), "diabetic": df["diabetic"].astype(bool),
        "treated_bp": df["treated_bp"].astype(bool), "n_rx_reason_codes": [len(rx_codes.get(s, ())) for s in df["SEQN"]],
    })
    X = df[FEATURES].astype(float).reset_index(drop=True)
    b = DatasetBundle(
        X=X, y=df["cvd_risk_flag"].astype(int).reset_index(drop=True), patient_id=context["SEQN"].copy(),
        feature_meta={k: dict(v) for k, v in FEATURE_META.items()}, context=context.reset_index(drop=True),
        concepts=concepts, current_meds=current,
        subgroups={"sex": X["sex"].map({1.0: "Male", 0.0: "Female"}), "age_band": age_band(X["age"]),
                   "race": race.reset_index(drop=True)},
        name="nhanes", has_ontology=True, cohort_flow=flow,
        label_definition="PCE 10-year ASCVD risk >= 7.5% (Goff 2013); cvd_risk_flag",
        citation="NCHS NHANES 2017-March 2020 pre-pandemic (P_DEMO, P_BPXO, P_BMX, P_TCHOL, P_HDL, P_GHB, P_BPQ, P_DIQ, P_SMQ, P_MCQ, P_KIQ_U, P_RXQ_RX)",
    )
    validate_bundle(b)
    return b
