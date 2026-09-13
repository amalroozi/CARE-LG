"""
Phase 7 -- minimal local FastAPI backend wrapping the corrected v6 pipeline
(Phases 1-6). Serves the static frontend in this directory and a small JSON
API the frontend calls. Functional, not polished; no screenshots included
per the brief. See ../HOW_TO_RUN.md for how to start this.

Trained models (VAE + classifier) are cached in-process per (dataset, seed)
the first time they're requested, so repeated UI interactions after the
first patient are fast; the very first request for a given (dataset, seed)
pair takes a few seconds (UCI) to ~3-5s (real NHANES) while it trains.
"""
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed, LOW_RISK_THRESHOLD
from src.graph.query_attachment import build_edge_table, build_query_edges, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.recourse.preference import preference_profiles, make_cell_matrix_preference, make_query_row_preference
from src.recourse.explanations import (
    shap_risk_attribution, guideline_derivation_text, semifactual_abstention_explanation,
    nhanes_guideline_context,
)
from src.graph.clinical_constraints import unscale_features, compute_clinical_effort
from src.recourse.single_patient import run_single_patient
from src.reporting.report_html import write_report
import webbrowser

# Resolves experiments_v6_audit/DEEP_AUDIT.md issue #1 ("update the UI to import
# from src/, not experiments_v6/lib/ directly, so the deployable path and the
# demoed path are the same code") -- every import above now comes from src/,
# the canonical location as of Phase 1's merge (see DEEP_AUDIT_RESOLUTION.md).

RESULTS_DIR = REPO_ROOT / "experiments_v6" / "results"
METRIC = "riemannian"
MU = 0.5

app = FastAPI(title="CARE-LG v6 UI backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {}  # (dataset, seed) -> {"run":..., "edge_table":..., "profiles":..., "blindspot_df":...}


def get_run(dataset: str, seed: int):
    key = (dataset, seed)
    if key not in _cache:
        run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
        edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                      run.scaler, run.feature_cols, k=run.k_neighbors)
        profiles = preference_profiles(dataset, run.effort_weights)
        blindspot_path = RESULTS_DIR / f"{dataset}_blindspot.csv"
        blindspot_df = pd.read_csv(blindspot_path) if blindspot_path.exists() else None
        _cache[key] = {"run": run, "edge_table": edge_table, "profiles": profiles, "blindspot_df": blindspot_df}
    return _cache[key]


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": ["uci", "nhanes_real"]}


@app.get("/api/patients")
def list_patients(dataset: str, seed: int = 0, limit: int = 60):
    c = get_run(dataset, seed)
    run = c["run"]
    with torch.no_grad():
        all_risk = run.classifier(torch.tensor(run.X_test, dtype=torch.float32)).cpu().numpy().squeeze(-1)
    high_risk_set = set(int(i) for i in run.high_risk_test_idx)
    idxs = list(run.high_risk_test_idx[:limit])
    return {
        "dataset": dataset, "seed": seed,
        "patients": [
            {"patient_idx": int(i), "risk": float(all_risk[i]), "high_risk": int(i) in high_risk_set}
            for i in idxs
        ],
        "preference_profiles": list(c["profiles"].keys()),
    }


class RecourseRequest(BaseModel):
    dataset: str
    seed: int = 0
    patient_idx: int
    profile: str = "no_preference"
    generate_report: bool = False  # opt-in: also write + auto-open the standalone HTML report


@app.post("/api/recourse")
def recourse(req: RecourseRequest):
    if req.generate_report:
        # Auto-generate + auto-open the standalone single-patient HTML report.
        # Uses the same underlying pipeline (src.recourse.single_patient) as
        # the CLI entry point (experiments_v6/run_single_patient.py) -- one
        # source of truth for single-patient computation, reused here rather
        # than duplicated. This recomputes independently of the caching below
        # (acceptable since it's opt-in and infrequent, not the default path).
        report_result = run_single_patient(req.dataset, req.seed, req.patient_idx, req.profile)
        report_path = write_report(report_result, RESULTS_DIR)
        webbrowser.open(f"file://{report_path.resolve()}")

    c = get_run(req.dataset, req.seed)
    run = c["run"]
    edge_table = c["edge_table"]
    if req.profile not in c["profiles"]:
        raise HTTPException(400, f"unknown profile {req.profile}")
    weights = c["profiles"][req.profile]

    if req.patient_idx < 0 or req.patient_idx >= len(run.X_test):
        raise HTTPException(404, "patient_idx out of range for this dataset/seed")
    x0 = run.X_test[req.patient_idx]
    with torch.no_grad():
        risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
        mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))
    z0 = mu.squeeze(0).cpu().numpy()

    N = edge_table.N
    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)
    base_csr = make_cell_matrix_preference(edge_table, run.X_train, METRIC, weights,
                                            run.feature_metadata, run.feature_cols, mu=MU)
    qrow = make_query_row_preference(qedges, x0, run.X_train, METRIC, weights,
                                      run.feature_metadata, run.feature_cols, N, mu=MU)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    shap_vals = shap_risk_attribution(run.classifier, x0, run.X_train, run.feature_cols, n_background=30)
    top_shap = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:6]

    try:
        path, cost = find_path_augmented(aug, N, target_mask)
    except ValueError:
        # --- Abstention: certification + semifactual explanation (Phase 5, layer 3) ---
        low_risk_idxs = np.where(run.low_risk_mask)[0]
        sf = semifactual_abstention_explanation(x0, run.X_train, low_risk_idxs, run.feature_metadata,
                                                 run.scaler, run.feature_cols)
        certification = "unresolved"
        bdf = c["blindspot_df"]
        if bdf is not None:
            row = bdf[(bdf.seed == req.seed) & (bdf.patient_idx == req.patient_idx)]
            if len(row) > 0:
                r = row.iloc[0]
                certification = ("certified_infeasible" if bool(r["certified_infeasible"])
                                  else "confirmed_blindspot" if bool(r["confirmed_blindspot"])
                                  else "unresolved")
        guideline_context = None
        if req.dataset == "nhanes_real":
            guideline_context = nhanes_guideline_context(req.seed, req.patient_idx)
        return {
            "success": False,
            "risk_before": risk_before,
            "certification": certification,
            "semifactual_explanation": sf.text,
            "closest_miss_candidate_idx": sf.closest_miss_candidate_idx,
            "closest_miss_violation_count": sf.closest_miss_violation_count,
            "blocking_rule_frequency": sf.blocking_rule_frequency,
            "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
            "guideline_context": guideline_context,
        }

    target_node = path[-1]
    x_target = run.X_train[target_node]
    with torch.no_grad():
        risk_after = float(run.classifier(torch.tensor(x_target, dtype=torch.float32).unsqueeze(0)).item())

    steps = []
    real_points = [x0 if n == N else run.X_train[n] for n in path]
    for i, (a, b) in enumerate(zip(real_points[:-1], real_points[1:])):
        src_dict = unscale_features(a, run.scaler, run.feature_cols, run.feature_metadata)
        tgt_dict = unscale_features(b, run.scaler, run.feature_cols, run.feature_metadata)
        deltas = {k: float(tgt_dict[k] - src_dict[k]) for k in src_dict if k in tgt_dict and abs(tgt_dict[k] - src_dict[k]) > 1e-6}
        derivation = guideline_derivation_text(a, b, run.feature_metadata, run.scaler, run.feature_cols)
        steps.append({
            "step": i + 1,
            "feature_deltas": deltas,
            "guideline_derivation": derivation,
        })

    effort = float(compute_clinical_effort(x0, x_target, run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols))

    guideline_context = None
    if req.dataset == "nhanes_real":
        guideline_context = nhanes_guideline_context(req.seed, req.patient_idx)

    return {
        "success": True,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "path_len": len(path) - 1,
        "cost": float(cost),
        "effort": effort,
        "steps": steps,
        "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
        "guideline_context": guideline_context,
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
