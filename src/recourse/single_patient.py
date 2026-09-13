"""
Runs the corrected v6/src pipeline end-to-end for exactly ONE real patient and
returns a single, self-contained result dict -- the shared data source for both
the auto-generated single-patient HTML report (src/reporting/report_html.py)
and, optionally, the UI's /api/recourse endpoint. No server is required to use
this module; it trains its own (dataset, seed) models directly via
src.pipeline.run_dataset_seed, exactly like every other v6 script.

Every number in the returned dict comes from actually running this patient
through the real, unmodified pipeline (B1/B2-fixed query attachment,
neurosymbolic admissibility, SHAP, preference-reweighted cost, real-value
verification) -- nothing here is templated or invented.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table, build_query_edges, augmented_matrix
from src.graph.neurosymbolic import check_admissibility_symbolic, RULE_CITATIONS
from src.recourse.search_augmented import find_path_augmented
from src.recourse.preference import preference_profiles, make_cell_matrix_preference, make_query_row_preference
from src.recourse.explanations import (
    shap_risk_attribution, guideline_derivation_text, semifactual_abstention_explanation,
    nhanes_guideline_context,
)
from src.graph.clinical_constraints import unscale_features, compute_clinical_effort

RESULTS_DIR = REPO_ROOT / "experiments_v6" / "results"
INTERVENTIONS_PATH = REPO_ROOT / "configs" / "interventions.json"
METRIC = "riemannian"
MU = 0.5
N_CANDIDATES_SHOWN = 6


def _load_interventions() -> Dict:
    with open(INTERVENTIONS_PATH) as f:
        return json.load(f)


def _lookup_certification(dataset: str, seed: int, patient_idx: int) -> str:
    path = RESULTS_DIR / f"{dataset}_blindspot.csv"
    if not path.exists():
        return "unresolved"
    bdf = pd.read_csv(path)
    row = bdf[(bdf.seed == seed) & (bdf.patient_idx == patient_idx)]
    if len(row) == 0:
        return "unresolved"
    r = row.iloc[0]
    if bool(r["certified_infeasible"]):
        return "certified_infeasible"
    if bool(r["confirmed_blindspot"]):
        return "confirmed_blindspot"
    return "unresolved"


def run_single_patient(dataset: str, seed: int, patient_idx: int, profile: str = "no_preference") -> Dict:
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    profiles = preference_profiles(dataset, run.effort_weights)
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; available: {list(profiles.keys())}")
    weights = profiles[profile]
    interventions = _load_interventions()

    if patient_idx < 0 or patient_idx >= len(run.X_test):
        raise ValueError(f"patient_idx {patient_idx} out of range for {dataset}/seed={seed} (0..{len(run.X_test)-1})")

    x0 = run.X_test[patient_idx]
    src_dict = unscale_features(x0, run.scaler, run.feature_cols, run.feature_metadata)
    with torch.no_grad():
        risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))[0].squeeze(0).cpu().numpy()

    shap_vals = shap_risk_attribution(run.classifier, x0, run.X_train, run.feature_cols, n_background=40)
    top_shap = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:6]

    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)

    # Closest-by-Riemannian-distance candidates -- real evidence for the
    # "Graph + Riemannian Geometry" architecture-walkthrough section, whether
    # this patient ultimately succeeds or abstains.
    order = np.argsort(qedges.dist_riemannian)
    candidate_positions = list(order[:N_CANDIDATES_SHOWN])
    candidates_preview = []
    for pos in candidate_positions:
        node = int(qedges.neighbor_idx[pos])
        admissible_flag = not bool(qedges.violates[pos])
        blocking = []
        if not admissible_flag:
            tgt_dict = unscale_features(run.X_train[node], run.scaler, run.feature_cols, run.feature_metadata)
            res = check_admissibility_symbolic(src_dict, tgt_dict, run.feature_metadata, check_step_horizon=True)
            blocking = res.justifications
        candidates_preview.append({
            "node_idx": node, "admissible": admissible_flag, "chosen": False,
            "dist_riemannian": float(qedges.dist_riemannian[pos]), "dist_euclidean": float(qedges.dist_euclidean[pos]),
            "blocking_reasons": blocking,
        })

    base_csr = make_cell_matrix_preference(edge_table, run.X_train, METRIC, weights,
                                            run.feature_metadata, run.feature_cols, mu=MU)
    qrow = make_query_row_preference(qedges, x0, run.X_train, METRIC, weights,
                                      run.feature_metadata, run.feature_cols, edge_table.N, mu=MU)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    guideline_context = nhanes_guideline_context(seed, patient_idx) if dataset == "nhanes_real" else None

    result = {
        "dataset": dataset, "seed": seed, "patient_idx": int(patient_idx), "profile": profile,
        "raw_record": {k: float(v) for k, v in src_dict.items()},
        "risk_before": risk_before,
        "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
        "candidate_neighbors": candidates_preview,
        "guideline_context": guideline_context,
    }

    try:
        path, cost = find_path_augmented(aug, edge_table.N, target_mask)
    except ValueError:
        low_risk_idxs = np.where(run.low_risk_mask)[0]
        sf = semifactual_abstention_explanation(x0, run.X_train, low_risk_idxs, run.feature_metadata,
                                                 run.scaler, run.feature_cols)
        result.update({
            "success": False,
            "certification": _lookup_certification(dataset, seed, patient_idx),
            "semifactual_text": sf.text,
            "closest_miss_candidate_idx": sf.closest_miss_candidate_idx,
            "closest_miss_violation_count": sf.closest_miss_violation_count,
            "blocking_rule_frequency": sf.blocking_rule_frequency,
            "n_low_risk_candidates_checked": len(low_risk_idxs),
        })
        return result

    chosen_target_node = int(path[-1])
    x_target = run.X_train[chosen_target_node]
    for c in result["candidate_neighbors"]:
        if c["node_idx"] == chosen_target_node:
            c["chosen"] = True
    if not any(c["node_idx"] == chosen_target_node for c in result["candidate_neighbors"]):
        pos = int(np.where(qedges.neighbor_idx == chosen_target_node)[0][0])
        result["candidate_neighbors"].append({
            "node_idx": chosen_target_node, "admissible": True, "chosen": True,
            "dist_riemannian": float(qedges.dist_riemannian[pos]), "dist_euclidean": float(qedges.dist_euclidean[pos]),
            "blocking_reasons": [],
        })
    result["candidate_neighbors"].sort(key=lambda c: c["dist_riemannian"])

    with torch.no_grad():
        risk_after = float(run.classifier(torch.tensor(x_target, dtype=torch.float32).unsqueeze(0)).item())

    tgt_dict = unscale_features(x_target, run.scaler, run.feature_cols, run.feature_metadata)
    admis = check_admissibility_symbolic(src_dict, tgt_dict, run.feature_metadata, check_step_horizon=True)
    rule_checklist = (
        [{"rule_id": rid, "status": "pass", "citation": RULE_CITATIONS.get(rid, "")} for rid in admis.passed_rules]
        + [{"rule_id": rid, "status": "fail", "citation": RULE_CITATIONS.get(rid, "")} for rid in admis.blocking_rules]
    )

    feature_deltas = {}
    changed_interventions = []
    for f in run.feature_cols:
        if f in src_dict and f in tgt_dict:
            delta = float(tgt_dict[f] - src_dict[f])
            if abs(delta) > 1e-6:
                feature_deltas[f] = {"before": float(src_dict[f]), "after": float(tgt_dict[f]), "delta": delta}
                if f in interventions:
                    changed_interventions.append({"feature": f, **interventions[f]})

    guideline_derivation = guideline_derivation_text(x0, x_target, run.feature_metadata, run.scaler, run.feature_cols)
    effort = float(compute_clinical_effort(x0, x_target, run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols))
    dist_r = float(qedges.dist_riemannian[int(np.where(qedges.neighbor_idx == chosen_target_node)[0][0])])
    dist_e = float(qedges.dist_euclidean[int(np.where(qedges.neighbor_idx == chosen_target_node)[0][0])])

    result.update({
        "success": True,
        "risk_after": risk_after,
        "path_len": len(path) - 1,
        "cost": float(cost),
        "effort": effort,
        "chosen_target_node": chosen_target_node,
        "feature_deltas": feature_deltas,
        "interventions": changed_interventions,
        "rule_checklist": rule_checklist,
        "guideline_derivation": guideline_derivation,
        "dist_riemannian_chosen": dist_r,
        "dist_euclidean_chosen": dist_e,
    })
    return result
