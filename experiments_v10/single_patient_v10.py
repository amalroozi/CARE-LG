"""
v10 counterpart to src.recourse.single_patient.run_single_patient: runs the
real per-patient recourse pipeline (B1/B2-fixed query attachment,
neurosymbolic admissibility, SHAP, guideline derivation, real-value
verification) for exactly ONE cardio-dataset test patient, given a
pre-built DatasetRunV10 (from experiments_v10.run_v10_cardio) so the
VAE/classifier aren't retrained per patient. Returns the same result-dict
shape src.reporting.report_html expects, so build_success_report /
build_abstain_report work unmodified.

Path search is find_path_sra (src.graph.source_relative_admissibility), the
SRA-fixed replacement for find_path_augmented -- same fix applied in
experiments_v10/run_v10_cardio.py's Stage 1, so these patient walkthroughs
and the aggregate numbers reflect the same (fixed) search.
"""
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import torch

from src.graph.query_attachment import build_edge_table, build_query_edges
from src.graph.neurosymbolic import check_admissibility_symbolic, RULE_CITATIONS
from src.graph.source_relative_admissibility import find_path_sra
from src.recourse.preference import preference_profiles, make_cell_matrix_preference, make_query_row_preference
from src.recourse.explanations import shap_risk_attribution, guideline_derivation_text, semifactual_abstention_explanation
from src.graph.clinical_constraints import unscale_features, compute_clinical_effort

INTERVENTIONS_PATH = REPO_ROOT / "experiments_v10" / "cardio_interventions.json"
METRIC = "riemannian"
MU = 0.5
N_CANDIDATES_SHOWN = 6


def _load_interventions() -> Dict:
    with open(INTERVENTIONS_PATH) as f:
        return json.load(f)


def run_single_patient_v10(run, patient_idx: int, profile: str = "no_preference") -> Dict:
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    # "cardio" has no exercise/medication-averse feature mapping defined (see
    # src/recourse/preference.py's _EXERCISE_LINKED/_MEDICATION_LINKED), so only
    # "no_preference" (base weights, unaffected either way) is meaningful here.
    profiles = preference_profiles("cardio", run.effort_weights)
    weights = profiles[profile]
    interventions = _load_interventions()

    if patient_idx < 0 or patient_idx >= len(run.X_test):
        raise ValueError(f"patient_idx {patient_idx} out of range (0..{len(run.X_test)-1})")

    x0 = run.X_test[patient_idx]
    src_dict = unscale_features(x0, run.scaler, run.feature_cols, run.feature_metadata)
    with torch.no_grad():
        risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()

    shap_vals = shap_risk_attribution(run.classifier, x0, run.X_train, run.feature_cols, n_background=40)
    top_shap = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:6]

    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)

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
    target_mask = np.concatenate([run.low_risk_mask, [False]])

    result = {
        "dataset": "cardio", "seed": run.seed, "patient_idx": int(patient_idx), "profile": profile,
        "raw_record": {k: float(v) for k, v in src_dict.items()},
        "risk_before": risk_before,
        "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
        "candidate_neighbors": candidates_preview,
        "guideline_context": None,
    }

    try:
        path, cost = find_path_sra(base_csr, qedges, qrow, edge_table.N, target_mask)
    except ValueError:
        low_risk_idxs = np.where(run.low_risk_mask)[0]
        sf = semifactual_abstention_explanation(x0, run.X_train, low_risk_idxs, run.feature_metadata,
                                                 run.scaler, run.feature_cols)
        result.update({
            "success": False, "certification": "certified_infeasible_or_unresolved",
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
        "success": True, "risk_after": risk_after, "path_len": len(path) - 1, "cost": float(cost),
        "effort": effort, "chosen_target_node": chosen_target_node, "feature_deltas": feature_deltas,
        "interventions": changed_interventions, "rule_checklist": rule_checklist,
        "guideline_derivation": guideline_derivation,
        "dist_riemannian_chosen": dist_r, "dist_euclidean_chosen": dist_e,
    })
    return result
