"""
Collects the real, end-to-end "patient journey" data for the report's
centerpiece section, by re-running the EXISTING, unmodified v6 pipeline and
explanation code (experiments_v6/lib/*, already built and regression-verified
in the prior v6 session) on two real, already-identified patients:

  - SUCCESS patient: UCI seed 0, test patient #0 -- confirmed, from the
    already-computed experiments_v6/results/uci_per_patient.csv (cell
    riemannian_hard, seed 0), to succeed with path [query, 225], path_len=1.
  - ABSTAIN patient: UCI seed 0, test patient #1 -- confirmed, same CSV, to
    have success=False under the identical cell/seed, and already used for
    one of the three semifactual examples in experiments_v6/report/examples.json.

No new modeling, metric, or rule logic is introduced. This script only calls
already-existing functions (build_query_edges, check_admissibility_symbolic,
semifactual_abstention_explanation, shap_risk_attribution, unscale_features)
and a small amount of pure formatting/labeling code (unit labels, plain-
English sentence assembly from the REAL per-feature deltas and REAL rule
pass/fail results already computed by those functions) to make the output
citable and legible.

Writes experiments_v6/report/journey.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.graph.neurosymbolic import check_admissibility_symbolic, RULE_CITATIONS
from src.recourse.explanations import shap_risk_attribution, semifactual_abstention_explanation
from src.graph.clinical_constraints import unscale_features, compute_clinical_effort

RESULTS_DIR = REPO_ROOT / "results" / "tables"

UNITS = {
    "age": "yrs", "resting_bp": "mmHg", "cholesterol": "mg/dL", "max_heart_rate": "bpm",
    "oldpeak": "mm ST-depression", "sex": "(1=male, 0=female)", "fasting_blood_sugar": "(1=elevated fasting glucose)",
    "exercise_angina": "(1=exercise-induced angina present)", "slope": "(ST-segment slope, ordinal)",
    "systolic_bp": "mmHg", "diastolic_bp": "mmHg", "bmi": "kg/m^2", "glycemic_hba1c": "% HbA1c",
}

RULE_FEATURE_LABEL = {
    "R-IMMUTABLE": "immutable attribute", "R-AGE-MONOTONIC": "age (non-decreasing)",
    "R-AGE-HORIZON": "age (single-step horizon)", "R-BP-DIRECTION": "blood pressure direction",
    "R-LIPID-DIRECTION": "cholesterol direction", "R-GLUCOSE-DIRECTION": "glucose/HbA1c direction",
    "R-BMI-DIRECTION": "BMI direction", "R-DIRECTIONAL-OTHER": "other directional feature",
}


def real_delta_dict(x_src, x_tgt, run):
    src = unscale_features(x_src, run.scaler, run.feature_cols, run.feature_metadata)
    tgt = unscale_features(x_tgt, run.scaler, run.feature_cols, run.feature_metadata)
    out = {}
    for f in run.feature_cols:
        if f in src and f in tgt:
            out[f] = {"before": float(src[f]), "after": float(tgt[f]), "delta": float(tgt[f] - src[f]),
                       "unit": UNITS.get(f, "")}
    return out


def plain_english_step(deltas: dict, admissible_result, feature_metadata) -> str:
    """Builds one sentence from THIS step's real deltas and real rule results
    (not a generic template unconnected to the patient's actual values)."""
    clauses = []
    for f in feature_metadata.get('directional_reduce', []) + feature_metadata.get('directional_increase', []):
        if f in deltas and abs(deltas[f]["delta"]) > 1e-6:
            direction = "decreased" if deltas[f]["delta"] < 0 else "increased"
            clauses.append(f"{f.replace('_',' ')} {direction} by {abs(deltas[f]['delta']):.1f} {deltas[f]['unit']}".strip())
    for f in feature_metadata.get('continuous_mutable', []) + feature_metadata.get('categorical_mutable', []):
        if f in deltas and abs(deltas[f]["delta"]) > 1e-6 and f not in (
            feature_metadata.get('directional_reduce', []) + feature_metadata.get('directional_increase', [])
        ):
            clauses.append(f"{f.replace('_',' ')} changed by {deltas[f]['delta']:+.1f} {deltas[f]['unit']}".strip())
    age_clause = ""
    if "age" in deltas and abs(deltas["age"]["delta"]) > 1e-6:
        age_clause = f", occurring over {deltas['age']['delta']:.1f} years"
    if admissible_result.admissible:
        body = "; ".join(clauses) if clauses else "no mutable feature changed"
        return (f"This step is accepted: {body}{age_clause}. Every applicable guideline rule "
                f"({', '.join(admissible_result.passed_rules)}) passed for this specific transition.")
    else:
        reasons = "; ".join(admissible_result.justifications)
        return f"This step is BLOCKED: {reasons}"


def collect_success_patient(run, edge_table, patient_idx, chosen_target_node):
    x0 = run.X_test[patient_idx]
    with torch.no_grad():
        risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
        mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32, device=run.device).unsqueeze(0))
    z0 = mu.squeeze(0).cpu().numpy()

    shap_vals = shap_risk_attribution(run.classifier, x0, run.X_train, run.feature_cols, n_background=40)
    top_shap = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:6]

    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)

    # Candidate neighbors: the 4 closest by Riemannian distance, PLUS the chosen target
    # if it isn't already among them -- real distances/violates flags from qedges.
    order = np.argsort(qedges.dist_riemannian)
    candidate_positions = list(order[:6])
    chosen_pos = int(np.where(qedges.neighbor_idx == chosen_target_node)[0][0])
    if chosen_pos not in candidate_positions:
        candidate_positions.append(chosen_pos)

    candidates = []
    src_dict = unscale_features(x0, run.scaler, run.feature_cols, run.feature_metadata)
    for pos in candidate_positions:
        node = int(qedges.neighbor_idx[pos])
        admissible_flag = not bool(qedges.violates[pos])
        blocking = []
        if not admissible_flag:
            tgt_dict = unscale_features(run.X_train[node], run.scaler, run.feature_cols, run.feature_metadata)
            res = check_admissibility_symbolic(src_dict, tgt_dict, run.feature_metadata, check_step_horizon=True)
            blocking = res.justifications
        candidates.append({
            "node_idx": node, "admissible": admissible_flag, "chosen": node == chosen_target_node,
            "dist_riemannian": float(qedges.dist_riemannian[pos]), "dist_euclidean": float(qedges.dist_euclidean[pos]),
            "blocking_reasons": blocking,
        })
    candidates.sort(key=lambda c: c["dist_riemannian"])

    # The chosen hop's full checklist + real numbers
    x_target = run.X_train[chosen_target_node]
    tgt_dict = unscale_features(x_target, run.scaler, run.feature_cols, run.feature_metadata)
    admis = check_admissibility_symbolic(src_dict, tgt_dict, run.feature_metadata, check_step_horizon=True)
    checklist = []
    for rid in admis.passed_rules:
        checklist.append({"rule_id": rid, "status": "pass", "citation": RULE_CITATIONS.get(rid, "")})
    for rid in admis.blocking_rules:
        checklist.append({"rule_id": rid, "status": "fail", "citation": RULE_CITATIONS.get(rid, "")})

    deltas = real_delta_dict(x0, x_target, run)
    plain_english = plain_english_step(deltas, admis, run.feature_metadata)

    with torch.no_grad():
        risk_after = float(run.classifier(torch.tensor(x_target, dtype=torch.float32).unsqueeze(0)).item())
    effort = float(compute_clinical_effort(x0, x_target, run.effort_weights, run.feature_metadata, run.scaler, run.feature_cols))

    # 2D PCA projection of a sample of real training Z points, plus this patient's
    # z0 and the chosen target's z -- standard linear-algebra projection of real
    # embeddings (same technique as experiments_v3's figC_path_geometry.png),
    # not a new statistic.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(run.Z_train), size=min(150, len(run.Z_train)), replace=False)
    Z_sample = run.Z_train[sample_idx]
    Z_mean = Z_sample.mean(axis=0)
    U, S, Vt = np.linalg.svd(Z_sample - Z_mean, full_matrices=False)
    proj = lambda z: ((z - Z_mean) @ Vt[:2].T).tolist()

    return {
        "dataset": run.dataset, "seed": run.seed, "patient_idx": int(patient_idx),
        "raw_record": {k: float(v) for k, v in src_dict.items()},
        "risk_before": risk_before, "risk_after": risk_after,
        "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
        "latent_z": z0.tolist(),
        "latent_pca_sample": [proj(z) for z in Z_sample[:60]],
        "latent_pca_query": proj(z0),
        "latent_pca_target": proj(run.Z_train[chosen_target_node]),
        "candidate_neighbors": candidates,
        "chosen_target_node": int(chosen_target_node),
        "rule_checklist": checklist,
        "feature_deltas": deltas,
        "plain_english": plain_english,
        "dist_riemannian_chosen": float(qedges.dist_riemannian[chosen_pos]),
        "dist_euclidean_chosen": float(qedges.dist_euclidean[chosen_pos]),
        "effort": effort,
    }


def collect_abstain_patient(run, patient_idx):
    x0 = run.X_test[patient_idx]
    with torch.no_grad():
        risk_before = float(run.classifier(torch.tensor(x0, dtype=torch.float32).unsqueeze(0)).item())
    shap_vals = shap_risk_attribution(run.classifier, x0, run.X_train, run.feature_cols, n_background=40)
    top_shap = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:6]
    low_risk_idxs = np.where(run.low_risk_mask)[0]
    sf = semifactual_abstention_explanation(x0, run.X_train, low_risk_idxs, run.feature_metadata, run.scaler, run.feature_cols)
    src_dict = unscale_features(x0, run.scaler, run.feature_cols, run.feature_metadata)
    return {
        "dataset": run.dataset, "seed": run.seed, "patient_idx": int(patient_idx),
        "raw_record": {k: float(v) for k, v in src_dict.items()},
        "risk_before": risk_before,
        "shap_top_features": [{"feature": f, "value": v} for f, v in top_shap],
        "n_candidates_checked": len(low_risk_idxs),
        "closest_miss_candidate_idx": sf.closest_miss_candidate_idx,
        "closest_miss_violation_count": sf.closest_miss_violation_count,
        "blocking_rule_frequency": sf.blocking_rule_frequency,
        "semifactual_text": sf.text,
    }


def main():
    run = run_dataset_seed("uci", seed=0, device=torch.device("cpu"))
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)

    # Confirm patient 0 / 225 and patient 1's abstention against the already-computed CSV.
    df = pd.read_csv(RESULTS_DIR / "uci_per_patient.csv")
    row0 = df[(df.seed == 0) & (df.cell == "riemannian_hard") & (df.patient_idx == 0)].iloc[0]
    assert bool(row0["success"]), "expected patient 0 to succeed under riemannian_hard seed 0"
    chosen_target = int(json.loads(row0["path"])[-1])
    row1 = df[(df.seed == 0) & (df.cell == "riemannian_hard") & (df.patient_idx == 1)].iloc[0]
    assert not bool(row1["success"]), "expected patient 1 to abstain under riemannian_hard seed 0"

    success = collect_success_patient(run, edge_table, 0, chosen_target)
    abstain = collect_abstain_patient(run, 1)

    out = {"success_patient": success, "abstain_patient": abstain}
    with open(Path(__file__).parent / "journey.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote journey.json:", "success target=", chosen_target, "risk", success["risk_before"], "->", success["risk_after"],
          " | abstain patient risk", abstain["risk_before"])


if __name__ == "__main__":
    main()
