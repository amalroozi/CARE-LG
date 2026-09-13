"""
Collects a few concrete, citable examples for the static HTML report by
calling the EXISTING, unmodified experiments_v6/lib/explanations.py and
neurosymbolic.py code (already built and regression-verified against the
hardcoded predicate at 100% agreement, 4000/4000 pairs, in the prior v6
session -- see experiments_v6/FINDINGS.md Phase 4). This script invents no
new logic, model, or metric -- it re-runs the same trained pipeline
(run_dataset_seed, deterministic given its seed) and the same explanation
functions to capture real output to a file, since the prior session printed
these to the terminal without saving them.

Writes experiments_v6/report/examples.json.
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

import experiments_v6.lib.dataset_registration  # noqa: F401
from experiments_v6.lib.pipeline_v6 import run_dataset_seed
from experiments_v6.lib.explanations import semifactual_abstention_explanation, guideline_derivation_text
from src.graph.clinical_constraints import unscale_features

RESULTS_DIR = REPO_ROOT / "experiments_v6" / "results"


def main():
    out = {"semifactual_examples": [], "guideline_derivation_examples": []}

    # --- 3 real semifactual abstention explanations, one for UCI (seed 0), two
    # for real NHANES (seed 0), drawn from patients the ALREADY-COMPUTED
    # results/{dataset}_blindspot.csv marks certified_infeasible == True. ---
    for dataset, seed, n_examples in [("uci", 0, 1), ("nhanes_real", 0, 2)]:
        run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
        bdf = pd.read_csv(RESULTS_DIR / f"{dataset}_blindspot.csv")
        certified = bdf[(bdf.seed == seed) & (bdf.certified_infeasible == True)]
        low_risk_idxs = np.where(run.low_risk_mask)[0]
        for pidx in certified["patient_idx"].head(n_examples):
            x0 = run.X_test[int(pidx)]
            sf = semifactual_abstention_explanation(x0, run.X_train, low_risk_idxs, run.feature_metadata,
                                                     run.scaler, run.feature_cols)
            out["semifactual_examples"].append({
                "dataset": dataset, "seed": seed, "patient_idx": int(pidx),
                "text": sf.text,
                "closest_miss_candidate_idx": sf.closest_miss_candidate_idx,
                "closest_miss_violation_count": sf.closest_miss_violation_count,
                "blocking_rule_frequency": sf.blocking_rule_frequency,
            })

    # --- 1 real guideline-derivation example: UCI seed 0 patient #0, the
    # actual successful step already used in the medication_averse row of
    # results/preference_demo.csv (target_node=225). ---
    run = run_dataset_seed("uci", 0, device=torch.device("cpu"))
    pref_df = pd.read_csv(RESULTS_DIR / "preference_demo.csv")
    row = pref_df[(pref_df.dataset == "uci") & (pref_df.seed == 0) &
                   (pref_df.patient_idx == 0) & (pref_df.profile == "medication_averse")].iloc[0]
    x0 = run.X_test[0]
    x_target = run.X_train[int(row["target_node"])]
    derivation = guideline_derivation_text(x0, x_target, run.feature_metadata, run.scaler, run.feature_cols)
    src_real = unscale_features(x0, run.scaler, run.feature_cols, run.feature_metadata)
    tgt_real = unscale_features(x_target, run.scaler, run.feature_cols, run.feature_metadata)
    real_deltas = {k: float(tgt_real[k] - src_real[k]) for k in run.feature_metadata['continuous_mutable']
                   if k in src_real and k in tgt_real}
    out["guideline_derivation_examples"].append({
        "dataset": "uci", "seed": 0, "patient_idx": 0, "target_node": int(row["target_node"]),
        "feature_deltas_real_units": real_deltas,
        "derivation": derivation,
    })

    with open(Path(__file__).parent / "examples.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out['semifactual_examples'])} semifactual + "
          f"{len(out['guideline_derivation_examples'])} guideline-derivation examples")


if __name__ == "__main__":
    main()
