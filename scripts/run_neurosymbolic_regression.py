"""
Phase 4 regression check: confirms the symbolic admissibility rules in
experiments_v6/lib/neurosymbolic.py::check_admissibility_symbolic agree
EXACTLY with the hardcoded predicate they re-encode
(src/graph/clinical_constraints.py::check_clinical_violations), on every
covered constraint type (immutable / non_decreasing / age_horizon /
directional_reduce / directional_increase).

Method: for both datasets, seed 0, samples a set of (source, target) pairs
that actually occur as candidate edges in the corrected v6 pipeline
(every training-node pair the query-attachment step considers, i.e. the
same real-valued transitions Phase 1 evaluates), evaluates both the
hardcoded predicate and the symbolic reasoner on each, and reports the
agreement rate plus any discrepant pairs verbatim (there should be none).

Usage:
  .venv/bin/python -m experiments_v6.run_neurosymbolic_regression
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.graph.clinical_constraints import check_clinical_violations, unscale_features
import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.neurosymbolic import check_admissibility_symbolic

RESULTS_DIR = REPO_ROOT / "results" / "tables"
N_PAIRS_PER_DATASET = 2000
RNG_SEED = 0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    summary = {}
    all_discrepancies = []

    for dataset in ["uci", "nhanes_real"]:
        run = run_dataset_seed(dataset, seed=0, device=torch.device("cpu"))
        N = len(run.X_train)
        n_pairs = min(N_PAIRS_PER_DATASET, N * N)

        # Sample real (source, target) pairs: mix of query->train (patient entry
        # transitions) and train->train (interior transitions), matching what
        # the corrected v6 graph actually evaluates admissibility on.
        n_query = n_pairs // 2
        n_interior = n_pairs - n_query

        pairs = []  # (x_source, x_target) as raw scaled feature vectors
        test_idx = rng.integers(0, len(run.X_test), size=n_query)
        train_idx_a = rng.integers(0, N, size=n_query)
        for ti, tj in zip(test_idx, train_idx_a):
            pairs.append((run.X_test[ti], run.X_train[tj]))

        train_idx_b = rng.integers(0, N, size=n_interior)
        train_idx_c = rng.integers(0, N, size=n_interior)
        for ti, tj in zip(train_idx_b, train_idx_c):
            pairs.append((run.X_train[ti], run.X_train[tj]))

        n_agree, n_disagree = 0, 0
        for x_src, x_tgt in pairs:
            hard_result = check_clinical_violations(
                x_src, x_tgt, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True
            )
            src_dict = unscale_features(x_src, run.scaler, run.feature_cols, run.feature_metadata)
            tgt_dict = unscale_features(x_tgt, run.scaler, run.feature_cols, run.feature_metadata)
            symbolic_result = check_admissibility_symbolic(
                src_dict, tgt_dict, run.feature_metadata, check_step_horizon=True
            )
            symbolic_violates = not symbolic_result.admissible

            if bool(hard_result) == bool(symbolic_violates):
                n_agree += 1
            else:
                n_disagree += 1
                all_discrepancies.append({
                    "dataset": dataset,
                    "hardcoded_violates": bool(hard_result),
                    "symbolic_violates": bool(symbolic_violates),
                    "blocking_rules": symbolic_result.blocking_rules,
                    "source": {k: float(v) for k, v in src_dict.items()},
                    "target": {k: float(v) for k, v in tgt_dict.items()},
                })

        agreement_pct = 100.0 * n_agree / (n_agree + n_disagree)
        summary[dataset] = {"n_pairs": n_agree + n_disagree, "n_agree": n_agree,
                             "n_disagree": n_disagree, "agreement_pct": agreement_pct}
        print(f"[regression] {dataset}: {n_agree}/{n_agree+n_disagree} agree "
              f"({agreement_pct:.4f}%), {n_disagree} discrepancies", flush=True)

    with open(RESULTS_DIR / "neurosymbolic_regression_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "neurosymbolic_regression_discrepancies.json", "w") as f:
        json.dump(all_discrepancies, f, indent=2)
    print(f"[regression] wrote summary + {len(all_discrepancies)} discrepancies to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
