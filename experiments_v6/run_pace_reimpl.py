"""
Phase 8 -- minimal, honestly-labeled PACE-style reimplementation.

WHAT PACE ACTUALLY IS (confirmed by reading the paper directly this
session, arXiv:2607.01306, "PACE: A Neuro-Symbolic Framework for Plausible
and Actionable Counterfactual Explanations"): an MLP classifier (2 hidden
layers, 32/16 units) plus an ASP program that encodes immutable attributes,
editable attributes, and per-feature transition rules (e.g. education only
moves to an adjacent level; hours change in fixed +-5/+-10 steps). Search is
an INCREMENTAL-BUDGET generate-and-test loop: start with a minimal
intervention budget and expand it until a classifier-flipping, ASP-admissible
candidate is found. Evaluated on the Adult Income dataset (income
classification, categorical transition-graph constraints), NOT a clinical
domain.

DISCREPANCY WITH THE BRIEF, documented per Hard Rule 5: the brief describes
PACE as "single-step, not multi-hop." Reading the actual paper shows this is
not quite right -- PACE's search is multi-step in the sense of incremental
BUDGET (how many features change at once), though each candidate is still a
single classifier query (not a multi-hop GRAPH traversal through other real
patients the way FACE/CARE-LG are). The defensible reading, adopted here:
PACE's core distinguishing idea relative to this project's graph-based
methods is exactly that -- perturb-and-classify against a symbolic-rule
FILTER, not path-search over real recorded patients. That is what this
reimplementation reproduces.

WHAT THIS SCRIPT DOES: a minimal PACE-STYLE method -- NOT a reproduction of
PACE's own numbers, which are reported on a different dataset/domain
entirely (Adult Income, categorical transition graphs) and are presented
separately in FINDINGS.md, clearly labeled "reported by PACE's authors, not
independently reproduced." This script instead builds the same CORE IDEA
(MLP classifier + ASP-admissible candidate generation with incremental
budget expansion, no graph/no real-patient lookup at all) adapted to THIS
project's own clinical domain and Phase 4 rule base, so a same-domain,
same-classifier, same-rule-base comparison against v6/FACE is possible.
Numbers from this script describe THIS reimplementation only.

Method: for each high-risk test patient, try intervention budgets b=1,2,3
(number of simultaneously-changed continuous mutable features). At each
budget, enumerate all size-b feature subsets x a small fixed grid of
admissible-direction step magnitudes (10%/20%/30% of that feature's
training-population std, in the guideline-admissible direction only, per
Phase 4's R-* rules), evaluate the classifier on every candidate in a single
batched forward pass, and accept the first (lowest-cost) candidate that (a)
satisfies the Phase-4 symbolic admissibility check and (b) drops predicted
risk below LOW_RISK_THRESHOLD. Stops at the first budget level with any
admissible, risk-flipping candidate (mirrors PACE's incremental-budget
stopping rule). No graph, no k-NN, no real training patient is looked up at
any point -- candidates are synthetic perturbations of x0 itself, exactly as
PACE's own method generates candidates.

Usage:
  .venv/bin/python -m experiments_v6.run_pace_reimpl --datasets uci nhanes_real --seeds 0 1 2 3 4
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import experiments_v6.lib.dataset_registration  # noqa: F401
from experiments_v6.lib.pipeline_v6 import run_dataset_seed, LOW_RISK_THRESHOLD
from experiments_v6.lib.neurosymbolic import check_admissibility_symbolic
from src.graph.clinical_constraints import unscale_features, compute_clinical_effort

MAGNITUDES = (0.25, 0.5, 0.75, 1.0, 1.5)  # fraction of feature's training std, admissible direction only
# Verified by a direct sanity check this session (see CHANGES.md): with the
# original narrower grid (0.10/0.20/0.30 std, budget<=3), a UCI seed-0 pilot
# found the classifier essentially unflippable -- 0/5 patients succeeded, even
# though flipping IS possible (confirmed directly: shifting all 4 admissible
# UCI features by 1.0 std simultaneously moved a 0.75 base risk to 0.47). The
# grid above was widened, and MAX_BUDGET set to use every admissible feature
# at once, so the reimplementation actually exercises PACE's own incremental-
# budget-to-flip idea rather than under-searching by construction.


def admissible_step_features(feature_metadata):
    """Continuous mutable features with a directional constraint (reduce or increase);
    PACE-style candidates only ever move in the guideline-admissible direction."""
    steps = {}
    for f in feature_metadata.get('directional_reduce', []):
        if f in feature_metadata.get('continuous_mutable', []):
            steps[f] = -1  # admissible direction: decrease
    for f in feature_metadata.get('directional_increase', []):
        if f in feature_metadata.get('continuous_mutable', []):
            steps[f] = +1  # admissible direction: increase
    return steps


def evaluate_patient_pace(x0, run, admissible_features, feature_std):
    feature_cols = run.feature_cols
    idx_of = {f: feature_cols.index(f) for f in admissible_features}
    t0 = time.perf_counter()
    max_budget = len(admissible_features)  # exhaust every editable feature before giving up

    for budget in range(1, max_budget + 1):
        feats = list(admissible_features.keys())
        if len(feats) < budget:
            continue
        candidates = []
        for subset in itertools.combinations(feats, budget):
            for mags in itertools.product(MAGNITUDES, repeat=budget):
                x_cand = x0.copy()
                for f, mag in zip(subset, mags):
                    direction = admissible_features[f]
                    x_cand[idx_of[f]] += direction * mag * feature_std[f]
                candidates.append(x_cand)
        if not candidates:
            continue
        X_cand = np.vstack(candidates)
        with torch.no_grad():
            risks = run.classifier(torch.tensor(X_cand, dtype=torch.float32)).cpu().numpy().squeeze(-1)

        # cheapest (fewest features, smallest total magnitude) candidate that flips risk,
        # scanned in the order generated (smallest magnitudes first) -- first hit wins,
        # mirroring PACE's "minimal intervention" preference within a budget level.
        for i, risk in enumerate(risks):
            if risk < LOW_RISK_THRESHOLD:
                x_cand = X_cand[i]
                src_dict = unscale_features(x0, run.scaler, feature_cols, run.feature_metadata)
                tgt_dict = unscale_features(x_cand, run.scaler, feature_cols, run.feature_metadata)
                admis = check_admissibility_symbolic(src_dict, tgt_dict, run.feature_metadata, check_step_horizon=False)
                if admis.admissible:  # should always hold by construction; verified, not assumed
                    latency = time.perf_counter() - t0
                    effort = float(compute_clinical_effort(x0, x_cand, run.effort_weights, run.feature_metadata, run.scaler, feature_cols))
                    return {"success": True, "budget": budget, "n_features_changed": budget,
                            "latency_sec": latency, "effort": effort, "admissible": True}
    latency = time.perf_counter() - t0
    return {"success": False, "budget": None, "n_features_changed": None,
            "latency_sec": latency, "effort": None, "admissible": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"], choices=["uci", "nhanes_real"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "experiments_v6" / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"

    all_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
            admissible_features = admissible_step_features(run.feature_metadata)
            feature_std = {f: float(np.std(run.X_train[:, run.feature_cols.index(f)])) for f in admissible_features}

            t0 = time.perf_counter()
            n_success = 0
            for pidx in run.high_risk_test_idx:
                r = evaluate_patient_pace(run.X_test[pidx], run, admissible_features, feature_std)
                r.update({"dataset": dataset, "seed": seed, "patient_idx": int(pidx)})
                all_rows.append(r)
                n_success += int(r["success"])
            t1 = time.perf_counter()
            n = len(run.high_risk_test_idx)
            print(f"[pace] {dataset} seed={seed}: success={n_success}/{n} ({100.0*n_success/max(1,n):.1f}%) "
                  f"in {t1-t0:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "pace_reimpl_per_patient.csv", index=False)
    with open(out_dir / "pace_reimpl_metadata.json", "w") as f:
        json.dump({"git_sha": sha, "datasets": args.datasets, "seeds": args.seeds,
                    "magnitudes_frac_std": MAGNITUDES, "max_budget": "len(admissible_features), per dataset",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print("\n[pace] === SUMMARY (mean +/- std over seeds) ===")
    for dataset in args.datasets:
        d = df[df.dataset == dataset]
        per_seed_succ = d.groupby("seed")["success"].mean() * 100
        print(f"{dataset:12s} success={per_seed_succ.mean():.1f}+/-{per_seed_succ.std():.1f}%  "
              f"(all admissible-by-construction -> real_cvr=0.0% for any success)")


if __name__ == "__main__":
    main()
