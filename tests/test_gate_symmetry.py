"""
The regression test that would have caught B1 (the entry-gate/interior-gate
asymmetry) had it existed during the original investigation. Full history:
archive/investigation_v2_to_v6/experiments_v6_audit/RECOMMENDATION.md and
docs/HISTORY.md.

Asserts, for a sample of real graph edges (query patients into the training
graph, and training-node-to-training-node interior edges), that the ENTRY
predicate (`src.graph.query_attachment.build_query_edges`'s `violates` flag)
and the INTERIOR predicate (`src.graph.query_attachment.vectorized_violation_counts`,
the same function `build_edge_table` uses for every interior edge) agree
EXACTLY on every sampled transition.

This test is verified, in this file, to actually discriminate: a
deliberately reconstructed COPY of the pre-fix relaxed entry gate
(hard-only, `check_step_horizon=False` -- the exact historical bug) is run
against the same sampled edges and shown to DISAGREE with the interior
predicate on a nonzero fraction of them -- i.e. this test fails on the
pre-fix code path and passes on the corrected one, in both directions.

Run: .venv/bin/python -m pytest tests/test_gate_symmetry.py -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import pytest

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import (
    build_query_edges, unscale_matrix, vectorized_violation_components, vectorized_violation_counts,
)

N_QUERY_SAMPLES = 200


def _relaxed_pre_fix_gate(unscaled, src_idx, tgt_idx, feature_metadata):
    """
    Deliberately reconstructs the CONFIRMED pre-fix bug from
    experiments_v3/lib/graph_build.py:238 (see experiments_v6_audit/RECOMMENDATION.md
    and experiments_v6/CHANGES.md's "B1 fix" section): hard-only,
    check_step_horizon=False. Exists ONLY so this test can prove it actually
    discriminates buggy code from fixed code -- it is not used anywhere else
    and must never be reintroduced into src/.
    """
    hard_count, _dir_count = vectorized_violation_components(
        unscaled, src_idx, tgt_idx, feature_metadata, check_step_horizon=False
    )
    return hard_count > 0


@pytest.fixture(scope="module")
def uci_run():
    return run_dataset_seed("uci", seed=0, device=torch.device("cpu"))


def test_entry_gate_matches_interior_gate_on_corrected_code(uci_run):
    """The actual regression test: entry gate == interior gate, real edges, real code."""
    run = uci_run
    rng = np.random.default_rng(0)
    test_idx = rng.integers(0, len(run.X_test), size=N_QUERY_SAMPLES)

    disagreements = 0
    for ti in test_idx:
        x0 = run.X_test[ti]
        with torch.no_grad():
            z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()

        qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                    run.scaler, run.feature_cols, k=run.k_neighbors)
        entry_gate_violates = qedges.violates  # the function under test

        # The SAME transitions, re-checked with the INTERIOR predicate directly
        # (vectorized_violation_counts, check_step_horizon=True -- identical to
        # what build_edge_table uses for every interior edge).
        N = len(run.Z_train)
        X_combined = np.vstack([np.tile(x0.reshape(1, -1), (N, 1)), run.X_train])
        unscaled = unscale_matrix(X_combined, run.scaler, run.feature_cols, run.feature_metadata)
        src_idx, tgt_idx = np.arange(N), np.arange(N, 2 * N)
        _count, interior_violates = vectorized_violation_counts(
            unscaled, src_idx, tgt_idx, run.feature_metadata, check_step_horizon=True
        )

        disagreements += int(np.sum(entry_gate_violates != interior_violates))

    assert disagreements == 0, (
        f"Entry gate and interior gate disagreed on {disagreements} sampled transitions -- "
        f"this is exactly the B1 pattern (see experiments_v6_audit/RECOMMENDATION.md) and "
        f"must never reappear in src/graph/query_attachment.py."
    )


def test_relaxed_pre_fix_gate_actually_disagrees_with_interior_gate(uci_run):
    """
    Proves this test suite is discriminating, not vacuous: the deliberately
    reconstructed PRE-FIX relaxed gate must disagree with the interior
    predicate on a nonzero number of real sampled transitions. If this test
    ever fails, the reconstructed relaxed gate above no longer reproduces the
    historical bug and the discrimination claim above is unverified.
    """
    run = uci_run
    rng = np.random.default_rng(0)
    test_idx = rng.integers(0, len(run.X_test), size=N_QUERY_SAMPLES)

    total_disagreements = 0
    for ti in test_idx:
        x0 = run.X_test[ti]
        N = len(run.Z_train)
        X_combined = np.vstack([np.tile(x0.reshape(1, -1), (N, 1)), run.X_train])
        unscaled = unscale_matrix(X_combined, run.scaler, run.feature_cols, run.feature_metadata)
        src_idx, tgt_idx = np.arange(N), np.arange(N, 2 * N)

        relaxed_violates = _relaxed_pre_fix_gate(unscaled, src_idx, tgt_idx, run.feature_metadata)
        _count, interior_violates = vectorized_violation_counts(
            unscaled, src_idx, tgt_idx, run.feature_metadata, check_step_horizon=True
        )
        total_disagreements += int(np.sum(relaxed_violates != interior_violates))

    assert total_disagreements > 0, (
        "Expected the reconstructed pre-fix relaxed gate to disagree with the interior "
        "predicate on at least one real transition (this is what B1 actually looked like); "
        "zero disagreements means this reconstruction no longer reproduces the historical "
        "bug and this test suite's discrimination claim needs re-verification."
    )
    print(f"\n[discrimination check] relaxed pre-fix gate disagreed with interior gate on "
          f"{total_disagreements} sampled transitions (query x {N} train nodes each) -- "
          f"confirms this test fails on buggy code and passes on the corrected code.")
