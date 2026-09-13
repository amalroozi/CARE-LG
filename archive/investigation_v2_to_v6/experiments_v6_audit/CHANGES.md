# CHANGES.md — experiments_v6_audit

No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, `data/`, `experiments_v2/`,
`experiments_v3/`, `experiments_v4/`, or `experiments_v5/` were modified. Everything new
lives under `experiments_v6_audit/`. No git commands beyond read-only
`status`/`log`/`diff`/`rev-parse` were run; nothing was committed or pushed.

## Pre-existing repo state noted, not caused by this session

At session start, `git status` showed `benchmarks/uci/results.csv` and `results.json`
as modified relative to HEAD (GrowingSpheres' row and DiCE/FACE latency changed; the
CARE-LG row is unchanged). This is consistent with an unseeded rerun of
`benchmarks/run_benchmarks.py` or `scripts/generate_paper_artifacts.py` at some point
before this session (GrowingSpheres' `np.random.randn` call is documented as unseeded
in `experiments_v2/REPO_MAP.md` D4). This audit did not cause it, did not investigate
its origin further (out of scope), and did not revert it (read-only git only, per this
session's hard rules).

## Files

- **`PROJECT_TIMELINE.md`** — Phase 1 deliverable: terse, dated reconstruction of what
  each of v2-v5 changed, found, and left unresolved.
- **`verify_rootcause_b.py`** — first-pass verification: reruns v3's exact R+hard cell
  (same pipeline, seeds, patients) with decoded-space CVR computed two ways side by
  side (decode every node vs. real recorded values for real nodes). Revealed the
  unexpected result that motivated the deeper investigation below (real-value CVR was
  HIGHER than decoded CVR, not lower — the opposite of the original hypothesis).
- **`verify_entry_gate_fix.py`** — the main verification script. Reimplements
  `experiments_v3/lib/graph_build.py::build_query_edges` with a `strict_gate` flag
  toggling between the original relaxed gate (`hard_count > 0`,
  `check_step_horizon=False`) and the fixed strict gate
  (`(hard_count + dir_count) > 0`, `check_step_horizon=True`, matching interior
  edges exactly), and reports decoded-CVR and real-value-CVR under both, for the R+hard
  cell, both datasets, 5 seeds. This is the evidence base for `RECOMMENDATION.md`.
- **`results/uci_rootcause_b_verification.csv`** — raw per-patient output of the
  first-pass script (UCI, seed 0 only — the unexpected direction of this result is what
  motivated the deeper entry-gate investigation below, so the script was never run on
  real NHANES; superseded by the entry-gate-fix results for all final numbers, both
  datasets).
- **`results/uci_entry_gate_fix.csv`**, **`results/nhanes_real_entry_gate_fix.csv`** —
  raw per-patient output of the main verification, both gates, all 5 seeds, both
  datasets. This is the primary evidence artifact.
- **`results/entry_gate_fix_log.txt`** — full console log of the main verification run.

## Ad hoc diagnostics run inline (not saved as standalone scripts, output preserved in this document / RECOMMENDATION.md)

- Direct inspection of `src/graph/calg.py` and `src/recourse/search.py` confirming the
  ORIGINAL pipeline's `decode_recourse_trajectory` never calls `vae_model.decode()`
  despite accepting it as a parameter (uses `scaled_features[node_idx]` directly).
- `grep` across `experiments_v2/run_grid.py`, `experiments_v3/run_grid.py`,
  `experiments_v4/lib/dsr_graph.py`, `experiments_v4/run_dsr.py`,
  `experiments_v5/run_egd.py`, `experiments_v5/lib/egd_graph.py`,
  `experiments_v5/lib/search_egd.py` confirming all four sessions decode real graph
  nodes rather than looking up recorded values.
- Concrete single-patient trace (UCI seed 0, patient 0, path `[237, 114]`): real
  cholesterol delta +35.0 mg/dL (231→266) vs. decoded delta +0.3 mg/dL (≈239.5→≈239.7),
  and confirmation this edge was never pruned (`qedges.violates[pos]=False`) — the
  concrete evidence underlying `RECOMMENDATION.md`'s B1/B2 sections.
- Entry-hop vs. interior-hop violation-rate decomposition (UCI seed 0, all 25 high-risk
  patients with a successful path): entry-hop violation rate 68.8% (11/16, all
  `directional`), interior-hop violation rate 0.0% (0/1 paths with an interior hop) —
  confirmed the entry gate, not interior-edge pruning, as the locus of the defect.
- Risk-monotonicity check (Root Cause D): both cohorts, seed 0, original relaxed gate —
  13/445 high-risk patients had a multi-hop successful path at all; 1/13 showed a
  genuine intermediate-risk-increase violation.
- Posterior-collapse diagnostic (Root Cause C): per-latent-dimension mean KL divergence
  and std(mu) on the full training set, both cohorts, seed 0 — UCI dim 1 KL=0.0063
  (collapsed); real NHANES shows no comparable collapse (all dims KL 0.23-1.33).

## Full command list to regenerate the audit's evidence from scratch

```bash
# First-pass verification (side-by-side decoded vs. real-value CVR, original gate only)
.venv/bin/python -m experiments_v6_audit.verify_rootcause_b --datasets uci nhanes_real --seeds 0

# Main verification: original vs. fixed entry gate, both CVR measures, 5 seeds, both cohorts
.venv/bin/python -m experiments_v6_audit.verify_entry_gate_fix --datasets uci nhanes_real --seeds 0 1 2 3 4
```

Total wall-clock for the main verification: ~2 minutes (UCI: ~1.5s/seed; real NHANES:
~12-14s/seed), per `results/entry_gate_fix_log.txt`'s own timing.
