# CHANGES.md — experiments_v4 file-by-file explanation

No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, `data/`, `experiments_v2/`,
or `experiments_v3/` were modified. Everything new lives under `experiments_v4/`. The
real NHANES cohort (`experiments_v3/data/nhanes_real.csv`) is REUSED read-only — v4 does
not rebuild it. No git commands beyond read-only `status`/`log`/`diff`/`rev-parse` were
run; nothing was committed or pushed.

## `experiments_v4/lib/` — reused vs. new

Copied UNCHANGED (import paths fixed only) from `experiments_v3/lib/`: `seeding.py`,
`riemannian_batch.py`, `constraints_ext.py`, `search_v2.py`, `vae_v3.py` (the type-aware
decoder), `graph_build.py` (kept for its `unscale_matrix` helper, reused by
`dsr_rules.py`), `dataset_registration.py` (repointed at
`experiments_v3/data/nhanes_real.csv` rather than a v4-local copy).

New:
- **`dsr_rules.py`** — the constraint predicate in all four modes (`raw`,
  `decoded_naive`, `decoded_banded`, `banded_no_identity`); METHOD.md §1-2.
- **`calibration.py`** — Component 2's conformal tolerance-band derivation; METHOD.md §2a.
- **`pipeline_v4.py`** — the 3-way train/calibration/test split and decoder-choice
  ('v2'/'v3') pipeline; METHOD.md §4.
- **`dsr_graph.py`** — edge-table construction with the one-shot decoded-node cache
  (Component 1's O(N) cost note) and query attachment under any mode.
- **`repair.py`** — Component 3's verify-and-repair loop; METHOD.md §3.

## Top-level scripts

- **`run_dsr.py`** — main runner: E1 (component comparison), E2 (`--tau-sweep`), E4
  (Riemannian vs Euclidean cells), E5 groundwork (`--decoders`). `--output-suffix` avoids
  collisions between the main run and the E5/E4-follow-up runs sharing the same result
  directory.
- **`run_e5_decoder_dependence.py`** — thin wrapper: calls `run_dsr.main()` with
  `--decoders v2 --output-suffix _v2decoder`, main cells only (no tau-sweep — E5 only
  needs the q=0.95 operating-point comparison).
- **`run_e4_best_tau.py`** — E1/E2's tau-sweep only varied the Riemannian `dsr_full`
  cell. This runs `dsr_full_euclidean` at the SAME tau quantile where Riemannian
  achieved its best success rate (from `tau_sweep_{dataset}.csv`), so E4 has a
  meaningfully powered paired test rather than only the near-zero-success comparison at
  q=0.95. On UCI, `dsr_full` has 0% success at every quantile tested, so this script
  reports that no powered E4 test is possible there rather than picking an arbitrary
  level.
- **`run_blindspot_dsr.py`** — E3: re-runs `experiments_v3`'s exact certified-
  infeasibility methodology (exhaustive feature-space check + two densification probes:
  doubled k, +500 classifier-verified VAE-prior synthetic nodes) on DSR's abstentions.
  **Performance bug found and fixed during this session**: the first implementation
  rebuilt both densification probe graphs (a fresh k-NN structure + full batched
  Riemannian distance computation) **once per abstaining patient**, even though the
  densified graph does not depend on which patient is being tested. On real NHANES,
  where `dsr_full` abstains on ~100% of ~400-445 high-risk patients per seed, this made
  the analysis intractable (>2 minutes with no single seed/cell completing). Fixed by
  building both probe graphs ONCE per (dataset, cell, seed) — `build_densified_probes` —
  and reusing them across every abstaining patient in that seed
  (`densify_and_retry(..., table_2k, fr_synth, table_synth, ...)`); this cut the
  per-seed/cell wall-clock from >120s-unresolved to ~5-8s.
  **A second, explicit, reported reduction**: even after that fix, `dsr_full`'s abstention
  count on real NHANES is ~400-445/seed; the exhaustive-feasibility + densification
  analysis is capped at a random (seeded) sample of `--max-abstentions-per-seed 25` per
  (dataset, cell, seed) — the *true* abstention count and rate are still recorded and
  reported (`n_abstain_true`, `pct_abstain` in `*_blindspot_summary.csv`), only the
  detailed certified/blindspot/unresolved decomposition is computed on the capped
  sample, and every summary row records `was_subsampled=True` where this applies.
- **`aggregate.py`** — builds `table1_{dataset}.csv`, `tau_sweep_{dataset}.csv`,
  `constraint_breakdown.csv`, `paired_tests.csv`, `decoder_dependence.csv`. Runs the
  old-vs-`dsr_full` and Riemannian-vs-Euclidean Wilcoxon tests BOTH at the nominal q=0.95
  operating point (where `dsr_full` has near-zero successes on both datasets — reported
  as a real 0-pairs result, not hidden) AND at `dsr_full`'s best-powered tau from the
  sweep (where a real paired comparison is possible).
- **`make_figures.py`** — Figures 1-6, matplotlib only, PNG(300dpi)+PDF.

## A design decision applied uniformly, restated from METHOD.md §2c

Component 2's bands, applied as a hard pre-filter (`dsr_c1c2`), collapse the graph to
near-zero edges at every tau level tested, because a small number of continuous
features have decoder reconstruction error exceeding their original epsilon tolerance
even at the loosest calibration quantile (`tau_vs_eps_diagnostic.csv`). `dsr_full` uses
Component 1 for graph connectivity and Component 2's bands as the Component-3
verify-and-repair standard instead of a pre-filter. `dsr_c1c2` is kept as its own
Table-1/Fig-2 row specifically so this discrepancy and its resolution are visible
results, not a silently-avoided literal reading.

## Metric comparability

Decoded-space CVR is computed with `experiments_v3`'s ORIGINAL, UNCHANGED checker
(`constraints_ext.py::check_decoded_violations`) for every cell including the old
method — the method under test changes; the yardstick it is measured against does not.
`Fig 4`'s per-constraint-type breakdown uses a DIFFERENT, complementary computation
(hop-wise, OR'd across every step) from Table 1's headline `decoded_cvr_endpoint`
(source-vs-terminal only) — the two can legitimately disagree on multi-hop paths
(cumulative small per-hop drift can fail an endpoint check that no individual hop
failed); this distinction is stated in `aggregate.py`'s `build_constraint_breakdown`
docstring and in `FINDINGS.md`, not conflated.

## Full command list to regenerate every number and figure from scratch

```bash
# E1 (component comparison) + E2 (tolerance sweep) + E4 (Riemannian cells): main run
.venv/bin/python -m experiments_v4.run_dsr --datasets uci nhanes_real --seeds 0 1 2 3 4 \
    --decoders v3 --tau-sweep

# E5: decoder-dependence check (v2 decoder, main cells only, separate output files)
.venv/bin/python -m experiments_v4.run_e5_decoder_dependence --datasets uci nhanes_real --seeds 0 1 2 3 4

# E4 follow-up: Euclidean at dsr_full's best-powered tau (needs tau_sweep_*.csv from the main run)
.venv/bin/python -m experiments_v4.run_e4_best_tau --datasets nhanes_real --seeds 0 1 2 3 4

# E3: certified-infeasibility / blindspot decomposition under DSR abstentions
.venv/bin/python -m experiments_v4.run_blindspot_dsr --datasets uci nhanes_real --seeds 0 1 2 3 4 \
    --cells dsr_c1 dsr_full

# Aggregate: Table 1, tau-sweep table, constraint breakdown, paired Wilcoxon tests, E5 table
.venv/bin/python -m experiments_v4.aggregate

# Figures 1-6 (PNG + PDF)
.venv/bin/python -m experiments_v4.make_figures

# Assemble the PDF report
.venv/bin/python -m experiments_v4.make_report
```

## Hyperparameters (all in code, restated here for one-stop reference)

- Calibration split: 15% of the original 80% train partition (`CALIB_FRAC = 0.15`,
  `pipeline_v4.py`), stratified, seeded per-seed.
- Tau quantiles swept (E2): 0.05, 0.10, 0.25, 0.50, 0.80, 0.90, 0.95, 0.99
  (`run_dsr.py::TAU_LEVELS`); 0.95 is the nominal "safe-sounding" operating point Table 1
  reports at.
- Repair: up to 3 project→re-encode→re-verify iterations per violating step
  (`repair.py::verify_and_repair(max_iters=3)`); risk-monotonicity tolerance 1e-3.
- Densification probes (E3): 2x intra-graph k; +500 VAE-prior synthetic nodes,
  classifier-verified low-risk (`run_blindspot_dsr.py`), identical to `experiments_v3`.
- k-NN neighbors: 40 (UCI) / 80 (real NHANES), unchanged from v2/v3.
- Everything else (VAE architecture/epochs, classifier architecture/epochs, high/low-risk
  thresholds 0.55/0.45) is unchanged from `experiments_v3`.

## Interpretation choices flagged for the record (beyond §2c above)

- **Fig 2 / Fig 4's "n/a" cells** (e.g. `dsr_c1c2` on UCI has 0 successes) are reported
  as literal zeros/blank, not imputed or estimated.
- **E5's v2-decoder run does not re-run the tau-sweep** — E5's question ("does DSR need a
  good decoder") is answered by the main-cell comparison at the standard operating
  point; a full second sweep under v2 was judged out of scope for the marginal
  additional evidence it would provide, given the session budget.
- **The blindspot-under-DSR subsampling (`--max-abstentions-per-seed 25`)** is the one
  place in this session where the brief's "reduce and say so explicitly" fallback
  instruction was invoked — done here, not silently.
