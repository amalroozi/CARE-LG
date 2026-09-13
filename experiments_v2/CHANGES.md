# CHANGES.md — file-by-file explanation of every addition

**No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, or `data/` were modified.**
Everything new lives under `experiments_v2/`. The original pipeline (`benchmarks/run_benchmarks.py`,
`scripts/test_day2.py`, etc.) still runs exactly as before and reproduces the old committed
outputs (verified in `REPO_MAP.md` §3). No git commands beyond read-only `status`/`log`/`diff`/
`rev-parse` were run in this session; nothing was committed or pushed.

## New files

### `experiments_v2/REPO_MAP.md`
Phase 0 reconnaissance: where every component lives, what's reusable, and nine documented
discrepancies (D1–D9) between the brief, the paper's claimed numbers, and what the code
actually does — including that "NHANES" is a synthetic dataset (D1), that the original
pipeline never seeds torch (D4), and the reproduction check showing the paper's UCI 47.6%
headline does not reproduce under retraining (16-point swing).

### `experiments_v2/lib/seeding.py`
`set_all_seeds(seed)` — seeds `random`, `numpy`, and `torch` (CPU/CUDA/MPS). Called at the
start of every new (dataset, seed) run. The original repo never did this (D4); all new code
does.

### `experiments_v2/lib/riemannian_batch.py`
`batched_riemannian_distances(vae, z_i, z_j)` — a vectorized (`torch.func.vmap(jacrev(...))`)
reimplementation of `src/graph/riemannian.py::riemannian_distance`, verified numerically
identical (max abs error 5.96e-8 on a random sample) but ~2 orders of magnitude faster.
Pure performance optimization; `src/graph/riemannian.py` is untouched and still used by the
original pipeline.

### `experiments_v2/lib/constraints_ext.py`
- `count_violations(...)` — a granular integer count of violated sub-conditions, built by
  calling the exact same per-feature comparisons as `check_hard_violations`/
  `check_soft_violations` in `src/graph/clinical_constraints.py` (values and epsilons copied
  verbatim, not re-derived), used as `v(x_i,x_j)` for the soft-penalty edge weight.
- `decode_node`/`decode_nodes_batch` — thin wrappers around `vae.decode`.
- `check_decoded_violations(...)` — Phase 3's decoded-space check. Re-checks constraints on
  VAE-DECODED reconstructions rather than the original encoded features, using explicit,
  documented noise tolerances (module docstring): binary immutables (sex,
  fasting_blood_sugar) rounded to nearest class (tolerance 0.5) before comparing; age keeps
  the original 0.1-year / 3.05-year-horizon tolerances; directional features reuse the
  original per-feature epsilons already in the codebase.

### `experiments_v2/lib/graph_build.py`
The Phase 1 ablation engine.
- `unscale_matrix(X, scaler, ...)` — vectorized equivalent of calling
  `unscale_features()` on every row of `X`. **Important precision note**: sklearn's
  `StandardScaler.inverse_transform` PRESERVES input dtype (float32 stays float32); we
  replicate that exactly (verified bit-exact against `scaler.inverse_transform`), because
  naively upcasting to float64 shifts values by ~1e-7 and can flip a boundary comparison
  like `resting_bp` delta exactly at its 5.0 epsilon.
- `vectorized_violation_components(...)` / `vectorized_violation_counts(...)` — vectorized,
  numerically-verified (assertion against the scalar functions on a random sample at every
  edge-table build) reimplementation of `check_clinical_violations`, split into
  hard-vs-directional components so the query-attachment gate (below) can use a different
  rule than interior graph edges.
- `build_edge_table(...)` — builds the k-NN structure ONCE per (dataset, seed) and computes,
  for every edge, both the Riemannian and Euclidean distance and the violation count/flag.
  A grid cell (metric, constraints, lambda) is then a cheap reweighting
  (`make_cell_matrix`) of this cached table — the expensive part (k-NN + Jacobian) runs once
  per seed, not once per cell.
- `build_query_edges(...)` / `make_query_row(...)` / `augmented_matrix(...)` — attaches a
  held-out test patient to the graph as an ephemeral (N+1)-th source node. **This required a
  new design decision, documented in REPO_MAP.md D8**: the original `src/graph/calg.py`
  never handles external test patients at all; only the separate benchmark harness
  (`benchmarks/run_benchmarks.py`) does, with its own more lenient entry rule
  (`check_hard_violations`, `check_step_horizon=False`, searched across the WHOLE train set,
  not a local k-NN neighborhood). We adopted: (a) full attachment to every train node (not
  k-restricted — restricting to k nearest Z-neighbors left most patients with zero valid
  entries, collapsing hard-mode success to ~4% for reasons unrelated to the ablation); (b)
  the original harness's relaxed hard-only/no-horizon gate for the hard-mode PRUNE decision;
  (c) full hard+directional severity for the soft-mode penalty. This is the single most
  consequential design choice in this codebase addition — see REPO_MAP.md D8 for the full
  numeric justification (a seed-0 pilot went from 4% → 16% → 64% success across three
  iterations of this design before landing on the version described here).

### `experiments_v2/lib/search_v2.py`
`find_path_augmented(...)` — Dijkstra over the (N+1, N+1) augmented matrix, structurally
identical to `src/recourse/search.py::find_recourse_path` but aware of the ephemeral query
node.

### `experiments_v2/lib/pipeline.py`
`run_dataset_seed(dataset, seed)` — trains a fresh, seeded VAE + classifier per (dataset,
seed) using the UNMODIFIED `src.vae.model.train_vae` / `src.blackbox_model.train_blackbox_model`
(same hyperparameters as `benchmarks/run_benchmarks.py::run_all_benchmarks`: 20 classifier
epochs, 25 VAE epochs, k=40 UCI / k=80 NHANES, high-risk threshold 0.55, low-risk threshold
0.45), then encodes the full train/test sets.

### `experiments_v2/run_grid.py` (Phase 1+2+3)
Main orchestrator. For each (dataset × seed × grid cell) it trains models, builds the edge
table, reweights it for that cell, and runs recourse search for every high-risk test
patient, decoding every path node and re-checking constraints (Phase 3), computing KDE
(trajectory mean log-density), clinical effort (decoded endpoints, using the existing
`compute_clinical_effort` and the paper's existing `CLINICAL_EFFORT_WEIGHTS`), and latency
(`time.perf_counter`, one untimed warm-up query per (dataset, seed, cell) before the timed
per-patient loop). Writes `experiments_v2/results/{dataset}_per_patient.csv` (long format,
checkpointed after every seed) and `run_metadata.json` (git SHA, seeds, grid cells, device,
torch version, timestamp).
Grid: `{riemannian, euclidean} × {hard, soft(λ∈{0.1,1,10,100}), none}` = 12 cells.
Runs on CPU by default (see REPO_MAP.md D7: CPU is ~7× faster than MPS on this machine for
this workload; this affects wall-clock only, not any reported number, since latency is
timed per-cell/per-patient regardless of device).

### `experiments_v2/run_blindspot.py` (Phase 4)
For every R+hard (riemannian metric, hard constraints — the actual CARE-LG cell) abstention:
exhaustively checks feature-space feasibility of every low-risk train node
(`check_clinical_violations`, `check_step_horizon=False` — a single "is this pairing
feasible at all" test, not a bounded-time hop) to classify CERTIFIED INFEASIBLE vs.
BLINDSPOT CANDIDATE, then runs two densification probes on blindspot candidates: (a)
doubled intra-graph k, (b) +500 synthetic nodes sampled from the VAE prior `z~N(0,I)` and
decoded, eligible as terminal targets only if the classifier verifies decoded risk below
the low-risk threshold. Writes `{dataset}_blindspot.csv` (per-patient, only when
abstentions occurred) and `{dataset}_blindspot_seed_summary.csv`.

### `experiments_v2/run_baselines.py` (Phase 5)
Reuses the EXISTING, unmodified `run_dice`/`run_face`/`run_growing_spheres` from
`benchmarks/run_benchmarks.py` — not reimplemented — run under the same seeded pipeline and
same high-risk test patients as the main grid. Reports both `cvr_hard` (matching the
original repo's `compute_cvr` exactly) and `cvr_full` (hard+directional, for apples-to-apples
comparison with the grid cells' decoded CVR). **REVISE, named in the brief, is not
implemented anywhere in this codebase** (no `dice_ml`, no `carla`, nothing VAE-latent-
gradient-based beyond CARE-LG itself) and was not reimplemented from scratch — a faithful
implementation is well beyond the brief's own "<1 hour" threshold for absent baselines.
`GrowingSpheres` (already in the repo) substitutes for it in Table 1, clearly labeled.
R+soft(λ) from the main grid is used, per the brief, as the "emulation" of Pegios-et-al-style
Riemannian soft-constrained recourse — explicitly labeled as an emulation, not their code.

### `experiments_v2/aggregate.py`
Aggregates `{dataset}_per_patient.csv` and `{dataset}_baselines.csv` into
`table1_{dataset}.csv` (mean±std over seeds), `paired_tests.csv` (Wilcoxon signed-rank,
`scipy.stats.wilcoxon`, paired by `(seed, patient_idx)`, only on patients where BOTH
compared cells succeeded — RQ2: R+hard vs E+hard on KDE/effort; RQ1 supporting: R+hard vs
R+soft(λ=1) on KDE/effort), and `rq_summary.json` (also pulls 3 concrete decoded-CVR example
paths for the report).

### `experiments_v2/make_table1.py`
Builds the brief's exact Table 1 row/column spec (`table1_final_{dataset}.csv` +
`table1_{dataset}.tex`) from the aggregated files.

### `experiments_v2/make_figures.py` (Phase 6)
Figures A–E as PNG (300dpi) + PDF into `experiments_v2/figures/`, matplotlib only (no
seaborn).

### `experiments_v2/make_report.py` (Phase 7)
Assembles `experiments_v2/REPORT.pdf` via `matplotlib.backends.backend_pdf.PdfPages` (no
`pdflatex`/`latexmk` on PATH in this environment — verified in Phase 0). LaTeX table
fragments (`table1_{dataset}.tex`) are still emitted uncompiled, per the brief's fallback
instruction.

### `experiments_v2/repro_check/run_repro.py`
Phase 0's reproduction check: reruns the unmodified `benchmarks/run_benchmarks.py` pipeline
end-to-end, monkeypatching only its `PROJECT_ROOT` module variable so output paths (and
model-checkpoint paths, forcing FRESH training rather than reusing cached weights) are
redirected under `experiments_v2/repro_check/`, so `benchmarks/` and `models/` are never
touched. No logic in `benchmarks/run_benchmarks.py` is changed.

## Full command list to regenerate every number and figure from scratch

```bash
# Phase 0: reproduction check (writes only under experiments_v2/repro_check/)
.venv/bin/python experiments_v2/repro_check/run_repro.py

# Phase 1+2+3: main ablation grid, both datasets, 5 seeds, 12 cells each
.venv/bin/python -m experiments_v2.run_grid --datasets uci nhanes --seeds 0 1 2 3 4

# Phase 4: blindspot vs. true-infeasibility decomposition
.venv/bin/python -m experiments_v2.run_blindspot --datasets uci nhanes --seeds 0 1 2 3 4

# Phase 5: baselines (DiCE / FACE / GrowingSpheres), same seeds & patients as the grid
.venv/bin/python -m experiments_v2.run_baselines --datasets uci nhanes --seeds 0 1 2 3 4

# Aggregate: Table 1 CSVs, paired Wilcoxon tests, RQ summary JSON
.venv/bin/python -m experiments_v2.aggregate

# Final Table 1 (brief's exact row/column spec) as CSV + LaTeX
.venv/bin/python -m experiments_v2.make_table1

# Phase 6: figures A-E (PNG + PDF)
.venv/bin/python -m experiments_v2.make_figures

# Phase 7: assemble the PDF report
.venv/bin/python -m experiments_v2.make_report
```

Total wall-clock for the full grid + blindspot + baselines on this machine (Apple M2, CPU):
under 2 minutes combined (see `experiments_v2/results/run_grid_log.txt` and
`run_baselines_log.txt` for per-cell/per-seed timings actually observed in this session).
The 5-seed grid was run in full for both datasets — no seed count reduction was needed
(brief's Phase 2 fallback instruction to run fewer seeds under budget pressure did not
apply here).

## Interpretation choices flagged for the record (beyond D1–D9 in REPO_MAP.md)

- **Soft-penalty severity `v(x_i,x_j)`** is an unweighted count of violated sub-conditions
  (immutable changes, age-decrease, age-horizon, each directional breach), not a
  magnitude-weighted severity score. The brief says "counts/weights constraint violations"
  without specifying which; we chose the count, since a magnitude-weighted score would
  require inventing a new, unjustified per-feature scale that isn't in the existing code.
- **KDE of "the trajectory"** (Phase 2) is computed as the mean KDE log-density of ALL
  decoded path nodes (source through terminal), not just the terminal point (which is what
  the original `benchmarks/run_benchmarks.py` computes, pooled across patients). This is a
  literal reading of "KDE log-likelihood of the trajectory."
- **Clinical effort** uses the paper's existing `CLINICAL_EFFORT_WEIGHTS` (per
  `configs/dataset_config.py`) and the existing `compute_clinical_effort` function, applied
  to VAE-DECODED source/terminal states (per Phase 3's framing that decoded reconstructions
  are what a patient would actually be shown), not the original scaled features.
