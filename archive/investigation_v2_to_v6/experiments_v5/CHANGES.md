# CHANGES.md — experiments_v5 file-by-file explanation

No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, `data/`, `experiments_v2/`,
`experiments_v3/`, or `experiments_v4/` were modified. Everything new lives under
`experiments_v5/`. DiCE/FACE/REVISE numbers are REUSED read-only from
`experiments_v3/results/`; DSR-full's best operating point is REUSED read-only from
`experiments_v4/results/` — both clearly labeled "REUSED... NOT rerun" in
`table1_wide_{dataset}.csv` and every report page that cites them. No git commands
beyond read-only `status`/`log`/`diff`/`rev-parse` were run; nothing was committed.

## Phase 0

- **`PRIOR_ART_COUNTERFLOWNET.md`** — close read (arXiv HTML, not just abstracts) of
  CounterFlowNet (2602.17244), FCEGAN (2502.17613), FastDCFlow (2404.13224).
- **`NOVELTY_CLAIM.md`** — the corrected, checked novelty claim, written BEFORE Phase 1
  began, per the brief's explicit checkpoint requirement.

## `experiments_v5/lib/`

Copied unchanged (import paths fixed): `seeding.py`, `riemannian_batch.py` (kept for
reference; NOT used by EGD's own Riemannian computation, which needs a two-argument
decoder — see `egd_graph.py`), `constraints_ext.py`, `search_v2.py`,
`dataset_registration.py`, `graph_build.py` (reused for the "old_latent_pruning"
controlled-comparison cell's ORIGINAL v3 hard-pruning logic), `vae_v3.py` (the
type-aware decoder, trained fresh alongside EGD for that same comparison cell).

New:

- **`egd_decoder.py`** — `ExactGuaranteeDecoder`, `train_egd`. See `METHOD.md` §1, §4
  for the architecture and the training-regime dead end (self-reconstruction, then
  k-NN pairing, both tried and rejected empirically) found and documented before
  settling on random-pairing.
- **`egd_graph.py`** — `batched_riemannian_distances_egd` (vmap+jacrev adapted for a
  decoder that takes TWO arguments, `z` and `x_source`, differentiating only w.r.t.
  `z`), edge table construction (NO pruning — EGD needs none), query attachment.
- **`search_egd.py`** — `find_recourse_egd`: ranks candidates by latent distance,
  sequentially decodes each, returns the first that passes the (not architecturally
  guaranteed) risk check; abstains with `abstain_stage` distinguishing `'no_path'` from
  `'risk_check'`.
- **`pipeline_v5.py`** — per-(dataset, seed) setup. Plain 80/20 split (no calibration
  carve-out needed — EGD calibrates nothing). Trains BOTH `EGD` and the type-aware
  `old_vae` on the SAME split/seed/classifier, for a controlled "old vs. EGD"
  comparison (mirroring `experiments_v4`'s practice of re-running the old method
  inside the new pipeline rather than quoting a prior session's numbers for that
  specific comparison).

## Top-level scripts

- **`run_egd.py`** — main runner. `run_egd_cell` (EGD, both metric ablations) +
  `run_old_cell` (freshly-trained old method, same pipeline). Writes
  `{dataset}_per_patient.csv`.
- **`run_blindspot_egd.py`** — Phase 3, certified-infeasibility decomposition on EGD's
  abstentions. **The same performance bug already found and fixed in
  `experiments_v4/run_blindspot_dsr.py` was reintroduced and re-fixed here**: an
  initial version rebuilt both densification probe graphs per abstaining patient
  rather than once per (dataset, seed); on real NHANES (abstention rate 97-100%),
  this made a single seed take >70s with no sign of finishing. Fixed identically to
  the v4 precedent: `build_densified_probes` runs once per seed, reused across every
  abstaining patient (`densify_and_retry`); this cut wall-clock to ~11-15s/seed. A
  synthetic densification node has no real source identity — decoded relative to an
  all-zero (population-mean, since features are standardized) reference, the only
  generic reference point available (documented in the module).
  **A second, explicit, reported reduction**: given real NHANES's abstention rate is
  97-100% (essentially every high-risk patient), the exhaustive-feasibility +
  densification analysis is capped at a random (seeded) sample of
  `--max-abstentions-per-seed 60` per (dataset, seed); the TRUE abstention rate is
  always reported in full (`n_abstain_true`, `pct_abstain` in the summary CSV), only
  the certified/blindspot/unresolved split is sampled, and every summary row records
  `was_subsampled`.
- **`aggregate.py`** — `table1_{dataset}.csv` (EGD/old, this session's own numbers,
  mean±std over 5 seeds), `table1_wide_{dataset}.csv` (adds v3's DiCE/FACE/REVISE and
  v4's DSR-full-at-its-best-tau, explicitly labeled as reused), `paired_tests.csv`
  (EGD-riemannian vs. EGD-euclidean; EGD vs. old, wherever enough paired successes
  exist for a real test).
- **`make_figures.py`** — Figures 1-5, matplotlib only, PNG(300dpi)+PDF.

## A note on `nhanes_real_blindspot_summary.csv`

`run_blindspot_egd.py` accumulates `summary_rows` across the WHOLE run (all datasets),
writing the CUMULATIVE list to `{dataset}_blindspot_summary.csv` after each dataset's
seeds finish — so the file written last (`nhanes_real_...`) contains BOTH UCI's AND
real NHANES's rows (each correctly labeled by its own `dataset` column; no computation
is wrong). `make_figures.py` and this document's own reading of the file filter
explicitly on `dataset==<name>` rather than trusting the filename alone — noted here so
a reader of the raw CSV isn't confused by row counts that don't match the filename.

## Full command list to regenerate every number and figure from scratch

```bash
# Phase 0 (research only, no code to run -- see PRIOR_ART_COUNTERFLOWNET.md / NOVELTY_CLAIM.md)

# Phase 2+4: main EGD vs. old comparison, both metrics, both cohorts, 5 seeds
.venv/bin/python -m experiments_v5.run_egd --datasets uci nhanes_real --seeds 0 1 2 3 4

# Phase 3: certified-infeasibility decomposition on EGD's abstentions
.venv/bin/python -m experiments_v5.run_blindspot_egd --datasets uci nhanes_real --seeds 0 1 2 3 4

# Aggregate: Table 1 (this session), wide comparison table (+ v3/v4 reused numbers), paired tests
.venv/bin/python -m experiments_v5.aggregate

# Figures 1-5 (PNG + PDF)
.venv/bin/python -m experiments_v5.make_figures

# Assemble the PDF report
.venv/bin/python -m experiments_v5.make_report
```

## Interpretation choices flagged for the record

- **CounterFlowNet reimplementation was judged out of scope** given the session budget
  already spent on Phase 0's close read and Phase 1's genuine training-regime debugging
  (self-reconstruction → k-NN pairing → random pairing, each requiring real diagnosis,
  not a quick swap). Per the brief's own fallback instruction, this is stated plainly:
  any CounterFlowNet comparison in this report is qualitative (from their paper's
  numbers on financial datasets, not a controlled rerun on this project's cohorts), not
  a controlled head-to-head.
- **The "old_latent_pruning" comparison cell is freshly trained in this session's own
  pipeline**, not copied from `experiments_v3`/`v4`'s published numbers, so it is
  directly, validly paired with EGD's own results (same seed, same test patients, same
  classifier) for the Wilcoxon tests in `paired_tests.csv`. The WIDE table's DiCE/FACE/
  REVISE/DSR-full rows are NOT from this session's runs and are not validly paired with
  anything in this session — they are context, not a controlled comparison, and are
  labeled as such in every table that includes them.
- **`egd_euclidean` is a genuine ablation cell**, run with the full protocol (not a
  quick spot-check) precisely because its dramatic difference from `egd_riemannian`
  (0% vs. 31%/1% success) was unexpected and needed a real, verified mechanistic
  explanation (path-length distribution, `METHOD.md`/`FINDINGS.md`) before being
  reported as a finding rather than dismissed as noise.
