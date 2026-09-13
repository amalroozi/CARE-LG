# CHANGES.md — experiments_v6

No files under `src/`, `benchmarks/`, `scripts/`, `configs/`, `data/`, `experiments_v2/`,
`experiments_v3/`, `experiments_v4/`, `experiments_v5/`, or `experiments_v6_audit/` were
modified. Everything new lives under `experiments_v6/`. No git commands beyond read-only
`status`/`log`/`diff`/`rev-parse` were run; nothing was committed, staged, or pushed.
Repo HEAD throughout this session: `539d87b795b78bb8f9446f99b0fabeded2c84841` (unchanged).

## Phase 1 — corrected foundation

### Step 4: independent B1-pattern check on v4 and v5's own query-attachment code

`experiments_v6_audit/RECOMMENDATION.md` explicitly flagged this as read-but-not-
independently-verified. Verified directly this session by reading the actual code
(not from memory):

- **`experiments_v4/lib/dsr_graph.py`** (`prune_for_mode`, interior edges, line 89:
  `check_step_horizon=True`; `build_dsr_query_edges`, query entry, line 144:
  `check_step_horizon=False`) — both call the same `violation_breakdown` function
  in `experiments_v4/lib/dsr_rules.py`. Reading that function (lines 100-200) shows
  directional violations (`f_dir`, lines 178-190) are computed **unconditionally**,
  regardless of `check_step_horizon` — only the age-horizon single-step cap
  (`f_horizon`, line 173) is gated by that flag. **v4 does NOT have the B1 bug.**
  Its entry/interior asymmetry is real but narrow and defensible: it skips only the
  per-step age-horizon cap on the entry hop (which v4's own docstring at
  `dsr_graph.py:116-124` argues is semantically different for an entry hop — "which
  existing case most resembles this patient" is not a bounded time step — versus
  directional/immutability checks, which it always applies). No fix needed; documented
  here per the brief's "document precisely what was found either way" instruction.
- **`experiments_v5/lib/egd_graph.py`** — grepped for
  `gate_violates|violates|check_clinical|hard_count|dir_count`: **zero matches**.
  v5's EGD has no constraint-based edge-pruning/gating predicate at all — its
  guarantee is architectural (masked-identity + signed-residual decoder heads that
  make violations structurally impossible to construct), not a checkable gate
  function that could have an entry/interior asymmetry. **B1's bug pattern does not
  apply to v5's architecture** — there is no gate to compare.

Conclusion: neither v4 nor v5 needed the B1 fix applied. Both are left completely
unmodified. Per the brief, neither is carried forward as an active component in
`experiments_v6` regardless (per `RECOMMENDATION.md`'s conclusion) — their code is
kept for reference only.

### Steps 1-3, 5-6: corrected baseline pipeline

`experiments_v6/lib/` is a copy of `experiments_v3/lib/` (`seeding.py`,
`riemannian_batch.py`, `constraints_ext.py`, `graph_build.py`, `search_v2.py`,
`vae_v3.py`, `dataset_registration.py`, `pipeline_v3.py` → `pipeline_v6.py`) with
import paths rewritten from `experiments_v3.lib` to `experiments_v6.lib`, and TWO
functional changes:

**Change 1 — B1 fix**, in `experiments_v6/lib/graph_build.py::build_query_edges`:
the query's entry-edge gate now uses `gate_violates = (hard_count + dir_count) > 0`
with `check_step_horizon=True` — the exact same full predicate used for every
interior graph edge (`check_clinical_violations` in `src/graph/calg.py`) — replacing
v2/v3's `gate_violates = hard_count > 0`, `check_step_horizon=False` (hard-only,
no directional check, no per-step age cap). This is applied unconditionally; there
is no toggle back to the old relaxed gate anywhere in `experiments_v6`.

**Change 2 — B2 fix**, in `experiments_v6/run_grid.py::evaluate_patient`: path
states for real graph nodes (the query itself, and every train-set node visited)
are read directly from `run.X_test`/`run.X_train` — never decoded through the VAE
— matching `src/recourse/search.py::decode_recourse_trajectory`, the ORIGINAL
pipeline's own patient-facing function. This is the metric now used for
success/failure-adjacent reporting (`real_cvr_hop`, `real_cvr_endpoint`,
`real_violation_categories`, `effort`, `kde`). The legacy decode-every-node
computation (`decoded_cvr_hop`, `decoded_cvr_endpoint`, `decoded_violation_categories`)
is ALSO computed and written to the same CSV, purely for continuity with v2-v5's
historical headline metric — it plays no role in any pass/fail determination in
this script. No synthetic/interpolated nodes appear in the Phase 1 baseline grid
(that is introduced only by Phase 2's densification probes), so B2's "decode only
genuinely synthetic nodes" clause has nothing to decode in this script; the
decode-only-when-synthetic logic will be exercised for real in Phase 2.

Smoke test (`--datasets uci --seeds 0 --smoke`, 5 patients, first 3 grid cells)
confirmed the expected qualitative signature before committing to the full run:
`riemannian_hard` (the fully constrained cell) → `real_cvr=0.0%`; soft-penalty
cells (violations allowed at a cost, not pruned) → nonzero `real_cvr`, exactly as
`experiments_v6_audit` measured. Full 5-seed, both-dataset numbers are in
`FINDINGS.md` once the full grid run completes (see `results/run_grid_log.txt` for
the raw console log and `results/run_metadata.json` for the logged git SHA / seeds
/ config of that run).

### Full 5-seed grid: confirmed real CVR

Full run (`--datasets uci nhanes_real --seeds 0 1 2 3 4`, `experiments_v6/run_grid.py`,
git SHA `539d87b795b78bb8f9446f99b0fabeded2c84841`) confirms, for the R+hard cell
(`riemannian_hard`/`euclidean_hard`, the actual CARE-LG operating point):

| dataset      | cell             | success (mean±std over 5 seeds) | real_cvr | decoded_cvr (legacy) |
|--------------|------------------|----------------------------------|----------|------------------------|
| uci          | riemannian_hard  | 38.6% ± 13.1%                    | **0.00% ± 0.00%** | 47.3% ± 10.3% |
| uci          | euclidean_hard   | 38.6% ± 13.1%                    | **0.00% ± 0.00%** | 62.1% ± 11.5% |
| nhanes_real  | riemannian_hard  | 37.7% ± 3.4%                     | **0.00% ± 0.00%** | 65.4% ± 7.9%  |
| nhanes_real  | euclidean_hard   | 37.7% ± 3.4%                     | **0.00% ± 0.00%** | 62.8% ± 6.9%  |

`real_cvr` is exactly 0.00% ± 0.00% on every seed of both cohorts, as predicted.
The `decoded_cvr` column is the legacy metric v2-v5 reported as their headline
finding; it is nonzero purely because it re-decodes real, already-known graph
nodes and measures VAE reconstruction noise, not a real constraint violation
(this was the audit's central finding, now reconfirmed on the FULL grid rather
than the audit's spot-check). Full per-patient rows: `results/uci_per_patient.csv`,
`results/nhanes_real_per_patient.csv`. Soft-penalty and unconstrained cells show
nonzero real_cvr as expected (violations are allowed there at a cost, not pruned) --
see those files for the complete cell-by-cell breakdown.

### `nhanes_real.csv` data path

`experiments_v6/lib/dataset_registration.py` reads the real-NHANES CSV from
`experiments_v3/data/nhanes_real.csv` directly (read-only reference to the v3-built
cohort file — the file itself is not copied or modified) rather than expecting a copy
under `experiments_v6/data/`, since rebuilding that cohort is out of this session's
scope and the brief does not ask for it.

## Phase 2 — certified infeasibility vs. blindspot

Ported `experiments_v3/run_blindspot.py`'s protocol unchanged onto `experiments_v6/lib`
(the only functional difference is the B1-fixed entry gate it now inherits
automatically through `graph_build.py`). See `experiments_v6/run_blindspot.py`'s
module docstring for the clarification that this protocol was ALREADY unifying
"no entry edge" and "no interior path" failure modes even before this session:
`find_path_augmented` runs one Dijkstra call over the whole augmented graph and
has no code path that distinguishes the two -- both surface identically as "no
finite distance from the query to any target." Full results in `FINDINGS.md`;
raw data in `results/{dataset}_blindspot.csv` and `_blindspot_seed_summary.csv`.

## Phase 3 — literature diff (time-boxed, ~45 min spent)

Read `arxiv.org/html/2302.11213` (Diverse Interpolation)'s actionability-graph
section directly (HTML full text, not just the abstract this time -- the PDF
extraction that failed in `experiments_v6_audit/SEARCH_FINDINGS.md` was
retried via arXiv's HTML rendering, which worked). Confirmed, quoting the
paper directly:
- Nodes = training samples; edges = "the feasible action between two
  instances," gated by "cost threshold constraints as well as immutable
  feature constraints" -- structurally the same shape as this project's CALG
  (k-NN graph + immutable/directional gating), not a different construction.
- Diversity is handled at PROTOTYPE SELECTION (their DPP/QP step, upstream of
  pathfinding), not at the graph or path level -- "the paths suggested by our
  method are more likely to be diverse... [than FACE's]" because the
  prototypes themselves were chosen to be diverse, not because of a different
  edge or path-search mechanism.
- Pathfinding is plain shortest-path (their own words: "the shortest paths
  connecting x0 to {xk}"), same as this project's Dijkstra.
- **The zero-feasible-path case is not discussed anywhere in the paper.**

Conclusion: no small, cheap, clearly-beneficial change to adopt. The paper's
edge/path construction is architecturally equivalent to what this project
already does; its one genuinely different idea (diversity-aware prototype
selection) addresses a different problem (choosing among multiple reachable
targets) than this project's actual gap (zero reachable targets for a
majority of R+hard abstentions, per Phase 2). This matches
`SEARCH_FINDINGS.md`'s prediction exactly. No code changed as a result of
this phase.

## Phase 4 — neurosymbolic guideline layer

Built `experiments_v6/lib/neurosymbolic.py` using `clingo` (installed this
session, `pip install clingo shap`, both installed cleanly with no compiler
step). Two kinds of rules, kept deliberately separate (see that file's module
docstring for the full citation list -- [PCE2013], [HTN2017], [ADA2023],
[CHOL2018]):
- **Admissibility rules (`R-*`)**: a rule-for-rule ASP re-encoding of
  `src/graph/clinical_constraints.py::check_clinical_violations` (immutable /
  non-decreasing / age-horizon / directional-reduce / directional-increase),
  each returning which named rule (with citation) blocked a transition.
- **Guideline annotations (`G-*`)**: non-blocking clinical facts (statin
  indication at the 7.5% 10-yr ASCVD/PCE threshold [PCE2013], consistent with
  `experiments_v3/DATA_PROVENANCE.md`'s own label definition; BP treatment
  targets differentiated by diabetes status, <130/80 vs. <140/90 [HTN2017])
  used only for patient-facing explanation (Phase 5), never for edge gating.

**Regression check** (`experiments_v6/run_neurosymbolic_regression.py`): 4,000
sampled (source, target) pairs (2,000 per dataset -- half query-entry
transitions, half interior transitions, matching what the corrected graph
actually evaluates), comparing the symbolic reasoner's admissibility decision
against the hardcoded `check_clinical_violations` predicate directly.
**Result: 2000/2000 (100.0000%) agreement on both datasets, 0 discrepancies.**
Raw output: `results/neurosymbolic_regression_summary.json`,
`results/neurosymbolic_regression_discrepancies.json` (empty).

Performance note, documented as a deliberate scope choice, not an omission:
clingo is invoked per-edge and is NOT substituted into the vectorized numpy
hot path (`graph_build.py`) that the full ablation grid uses -- doing so
would make the full grid orders of magnitude slower for no measurement
benefit, since the vectorized code already agrees with the symbolic layer by
construction (both re-encode the same hardcoded predicate, confirmed above).
The symbolic layer is used directly wherever admissibility needs to be
EXPLAINED for one patient/edge at a time: the regression check itself, Phase
5's guideline-derivation text, and the Phase 7 UI.

## Phase 6 — preference elicitation: discrepancy with the brief

The brief's framing assumes an existing "C_clin edge-cost term" inside the
Dijkstra weight that can be reweighted per patient. Reading
`experiments_v6/lib/graph_build.py::make_cell_matrix`/`make_query_row` shows
this is not the case in this codebase: the traversal cost W_ij is PURE
latent-space distance (Riemannian or Euclidean), optionally penalized by a
raw violation COUNT in "soft" mode; `compute_clinical_effort` is computed
only as a POST-HOC reporting metric on the realized path, never as part of
the cost the search actually optimizes. Most defensible interpretation,
adopted in `experiments_v6/lib/preference.py`: add a NEW, clearly-labeled
additive term `mu * weighted_L1_effort(x_i,x_j; profile)` on top of the
EXISTING hard-mode admissible edge set (unchanged from Phase 1 -- preferences
can only change which admissible path is cheapest, never make an
inadmissible transition admissible). See that file's docstring for the full
statement of this deviation and the illustrative, simplified
feature-to-preference-category mapping used for the three demo profiles.

## Deviations from the brief (Hard Rule 5) — summary

1. `nhanes_real.csv` is read directly from `experiments_v3/data/` rather than
   copied into `experiments_v6/data/` (straightforward reuse, not a conflict).
2. Phase 6's "C_clin edge-cost term" does not exist in the codebase as the
   brief assumes; a new additive cost term was introduced for the preference
   demo only, documented in full above and in `lib/preference.py`.
3. Phase 8's characterization of PACE as "single-step, not multi-hop" does not
   match the actual paper (arXiv:2607.01306, read directly this session):
   PACE uses an incremental-BUDGET generate-and-test loop (1..k simultaneously
   changed features), not a literal single feature change, and was evaluated
   on Adult Income (categorical transition-graph constraints), not a clinical
   domain. See the Phase 8 section below for the interpretation adopted.

## Phase 6 — preference elicitation results

`experiments_v6/run_preference_demo.py`: 5 patients per dataset (seed 0),
3 synthetic profiles (`no_preference`, `exercise_averse`, `medication_averse`,
see `lib/preference.py` for the exact feature-multiplier mapping), MU=0.5.
Raw output: `results/preference_demo.csv`.

**Honest finding, not smoothed over**: across the 10 demoed patients (5 UCI +
5 NHANES), the SELECTED target node changed under reweighting for only
1 of 10 (UCI patient #30: `medication_averse` picked target node 142 instead
of node 19). For the other 9, all three profiles selected the SAME target
despite substantially different reported cost (e.g. NHANES patient #12:
cost 9.4 -> 13.5 -> 26.4 across profiles, same target throughout). This is a
direct, expected consequence of Phase 1/2's own finding: post-B1-fix,
admissible targets are comparatively sparse for a given patient (many
patients have zero; those with a path often have one strongly-dominant
low-latent-distance option), so there is often little competitive
alternative for a cost reweighting to promote instead. The mechanism works
exactly as specified (cost changes, safety/admissibility never does; one
concrete case, patient #30, shows a full target switch) -- this is reported
plainly rather than cherry-picking only the switching case.

## Phase 8 — FACE / PACE comparison

**FACE rerun** (`experiments_v6/run_face_rerun.py`): FACE (Poyiadzi et al.,
AIES 2020) never involves the VAE at all -- its k-NN graph and recourse
target are built directly in ORIGINAL feature space
(`benchmarks/run_benchmarks.py::run_face`, confirmed by reading that code
this session), so there was never a decoded-vs-real CVR distinction for it.
"Rerun under the corrected protocol" is interpreted as: evaluate FACE's own
unmodified, unconstrained design (full ephemeral-query attachment, no
admissibility gating at all -- faithful to the published method) on the SAME
v6 test cohort/seeds/real-value-CVR metric as Phase 1, for a controlled,
apples-to-apples comparison, rather than reusing the older 100-instance
`benchmarks/run_benchmarks.py` numbers. Result: 100% success on both cohorts
(nothing is ever pruned), 96.3%/99.9% real-value CVR (UCI/NHANES) -- FACE
finds a path almost every time, and that path is clinically inadmissible
almost every time.

**PACE**: read the actual paper (arXiv:2607.01306, "PACE: A Neuro-Symbolic
Framework for Plausible and Actionable Counterfactual Explanations") directly
this session -- see the discrepancy note above. A minimal, same-domain
reimplementation of PACE's core idea (MLP classifier + ASP-admissible
candidate generation, incremental budget expansion, NO graph/NO real-patient
lookup) was built (`experiments_v6/run_pace_reimpl.py`) and run on this
project's own cohorts and Phase-4 rule base -- this is NOT a reproduction of
PACE's own Adult Income numbers, which are reported separately in
`FINDINGS.md`, clearly labeled "reported by PACE's authors, not
independently reproduced." A first pass with a narrow perturbation grid
(magnitudes 10/20/30% of feature std, budget<=3) found the classifier
essentially unflippable this way (0/5 UCI pilot patients succeeded); a direct
sanity check confirmed flipping IS possible with larger simultaneous changes
(shifting all 4 admissible UCI features by 1.0 std moved a 0.75 base risk to
0.47), so the grid was widened (25/50/75/100/150% of std, budget up to every
admissible feature) before the real run -- documented here rather than
silently discarding the first attempt.

## Phase 7 — UI

`experiments_v6/ui/server.py` (FastAPI, installed this session along with
`uvicorn`) + `experiments_v6/ui/index.html` (vanilla JS, no framework/build
step). The backend imports `experiments_v6/lib/` directly -- it is not a
separate reimplementation of the pipeline, so anything the UI shows is
reproducible from the command line the same way. Trained models are cached
in-process per (dataset, seed) after their first request. Tested end-to-end
this session via direct HTTP calls (`curl`) against a live local instance:
both a successful-plan response (UCI seed 0 patient #0, all 3 explanation
layers populated, all Phase-4 rules shown as passed) and an abstention
response (UCI seed 0 patient #9, real certified-infeasible classification
looked up from `results/uci_blindspot.csv`, semifactual explanation
matching the standalone `explanations.py` test) were confirmed working
before this file was written. No screenshots, per the brief.

## Full reproduction command list

```bash
# one-time installs (all installed cleanly this session, no compiler needed)
.venv/bin/pip install clingo shap fastapi "uvicorn[standard]"

# Phase 1
.venv/bin/python -m experiments_v6.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4
# Phase 2
.venv/bin/python -m experiments_v6.run_blindspot --datasets uci nhanes_real --seeds 0 1 2 3 4
# Phase 4
.venv/bin/python -m experiments_v6.run_neurosymbolic_regression
# Phase 6
.venv/bin/python -m experiments_v6.run_preference_demo --datasets uci nhanes_real --seeds 0 --n_patients 5
# Phase 8
.venv/bin/python -m experiments_v6.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m experiments_v6.run_pace_reimpl --datasets uci nhanes_real --seeds 0 1 2 3 4
# Report
.venv/bin/python -m experiments_v6.make_report
# UI (Phase 7)
.venv/bin/python -m uvicorn experiments_v6.ui.server:app --host 127.0.0.1 --port 8731
```

Total wall-clock this session, all scripts above: ~20-25 minutes (dominated
by Phase 1's real-NHANES grid, ~9 min, and Phase 2's densification probes on
real NHANES, ~5 min); see each `results/run_*_log.txt` for exact per-seed
timing.

## Follow-up session: DEEP_AUDIT resolution

A later session resolved the 10 issues found by a deep repo audit
(`experiments_v6_audit/DEEP_AUDIT.md`), summarized in full in
`experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md`. Headline changes affecting
files described above: the corrected pipeline, neurosymbolic layer,
explanation system, and preference elicitation were merged into `src/`
(`experiments_v6/lib/*.py` are now re-export shims, not the canonical
implementation); `experiments_v6/ui/server.py` now imports from `src/`
directly; `statin_indication`/`bp_target` were wired into the UI for real
NHANES patients; `experiments_v6/CLASSIFIER_AUDIT.md`,
`experiments_v6/run_classifier_audit.py`, and
`experiments_v6/tests/test_gate_symmetry.py` were added; `requirements.txt`
was fixed and verified in a fresh virtualenv. The full 5-seed/both-cohort
grid was rerun after the `src/` merge and reproduced this document's
already-published numbers exactly (38.6%±13.1% UCI / 37.7%±3.4% NHANES
success, 0.00%±0.00% real CVR both cohorts) — no number in this file changed
as a result of that session; only file locations, framing/wording in three
specific places (success-rate framing, the "confirmed four times" language,
and the neurosymbolic section's mention of the newly-wired statin/BP rules),
and additive documentation changed.
