# HISTORY.md — how CARE-LG got here

A short, plain-language timeline of the investigation that produced the
current `src/`. Full detail for every claim below lives in
`archive/investigation_v2_to_v6/`, preserved as a historical snapshot — it
is **not meant to be run as-is** from its new location; current
reproduction uses `src/` and the scripts in `scripts/` (see `README.md`).

## v2 — first correction pass

Found the original pipeline was unseeded and its "NHANES" dataset was fully
synthetic. Built a seeded multi-run harness and introduced **decoded-space
verification** — re-checking constraints on the VAE's decoded reconstruction
of a path, on the premise that "the patient sees the decoded reconstruction,
not the raw value." Headline finding: 78-92% of successful paths violated a
constraint once decoded, despite 0% violations on the graph's own internal
(encoded) representation. Left unresolved: whether decoding a real,
already-known patient's node was ever necessary, and whether the
query-entry gate used the same rule as every other edge.

## v3 — real data + better decoder

Replaced the synthetic NHANES with a real, PCE/ASCVD-labeled cohort
(`data/nhanes_real.csv`) and added a type-aware VAE decoder head. This
roughly halved decoded-space violations on UCI (83.6%→41.9%) — age
reconstruction was the dominant failure there — but barely moved Real
NHANES (78.1%→81.0%), where sex reconstruction remained near-chance
instead. Same two questions from v2 remained unasked.

## v4 (DSR) and v5 (EGD) — chasing the decoder further

Two successive attempts to close the decoded-CVR gap AT THE DECODER: v4
added conformal calibration bands and a repair step; v5 built an
architecturally exact decoder (masked/residual heads) that GUARANTEED 0.0%
decoded CVR by construction wherever it succeeded. Both paid severe costs
elsewhere — v5's success rate collapsed on Real NHANES (45.5%→1.0%), effort
rose ~50×, plausibility got worse, and the Riemannian-vs-Euclidean metric
choice flipped. Neither actually closed the gap in a way that held up
end-to-end; both were making the wrong thing better.

## The audit — the actual root cause

A dedicated diagnostic session (no new building, just verification) found
the gap was never about decoder quality at all. It was two code-level bugs
introduced when the evaluation harness (not the original library) built its
own query-attachment mechanism to support proper train/test splitting:

- **B1**: the query patient's entry edge was checked against a WEAKER rule
  (immutability + age-decrease only) than every interior graph edge (which
  also checks directional constraints like rising blood pressure). A patient
  could be silently attached through a real, measured violation.
- **B2**: every path node — including real, already-known training
  patients — was being re-decoded through the VAE for verification, even
  though the true value was already on file. Reconstruction noise was
  masking real violations (a genuine +35 mg/dL cholesterol change decoded to
  an apparent +0.3 mg/dL).

Fixing both together drove real-value CVR to exactly 0.00% on both cohorts,
at the honest cost of success rate dropping (71.8%→38.6% UCI, 45.5%→37.7%
NHANES) — patients with no genuinely safe entry are now correctly rejected
instead of silently accepted.

## v6 — the corrected foundation, plus the current feature set

Applied the B1/B2 fix as the sole mode (confirmed 0.00% real CVR, 5 seeds,
both cohorts). Verified the success-rate cost is genuine structural
infeasibility (87.9%/98.5% of abstentions certified infeasible), not a
graph-coverage problem. Built the features that make up the current system:
a neurosymbolic (ASP) guideline layer regression-verified against the
hardcoded predicate; a three-layer explanation system (SHAP, guideline
derivation, semifactual abstention explanations); per-patient preference
elicitation over the clinical-effort cost term; a functional local UI; and
an auto-generated single-patient HTML report. Reran FACE under a controlled
protocol and built a minimal PACE-style reimplementation for comparison.

## The deep audit — closing the remaining gaps

A follow-up review found the corrected pipeline still lived only in
`experiments_v6/`, disconnected from the actually-importable `src/`; the
classifier's own validity had never been checked; two framing claims
overstated their evidence; two implemented guideline rules were never
surfaced in any output; and `CLINICAL_EFFORT_WEIGHTS` had no cited
rationale. All were resolved: the corrected pipeline was merged into `src/`
(reproducing identical numbers); the classifier was audited directly (AUC,
Brier, calibration, sex-subgroup performance, and a quantified — but
deliberately unresolved — threshold reconciliation); the overstated claims
were reworded; the guideline rules were wired into the UI; the effort
weights were documented as unvalidated; and a permanent regression test was
added asserting the entry-gate/interior-gate symmetry that B1 had violated.

## The baseline rebuild — bringing DiCE, FACE, and Growing Spheres up to standard

`benchmarks/run_benchmarks.py`'s `run_dice`, `run_face`, and
`run_growing_spheres` predate this project's documented investigation
phase entirely (git history places them in one of the repository's very
first commits) and, on inspection, cite no source paper anywhere and
include no algorithm-fidelity documentation — inconsistent with every
baseline built since (REVISE, PACE, the corrected FACE rerun), each of
which cites its source and documents its own deviations explicitly. This
session rebuilt all three from their actual papers, into `src/baselines/`.

**The most consequential finding, reported plainly**: the original
`run_dice` was a single L2-penalized gradient-descent counterfactual with
NO diversity mechanism at all. Diversity across several simultaneous
counterfactuals is DiCE's entire defining contribution over plain
gradient-descent search (which this project's REVISE baseline already is,
just in latent space) — so the original was not a faithful DiCE
implementation by the paper's own definition of the method, not merely an
imprecise one.

Real, executed old-vs-new comparison (same patients, same seeds, same
0.45 target threshold applied to both for a fair comparison):

| Dataset | Method | Success (old→new) | Real-value CVR (old→new) |
|---|---|---|---|
| UCI | Growing Spheres | 82.7% → 94.7% | 95.2% → 100.0% |
| UCI | DiCE | 66.7% → 98.7% | 98.0% → 74.3% |
| Real NHANES | Growing Spheres | 98.7% → 100.0% | 100.0% → 100.0% |
| Real NHANES | DiCE | 49.3% → 97.3% | 100.0% → 69.9% |

DiCE's success rate roughly doubled once actually implemented with its
paper's real joint, diverse optimization (more simultaneous attempts per
patient genuinely helps find a valid counterfactual); its real-value CVR
dropped meaningfully once DiCE's own actionability mechanism (holding
immutable/non-decreasing features fixed, a real part of the published
method the original never implemented) was applied. Growing Spheres
changed less — it was always an unconstrained, heuristic method in both
versions, and remains one; the rebuild's more faithful shell-sampling
mostly just makes success slightly more reliable. Full metrics (KDE,
effort, latency) for the new implementations: `docs/FINDINGS.md`.

Nothing in `docs/FINDINGS.md` or `results/` previously cited numbers from
these three functions — FACE's currently-published numbers were already
sourced from the corrected rerun before this session, and no DiCE/Growing
Spheres numbers had been published anywhere in current docs. This rebuild
therefore adds new, real baseline numbers; it does not contradict or
require correcting any previously-published figure.

## What `src/` contains today, and why

`src/` is the single, current implementation of everything above: the
classifier (`src/classifier/`), the type-aware VAE (`src/vae/`), the
B1/B2-fixed graph construction and neurosymbolic layer (`src/graph/`), the
search, explanation, and preference-elicitation logic (`src/recourse/`),
the single-patient HTML report generator (`src/reporting/`), and the
cited, faithful comparison baselines (`src/baselines/`). It is
self-contained and does not depend on `archive/`. The scripts that
reproduce every number in `docs/FINDINGS.md` live in `scripts/`; the UI
lives in `ui/`; results already generated live in `results/`.
`benchmarks/run_benchmarks.py` holds the original, now-superseded baseline
implementations, kept unmodified for historical reference only.
