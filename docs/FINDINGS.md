# FINDINGS.md — CARE-LG, consolidated current findings

This is a consolidation, not new research: every number below was already
published in `archive/investigation_v2_to_v6/experiments_v6/FINDINGS.md` and
`archive/investigation_v2_to_v6/experiments_v6_audit/`, produced by the
`experiments_v6` session (git SHA `539d87b795b78bb8f9446f99b0fabeded2c84841`)
and its follow-up deep audit. Nothing here is recomputed or invented; the
numbers have also since been reproduced bit-identically from `src/` as part
of this repo's reorganization (see `CHANGES.md` for that verification).
For the full narrative of how these numbers were arrived at, see
`docs/HISTORY.md`; for full session-by-session detail, see `archive/`.

## 1. The corrected pipeline (B1/B2 fix)

Two code-level bugs (B1: a weaker constraint gate on a query's entry edge
than on every interior edge; B2: unnecessary VAE-decoding of real,
already-known patients, whose reconstruction noise masked genuine
constraint violations) were found and fixed. Full 5-seed grid, both real
cohorts, R+hard (the actual CARE-LG operating point):

| Dataset | Success rate | Success rate range (5 seeds) | Real-value CVR | Decoded CVR (legacy metric) |
|---|---|---|---|---|
| UCI Heart | 38.6% ± 13.1% | 21.7% – 55.0% | **0.00% ± 0.00%** | 47.3% ± 10.3% (riemannian) / 62.1% ± 11.5% (euclidean) |
| Real NHANES | 37.7% ± 3.4% | 32.4% – 41.2% | **0.00% ± 0.00%** | 65.4% ± 7.9% (riemannian) / 62.8% ± 6.9% (euclidean) |

**Real-value CVR is exactly 0.00% ± 0.00% on every one of the 10 (dataset ×
seed) runs.** The nonzero "decoded CVR" is what earlier sessions (v2-v5)
reported as their headline finding — it measures VAE reconstruction noise on
real, already-known patients, not an actual constraint violation.

**Its own finding, not just "the cost of 0% CVR"**: under the corrected
gate, roughly **6 in 10 high-risk test patients receive no recourse plan at
all** (61.4% mean abstention rate UCI, 62.3% Real NHANES). UCI's wide
21.7-55.0% range reflects a small (20-25-patient) per-seed high-risk cohort
— small-sample noise, not five genuinely different operating points.

Before the fix (relaxed gate): success 71.8% (UCI) / 45.5% (NHANES),
real-value CVR 58.8-71.3% on both cohorts.

## 2. Certified infeasibility vs. blindspot

For every patient who abstains under the corrected gate, an exhaustive
feature-space check plus two densification probes (2× k-NN; +500
classifier-verified VAE-prior synthetic nodes) classify the abstention:

| Dataset | Certified infeasible | Confirmed blindspot | Unresolved |
|---|---|---|---|
| UCI | **87.9%** of abstentions | **0.0%** | 12.1% |
| Real NHANES | **98.5%** of abstentions | **0.0%** | 1.5% |

The success-rate cost is overwhelmingly genuine, certified structural
infeasibility — not a fixable search-radius or graph-coverage artifact.
Worded precisely: the SAME densification-probe methodology was applied
independently in four separate sessions (v3, v4, v5, v6) and found zero
blindspots each time — a well-replicated result under **one search
strategy**, not four independent verification methods. An independent
approach (a different densification strategy, or larger-scale exact
enumeration) would strengthen this further and has not been attempted.

## 3. Riemannian vs. Euclidean metric

Paired Wilcoxon signed-rank tests (originally run in `experiments_v3`):

| Dataset | Metric | n pairs | p-value | Mean diff (R−E) |
|---|---|---|---|---|
| UCI | KDE | 81 | 0.649 (n.s.) | −0.0007 |
| UCI | Effort | 81 | 3.5×10⁻⁴ | −0.025 (R slightly better) |
| Real NHANES | KDE | 980 | 0.595 (n.s.) | +0.008 |
| Real NHANES | Effort | 980 | 6.8×10⁻¹³ | +0.204 (R slightly worse) |

No difference on plausibility either dataset; a real but small,
sign-flipping effect on effort. Success rate and KDE are structurally
identical between the two metrics (hard-mode pruning uses the same
admissible edge set regardless of ranking metric).

## 4. Neurosymbolic guideline layer

An ASP (`clingo`) re-encoding of the exact admissibility predicate, plus
two non-blocking guideline annotations (statin indication at the 7.5%
PCE/ASCVD threshold; BP target differentiated by diabetes status).
**Regression check: 2000/2000 (100.0000%) agreement with the hardcoded
predicate on each dataset, 4000/4000 combined, 0 discrepancies.** The
statin/BP-target annotations are wired into the UI for real NHANES patients
(not UCI — its label isn't ASCVD-based).

## 5. Classifier audit

First independent audit of the `RiskClassifier`'s own predictive validity:

| Dataset | AUC-ROC | Brier score | Empirical 7.5%-crossing (classifier probability) |
|---|---|---|---|
| UCI | 0.834 ± 0.041 | 0.182 ± 0.010 | 0.245 ± 0.040 |
| Real NHANES | 0.960 ± 0.004 | 0.078 ± 0.006 | 0.075 ± 0.000 (exact across all 5 seeds) |

No meaningful sex-based fairness gap on NHANES's large subgroups (AUC
difference ~0.003, within seed-to-seed noise). **A real, quantified
threshold discrepancy**: the pipeline's operational 0.55/0.45 gate sits
roughly **7.3× (NHANES) / 2.2× (UCI)** higher on the classifier's
probability scale than the point where observed outcomes actually cross the
7.5% clinical threshold the data label itself is built from. Not
recalibrated — doing so would change which patients qualify as "high-risk"
throughout, invalidating every number above; documented as a deliberate,
open decision, not implemented. Full detail: `docs/CLASSIFIER_AUDIT.md`.

## 6. FACE / PACE comparison

| Method | Dataset | Success rate | Real-value CVR | Certified-infeasibility rate |
|---|---|---|---|---|
| **CARE-LG (corrected)** | UCI | 38.6% ± 13.1% | **0.00% ± 0.00%** | 87.9% of abstentions |
| **CARE-LG (corrected)** | Real NHANES | 37.7% ± 3.4% | **0.00% ± 0.00%** | 98.5% of abstentions |
| FACE (controlled rerun) | UCI | 100.0% ± 0.0% | 96.3% ± 4.1% | n/a |
| FACE (controlled rerun) | Real NHANES | 100.0% ± 0.0% | 99.9% ± 0.2% | n/a |
| PACE-style reimpl. (this project's domain) | UCI | 80.4% ± 14.0% | 0.00%¹ | n/a |
| PACE-style reimpl. (this project's domain) | Real NHANES | 57.6% ± 6.5% | 0.00%¹ | n/a |
| PACE (reported by its authors, Adult Income — not reproduced) | Adult Income | validity 24.0% | plausibility 100%² | n/a |

¹ 0.00% by construction (candidates are filtered through admissibility
before acceptance — a search-space restriction, not a measured rate).
² Not directly comparable to real-value CVR; PACE's metric is conditioned
on acceptance.

FACE enforces no clinical constraints at all, so it finds a path almost
every time — and that path is clinically inadmissible almost every time.
CARE-LG trades success rate for a genuine, verified 0% real-world violation
guarantee, and is the only method here that distinguishes genuinely
impossible patients from not-yet-found ones, or explains its decisions.

## What is NOT working / not yet done (stated plainly, not buried)

- VAE decoder quality was chased across two sessions (v2→v3 type-aware
  decoder; v5's architecturally-exact EGD) and never was the actual
  problem — see `docs/HISTORY.md`.
- Densification probes have found zero blindspots four separate times,
  always with the same method (section 2 above).
- `CLINICAL_EFFORT_WEIGHTS` (`configs/dataset_config.py`) are illustrative,
  uncited constants — every effort number and the preference-elicitation
  mechanism inherits this.
- The classifier threshold discrepancy (section 5) is quantified but
  unresolved.
- Whether the VAE/latent-space search is actually necessary for the
  *search itself* (as opposed to being incidental machinery) has never
  been isolated and tested against a neurosymbolic-gated raw-feature-space
  baseline.
