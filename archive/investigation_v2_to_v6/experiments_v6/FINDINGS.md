# FINDINGS.md — experiments_v6

All numbers below come from code executed this session, git SHA
`539d87b795b78bb8f9446f99b0fabeded2c84841`. Exact commands and raw file
locations are in `HOW_TO_RUN.md` / `CHANGES.md`.

## Phase 1 — the corrected foundation

Applying B1 (entry-gate fix) and B2 (no unnecessary decoding of real nodes)
as the sole mode of `experiments_v6`'s pipeline, R+hard (the actual CARE-LG
operating point), 5 seeds, both cohorts:

| dataset | success rate (mean ± std) | success rate range (5 seeds) | real-value CVR | decoded CVR (legacy, informational only) |
|---|---|---|---|---|
| UCI Heart | 38.6% ± 13.1% | **21.7% – 55.0%** | **0.00% ± 0.00%** | 47.3% ± 10.3% (riemannian) / 62.1% ± 11.5% (euclidean) |
| Real NHANES | 37.7% ± 3.4% | 32.4% – 41.2% | **0.00% ± 0.00%** | 65.4% ± 7.9% (riemannian) / 62.8% ± 6.9% (euclidean) |

*UCI's 21.7%-55.0% range on a 20-25-patient high-risk test cohort per seed is
small-sample noise wide enough that any single seed's success number, taken
alone, is close to meaningless — only the 5-seed distribution is meaningful.
Real NHANES's much larger per-seed cohort (~420-445 high-risk patients) gives
a correspondingly tighter, more trustworthy range.*

**Its own sub-finding, not just "the cost of 0% CVR"**: under the corrected
gate, **roughly 6 in 10 high-risk test patients receive no recourse plan at
all** (61.4% mean abstention rate, UCI; 62.3%, Real NHANES — see Phase 2
below for what happens to them). This is a real clinical-utility limitation
in its own right. It has been more consistently presented elsewhere in this
project's documents as the "cost" paid for reaching 0% real-world constraint
violation than as a standalone finding — stated explicitly here per
`experiments_v6_audit/DEEP_AUDIT.md` issue #5: a system that safely helps
~4 in 10 flagged patients and is silent for the other ~6 is a materially
different clinical proposition than "0% CVR" alone conveys.

**Real-value CVR is exactly 0.00% ± 0.00% on every one of the 10 (dataset ×
seed) runs.** This confirms `experiments_v6_audit/RECOMMENDATION.md`'s
measurement, now on the full grid rather than a spot-check. The nonzero
"decoded CVR" figure is exactly what v2-v5 reported as their headline
finding — it measures VAE reconstruction noise on real, already-known
patients, not an actual constraint violation, since those patients' true
values were never actually unknown.

## Phase 2 — certified infeasibility vs. blindspot

Of the patients who abstain under the corrected R+hard gate:

| dataset | mean abstention rate | certified infeasible | confirmed blindspot | unresolved |
|---|---|---|---|---|
| UCI | 61.4% of high-risk test patients | **87.9%** of abstentions | **0.0%** | 12.1% |
| Real NHANES | 62.3% of high-risk test patients | **98.5%** of abstentions | **0.0%** | 1.5% |

**The success-rate drop from Phase 1 is overwhelmingly explained by genuine,
certified structural infeasibility — not by a fixable search-radius or
graph-coverage limitation.** Zero blindspots were confirmed on either
cohort, in any of the 5 seeds. Per `experiments_v6_audit/DEEP_AUDIT.md` issue
#7: the same densification-probe methodology (2× k-NN; +500
classifier-verified VAE-prior synthetic nodes) has now been applied
independently in four separate sessions (v3, v4, v5, v6) and found zero
blindspots each time — **a well-replicated result under one search strategy,
not four independent verification methods.** "Our own search method cannot
find a counterexample to our own search method's claim" is real, consistent
evidence, but it is weaker than "confirmed four times" can sound on a first
read. An independent verification approach not yet attempted in this
project — e.g. a structurally different densification strategy (real
longitudinal/repeated-measures data if it becomes available, per
`experiments_v6_audit/SEARCH_FINDINGS.md`'s literature scan; or exact
enumeration at a larger synthetic-node scale than 500) — would strengthen
this finding further. The small unresolved remainder (12.1% UCI, 1.5%
NHANES) is not certified infeasible under the exhaustive pairwise check but
also wasn't confirmed reachable by either probe — genuinely ambiguous, not
silently folded into either category.

`find_path_augmented` runs a single Dijkstra call over the whole augmented
graph; it has no code path that distinguishes "no entry edge exists" from
"entry succeeded but nothing beyond it reaches a low-risk target" — both
were always the same failure signature (no finite distance from the query
to any target). So the entry-failure/path-failure unification the brief
asked for was already true by construction of the search machinery; what
Phase 2 adds is running the full exhaustive-check-plus-densification
protocol against the CORRECTED (B1-fixed) gate for the first time.

## Phase 3 — literature diff

Read arXiv:2302.11213 ("Feasible Recourse Plan via Diverse Interpolation")'s
actionability-graph construction in full (HTML rendering, not just the
abstract). Its edges are gated by "cost threshold constraints as well as
immutable feature constraints" — architecturally the same shape as this
project's own CALG (k-NN + immutable/directional gating). Its one genuinely
different idea (diversity-aware prototype selection via DPP/QP) operates
upstream of pathfinding and addresses choosing among several REACHABLE
targets — a different problem than this project's actual gap (Phase 2: the
large majority of abstentions have ZERO reachable targets). The paper does
not discuss the zero-feasible-path case at all. **No change adopted; no
small, cheap improvement was found.**

## Phase 4 — neurosymbolic guideline layer

`clingo`-based ASP encoding of the exact admissibility predicate
(`check_clinical_violations`), plus non-blocking guideline annotations
(statin indication at the 7.5% PCE/ASCVD threshold; BP targets differentiated
by diabetes status). **Regression check: 2000/2000 (100.0000%) agreement
with the hardcoded predicate on each dataset, 4000/4000 combined, 0
discrepancies.** Every admissible/blocked decision the reasoner makes is
provably identical to what Phase 1's numpy code already decides — this
layer adds EXPLANATION (which rule fired, with citation), not a change in
behavior. Full rule table and citations: `METHOD.md`.

**Update (this session, `experiments_v6_audit/DEEP_AUDIT.md` issue #8)**: the
`statin_indication` and `bp_target` guideline annotations were implemented
and regression-verified in the prior session but never actually surfaced in
any output — a grep for their function names outside `neurosymbolic.py`
returned zero hits. They are now wired into the UI and API
(`src.recourse.explanations.nhanes_guideline_context`, called from
`experiments_v6/ui/server.py`) for real NHANES patients, the only cohort
with the source fields (`ascvd_10yr_risk_pct`, `diabetic`) these rules need.
Confirmed working on a real patient this session (NHANES seed 0, patient
#5): 17.7% ASCVD risk → statin indicated per [PCE2013]; non-diabetic →
<140/90 mmHg BP target per [HTN2017]. **UCI is deliberately NOT wired to
this layer** — its label is not ASCVD/PCE-based, so a "10-year ASCVD risk"
number does not exist for UCI patients and fabricating one would violate
this project's own no-fabrication rule; this is documented as a real data
limitation, not an oversight.

## Phase 5 — explanation system

All three layers verified working end-to-end via the UI backend
(`experiments_v6/ui/server.py`) and standalone smoke tests this session.
One concrete worked example (UCI, seed 0, patient #9, `no_preference`):

> *Even under the most favorable single-step comparison found in the
> training data (patient #194, 1 violated sub-condition(s) — the fewest of
> any of the 113 low-risk candidates checked), risk would remain above the
> target threshold, because: [R-AGE-MONOTONIC] age: Age cannot decrease
> between source and target (structural constraint). Across the full
> candidate set, the most frequent obstruction(s) were: R-AGE-MONOTONIC
> (97% of 113 checked candidates); R-BP-DIRECTION (93% of 113 checked
> candidates); R-IMMUTABLE (45% of 113 checked candidates).*

This is generated from a live recomputation against every one of that
patient's 113 low-risk training candidates (not a template) — spot-checked
by hand this session against the underlying per-candidate violation counts
and confirmed accurate (patient #194 genuinely has the fewest violations,
1, of any candidate; the age-decrease requirement genuinely recurs in 97%
of candidates).

## Phase 6 — preference elicitation

Discrepancy with the brief documented in `CHANGES.md` (no existing `C_clin`
cost term to reweight; a new additive term was introduced for this demo
only, admissibility unchanged). Demo: 5 patients × 3 profiles per dataset
(seed 0). **Honest result**: the selected TARGET changed for 1 of 10 demoed
patients (UCI #30, `medication_averse`); for the other 9, cost changed
substantially across profiles (e.g. 9.4 → 13.5 → 26.4 for one NHANES
patient) but the cheapest admissible target stayed the same. This is a
direct consequence of Phase 1/2: post-fix, admissible targets are
comparatively sparse, so there is often no competitive alternative for a
reweighted cost to promote instead — the mechanism works exactly as
specified, but its practical effect on THIS project's graphs is limited by
how few admissible options a given patient typically has.

## Phase 7 — UI

Functional FastAPI backend + single-page frontend, all five specified
panels implemented and tested end-to-end this session (patient selector,
risk gauge, preference control, expandable step-by-step plan with guideline
derivations, and an abstention panel showing the real certification result
and semifactual explanation, not a generic error). No screenshots, per the
brief. See `HOW_TO_RUN.md`.

## Phase 8 — FACE / PACE comparison

| method | dataset | success rate | real-value CVR | certified-infeasibility rate | explanation coverage |
|---|---|---|---|---|---|
| **v6 (corrected, R+hard)** | UCI | 38.6% ± 13.1% (range 21.7-55.0%*) | **0.00% ± 0.00%** | 87.9% of abstentions | full (3-layer, this project only) |
| **v6 (corrected, R+hard)** | Real NHANES | 37.7% ± 3.4% | **0.00% ± 0.00%** | 98.5% of abstentions | full |
| FACE (controlled rerun) | UCI | 100.0% ± 0.0% | 96.3% ± 4.1% | n/a (not certified by FACE) | none |
| FACE (controlled rerun) | Real NHANES | 100.0% ± 0.0% | 99.9% ± 0.2% | n/a | none |
| PACE-style reimpl. (this project's domain) | UCI | 80.4% ± 14.0% | 0.00%¹ | n/a | none (single risk-flip check only) |
| PACE-style reimpl. (this project's domain) | Real NHANES | 57.6% ± 6.5% | 0.00%¹ | n/a | none |
| PACE (**reported by its authors**, Adult Income — different domain) | Adult Income | validity 24.0% | plausibility 100%² | n/a | none |

¹ 0.00% by construction — every generated candidate is filtered through the
Phase-4 admissibility check before being accepted, so an inadmissible
"success" cannot occur; this is not a measured rate in the same sense as
the graph-search methods above, it is a search-space restriction.
² PACE's own reported "plausibility" metric (fraction of accepted
counterfactuals satisfying their symbolic constraints) is a different
quantity from this table's real-value CVR (fraction of the FULL cohort with
constraint-violating trajectories); the two are not directly comparable
even though both measure constraint satisfaction, because PACE's metric is
conditioned on acceptance while CVR here is unconditional relative to
attempted patients.
* UCI's per-seed success rate ranges 21.7%-55.0% (see Phase 1 above) on a
  20-25-patient high-risk cohort per seed — small-sample noise wide enough
  that this single mean should not be read as a precise operating point.

**Reading this table honestly**: FACE finds a path for essentially every
patient because it enforces no constraints at all — its real-value CVR
confirms that a "path" and a "clinically safe path" are very different
things once actually checked against real patient values. The PACE-style
reimplementation (same domain, same classifier, same Phase-4 rule base as
v6) sits between FACE and v6 on success rate precisely because it searches
a much larger, unconstrained-by-graph-topology candidate space (any
admissible perturbation of x0, not only transitions to real recorded
patients) — it is not directly comparable to v6's graph-search paradigm,
and its "0% CVR" is trivial by construction rather than an achievement.
PACE's own officially reported numbers (Adult Income, a demographic/
financial domain with categorical transition-graph constraints, not a
clinical cohort) are presented for reference only and are not a like-for-
like comparison to any row above — this is stated plainly rather than
implying comparability that doesn't exist.

**Unique to v6**: certified-infeasibility classification (neither FACE nor
PACE distinguishes "genuinely impossible" from "not found") and the full
three-layer explanation system (neither baseline offers risk attribution,
guideline derivation, or a semifactual abstention explanation at all).

## What was and wasn't achieved this session — honest summary

**Achieved, with real numbers**: the B1/B2 fix confirmed as the project's
corrected foundation (real CVR = 0.00% on the full grid, not a spot-check);
v4/v5 independently checked and confirmed NOT to need the same fix;
certified-infeasibility protocol re-run against the corrected gate,
confirming the success-rate cost is genuine structural infeasibility
(87.9%/98.5% of abstentions), not a graph-coverage artifact; a working
neurosymbolic layer that regression-matches the hardcoded predicate exactly
(100% agreement, 4000/4000 sampled pairs); all three explanation layers
built and verified end-to-end, including a hand-spot-checked semifactual
example; a working preference-elicitation mechanism with an honestly-
reported (not cherry-picked) demonstration; a functional UI exercising the
full corrected pipeline; FACE rerun under a controlled, apples-to-apples
protocol; a minimal, clearly-labeled PACE-style reimplementation adapted to
this project's own domain (not a reproduction of PACE's published numbers).

**Not achieved / explicitly out of scope this session**: architecture
diagrams and UI screenshots (deferred per the brief); a full reproduction
of PACE on its own Adult Income setup (out of scope — a different domain
entirely; its own reported numbers are presented separately and labeled as
such, never fabricated as if measured here); Phase 3 found no adoptable
literature change, which is itself a reportable (negative) result rather
than a gap in effort.
