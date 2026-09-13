# FINDINGS.md — v3 verdicts, diffed explicitly against v2

All numbers below come from code executed in this session (`experiments_v3/results/`,
`run_metadata.json` for git SHA/seeds/config). Git SHA: `539d87b795b78bb8f9446f99b0fabeded2c84841`
(unchanged from v2 — no commits were made in either session). 5 seeds (0–4), both datasets
(UCI, and the real NHANES cohort registered as `nhanes_real`), full 12-cell grid.

This document **diffs against `experiments_v2/FINDINGS.md`** per the brief — it does not
replace it. `experiments_v2/FINDINGS.md` remains an accurate record of what the uncorrected
pipeline (synthetic NHANES, original decoder, no REVISE) produced.

---

## RQ1: Does hard pruning prevent constraint violations that a Riemannian metric with soft penalties still commits?

**v2 answer:** No — decoded CVR was 78–92% in every cell tested, hard or soft.
**v3 answer: Still no, but meaningfully closer on UCI, barely closer on real data.**

| | UCI decoded CVR (R+hard) | Real NHANES decoded CVR (R+hard) |
|---|---:|---:|
| v2 (synthetic NHANES, original decoder) | 83.6% ± 26.9 | 78.1% ± 10.4 |
| v3 (real NHANES, type-aware decoder) | **41.9% ± 9.1** | **81.0% ± 7.0** |

The type-aware decoder (Phase B) cut age reconstruction error 4–6× (UCI 6.83→1.63 years;
Real NHANES 4.12→0.72 years), and on UCI this translated directly into roughly halving
decoded CVR — the dominant violation category on UCI was always age (non-decreasing), so
fixing age fixed most of the problem there. On Real NHANES, decoded CVR barely moved
(78.1%→81.0%, statistically flat), because a **new** dominant violation category appeared:
the immutable `sex` flag crosses its rounding boundary in a real fraction of decoded paths
(concrete examples in `REPORT.pdf`), and sex reconstruction accuracy improved only
marginally (62.4%→68.2%, still near-chance) — this looks like a 4-dimensional latent
information bottleneck that a better decoder head alone cannot fix.

Fig D still shows R+hard sitting close to or below the R+soft curve on both datasets (an
improvement over v2, where UCI's hard line sat *above* soft), but decoded CVR never
approaches zero on either dataset at any λ tested.

**Verdict, updated**: the decoder fix is real and worth keeping (it demonstrably reduces
decoded-space harm on the cohort where the problem was age-driven), but it does not turn
"0% CVR" into a true patient-facing claim on either dataset. The claim still needs the same
rewording v2 recommended: "0% CVR *on the graph's internal, encoded representation*," now
with the added nuance that *which* feature drives the residual gap is dataset-dependent
(age on UCI, sex on Real NHANES) — a decoder fix targeted at one feature does not
generalize to fixing the others.

---

## RQ2: Does the Riemannian metric produce measurably better recourse than Euclidean?

**v2 answer:** No significant difference in KDE or effort; median difference exactly zero
in every paired test.
**v3 answer: Still mostly no, but now with a statistically real (if small and inconsistent) effort effect.**

| Dataset | Metric | n pairs | p-value | Mean diff (R−E) | Practical scale |
|---|---|---:|---:|---:|---|
| UCI | KDE | 81 | 0.649 (n.s.) | −0.0007 | negligible |
| UCI | Effort | 81 | **3.5×10⁻⁵** | **−0.025** | Riemannian slightly *better* (effort scale ≈0.1–0.6) |
| Real NHANES | KDE | 980 | 0.595 (n.s.) | +0.008 | negligible |
| Real NHANES | Effort | 980 | **6.8×10⁻¹³** | **+0.204** | Riemannian slightly *worse* (effort scale ≈1.8–5.0) |

Unlike v2 (where every paired comparison was non-significant), the larger, real-data-driven
sample sizes here (real NHANES has ~10× more high-risk patients per seed) give enough
statistical power to detect a real, non-zero effort difference between the two metrics.
But the effect is (a) small relative to the effort scale on both datasets, and (b) **flips
sign between datasets** — Riemannian is marginally better on UCI and marginally worse on
Real NHANES. Success rate remains structurally identical between R+hard and E+hard on both
datasets (a mathematical necessity of the hard-pruning rule, unchanged from v2). KDE
plausibility remains statistically indistinguishable on both datasets, exactly as in v2.

**Verdict, updated**: v2 said "not supported by this experiment." v3 says "detectable but
not reliable" — there is now a real, non-zero signal, but it is too small and
direction-inconsistent across cohorts to support a claim that the Riemannian metric
produces *better* recourse. This is a more precise, not more favorable, verdict than v2's.

---

## RQ3: When CARE-LG abstains on sparse/real data, is that certified infeasibility or a blindspot?

**v2 answer:** 100% certified infeasible on UCI (mean 6.4 abstentions/seed); zero
abstentions on synthetic NHANES (nothing to decompose).
**v3 answer: Still 100% certified infeasible on UCI — AND now proven on a much larger, real-data abstention rate too.**

| Dataset | Mean abstentions/seed | % of high-risk patients | % certified infeasible | % confirmed blindspot | % unresolved |
|---|---:|---:|---:|---:|---:|
| UCI | 6.4 | 13–36% (varies by seed) | **100.0%** | 0.0% | 0.0% |
| Real NHANES | **235.2** | **51.7–58.9%** | **100.0%** | 0.0% | 0.0% |

This is the single biggest new result in v3. Real NHANES data is genuinely much harder for
CARE-LG than the synthetic stand-in: over half of high-risk patients get no path at all
(vs. literally zero abstentions on synthetic NHANES in v2) — real correlational structure
between age, BP, cholesterol, sex, etc. makes simultaneous constraint satisfaction far
rarer than independent-normal synthetic data implied. But across all 1,176 real-NHANES
abstentions (5 seeds combined), the exhaustive feature-space feasibility check found **zero**
cases where a valid low-risk target existed anywhere in the 3,631-patient cohort that the
graph search failed to find. Both densification probes (doubled k; +500 classifier-verified
synthetic nodes) were exercised on a much larger and more demanding pool of abstentions than
in v2, and still found nothing to resolve.

**Verdict, unchanged and strengthened**: RQ3 remains the project's strongest, most
defensible claim, and this session raised the bar on it substantially — a claim that held
at a 13–36% abstention rate on synthetic-adjacent real data (UCI) also holds at a 52–59%
abstention rate on genuinely correlated real clinical data. This is the one mechanism in
this project that has now survived two full rounds of adversarial re-checking (fresh
seeds/decoder in v2→v3, and real vs. synthetic data within v3).

---

## REVISE (genuinely implemented for the first time)

v2 could not report REVISE numbers at all — it was never implemented in the codebase and
was not re-implemented from scratch, per the brief's own "don't reimplement absent
baselines" instruction. v3 implements it faithfully (Joshi et al. 2019, `experiments_v3/lib/revise.py`)
and reports real numbers for the first time:

| Dataset | Success | Decoded CVR | KDE | Effort | Latency |
|---|---:|---:|---:|---:|---:|
| UCI | 95.4% ± 7.9 | 100.0% | −7.311 ± 0.135 | 9.31 ± 0.61 | 0.0115s ± 0.0040 |
| Real NHANES | 90.5% ± 2.3 | 100.0% | −6.804 ± 0.111 | 5.81 ± 0.31 | 0.0087s ± 0.0007 |

REVISE achieves high success and the best or near-best KDE plausibility of any method
(latent-space search does keep counterfactuals on-manifold, as the paper claims), at the
cost of the highest clinical effort of any method and, like every non-graph baseline, 100%
decoded-space constraint violation — REVISE has no constraint-awareness mechanism at all,
so its counterfactuals routinely change immutable/monotone features. This means the paper's
originally-*claimed* REVISE numbers (94.1%/45.2%/2.1000s on NHANES, 88.3%/58.4%/1.8500s on
UCI, cited in this session's brief) **do not correspond to anything that was ever actually
run in this repository before this session** — they cannot be verified or reproduced, and
should not appear in any future report. The numbers above are the first REVISE numbers in
this project's history backed by executed code.

---

## Summary table: v2 verdict → v3 verdict

| Claim | v2 verdict | v3 verdict | What changed |
|---|---|---|---|
| "0% CVR" (graph-internal) | Survives, narrower than presented | **Unchanged**: still survives, still narrower | N/A (structural, decoder-independent) |
| "0% CVR" (patient-facing, decoded) | Contradicted (78–92% decoded CVR) | **Contradicted, but less severely on UCI** (42% UCI, 81% Real NHANES) | Type-aware decoder halved UCI's gap; barely moved Real NHANES's |
| Riemannian metric beats Euclidean | Not supported (all tests n.s.) | **Detectable but not reliable** (significant, small, sign-flips by dataset) | Larger real-data sample sizes gave the tests power v2's synthetic data lacked |
| "Ultra-fast" recourse | Survives | **Survives** | Latency still 0.001–0.013s across all grid cells |
| UCI success ≈47.6% (point estimate) | Needs rewording (63.6–87.0% observed) | **Unchanged**: needs rewording, same range (decoder-independent) | N/A |
| NHANES ≈99% success / ~1% failure | Contradicted (100% success on synthetic data) | **Contradicted the other way**: 45.5% success on REAL data | Real correlated data is much harder than the synthetic stand-in ever suggested |
| Abstentions = certified infeasibility, not blindspots | Survives strongly (UCI only, synthetic NHANES had none) | **Survives even more strongly**: now proven at a 52–59% real-data abstention rate too | Real data gave a much larger, harder test population for this claim, which it passed completely |
| Dataset is "CDC NHANES" | Contradicted (fully synthetic) | **Resolved**: now genuinely real NHANES data (N=4,539, DATA_PROVENANCE.md) | Phase A |
| REVISE baseline | Absent, not implemented, not reimplemented | **Resolved**: genuinely implemented and run (this document, above) | Phase C |

## Overall recommendation for a revised paper (updated from v2)

RQ3 (the abstention/recourse-verification mechanism) should remain the paper's primary
contribution — it is now validated on real data at a realistic, high abstention rate, which
is a *stronger* result than v2 could report. RQ1 should be reframed as "hard pruning offers
a real but incomplete and dataset-dependent decoded-space safety margin over soft
penalties, whose size depends on which specific feature's reconstruction fidelity
dominates the residual gap" — a more precise, defensible claim than either "0% CVR" or
outright "no benefit." RQ2 should be reframed similarly: "the Riemannian metric produces a
statistically detectable but practically small and sign-inconsistent effort difference
versus Euclidean distance" — neither "the metric matters" nor "the metric is pointless" is
supported; the honest claim is in between. The paper's REVISE comparison can now be made
honestly for the first time, and should be, since the originally-cited REVISE numbers do
not correspond to any code ever run in this project.
