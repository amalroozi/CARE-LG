# FINDINGS.md — Decode-Safe Recourse (DSR) results, plain language

All numbers below come from code executed in this session (`experiments_v4/results/`,
`run_metadata.json` for git SHA/seeds/config). Git SHA: `539d87b795b78bb8f9446f99b0fabeded2c84841`
(no commits in this or any prior session). 5 seeds (0-4), both cohorts (UCI, real
NHANES). E3's detailed per-abstention decomposition on real NHANES is computed on a
25-patient random sample per seed where the true abstention count exceeds that
(explicitly reported, `CHANGES.md`).

## The headline number

**Does DSR close the 78-92% decoded-CVR gap `experiments_v3` found? Mostly yes, and the
single cheapest component (aligning the pruning rule with what gets checked) does almost
all of the work:**

| | Old method (latent pruning) | DSR-C1 (decoded pruning only) | DSR-full (+bands, +repair) |
|---|---:|---:|---:|
| UCI decoded CVR | 73.9% ± 17.6 | **18.4% ± 9.1** | 0.0% (at its only viable operating point — see below) |
| Real NHANES decoded CVR | 68.3% ± 11.3 | **2.5% ± 3.3** | **0.0%** (at q≤0.10, n=303 paired successes, Wilcoxon p≈4×10⁻³⁷) |

Component 1 alone — just checking the SAME decoded representation the graph already
prunes on for immutability, instead of the raw recorded values — cuts decoded CVR by
roughly 4-27x on both cohorts, **while also increasing success rate** (UCI 29.3%→75.5%;
real NHANES 37.7%→54.1%). This is not a trade-off; it is a straightforward correction of
a mismatch between what the old method checked and what it was judged on.

DSR-full (bands + repair) achieves **exactly 0% decoded CVR** wherever it succeeds, on
both cohorts, but its viable operating range is narrow and cohort-dependent (below).

## RQ (new) — Does decode-safe pruning alone (Component 1) fix the gap?

**Mostly, and it is the single biggest lever in this whole method.** See the headline
table. The mechanism is simple and verified: the old method prunes on `x` but is
verified on `f_θ(enc(x))`; Component 1 prunes on `f_θ(enc(x))` directly, so the two
checks now agree by construction wherever the decoder's rounding/thresholding behavior
is stable. The residual (18.4% UCI, 2.5% NHANES) is driven by exactly the features
Component 2 targets next (below).

## RQ1 (brief's framing) — Do the calibrated bands (Component 2) certify what pruning alone cannot?

**Partially, and the failure mode is precise, not vague.** Applying bands as a hard
PRE-FILTER (`dsr_c1c2`) collapses success to 0.0-0.3% on both cohorts at every tested
confidence level (0.05-0.99) — `experiments_v4/results/tau_vs_eps_diagnostic.csv` shows
why: for cholesterol (both cohorts), oldpeak (UCI), and BMI (real NHANES), the
calibrated reconstruction-error tolerance `τ_k` **exceeds** the original epsilon
tolerance even at the loosest quantile tested (q=0.05) — the decoder's real error on
those specific features is simply larger than the tight tolerance the original code
assumed, and a directional violation is an OR across all directional features, so one
poisoned feature blocks nearly every edge. This is a genuine, decoder-fidelity-driven
limit, not a calibration bug (`METHOD.md` §2a, §2c).

Resolution used for `dsr_full`: bands are applied at the verify-and-repair stage instead
of as a pre-filter (`METHOD.md` §2c). There, they work — DSR-full achieves 0% decoded
CVR at every operating point it succeeds at (Wilcoxon p≈4×10⁻³⁷ vs. the old method on
real NHANES, n=303 pairs, the strongest possible signed-rank result: every single paired
patient improved).

**The identity rule specifically (immutables via node identity, not decoded values)**:
isolating it (`dsr_bands_no_identity` — bands everywhere including immutables) shows
success collapsing to 0.0-0.09% and CVR jumping back up where it succeeds at all
(0% on the rare NHANES success, but the sample is too thin — n=2 — to generalize from).
The clean signal is `dsr_c1`'s constraint-type breakdown
(`constraint_breakdown.csv`): under DSR-C1, immutable, non-decreasing, and directional
hop-wise violations are all **0%** on UCI, and immutable and directional are 0% on real
NHANES too (only age-horizon remains at 13.5% there) — versus 30-54% and 8-42%
respectively under the old method. The identity rule is doing real, measurable work.

## RQ2 — Does the safety/success trade-off (E2) show a real, usable operating point?

**Yes, on real NHANES; no, on UCI — and the difference is informative, not noise.**

| τ quantile | Real NHANES success | Real NHANES decoded CVR (among successes) |
|---:|---:|---:|
| 0.05 | 23.4% ± 7.3 | 0.0% |
| 0.10 | 12.8% ± 6.7 | 0.0% |
| 0.25 | 2.8% ± 2.4 | 0.0% |
| 0.50 | 0.5% ± 0.7 | 12.5% ± 17.7 (small-n) |
| ≥0.80 | 0.0% | n/a |

UCI: **0.0% success at every single quantile from 0.05 to 0.99.** There is no operating
point on UCI where DSR-full returns anything. This is not a failure of the sweep design
— UCI's smaller calibration set (36 rows) and its specific feature mix (cholesterol,
oldpeak, max_heart_rate all have real, structural `τ ≫ ε`, `tau_vs_eps_diagnostic.csv`)
make even the loosest tested band unrepairable there, and Component 3's repairability is
itself feature-specific (below).

**The honest conclusion**: DSR's viable operating point exists but is well below the
naively "safe-sounding" q=0.95 standard, and its very existence is cohort-dependent. A
clinician deploying this method on a small cohort like UCI would currently get zero
usable recourse from DSR-full at any confidence level tested; on a cohort the size of
real NHANES, a genuine (if narrow, ≤25% success) safe operating regime exists.

## RQ3 — Does the certified-infeasibility guarantee survive under DSR? (E3)

**No, not cleanly — DSR-full introduces a new abstention category `experiments_v3`
never needed.**

| Cell | UCI cert. infeasible | UCI unresolved | Real NHANES cert. infeasible | Real NHANES unresolved |
|---|---:|---:|---:|---:|
| DSR-C1 | 100% | 0% | 85% ± 13 | 14% ± 13 (+1% blindspot-resolved) |
| DSR-full | 85% ± 15 | **15% ± 15** | 79% ± 18 | **21% ± 18** |

`experiments_v3`'s finding was 100% certified-infeasible, 0% blindspot, on both cohorts.
DSR-C1's abstentions come close to reproducing that (100% on UCI exactly; 85%/14%/1% on
real NHANES — a small but real softening). **DSR-full's abstentions do not**: 15-21% of
its abstentions are neither certified-infeasible (a valid low-risk target genuinely
exists, confirmed by the same exhaustive check) nor resolved by either densification
probe (more k-NN neighbors; +500 synthetic nodes) — they are cases where the graph
*could* reach a target, but Component 3's repair mechanism fails to certify the path
regardless of graph density. **This is a new failure category** — call it a
*repair-induced blindspot* — that the existing recourse-verification toolkit (built to
diagnose *connectivity* blindspots) has no probe for, because the problem is not
connectivity. This is directly explained by `METHOD.md` §3's finding that repair only
reliably fixes features with a dedicated decoder head (age): a target that requires
repairing cholesterol, for instance, will be unresolved by DSR-full regardless of how
densely connected the graph is.

**Verdict**: the certified-infeasibility claim is a property of the OLD/C1-style
graph-connectivity abstention, not of DSR-full's repair-gated abstention. Reporting
"abstention = certified infeasibility" for DSR-full without qualification would now be
false 15-21% of the time — this must be stated plainly in any future paper draft.

## E4 — Riemannian vs. Euclidean, re-tested under DSR

**Still null on plausibility, and now real evidence exists at a powered operating
point** (real NHANES, q=0.05, n=459 paired successes — UCI has no viable DSR-full
operating point, so E4 could not be tested there at all, and this absence is reported
rather than filled with an arbitrary choice, `run_e4_best_tau.py`):

| Metric | n pairs | p-value | Median diff |
|---|---:|---:|---:|
| KDE | 459 | 0.820 (n.s.) | 0.000 |
| Effort | 459 | 0.00019 (sig.) | 0.000 |

This exactly reproduces `experiments_v3`'s pattern under the OLD method: KDE
indistinguishable, effort statistically detectable but with a zero median (an
asymmetric-but-thin signal among many exact ties). The Riemannian-vs-Euclidean question
closes out the same way under DSR as it did under latent pruning: real but not
practically decisive.

## E5 — Decoder dependence: is DSR's benefit a property of the method or of Phase B's decoder?

**Both, in different proportions on the two cohorts — a genuinely useful, cohort-
dependent answer, not a clean "yes" or "no":**

| Cohort | Decoder | DSR-C1 decoded CVR |
|---|---|---:|
| UCI | v2 (untyped) | 54.1% ± 32.1 |
| UCI | v3 (type-aware) | 18.4% ± 9.1 |
| Real NHANES | v2 (untyped) | 3.1% ± 1.7 |
| Real NHANES | v3 (type-aware) | 2.5% ± 3.3 |

On UCI, decoder quality matters a great deal for DSR-C1 (54%→18%, a 3x reduction from
the type-aware fix alone). On real NHANES, it barely matters (3.1%→2.5%, within noise) —
the OLD method's own decoder-dependence tells the same story (real NHANES v2: 75.8% CVR
vs. v3: 68.3% CVR, a much smaller gap than UCI's v2: 100% vs. v3: 73.9%). The most
likely explanation is data volume: real NHANES's ~15x larger training set (3,086 vs. 201
graph-train rows) gives even the architecturally-plain v2 decoder enough signal to
reconstruct most features adequately, narrowing the gap Phase B's architecture change
was designed to close. **Conclusion: DSR-C1's benefit is a real property of the METHOD
(it helps under either decoder, on both cohorts), amplified by decoder quality when data
is scarce, and largely decoder-quality-independent when data is abundant.**

## Summary table: does DSR fix the paper's problem?

| Claim | Status | Evidence |
|---|---|---|
| Component 1 (decoded pruning) meaningfully reduces decoded CVR | **Survives strongly** | 4-27x reduction, both cohorts, also raises success rate |
| Component 2's bands certify continuous features at a stated confidence | **Partially — works via repair, not as a pre-filter; fails outright on features where τ≫ε at every quantile** | `tau_vs_eps_diagnostic.csv`, `dsr_c1c2` collapse |
| Component 2's identity rule fixes immutable-driven violations | **Survives** | 0% immutable violations under DSR-C1 on both cohorts (was 30%/UCI, 30%/NHANES under old) |
| Component 3 (repair) makes the method usable at a real operating point | **True on real NHANES (≤25% success at q≤0.10, 0% CVR); false on UCI (0% success at every quantile)** | Fig 3 |
| DSR-full's abstentions are certified infeasibility | **Contradicted — 15-21% are a new, repair-induced, densification-proof "unresolved" category** | E3 table above |
| Riemannian metric beats Euclidean, now under DSR | **Still not supported** (null KDE, thin zero-median effort signal) | E4 table |
| DSR's benefit is decoder-independent | **Partially — real but data-volume-dependent** | E5 table |

## What this means for the paper (honest recommendation)

The defensible claim is: *"Aligning the recourse graph's constraint checks with the
decoded representation patients actually see (Component 1) closes most of the gap
cheaply; a calibrated, conformal-style safety margin (Component 2) can certify the
remainder, but only where the decoder's actual reconstruction error is smaller than the
feature's intended tolerance, and only when applied as a post-hoc repair standard rather
than a pre-emptive filter; the resulting method (DSR-full) achieves a genuine zero-CVR
operating point on a large, dense cohort but has none on a small, sparse one, and its
abstentions are no longer cleanly interpretable as certified infeasibility."* This is an
empirical-study result, not an unqualified method win — see `ASSESSMENT.md` for why that
distinction matters for venue choice.
