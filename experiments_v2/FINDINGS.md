# FINDINGS.md — plain-language answers to the three research questions

All numbers below come from code executed in this session (see `experiments_v2/results/`
for raw and aggregated CSVs, `run_metadata.json` for the git SHA/seeds/config of the run
that produced them). Git SHA for this whole study: `539d87b795b78bb8f9446f99b0fabeded2c84841`.
5 seeds (0–4), both datasets, full 12-cell grid — no seed-count reduction was needed.

---

## RQ1: Does hard pruning prevent constraint violations that a Riemannian metric with soft penalties still commits?

**Short answer: No — not once you check where it actually matters (after decoding).**

The graph's hard-mode pruning is exact and true on the *original, encoded* feature values:
its original-space CVR is 0.0% by construction, for the trivial reason that a violating
edge is never added to the graph. That part of the paper's claim is correct and always will
be, structurally.

But the patient never sees the graph's internal book-keeping — they see the VAE's *decoded*
reconstruction of each step. Re-checking every constraint (immutables, non-decreasing age,
directional BP/cholesterol/etc.) on the decoded path nodes (Phase 3) gives:

| Dataset | R+hard decoded CVR (source→terminal) | R+soft(λ=1) decoded CVR | R+soft(λ=10) decoded CVR |
|---|---:|---:|---:|
| UCI    | 83.6% ± 26.9 | 90.0% ± 19.9 | 90.0% ± 19.9 |
| NHANES | 78.1% ± 10.4 | 61.5% ± 18.7 | 62.1% ± 18.4 |

Hard pruning does **not** clearly beat soft penalties in decoded space: on UCI it is
actually slightly *worse* than every soft λ tested, and on NHANES soft penalties at λ≥1 are
markedly *better* (Fig D). Increasing λ from 0.1 to 100 does not converge decoded CVR back
down toward the hard-mode structural guarantee on either dataset — the curve is essentially
flat past λ=1 (Fig D).

**Root cause, verified directly**: the VAE's reconstruction fidelity is poor relative to the
constraint tolerances. On the train set, decoder reconstruction mean absolute error is
≈5.8–6.4 years for `age` and ≈0.47–0.49 for the binary `sex` feature — i.e. barely better
than a coin flip at recovering which class a patient belongs to (full table:
`experiments_v2/results/vae_reconstruction_error.csv`). Three concrete UCI examples (seed 0,
R+hard, full feature deltas in `REPORT.pdf` §Decoded-Space Verification) show decoded age
*decreasing* by 2.6–3.6 years and decoded cholesterol *increasing* by 2.8–5.5 mg/dL along a
path the graph itself certified as monotone-age and cholesterol-decreasing.

**Verdict on the paper's claim**: "0% CVR" needs to be reworded to "0% CVR *on the graph's
internal, encoded representation of the path*" — it is not a guarantee about what the
decoded, patient-facing recourse actually does. This is the single most important
correction this study makes to the original paper's headline claim.

---

## RQ2: Does the Riemannian metric change which safe path is chosen, and are those paths measurably more plausible / lower-effort than Euclidean-weighted ones under identical hard constraints?

**Short answer: Mostly no, on this VAE/graph.**

Success rate is *structurally identical* between R+hard and E+hard on both datasets (UCI
71.8%±9.6 for both; NHANES 100.0%±0.0 for both) — this is mathematically guaranteed, not a
coincidence, because hard-mode edge *existence* depends only on the violation predicate,
which is metric-independent; the metric only affects which surviving edge is cheapest.

Paired Wilcoxon signed-rank tests (patients where both R+hard and E+hard succeeded, paired
by seed+patient) found:

| Dataset | Metric | n pairs | p-value | Median difference (R − E) |
|---|---|---:|---:|---:|
| UCI | KDE | 81 | 0.234 | 0.000 |
| UCI | Effort | 81 | 0.501 | 0.000 |
| NHANES | KDE | 675 | 0.054 | 0.000 |
| NHANES | Effort | 675 | 0.103 | 0.000 |

None of these are significant at α=0.05 (NHANES KDE is closest, at p=0.054), and the
**median difference is exactly zero in every case** — meaning for the large majority of
patients, R+hard and E+hard choose the *same* terminal node and effectively the same path.
Fig C visualizes this directly: for three example UCI patients, the R+hard and E+hard paths
are visually superimposed in the top-2 PCA plane of the latent space.

**Verdict on the paper's claim**: the specific claim that the Riemannian pullback metric
produces measurably more plausible or lower-effort recourse *than Euclidean latent distance,
under identical hard constraints*, is **not supported** by this experiment on either cohort.
This does not mean the Riemannian metric computation is wrong or pointless in general — only
that, as implemented here (a locally-linearized midpoint metric on a 4-dimensional latent
space from a small, weakly-regularized VAE), it does not diverge enough from Euclidean
distance to change outcomes on these two cohorts. This claim needs to be reworded from "the
Riemannian metric improves recourse quality" to "the Riemannian metric is available as an
option, but did not measurably outperform Euclidean distance in our controlled tests."

---

## RQ3: When CARE-LG abstains on sparse data, is that certified infeasibility or a blindspot?

**Short answer: Certified infeasibility, cleanly, on every case observed.**

On UCI (5 seeds), R+hard abstained on a mean of 6.4 high-risk test patients per seed (out of
a mean 22.6 high-risk patients per seed — i.e. abstention rate itself varies 13–36% by seed,
another symptom of the small-cohort seed sensitivity documented under RQ-adjacent findings
below). For **100% of these abstentions, in every one of the 5 seeds**, an exhaustive
feature-space check found *no* low-risk train patient anywhere in the 237-patient cohort
that satisfied the pairwise source→target constraints (immutables unchanged, age not
decreased, directional rules respected) — i.e. every abstention is a **certified
infeasibility**, not a search failure. Two densification probes (doubling the graph's k;
adding 500 synthetic nodes sampled from the VAE prior and verified low-risk by the
classifier) were implemented and ran correctly, but found zero blindspots to resolve,
because there were none among the abstentions to begin with.

On NHANES, R+hard produced **zero abstentions in all 5 seeds** — there was nothing to
decompose. This itself contradicts the paper's claimed ~1% NHANES failure rate; under our
corrected query-attachment protocol (REPO_MAP.md D8), the dense cohort simply always found a
path.

**Verdict on the paper's claim / the ICLR-2024 "gatekeeper" framing**: this is the
**strongest, most defensible finding in this study**. CARE-LG's abstentions on UCI are true
"no recourse exists" signals, not method blindspots — a direct, positive validation of the
recourse-verification distinction the brief asks about. If anything, this result is *more*
striking than what the original paper claimed, since it is backed by an exhaustive
(non-heuristic) feasibility check rather than a proxy metric.

---

## Summary: what survives, what needs rewording, what is contradicted

| Original claim | Status | Evidence |
|---|---|---|
| "0% Constraint Violations" (graph-internal) | **Survives**, but is a narrower claim than presented | RQ1, Table 1 `decoded CVR` column |
| "0% CVR" as a patient-facing safety guarantee | **Contradicted** | RQ1: 78–92% decoded-space CVR across every hard AND soft cell |
| Riemannian metric improves recourse quality over Euclidean | **Not supported** in controlled comparison | RQ2 paired tests, Fig C |
| "Ultra-fast" recourse (~0.0001–0.0005s) | **Survives** | All grid cells: 0.001–0.005s per patient, competitive with or faster than every baseline |
| UCI success rate ≈47.6% (single point estimate) | **Needs rewording** — unstable point estimate | Fig E: 63.6–87.0% across 5 seeds (mean 71.8±9.6); reproduction check shows the same seed=42 retrained twice gives 47.6% vs 63.6% |
| NHANES ≈99% success / ~1% failure | **Contradicted (in the good direction)**: 100% success, 0 failures, across all 5 seeds under the corrected pipeline | Phase 2 grid, Phase 4 blindspot summary |
| CARE-LG's abstentions represent true infeasibility, not blindspots | **Survives strongly** | RQ3: 100% certified-infeasible across 5 seeds, 0 blindspots found |
| Dataset is "CDC NHANES" | **Contradicted** | REPO_MAP.md D1: fully synthetic, parametric data generator |
| DiCE/FACE/REVISE baselines | **REVISE absent**, not implemented anywhere in the codebase | REPO_MAP.md component table; CHANGES.md |

**Overall recommendation for a revised paper**: reposition around RQ3 (the abstention/
recourse-verification story) as the primary contribution, since it is the only one of the
three mechanisms that held up to adversarial re-checking in this study. RQ1 and RQ2 should
be reframed as negative/mixed empirical results with the VAE reconstruction-error root cause
named explicitly, since that is a fixable, concrete next step (a higher-capacity or
categorical-aware decoder) rather than a fundamental flaw in the graph-based recourse idea.
