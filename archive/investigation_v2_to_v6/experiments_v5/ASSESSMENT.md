# ASSESSMENT.md — adversarial self-assessment (v5)

Written last, after Phases 0-4's results were in, updating `experiments_v4/ASSESSMENT.md`
explicitly rather than restating it. The reader to assume: the same hostile,
literature-literate reviewer as `v4`'s assessment, who has now also read
CounterFlowNet and will ask why this session's architecture is preferable to it.

---

## Novelty assessment — updated against Phase 0's close read

`v4/ASSESSMENT.md` concluded DSR was "not a new mechanism... an empirical
characterization of a known gap." Phase 0 of this session (`PRIOR_ART_COUNTERFLOWNET.md`,
`NOVELTY_CLAIM.md`) checked the brief's stronger claim — "exact-guarantee generation ...
paired with exhaustive provable certification of infeasibility" — against the five
closest papers found, BEFORE any code was written, and the claim was narrowed on the
spot, not after the fact:

- **Exact/architectural constraint satisfaction is not new.** CounterFlowNet
  (arXiv:2602.17244) already achieves it, for BOTH immutability and monotonicity, via
  discrete action masking in a GFlowNet with no VAE at all. FCEGAN (arXiv:2502.17613)
  achieves it for immutability alone, via post-hoc value resetting, on a GAN. EGD's
  masked-identity-plus-signed-residual construction is a THIRD independent arrival at
  "make the violation impossible by construction," not a first.
- **What IS confirmed absent from all five papers checked**: any distinction between
  true infeasibility and method failure. CounterFlowNet's own evaluation protocol
  "assumes valid CFs always exist for test instances" (direct quote,
  `PRIOR_ART_COUNTERFLOWNET.md`). This is the one part of the session's premise that
  held up to a literature check, and it is the basis for proceeding past Phase 0.

**Given what Phases 1-4 actually produced, does the recalibrated claim still hold as a
publishable contribution?** Partially, and less cleanly than hoped. The certified-
infeasibility pairing is real and was executed (Phase 3) — but its OWN result
(§ "Publishability," below) is that 40-42% of EGD's abstentions are "unresolved," not
certified — meaning the pairing produces a MORE HONEST decomposition of failure, not a
cleaner overall guarantee. The novelty is in the ANALYSIS, not in a success EGD can
claim. A reviewer would be right to note that this session's actual contribution is
closer to "we built the exact-guarantee mechanism CounterFlowNet's masking already
achieves, in a harder-to-train architecture, and used it to show that even a perfect
constraint guarantee does not make certified infeasibility clean" — a real, but much
more modest and much more negative-leaning finding than the brief's framing anticipated.

### The strongest reviewer argument against this session specifically, and the honest response

**Objection**: "You built a strictly worse system than CounterFlowNet (which has near-
99% validity on its own benchmarks) to prove a point about infeasibility that could have
been made by pairing YOUR EXISTING v3/v4 certified-infeasibility machinery with
CounterFlowNet's masking mechanism directly, without inventing a new, harder-to-train
decoder architecture that fails outright on your denser, more realistic cohort."

**Honest response**: this is a fair objection, and this document does not have a strong
rebuttal to it. The VAE-latent-graph architecture is this project's own five-session
throughline (Riemannian pullback metric, k-NN shortest-path search), and EGD's
contribution is specific to keeping that architecture while fixing its constraint
guarantee — a legitimate scoping choice for a project about THIS architecture, but not
a claim that EGD is the best available way to get an exact guarantee in general. A
reviewer asking "why not just adopt CounterFlowNet's masking directly" deserves the
answer "because this project is about characterizing THIS architecture's specific
failure modes, not about finding the best architecture for recourse in general" — which
is a defensible SCOPE statement, not a claim of superiority.

---

## Publishability assessment, given what was ACTUALLY measured (not hoped for)

### Venue fit

- **IMLH @ ICML (workshop) — still the best fit, unchanged from v4.** A session whose
  headline finding is "the guarantee works, the system built around it does not" is
  exactly workshop-scoped: rigorous, honest, narrow.
- **AIES / FAccT — reasonable backup, STRENGTHENED by this session's results.** The
  "certified infeasibility does not survive under repair" finding from `v4` NOW HAS A
  SECOND, INDEPENDENT CONFIRMATION under a completely different mechanism (EGD): both a
  statistical patch (DSR) and an architectural guarantee (EGD) produce a nontrivial
  "unresolved" category that the field's existing recourse-verification tools were not
  built to detect. Two independent methods hitting the same qualification is a STRONGER,
  more general claim than either session alone, and is worth foregrounding.
- **ML4H / CHIL — weaker fit, WORSENED from v4's assessment.** v4's DSR at least had a
  real, if narrow, working operating point (23% success at 0% CVR on real NHANES). EGD's
  real-NHANES success rate (1.0%) is low enough that a clinically-focused reviewer would
  reasonably read this as "the method does not work on the realistic cohort," full stop.
- **AAAI/IJCAI applied — not recommended, unchanged from v4.**

### Three most likely reviewer objections, and what this repo currently answers

1. **"EGD is worse than your OWN prior session's DSR on the harder cohort (1.0% vs.
   23.4% success on real NHANES) — why does this session's method exist?"** *Answered
   honestly, not defended*: EGD is not proposed as an improvement over DSR; it answers a
   DIFFERENT question (can an exact, non-statistical guarantee be built at all in this
   architecture, and does it change the infeasibility story). The data show yes to the
   first, and "not cleanly" to the second. This should be stated as directly in any
   paper draft as it is here.
2. **"Is the Riemannian-vs-Euclidean reversal a real finding or an artifact of a small,
   seed-variable sample?"** *Answered with a verified mechanism*: path-length
   distributions were checked directly (candidates' path lengths differ systematically
   by metric, `METHOD.md` §2, FINDINGS.md), not just outcome statistics — this is a
   causally-explained result, not a correlation. Not yet done: a controlled ablation
   that FORCES equal path lengths under both metrics to isolate the effect further —
   listed below as a strengthening step.
3. **"Your training-regime search (self/k-NN/random pairing) reads as ad hoc — how do
   you know 'random' is actually the best regime, not just the best of three guesses?"**
   *Partially answered*: the three regimes tested span a principled range (identical
   pairs → local pairs → global pairs) and the failure mode of each was diagnosed
   mechanistically, not just observed. What is NOT established: a fourth, curriculum-
   style regime (mixing local and global pairs, or weighting by risk-gap size) was not
   tried and might do meaningfully better, particularly on real NHANES — listed below.

### Method paper or empirical study? — direct, unhedged verdict

**This is an empirical study reporting a substantially negative result for EGD as a
practical method, with two genuinely positive, generalizable side-findings.** Not
hedged either direction: EGD does not achieve competitive success, plausibility, or
effort relative to the old method on either cohort, and is decisively worse than the
old method and than `v4`'s own DSR on the harder, more realistic cohort. Recommending a
method-paper framing here — "we propose EGD" — would be exactly the class of mistake
this five-session project exists to correct, and this document declines to make it. The
two findings that ARE strong enough to lead a paper are (a) the metric-choice reversal
(a genuine, mechanistically-explained, generalizable insight about how sequential,
source-conditioned decoders interact with graph search metrics) and (b) the SECOND
independent confirmation, via a completely different mechanism, that certified
infeasibility does not survive contact with any mechanism that can fail for reasons
other than "no answer exists" — this is the project's most durable, cross-session
finding, and this session strengthens it rather than EGD strengthening the project.

### Data-integrity / reproducibility issues a reviewer could reasonably raise

- **UCI's EGD success rate has std=37 points on a mean of 31%** — any five-seed mean
  reported without the full per-seed breakdown (Fig 4) would be actively misleading; the
  report includes it, but a paper draft compressing to one table row must not drop the
  std or the seed range.
- **The wide comparison table mixes numbers from three different sessions under three
  different protocols** (this session's fresh EGD/old rerun; v4's DSR-full under its
  3-way calibration split; v3's DiCE/FACE/REVISE under the original 80/20 split) — none
  of these are validly paired with each other statistically, and the table says so, but
  a reader skimming only the numbers could mistake it for a single controlled
  comparison. A paper draft must keep these visually and textually separated, not
  merged into one ranked list.
- **Real NHANES's blindspot decomposition is a 60-patient sample of a ~97-100% true
  abstention population** (up to ~445 patients) — the certified/unresolved SPLIT
  percentages carry real sampling uncertainty the current std bars only partially
  capture (std is over 5 SEEDS, each already a 60-patient sample, not over the full
  abstention population within a seed).
- **CounterFlowNet is discussed but not benchmarked** — any submitted paper WILL be
  asked "why not compare directly," and "out of session budget" is a true but weak
  answer a reviewer may not accept; this is listed as blocking below.

---

## Required next steps

**Blocking:**
1. Either implement a minimal CounterFlowNet-style masking baseline for a genuine
   head-to-head on THIS project's cohorts, or explicitly reframe the paper to NOT claim
   comparison to CounterFlowNet at all (currently the report discusses it qualitatively,
   which invites exactly the reviewer objection above without fully answering it).
2. Decide and state explicitly, in the paper's own framing (not just this assessment),
   that EGD is NOT proposed as a replacement for DSR or the old method — the abstract
   must not imply a method contribution the data don't support.
3. Re-verify the real-NHANES blindspot decomposition without the 60-patient cap if
   compute budget allows, given how central "40% unresolved, 0% blindspot-resolved" is
   to this session's strongest claim.
4. Run the equal-path-length ablation (Riemannian vs. Euclidean, both restricted to the
   same hop count) to strengthen or falsify the metric-reversal's causal mechanism
   beyond the current correlational-with-diagnosis evidence.

**Would strengthen:**
5. Try a curriculum training regime (mixing near and far source/target pairs, or
   weighting pairs by risk gap) to test whether real NHANES's near-total failure is
   fixable with a smarter training signal, not just a different network.
6. Extend Phase-B-style type-aware treatment (dedicated, upweighted heads) to MORE
   continuous features, not just age, and re-test whether that closes real NHANES's
   residual-magnitude gap — directly testable given `METHOD.md` §4's diagnosis that
   age's dedicated head is what makes it steerable.
7. A larger, independent replication of the UCI seed-variance finding (more than 5
   seeds) to characterize whether 31%±37% is a stable distributional estimate or itself
   noisy at n=5.
