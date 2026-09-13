# ASSESSMENT.md — adversarial self-assessment

Written last, after all E1–E5 results were in. The reader to assume: a reviewer at a
top ML/health venue who knows the counterfactual-recourse literature well, is tired of
"we add a VAE and call it a guarantee" papers, and will check every claim against the
numbers in `FINDINGS.md` before believing the framing in `REPORT.pdf`.

---

## Novelty assessment

### What the literature search found (web search, this session)

- **The problem DSR names — "latent-space guarantees don't transfer to output space" —
  is already explicitly named in the literature.** "Counterfactual Explanations via
  Latent Space Projection and Interpolation" (SharpShooter, arXiv:2112.00890) states
  directly that when interpolation happens in latent space, "sparsity and proximity in
  the original space cannot be guaranteed" — the exact gap this project's decoded-CVR
  finding measures. **It does not propose a fix**: no post-decoding constraint check, no
  reconstruction-error calibration, no repair step (confirmed by fetching the abstract).
  So: the *diagnosis* is not new to the literature in general terms; DSR's contribution
  can only be the *specific mechanism* that closes the gap, not the observation that the
  gap exists.

- **"Conformal counterfactuals" already exists as a term and a technique, but targets a
  different object.** "Faithful Model Explanations through Energy-Constrained Conformal
  Counterfactuals" (ECCCo, arXiv:2312.10648) combines conformal prediction with
  counterfactual generation — but conformal prediction there bounds the **classifier's
  output/decision** (a prediction set on the model's class assignment), not **per-feature
  decoder reconstruction error** used to enforce **hard tabular constraints**
  (immutability, monotonicity). These are different technical targets: ECCCo's
  guarantee is about whether the counterfactual crosses the decision boundary reliably;
  DSR's is about whether a specific recorded/decoded feature obeys a domain rule. A
  reviewer who has read ECCCo will still recognize "conformal + counterfactual" as a
  used combination and will want this distinction made explicit — it now is, here and
  in `METHOD.md` §2a.

- **Recourse verification and reachable sets (the ICLR 2024 paper this project's brief
  is built around, Kothari/Kulynych/Weng/Ustun, arXiv:2308.12820) already formalizes
  blindspot-vs-infeasibility.** Their own empirical study reports "nearly half" of
  method failures are blindspots on their datasets/methods. This project's E3 finding —
  that DSR-full introduces a *new*, repair-specific, densification-proof "unresolved"
  category that is neither certified-infeasible nor a connectivity blindspot — is a
  refinement of their taxonomy for a class of method (VAE-decoder-based repair) their
  paper does not analyze, not a contradiction of it, and not a wholly new taxonomy.

- **No paper was found that combines**: (a) a VAE-decoder-based graph/shortest-path
  recourse method, (b) conformal/empirical-quantile calibration of the decoder's own
  per-feature reconstruction error specifically (not the classifier's output) used to
  tighten hard tabular constraint thresholds, and (c) a clip-project-then-re-encode
  repair loop with a documented empirical characterization of when the re-encode step
  does and does not preserve the intended edit. Adjacent partial matches: hinge-loss
  monotonicity penalties baked into the *generation* objective (not a post-hoc repair,
  e.g. "Model-Based Counterfactual Explanations Incorporating Feature Space Attributes
  for Tabular Data," arXiv:2404.13224) and outright rejection/filtering of violating
  counterfactuals at inference time (not repair, per a strategy noted in "Feature-based
  Learning for Diverse and Privacy-Preserving Counterfactual Explanations,"
  arXiv:2209.13446).

### Honest novelty delta, stated as one paragraph

This is **not a new mechanism at the level of "conformal prediction" or "VAE-based
recourse"** — both exist, separately, in the literature already, and the diagnosis that
motivates this work (latent guarantees don't hold in output space) has also already been
named by at least one other paper. What is new is the **specific assembly and its
empirical characterization for this problem class**: nobody has (as far as this search
found) taken a VAE-decoder recourse graph, measured its ACTUAL per-feature
reconstruction error via empirical quantiles rather than assuming it away, used that
measured error to tighten hard tabular constraints, discovered that doing so as a
pre-filter is destructive because the measured error exceeds the field's original
tolerance assumptions for specific features, and built a repair mechanism whose success
this project then shows is gated by which features got a type-aware decoder head. That
last empirical chain — measure real decoder error → show naive calibration fails →
diagnose why (feature-specific architecture asymmetry) → show repair inherits the same
asymmetry — is the paper's actual contribution, and it is an *empirical characterization
of a known gap*, not a *new class of mechanism*. Both are publishable; they are not the
same claim, and this document does not conflate them.

### The strongest reviewer argument that this is not novel, and the best response

**Objection**: "This is just DiCE/FACE-style post-hoc filtering (reject violating
counterfactuals) plus a standard conformal calibration recipe, applied to a graph method
that already existed in your own prior work (`experiments_v2`/`v3`). Component 1 is a
one-line bugfix (check the thing you're going to verify against, not a different thing).
Component 2 is textbook conformal quantile calibration. Component 3 fails on 2 of 3
features tested and never succeeds at all on one of your two datasets. Where is the
method?"

**Best available response**: Component 1 being simple does not make it not a finding —
the fact that a "one-line" fix produces a 4–27x reduction is itself evidence that the
prior literature's typical practice (check constraints on encoded/original values,
report success on decoded ones) is a real, underappreciated correctness bug, not a
trivial one; that this project's OWN prior sessions (`experiments_v2`, `experiments_v3`)
made exactly this mistake for two full rounds before catching it is direct evidence it
is not obvious in practice, whatever it looks like in hindsight. The response to "Component
2/3 mostly fail" is not a defense — it is the honest empirical result, and the paper's
framing (below) should say so rather than force a method-paper narrative onto a partial
result.

---

## Publishability assessment

### Venue fit, given the ACTUAL measured results

- **IMLH @ ICML (workshop)** — **Best fit, recommend as primary.** A workshop on
  interpretable ML for healthcare is exactly the right size and rigor level for a
  paper whose contribution is "we found and partially fixed a specific correctness gap
  in an existing method, with an honest characterization of what does and does not
  generalize." Workshop reviewers expect exactly this kind of scoped, partially-negative
  empirical result; a full-length venue does not.
- **AIES / FAccT** — **Reasonable backup.** Both venues have an appetite for papers that
  interrogate a method's actual guarantees against its claimed ones (this is close to
  FACE's own AIES 2020 framing) and for negative/mixed results presented honestly. The
  E3 "certified-infeasibility does not survive under repair" finding is exactly the kind
  of result FAccT rewards.
- **ML4H / CHIL** — **Weaker fit as currently scoped.** Both expect either a strong
  clinical validation angle (this project's cohort, while now real NHANES data, still
  has the documented PCE-label and prevalent-disease-inclusion limitations from
  `experiments_v3/DATA_PROVENANCE.md`) or a clearly positive methodological result;
  DSR's 0%-success-on-UCI and 15–56%-unresolved-abstention findings would need to be
  framed very carefully to avoid reading as "the method doesn't work" to a clinically-
  focused reviewer, even though that framing would be defensible for the right audience.
- **AAAI/IJCAI applied track** — **Not recommended at this stage.** These venues expect
  either a positive, generalizable methodological advance or a much larger-scale
  empirical study; a 2-cohort, partially-negative result with 25-patient-subsampled E3
  analysis on one cohort is unlikely to clear the applied-track bar without substantially
  more experiments (see "Required next steps" below).

### Three most likely reviewer objections, and what answers this repo already has

1. **"Your repair mechanism only works for one feature (age) — is this a real method or
   an artifact of your own Phase-B architecture choice?"** *Answered*: E5's decoder-
   dependence comparison (Fig 6) and the controlled re-encode test in `METHOD.md` §3
   directly show the mechanism (dedicated, upweighted decoder heads are steerable;
   standard heads are not) — this is not hidden, it is the paper's own diagnosis.
2. **"Your 0.95-quantile 'safe' operating point never succeeds — why report it as the
   headline at all?"** *Answered, but this is a real weakness*: the report does lead
   with q=0.95 (Table 1) precisely because it is the naive, expected-safe choice, and
   states plainly that it never works, then reports the honest achievable point (q≤0.10)
   separately. A reviewer could still reasonably say the paper should lead with the
   achievable point instead — this is a framing choice to revisit before submission, not
   a data gap.
3. **"Is UCI too small to draw conclusions from? You have zero DSR-full successes at any
   confidence level there."** *Partially answered, partially not*: `FINDINGS.md`
   states the UCI null result plainly and attributes it to a specific, checked mechanism
   (calibration-set size, τ≫ε on 3 of 5 continuous features). What is **not yet
   established**: whether a larger UCI-scale calibration set (all of the original UCI
   data used for calibration, sacrificing graph size) would change this — that
   experiment was not run this session and would directly strengthen or falsify the
   "small-N-drives-it" explanation.

### Method paper or empirical study? — direct answer, not hedged

**This supports an empirical study, not a method paper**, and `FINDINGS.md`/`REPORT.pdf`
are written that way deliberately. The evidence: two of DSR's three components
(bands-as-filter, repair) either fail outright (UCI, every tau) or work only in a narrow,
cohort-dependent regime (real NHANES, q≤0.10, ≤23% success) on both datasets tested. A
method paper claims a technique that works and characterizes its cost; what this session
actually produced is closer to "we measured exactly where and why an intuitive fix
(Component 1) works great, and where a more ambitious fix (Components 2–3) meets a hard
architectural limit" — that is an empirical characterization paper. Recommending
otherwise would be exactly the class of mistake this whole project (`experiments_v2` →
`v3` → `v4`) exists to correct.

### Data-integrity / reproducibility issues a reviewer could reasonably raise

- **The calibration split is small** (36 rows, UCI) and low-quantile τ estimates
  inherit that noise — should be flagged as a limitation with an exact number (already
  is, in `REPORT.pdf`'s Limitations page) and ideally re-run with a larger calibration
  fraction as a robustness check before submission.
- **E3's per-abstention decomposition on real NHANES is subsampled** (25/seed, from a
  true count up to ~445/seed) for `dsr_full` — the true abstention *rate* is exact and
  reported; the certified/blindspot/unresolved *split* is estimated from a sample. A
  reviewer doing a careful read will notice the sample size in the appendix table and
  should be told this explicitly in the main text, not just a footnote.
- **The "old" method's numbers in this report are not the same numbers published in
  `experiments_v3`** (different train/calibration split, `METHOD.md` §4) — this is
  handled correctly (re-run under a controlled, identical-to-DSR protocol) but must be
  stated in the paper's methods section, not left for a reader to discover by diffing
  CSVs.
- **Fig 2's `dsr_c1c2` real-NHANES bar (90% CVR on 0.27% success, n≈1 patient/seed)** is
  a genuine statistical near-non-result being plotted next to real ones — the report
  already labels it as small-n, but a paper draft should consider omitting or heavily
  caveating this specific bar rather than let a reader eyeball it as comparable to the
  n=250+ bars beside it.

---

## Required next steps

**Blocking (must happen before any submission):**
1. Decide and commit to ONE headline operating point (q=0.95 "shown to fail" vs. q≤0.10
   "the one that works") for the abstract/intro framing — currently the report presents
   both, correctly, but a paper needs a single clear headline claim.
2. Re-run E3's real-NHANES decomposition without subsampling (or with a materially
   larger cap, e.g. 100/seed) if compute budget allows, since this is the paper's most
   novel finding (the "unresolved" category) and currently rests on n=25/seed samples.
3. Write the paper's abstract/intro to state the empirical-study framing explicitly
   (per "Method paper or empirical study?" above) — do not let Component 1's strong
   result carry a method-paper framing that Components 2–3's results don't support.
4. Add the SharpShooter (2112.00890) and ECCCo (2312.10648) citations and the precise
   novelty-delta paragraph above to the related-work section — a reviewer who knows this
   literature will find both papers within an hour if the paper doesn't address them
   first.

**Would strengthen (not blocking):**
5. Test whether a larger UCI calibration split changes the "0% success at every tau"
   result — directly tests this report's own explanation for why UCI fails.
6. Extend Phase B's type-aware decoder treatment to more continuous features (not just
   age) and re-run Components 2–3 — directly tests whether repair's feature-specific
   failure is fixable, which this session diagnosed but did not attempt to fix (out of
   scope this session, per the brief).
7. A held-out validation of the `τ_k(q) ≥ ε_k` diagnosis on a third dataset, to check
   whether "some features are just structurally uncalibratable at this tolerance" is a
   property of these two cohorts' decoders specifically or a more general phenomenon.
8. Consider the doubled-margin (`2τ_k`) variant named but not implemented in `METHOD.md`
   §2a, to see whether the single-margin choice materially understates the true required
   band width.
