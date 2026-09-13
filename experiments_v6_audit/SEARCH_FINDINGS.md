# SEARCH_FINDINGS.md — literature scan for recovering B1/B2-fix success-rate loss

Pure literature review. No code was written, modified, or executed in this session; no
experiment was run. Per the brief's hard rule, every idea below that would need testing
is marked as such and left untested.

## The precise problem, restated from `RECOMMENDATION.md`

With the entry-gate and unnecessary-decoding bugs fixed, `build_query_edges` already
attaches each query to **every** training node (full attachment, not k-restricted —
`experiments_v3/lib/graph_build.py`'s documented "D8" design). So the failure mode is
not "the search radius was too small" — it is that, searching the **entire** dataset
under the correct predicate, no single real patient simultaneously satisfies every
directional/immutability/monotonicity constraint from a given query in one hop. As the
number of independently-constrained features grows, this is a conjunctive-filtering
problem whose satisfying set shrinks multiplicatively — a real, structural, not merely
implementation, difficulty.

**Critical prior evidence already inside this project, weighed against every candidate
below**: `experiments_v3`, `v4`, and `v5` each independently tested graph densification
(doubling k; adding 500 VAE-prior-sampled synthetic nodes, classifier-verified
low-risk) as a way to resolve abstentions, and found **zero** blindspots resolved by
either probe, in every session, on both cohorts. "Add more candidate nodes" has already
been tried three times in this exact codebase and has never once worked. Any literature
candidate resembling that idea is judged skeptically against this internal track record.

## Candidates found and evaluated

### 1. FACE (Poyiadzi et al., AIES 2020, arXiv:1909.09369) — the paper this project already emulates

The PDF did not extract legible text via WebFetch (heavily compressed); based on the
paper's well-established public description (shortest-path recourse over a density-
weighted graph of training points, Dijkstra to the nearest feasible high-density
target) and this project's own `experiments_v2/REPO_MAP.md` citation of it: FACE does
**not** discuss what happens when a query's neighborhood is entirely infeasible — it
implicitly assumes the training graph is dense/connected enough that a path exists.
**Fit: none for this specific gap** — confirms the gap is real, doesn't address it.

### 2. Feasible Recourse Plan via Diverse Interpolation (Nguyen, Bui, Nguyen, arXiv:2302.11213, 2023)

Builds an "actionability graph" over training samples — nodes are training points,
edges are feasible actions between two instances — and finds a feasible path from the
query to a set of diverse favorable-class prototypes, balancing proximity and
diversity via a determinantal point process or quadratic program. **This is the
closest architectural match found**: same graph-over-real-points-with-constraint-edges
structure as this project's CALG. **Fit assessment**: it does NOT claim to solve the
zero-feasible-neighbor case — it assumes the actionability graph has reasonable
coverage and focuses on prototype diversity/proximity trade-offs given that a feasible
path already exists. Reading it would still be worthwhile for its actionability-graph
construction details, which may differ subtly from this project's and be more
robust — but this is a methodology comparison, not a fix for entry infeasibility.
**Effort to prototype the parts that ARE relevant (their graph construction)**:
1-2 days (would need their actionability-graph definition compared line-by-line against
`src/graph/calg.py`'s). **Risk**: may turn out architecturally equivalent, yielding no
improvement, consistent with prior densification attempts.

### 3. DAACE — Data-Agnostic Actionable Counterfactual Explanations (Valero-Leal, Larrañaga, Bielza, arXiv:2508.02634, Aug 2025, IEEE)

Replaces the discrete graph-over-training-points entirely with a **continuous density
landscape** (negative log-likelihood of a learned density estimator) as the cost
surface for path planning (NSGA-II), generating intermediate waypoints that need not
land on any real training point. Motivation is different from this project's (avoiding
training-data access for privacy, not recovering infeasible entries) but the mechanism
— synthetic, continuous waypoints instead of requiring a jump to an existing real
patient — is conceptually the one genuinely different idea in this search relative to
what this project has already tried. **Fit assessment**: moderate. It sidesteps the
"must match an existing patient in one hop" bottleneck that B1/B2's fix exposed, but it
inherits a real risk this project has already measured: any synthetic waypoint still
has to be **decoded** to a plausible, verifiable feature vector, and `experiments_v3`
Phase B / `v5`'s own diagnosis found this decoder struggles precisely with the large,
multi-feature, simultaneous deviations this exact bottleneck requires. **Effort**:
days, not hours — requires training a separate density estimator and a path-planning
loop (NSGA-II or equivalent), a materially larger undertaking than B1/B2's ~10-line
fix. **Not recommended as a first prototype** given the size of the lift relative to
its unconfirmed payoff on this project's specific data.

### 4. FGCE — Feasible Group Counterfactual Explanations (Fragkathoulas, Papanikou, Pitoura, Terzi, arXiv:2410.22591, 2024)

Builds a "density-weighted graph that encapsulates feasibility constraints" (same
family as FACE/this project) but reframes the OUTPUT as a small set of shared
counterfactuals covering a cohort, via greedy maximization or MIP. Feasibility is
still determined by "path existence in the feasibility graph" — **it does not expand
reachability, it selects efficient representative targets among instances that are
already reachable.** **Fit assessment: low.** Solves a different problem (auditing/
interpretability efficiency across a group), not recovery of individually-unreachable
queries. Not recommended.

### 5. Navigating Explanatory Multiverse Through Counterfactual Path Geometry (Sokol, Small, Xuan; ECML-PKDD 2025 journal track; arXiv:2306.02786)

Formalizes the **space of all valid counterfactual paths** between a factual point and
its counterfactuals as a "multiverse," with a graph-based implementation and a metric
("opportunity potential") for comparing path geometries — letting a user pick among
multiple valid journeys, not just the single shortest one. **Fit assessment: low-
moderate, for a specific and important reason**: per its own abstract, it reasons about
paths that **already exist** connecting factual and counterfactual — it is a
path-selection/diversity tool, not a reachability-expansion tool. Since this project's
query attachment is already full (every node considered), a smarter way to choose
among multiple valid paths would not help queries that currently have **zero** valid
paths at all. Would be relevant if this project's problem were "we find a path but it's
not a good one" — it is not; the problem is "we sometimes find no path." **Not
recommended for the specific gap**, though worth keeping in mind if a future session's
problem shifts toward path quality among survivors rather than survivor count.

### 6. Longitudinal Counterfactuals: Constraints and Opportunities (Asemota, Hooker, arXiv:2403.00105, 2024)

Uses real **longitudinal (repeated-measures) data** — actual observed within-person
changes over time — to ground plausibility for exactly this project's constraint types
(non-decreasing age, directional risk-factor trends), instead of requiring a match to
a different existing patient. Conceptually this is an elegant fit for the conjunctive-
constraint bottleneck: instead of needing ONE other patient to simultaneously satisfy
every direction, use typical observed per-visit deltas as a template. **Fit
assessment: conceptually strong, practically blocked.** Both of this project's cohorts
(UCI Heart, single cross-sectional snapshot per patient; real NHANES, also
cross-sectional per `experiments_v3/DATA_PROVENANCE.md`) have **no repeat-visit data at
all** — this technique requires a data modality neither cohort has. **Not viable
without a different or additional dataset** (e.g. NHANES has other cycles that could in
principle be linked longitudinally for the same individuals via restricted-use data,
but that is a data-acquisition project of its own, not a code fix). Noted for the
record as the best-matched IDEA in this search, with the clearest reason it cannot be
adopted here.

### Rejected outright (wrong technical object, confirmed by reading, not guessed from title)

- **ACE: Adapting Sampling for Counterfactual Explanations** (Guerrero, Rojas,
  arXiv:2509.26322, 2025) — Bayesian-optimization sample-efficiency for black-box model
  **query count**, unrelated to graph reachability.
- **Adaptive Entry Point Selection for Graph-based ANN Search** (Oguri, Matsui,
  arXiv:2402.04713, 2024) — vector-search retrieval speed/accuracy, not constraint
  feasibility; confirmed by direct abstract quote this is a different "entry point"
  concept entirely (traversal starting node for search EFFICIENCY, not a
  constraint-satisfying attachment point).
- **Explaining k-Nearest Neighbors: Abductive and Counterfactual Explanations**
  (Barceló et al., arXiv:2501.06078, 2025) — explains a kNN **classifier's own**
  predictions; this project's classifier is an MLP and the kNN structure here is a
  recourse-path graph, not the model being explained. Different technical object,
  confirmed by direct abstract read.

## What was NOT found

No paper directly addresses "a graph-based recourse method has already searched its
entire training set under a correct hard-constraint predicate and found no valid
single-hop entry for a query — what next?" as its own stated problem. The closest
matches either (a) assume reasonable graph coverage and focus on path/target quality
once a path exists (FACE, Diverse Interpolation, the Multiverse paper, FGCE), or (b)
sidestep the discrete-graph formulation entirely with continuous density-landscape
planning (DAACE) at a real implementation cost and a real, previously-measured risk
(decoder unreliability on large synthetic deviations). Nothing published after
`experiments_v5/PRIOR_ART_COUNTERFLOWNET.md` was written changes this picture.

## Ranked shortlist

1. **Read `Feasible Recourse Plan via Diverse Interpolation`'s actionability-graph
   construction in detail and diff it against `src/graph/calg.py`** — cheapest possible
   next step (reading, not building), might surface a genuinely different edge/
   constraint definition worth adopting even though the paper doesn't solve the
   zero-feasible-neighbor case directly.
2. **DAACE-style continuous waypoint planning, specifically for the query's entry step
   only** (not the whole path) — the one idea in this search that structurally differs
   from what this project has already tried three times and found ineffective. Real
   implementation cost (days) and a real, already-measured risk (decoder unreliability
   on large deviations) that must be weighed before committing.
3. Everything else in this list is not recommended as a near-term prototype target.

## Recommendation

**Prototype #1 first, and only if it reveals a genuine gap, escalate to #2.** The
honest, load-bearing finding of this search is that the literature does not offer a
ready-made fix for this project's specific, now precisely-characterized gap — three of
six candidates explicitly assume the problem away (feasible paths already exist), one
requires data this project's cohorts do not have, and the two structurally novel ideas
(DAACE's continuous planning; this project's own untested "synthetic entry waypoint"
variant inspired by it) both inherit the SAME decoder-reliability risk that sank
`experiments_v4`'s DSR and made `experiments_v5`'s EGD impractical on real NHANES. A
following session should treat "no clean literature fix exists" as itself the finding,
and if it chooses to prototype anyway, do so with the explicit expectation — grounded
in this project's own three prior densification failures — that success is not
assured.
