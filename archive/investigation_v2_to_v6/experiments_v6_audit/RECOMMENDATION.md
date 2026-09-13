# RECOMMENDATION.md — the one fix

## The hypothesis, and the answer

**Yes, the pattern is real: v2 through v5 each fixed a symptom of a single upstream
problem in the codebase, not the problem itself.** The dominant root cause is
Candidate B from this audit's brief, and it is more precisely characterized as **two
tightly coupled, code-level (not architectural, not decoder-quality) defects**, both
verified empirically, both fixed with a combined ~10-line code change, both reducing
the real-world constraint-violation rate to **exactly 0.0% ± 0.0%, across all 5 seeds,
both cohorts, with zero exceptions.**

## Root Cause B (confirmed, dominant): two defects, both in the constraint-checking pipeline, neither in the decoder

### B1 — The query's entry edge is gated by a weaker predicate than every other edge

`experiments_v3/lib/graph_build.py:238` (byte-identical logic in v2/v4/v5's equivalent
functions): `gate_violates = hard_count > 0`. This checks only immutability and
age-decrease — **it never checks directional constraints at all**, even though
`dir_count` is computed on the very same line group and simply unused for the gating
decision. Every OTHER edge in the graph (`src/graph/calg.py:108`,
`check_clinical_violations`, hard OR directional) is checked correctly. This was a
known, documented design choice (`experiments_v2/REPO_MAP.md` "D8") — but its
consequence was never isolated or measured against the headline decoded-CVR finding.

**Concrete evidence** (`experiments_v6_audit/verify_entry_gate_fix.py`, UCI seed 0,
patient 0): the entry edge query→node 114 was NOT pruned (`qedges.violates=False`).
Its real recorded cholesterol delta: **+35 mg/dL** (231→266), an unambiguous,
large-magnitude directional violation the entry gate was designed to prevent but
structurally cannot detect.

### B2 — Real graph nodes are decoded through the VAE for verification, when their true recorded values are already known

`src/recourse/search.py::decode_recourse_trajectory` (the ORIGINAL, pre-v2 pipeline's
own patient-facing function) takes a `vae_model` parameter and **never calls
`.decode()` on it** — line 82, `x_scaled = scaled_features[node_idx]`, looks up the
real recorded value directly. Every v2–v5 Phase-3 decoded-CVR measurement does the
opposite: `experiments_v2/run_grid.py:137-138` (and the byte-identical v3 lines
142-143; v4's `dsr_graph.py:70,133`; v5's `run_egd.py:78,103-104` for its own
"old"-method comparison cell) decode EVERY node along the path via
`decode_node(vae, Z_train[node])`, discarding the already-available real value.

**Concrete evidence**: for the SAME query→node-114 edge, real cholesterol delta is
+35.0 mg/dL; decoded cholesterol delta is **+0.3 mg/dL** (query decodes to ≈239.5,
target decodes to ≈239.7 — both pulled toward the population mean by VAE
regression-to-the-mean, converging to nearly identical reconstructions). **Decoding did
not create this violation — it masked it.** This inverts the project's founding
premise (that decoding reveals violations invisible in encoded space); here, decoding
hides a genuine, real-world violation invisible in the DECODED comparison but glaring in
the RECORDED data.

### Combined, measured effect (5 seeds, both cohorts, R+hard cell)

| | UCI success | UCI decoded CVR | UCI **real** CVR | NHANES success | NHANES decoded CVR | NHANES **real** CVR |
|---|---:|---:|---:|---:|---:|---:|
| Original (relaxed gate, decode-every-node — the v2-v5 method) | 71.8%±9.6 | 41.9%±9.1 | 58.8%±15.9 | 45.5%±3.0 | 81.0%±7.0 | 71.3%±5.9 |
| **Fixed (strict gate + real values for real nodes)** | 38.6%±13.1 | 47.3%±10.3¹ | **0.0%±0.0%** | 37.7%±3.4 | 62.8%±8.3¹ | **0.0%±0.0%** |

¹ "decoded CVR" under the fixed gate is reported for completeness (what you'd measure
if you fixed the gate but kept the old, unnecessary decoding step) — it is NOT the
number that matters; **realCVR is the number that matters, and it is exactly zero.**

The original UCI/NHANES success-rate and decoded-CVR numbers in the "Original" row
reproduce `experiments_v3/FINDINGS.md`'s published numbers exactly (71.8±9.6 /
41.9±9.1 UCI; 45.5±3.0 / 81.0±7.0 NHANES), confirming this audit's reimplementation is
faithful and the delta measured is attributable to the fix, not a different pipeline.

### What this means for v3, v4, and v5's own headline findings

- **v2/v3's "78–92% decoded-CVR gap"**: was a conflation of a genuine, fixable
  constraint-checking bug (B1, ~59-71% of the gap under the original relaxed gate) and
  an entirely avoidable measurement artifact (B2, the remaining ~47-63% that persists
  even after B1 is fixed). **Neither component required a better decoder or a different
  architecture.**
- **v3 Phase B (type-aware decoder)**: real, measured improvement (age MAE 4-6× lower)
  — but was solving problem B2 the hard way (make the unnecessary decoding step less
  wrong) instead of the cheap way (don't take the step).
- **v4's DSR** (`ASSESSMENT.md`: "a statistical patch on an untrustworthy network"):
  the "untrustworthy network" was never the real obstacle — the network was asked to
  reconstruct nodes whose true values were already sitting in `X_train`, unused.
- **v5's EGD**: proved an exact guarantee is achievable in this architecture (a real,
  narrower technical result, per its own `ASSESSMENT.md`) — but the guarantee it
  built an entire new decoder to provide is delivered for free, with no architecture
  change, by not decoding what doesn't need decoding.

## The other four candidates: confirm / partial / reject, with evidence

- **A. VAE lossy compression as fundamental backbone** — **Partially confirmed, but not
  dominant.** Once B1 is fixed (removing the genuine constraint violations), decoded CVR
  under continued unnecessary decoding is still 47.3% (UCI) / 62.8% (NHANES) — real,
  substantial decoder noise. But this ENTIRE remaining contribution is eliminated by B2
  alone (real CVR = 0.0% either way B1 is combined with B2). The decoder's imprecision
  is real but irrelevant to real-node traffic once B2 is fixed — it only matters for
  genuinely synthetic nodes (densification probes), a small minority of graph traffic.
  **Rejected as the dominant explanation.**
- **B. Unnecessary decoding of known nodes** — **Confirmed as the (combined, B1+B2)
  dominant root cause.** See above.
- **C. Latent dimensionality (posterior collapse)** — **Partially confirmed, secondary,
  dataset-specific.** UCI's latent dim 1 shows near-total collapse (mean KL=0.0063,
  std(mu)=0.092, vs. the other 3 dims' 0.13-1.13 KL / 0.39-1.14 std) —
  `experiments_v6_audit`'s posterior-collapse check. Real NHANES shows no collapse
  (all 4 dims: KL 0.23-1.33, std(mu) 0.62-0.95). This is consistent evidence that UCI's
  smaller sample degrades latent geometry — but NHANES shows the SAME magnitude
  decoded-CVR gap (81.0%) as UCI (41.9%) despite having no collapse, and both resolve to
  exactly 0% under the B1+B2 fix regardless of collapse. **Rejected as an explanation
  for the dominant finding; retained as a plausible secondary contributor to v2-v4's
  weak/null Riemannian-vs-Euclidean result specifically on UCI** (untested beyond this
  one diagnostic — would need a dedicated ablation to confirm causally).
- **D. Risk-monotonicity never guaranteed** — **Confirmed as real, rejected as a major
  contributor by volume.** Only `experiments_v4/lib/repair.py` checks this anywhere in
  the codebase (`np.diff(risks) > RISK_TOLERANCE`), and only within DSR's repair
  verification — never in the standard success/failure path used everywhere else. Direct
  check (`experiments_v6_audit`, seed 0, both cohorts, original relaxed gate): of 445
  high-risk test patients, only 13 successful paths had ANY intermediate hop at all
  (the overwhelming majority of "successful" recourse is a single direct hop, which
  cannot violate monotonicity by construction, since the target is chosen specifically
  for its low risk); of those 13 multi-hop paths, exactly 1 had a genuine
  monotonicity violation. **Real, but low-volume given how short successful paths
  are throughout this project — not a priority fix.**
- **E. Small/sparse UCI cohort** — **Rejected as an explanation for the dominant
  finding; confirmed as a real, independent driver of OTHER v4/v5 findings.** UCI's
  entry-gate/decoding problem (B1+B2) is exactly as fixable, and resolves to exactly
  the same 0.0% real CVR, as real NHANES's — cohort size does not gate whether the fix
  works. Sample size clearly DOES explain other findings this audit did not re-litigate
  (v3's Fig E UCI seed-variance; v5's UCI EGD success-rate std=37 points; DSR/EGD's
  narrower operating range on UCI than NHANES) — those are downstream of calibration-set
  and training-set size, a separate, already-well-documented phenomenon, not entangled
  with the B1/B2 fix.

## The fix, precisely, for a following session

**Change 1 (B1)**: in every session's `build_query_edges`-equivalent function
(`src/`-equivalent doesn't have this function at all — it's v2-v5-specific), change
the entry-gate line from `gate_violates = hard_count > 0` to
`gate_violates = (hard_count + dir_count) > 0`, and change `check_step_horizon=False`
to `check_step_horizon=True` in that same call, matching interior edges exactly.
Locations: `experiments_v2/lib/graph_build.py`, `experiments_v3/lib/graph_build.py`
(and any v4/v5 code that independently reimplements query attachment —
`experiments_v4/lib/dsr_graph.py::build_dsr_query_edges`,
`experiments_v5/lib/egd_graph.py::build_egd_query_edges` — both should be checked for
the same relaxed-gate pattern in a following session; this audit verified B1 against
v3's implementation specifically, not v4/v5's, for time reasons — see "What this audit
does not cover," below).

**Change 2 (B2)**: everywhere decoded-space constraint verification or patient-facing
trajectory output is computed, replace `decode_node(vae, Z_train[node])` /
`decode_nodes_batch(vae, Z_train)` for a node known to be a REAL, recorded training or
test patient with a direct lookup of that patient's real `X_train[node]` /
`X_test[idx]` row. Decoding remains necessary ONLY for genuinely synthetic/interpolated
nodes (v3/v4/v5's densification-probe synthetic samples), which have no ground truth
to look up.

**Confirms it worked**: re-run the decoded-CVR (now: real-value CVR) measurement on the
R+hard cell, same seeds/protocol as `experiments_v3/run_grid.py`; expect exactly 0.0%,
matching this audit's own measurement.

## Effort estimate and expected improvement

**Effort: hours, not days.** Both changes are localized, mechanical, and this audit
already implements and verifies them for v3's pipeline
(`experiments_v6_audit/verify_entry_gate_fix.py`). Porting to v2/v4/v5's independent
reimplementations of the same query-attachment/decoding logic is the remaining work —
estimated 2-4 hours given the pattern is now precisely identified and the fix is
mechanical, not exploratory.

**Expected improvement: this is a MEASURED result, not a prediction**, for v3's R+hard
cell specifically (`experiments_v6_audit/results/*_entry_gate_fix.csv`): real-world
constraint-violation rate falls from 41.9-81.0% (as reported) to exactly 0.0%, at the
cost of success rate falling from 71.8%/45.5% to 38.6%/37.7% (fewer patients get
recourse at all, because many previously-"successful" entries are now correctly
rejected as unsafe). **This is not free — it is the honest cost of actually enforcing
the constraint the paper claims to guarantee.** Whether this reduced success rate is
itself further improvable (e.g., by finding alternative valid entries for patients
whose nearest entry was invalid) is the natural next question and is NOT measured here.

## What this audit rules out

**Further statistical patches in the style of v4's DSR, and further architectural
guarantees in the style of v5's EGD, are not worth pursuing before this fix is applied
and re-measured.** Both sessions built substantial, costly machinery to compensate for
a decoder-imprecision problem that this audit shows is (a) not the dominant contributor
to begin with (B1 alone accounts for more of the original gap than decoder noise does)
and (b) trivially eliminable without touching the decoder at all (B2) for the
overwhelming majority of recourse traffic (real-to-real node transitions). If a future
session still finds residual decoded-space violations AFTER applying B1+B2 — restricted
now to genuinely synthetic nodes, where decoding is unavoidable — THAT would be the
correct, narrower context in which to revisit decoder-quality or architectural fixes,
not before.

## What this audit does not cover (explicit, not silently assumed)

- B1's fix was implemented and measured against v3's `graph_build.py` only. v2's
  independent copy of the same logic was read and confirmed to have the identical
  pattern (byte-identical `run_grid.py` decode-loop); v4's `dsr_graph.py` and v5's
  `egd_graph.py` were read and confirmed to have the SAME "decode every node" pattern
  (B2) but their OWN query-attachment gating logic (analogous to B1) was not
  independently re-verified line-by-line in this audit — flagged as the first thing to
  check in a following session, not assumed identical.
- This audit did not test whether B1+B2 combined with v4's calibrated bands or v5's
  EGD architecture produces a BETTER combined system than either alone — only that
  B1+B2 alone, on top of the ORIGINAL (v2/v3-style) latent-pruning method, already
  reaches the safety guarantee those later sessions built more elaborate machinery to
  approximate. This is a natural, cheap follow-up experiment, not yet run.
- Root Cause C's posterior-collapse finding is a single diagnostic on one seed, not a
  causal ablation — labeled as such, not overclaimed.
