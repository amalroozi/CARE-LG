# FINDINGS.md — Exact-Guarantee Decoder (EGD) results, plain language

All numbers from code executed in this session (`experiments_v5/results/`,
`run_metadata.json`). Git SHA: `539d87b795b78bb8f9446f99b0fabeded2c84841` (no commits, any
session). 5 seeds (0-4), both cohorts. Real NHANES's blindspot decomposition is capped
at a 60-patient random sample per seed where the true abstention count exceeds that
(true rate always reported in full — `CHANGES.md`).

## The headline number: does EGD's guarantee actually hold?

**Yes, exactly, with zero exceptions measured.** Decoded-space CVR is **0.0% ± 0.0%**
for every successful EGD path, both metrics, both cohorts, across the full 5-seed grid
— verified directly on the full training set (0/237 UCI, 0/3631 real NHANES
self-reconstruction violations) and on every search-returned path. This is the one
claim in this whole session that is unconditionally, empirically confirmed.

**Everything else is a real, substantial cost, and this document reports it plainly.**

## Success rate: unreliable on UCI, effectively broken on real NHANES

| | UCI success | Real NHANES success |
|---|---:|---:|
| Old (latent pruning, this session's fresh rerun) | 71.8% ± 9.6 | 45.5% ± 3.0 |
| EGD (Riemannian) | 31.1% ± 37.0 | 1.0% ± 1.4 |
| EGD (Euclidean) | 0.0% ± 0.0 | 0.2% ± 0.4 |

UCI's EGD success rate ranges from near-0% to over 90% depending on seed (std=37.0 on a
mean of 31.1 — the widest seed-to-seed swing measured anywhere in this project's
history). Real NHANES's EGD success is close to zero regardless of seed or metric — a
near-total collapse relative to the old method's 45.5%.

**Root cause, diagnosed directly (not inferred)**: every real-NHANES abstention's
`abstain_stage` was `'risk_check'` (100% of abstentions, all 5 seeds) — never
`'no_path'`. The k-NN latent graph, left fully unpruned, always finds a route; EGD's
learned residuals are simply too small, even accumulated over the search's allowed
hops, to move a high-risk patient's decoded trajectory across the risk threshold. Three
training regimes were tried to fix this (self-reconstruction, k-NN-neighbor pairing,
random pairing — `METHOD.md` §4); random pairing is the best found and is what these
numbers reflect. This is reported as a genuine, diagnosed limitation of the current
4-dimensional-latent, residual-per-hop architecture — the SAME underlying difficulty
(a small latent bottleneck struggling to represent large feature changes) that
`experiments_v3` Phase B found for the ordinary decoder and `experiments_v4` found for
DSR's repair mechanism, now manifesting a third time via a third distinct mechanism.

## The cost where EGD DOES succeed: plausibility and effort

Paired tests (same patients, both methods succeeded):

| Dataset | Metric | n pairs | p-value | Median diff (EGD − old) |
|---|---|---:|---:|---:|
| UCI | KDE | 30 | <0.000001 | −2.78 (worse) |
| UCI | Effort | 30 | <0.000001 | +13.82 (worse, ~50x old's mean) |
| Real NHANES | KDE | 12 | 0.0005 | −2.33 (worse) |
| Real NHANES | Effort | 12 | 0.0024 | +3.96 (worse) |

Both differences are significant, in the SAME (unfavorable to EGD) direction, on BOTH
cohorts. EGD's paths — where they exist at all — are measurably less plausible
(further from the data manifold) and require dramatically more clinical effort than the
old method's. This is a real, consistent, statistically overwhelming trade-off, not
noise: the exact guarantee is bought at the price of realism and parsimony.

## RQ (new) — Does the Riemannian-vs-Euclidean null finding survive under EGD?

**No — it REVERSES, for a clear, verified, mechanistic reason.** Every prior session
(v2-v4) found no meaningful difference between the two metrics. Under EGD:

| | UCI success | Real NHANES success |
|---|---:|---:|
| Riemannian | 31.1% ± 37.0 | 1.0% ± 1.4 |
| Euclidean | 0.0% ± 0.0 | 0.2% ± 0.4 |

Diagnosed directly: Euclidean-weighted Dijkstra systematically prefers SHORTER paths
(observed: consistently 2-node/1-hop candidates) than Riemannian-weighted Dijkstra
(observed: 2-7-node paths for the same query). Since EGD's guarantee-preserving
construction ACCUMULATES change per hop (§ METHOD.md §2), shorter paths structurally
cannot accumulate as much risk-reducing change. Every prior session's decoders were
path-independent (decode(z) alone), so path length never mattered for the metric
comparison; EGD's sequential, source-conditioned decoding makes path length — and
therefore the metric that shapes it — matter for the first time in this project's
history. This is a genuine, mechanistically-explained finding, not an artifact.

## RQ3 — Certified infeasibility under EGD

| | UCI | Real NHANES |
|---|---:|---:|
| Mean abstention rate | 69% of high-risk patients | 99% of high-risk patients |
| % certified infeasible | 58% ± 14 | 60% ± 8 |
| % confirmed blindspot (densification-resolved) | **0%** | **0%** |
| % unresolved | 42% ± 14 | 40% ± 8 |

**Zero blindspots resolved by either densification probe (doubled k; +500 synthetic
nodes) on EITHER cohort, at any seed.** This is a clean, important result: it confirms
the "unresolved" category is not a connectivity problem the graph could fix with more
neighbors or more synthetic candidates — it is a MECHANISM problem (the residual
construction cannot bridge the gap), and no amount of graph densification touches that.
This is the third time in this project's history (after `experiments_v3`'s original
finding on UCI and `experiments_v4`'s DSR-full) that "abstention = certified
infeasibility" needs qualification: under EGD, 40-42% of abstentions are real targets
the method simply cannot reach, not provably nonexistent ones.

## Summary: does EGD solve the problem this project set out to fix?

| Claim | Status | Evidence |
|---|---|---|
| Exact (not merely calibrated) constraint satisfaction | **Fully confirmed** | 0.0000% decoded CVR, every success, both cohorts, verified directly on the training set and every search path |
| Practical success rate competitive with the old method | **Contradicted** | UCI 31% vs. 72% (and wildly seed-variable); real NHANES 1% vs. 46% |
| Path plausibility (KDE) at least comparable to the old method | **Contradicted** | Significantly worse, both cohorts, p<0.001 |
| Clinical effort comparable to the old method | **Contradicted** | Significantly worse, both cohorts; ~50x on UCI |
| Riemannian metric matters more under EGD than under prior methods | **Confirmed, but favors a narrower, harder-to-reach operating point** — Euclidean is unusable (0% UCI) | Fig 3 |
| Certified infeasibility survives as a clean guarantee | **Contradicted** — 40-42% of abstentions are unresolved, not certified | Fig 5, both cohorts |

**Honest bottom line**: EGD proves the NARROW technical claim it set out to prove
(exact, architectural constraint satisfaction is achievable in this class of method) but
does NOT constitute a practical improvement over either the old latent-pruning method or
`experiments_v4`'s DSR-full — it trades away success rate, plausibility, and effort
efficiency for a guarantee that, per `NOVELTY_CLAIM.md`, was not even the field's first
exact-guarantee mechanism to begin with. See `ASSESSMENT.md` for what this means for
publishability and next steps.
