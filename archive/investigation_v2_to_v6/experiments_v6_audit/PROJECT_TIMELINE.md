# PROJECT_TIMELINE.md — factual reconstruction, v2 through v5

Scaffolding for `RECOMMENDATION.md`. Terse by design.

## v2 (first correction session)
- **Found**: `data/nhanes.csv` is fully synthetic, not real NHANES (`experiments_v2/REPO_MAP.md` D1). No `torch`/`random` seeding anywhere (D4) — UCI headline success (47.6%) didn't reproduce (63.6% on rerun).
- **Built**: seeded multi-run harness; "Phase 3" decoded-space verification — introduced `decode_node`/`decode_nodes_batch` to re-check constraints on **decoded** path states, on the premise that "the patient is shown the decoded reconstruction, not the raw recorded value."
- **Headline finding**: 78–92% decoded-space CVR despite 0% graph-internal CVR. Attributed to VAE reconstruction imprecision.
- **Left unresolved**: never checked whether real graph nodes' true values could be used instead of decoding them; never isolated the query-entry-edge gating rule's own correctness independent of decoding.

## v3 (real data + decoder fix)
- **Found**: real NHANES cohort built (N=4,539, PCE label, `DATA_PROVENANCE.md`). Type-aware decoder (dedicated head for `age`) cut age MAE 4–6×.
- **Result**: UCI decoded CVR fell 83.6%→41.9%; real-NHANES decoded CVR barely moved (78.1%→81.0%) — attributed to `sex` reconstruction remaining near-chance.
- **Left unresolved**: same as v2 — decoding of real nodes, and the entry-gate's relaxed predicate, neither examined.

## v4 (DSR: calibrated bands + repair)
- **Built**: conformal per-feature reconstruction-error bands; verify-and-repair loop; identity rule for immutables (real value copy-through — but only inside the NEW repair-verification code path, not in the underlying decoded-CVR measurement itself).
- **Found**: bands-as-pre-filter collapse the graph (τ≫ε for several features at every quantile). DSR-full reaches 0% CVR only at a narrow operating point (≤23% success, real NHANES; 0% success, UCI, at any τ).
- **`ASSESSMENT.md`** already flagged: "a statistical patch on an untrustworthy network."
- **Left unresolved**: never asked why the network is "untrustworthy" for real nodes specifically, whose true values were never actually unknown.

## v5 (EGD: architecturally exact guarantees)
- **Built**: masked-identity + signed-residual decoder heads, sequential path decoding.
- **Found**: guarantee holds exactly (0.0000% decoded CVR) — but success collapses on real NHANES (1.0% vs. old method's 45.5%), effort ~50× higher, KDE significantly worse, metric-choice reversal (Euclidean unusable).
- **`ASSESSMENT.md`**: "EGD is not a practical improvement... trades away success, plausibility, effort for a guarantee."
- **Left unresolved**: same gap — the entire architecture exists to make **decoding** exact; it was never asked whether decoding the real, already-known graph nodes was necessary at all.

## The pattern across all four sessions
Each session diagnosed the SAME symptom (decoded-space constraint violations on successful paths) and built a progressively more elaborate fix at the DECODER level (better decoder → statistical calibration+repair → architecturally exact decoder), without ever checking two structural questions upstream of the decoder entirely:
1. Does the pipeline need to decode a graph node whose real, recorded feature vector is already known? (`src/recourse/search.py::decode_recourse_trajectory` — the ORIGINAL pipeline's own patient-facing function — never does; every v2–v5 Phase-3 measurement does.)
2. Is the constraint predicate that gates the query's entry into the graph the SAME predicate that gates every other edge? (No — documented as "D8" in `experiments_v2/REPO_MAP.md`, but never load-bearing-tested against the headline decoded-CVR finding.)

This audit's Phase 2/3 (below, `RECOMMENDATION.md`) tests both directly.
