# METHOD.md — Exact-Guarantee Decoder (EGD)

## 0. What problem this targets, and the corrected scope (see NOVELTY_CLAIM.md)

`experiments_v2`-`v4` established that statistical/calibrated approaches to decoded-
space constraint satisfaction (DSR, `v4`) work only in a narrow, cohort-dependent
regime. EGD replaces the statistical approach with an architecture that makes
violations of immutability and monotonicity **structurally impossible**, independent of
training or calibration. Per `NOVELTY_CLAIM.md`, this specific mechanism is not the
first exact-guarantee construction in the literature (CounterFlowNet's discrete action
masking already achieves it, in a different architecture) — EGD's contribution is doing
this within a continuous VAE-latent-graph-shortest-path method, paired with the
certified-infeasibility analysis this project already validated, a combination not
found elsewhere.

## 1. The decoder: per-feature-type output rule

`ExactGuaranteeDecoder` (`experiments_v5/lib/egd_decoder.py`) keeps the encoder
architecture unchanged (Input → 64 → 32 → (mu, logvar), latent_dim=4). The decoder's
output for column `k`, given a target latent code `z` and a SOURCE state `x_source`
(the previous state in a trajectory), is one of:

- **Immutable** (`sex`, `fasting_blood_sugar`): `x_target[k] = x_source[k]`. No
  learnable parameters; zero reconstruction error is possible because there is no
  reconstruction — the output IS the input for this column, regardless of `z`.
- **Non-decreasing** (`age`) or **directional_increase** (e.g. UCI's `max_heart_rate`):
  `x_target[k] = x_source[k] + softplus(net_k(z))`. `softplus(·) > 0` everywhere, for
  any network weights, so the sum is `≥ x_source[k]` before any training happens.
- **Directional_reduce** (BP, cholesterol, etc. — must not increase):
  `x_target[k] = x_source[k] - softplus(net_k(z))`, mirrored.
- **Free** (anything else): `x_target[k] = net_k(z)`, unrestricted. Neither UCI's nor
  real NHANES's schema has any column here in practice — every mutable feature carries
  a directional or non-decreasing constraint — but the code path exists.

**Consequence, stated precisely**: because this construction operates in the SAME
representation the feature is stored in (z-score standardized for continuous columns;
raw units for categorical/immutable ones), and `StandardScaler`'s scale is strictly
positive, enforcing "scaled delta ≥/≤ 0" enforces the TRUE clinical-unit direction
exactly, with **zero tolerance** — stricter than the original codebase's small epsilon
allowances (e.g. cholesterol's 1.0 mg/dL, age's 0.1yr). This is a deliberate,
defensible simplification, not an oversight.

## 2. Sequential decoding along a path

Because `decode_step` needs a real `x_source`, a path's hop `k+1` is decoded relative
to hop `k`'s OWN decoded output — `decode_path` implements this chain: `x̂_1 =
decode_step(z_1, x0)`, `x̂_2 = decode_step(z_2, x̂_1)`, …. By construction, and
verified directly (§4): for a non-decreasing feature, `x̂_{k+1}[age] ≥ x̂_k[age]` for
every consecutive pair, and by transitivity of `≥`, this holds for the FULL path
end-to-end regardless of how many hops it takes or which nodes it visits — a genuinely
stronger, per-step-AND-endpoint guarantee than the original tolerance-based check ever
gave. Same argument, mirrored, for directional-reduce features.

**Direct consequence**: a graph "node" has no single canonical decoded value anymore —
its value depends on which path reached it. `egd_graph.py` resolves this by using pure
LATENT-space distance (encoder `mu`, with each edge's own true source for the one-step
Riemannian Jacobian) for graph structure and Dijkstra search, and computing the actual
decoded trajectory (and the one thing EGD does NOT guarantee — risk reduction) only
AFTER a candidate path is chosen, per query (`search_egd.py`).

## 3. What is checked, not guaranteed: risk reduction

Nothing in EGD's construction relates `classifier(x_target)` to `classifier(x_source)`.
`find_recourse_egd` ranks up to 20 candidate low-risk-mask targets by latent Dijkstra
distance, sequentially decodes each IN ORDER, and returns the first whose ACTUAL decoded
terminal state passes the classifier's risk check — exhausting the ranked list results
in abstention (`abstain_stage='risk_check'`), tracked separately from "no path exists at
all in the latent graph" (`abstain_stage='no_path'`). Empirically (§5), `no_path` never
occurred in this session's runs — the k-NN graph, left entirely unpruned since EGD needs
no constraint-based masking, is well-connected on both cohorts; every abstention
observed was a `risk_check` failure.

## 4. Training: a real dead end found, and its fix

`train_egd` needs (source, target) training PAIRS. Three regimes were tried, and the
choice is empirical (full account in `egd_decoder.py`'s docstring, summarized here):

| Regime | Mechanism | UCI result | Real NHANES result |
|---|---|---:|---:|
| `self` (source=target=x) | first attempt | 0/10 successes | 0/10 successes |
| `knn` (raw-feature-space k-NN pair) | tried to match deployment | 2-3/15-20 | 0/15-20 |
| `random` (shuffle within batch) | **used, default** | 13/20 (smoke) | 0-12/420-445 |

`self` only ever shows the residual heads "source and target are the same patient" —
near-zero residuals that don't generalize to real graph edges (always two DIFFERENT
patients). `knn` matches deployment's typical edge but gives residual TARGETS too small
to bridge a real high-risk-to-low-risk gap even accumulated over several hops. `random`
fixes the near-zero collapse (UCI succeeds well) but has a diagnosed weakness on real
NHANES: independent (x_source, x_target) means `Var(x_target - x_source | z_target)` is
dominated by `Var(x_source)`, noise unrelated to what `z_target` can predict — verified
this is a converged optimum, not underfitting, by testing 100 epochs and a deeper
per-feature head; neither changed the outcome (residuals got SMALLER, not larger, with
more capacity/training). `random` is kept as the default because it is the
best-performing regime actually found, and real NHANES's poor result under EVERY regime
tested is reported as a genuine, diagnosed limitation (§5), not a training bug still to
fix.

## 5. Empirical verification of the exact guarantee

Measured directly, not asserted: self-reconstruction AND random-pair decoded-CVR is
**0.0000%** on the full training set, both cohorts (237/237 UCI, 3631/3631 real NHANES
rows), confirmed again under a synthetic 3-hop path (age and cholesterol sequences
monotone across all hops). Across every successful search-returned path in the full
5-seed grid (`run_egd.py`), decoded-space CVR is **0.0% ± 0.0%**, both cohorts, both
metrics — the guarantee held with zero exceptions, at every scale tested.

## 6. What this does NOT solve, stated as plainly as the guarantee itself

The exact guarantee is real and unconditional. Practical viability is not: success
rate is highly seed-variable on UCI (0-90%+ range, mean 31.1% ± 37.0 across 5 seeds)
and collapses to near-zero on real NHANES (1.0% ± 1.4%, riemannian; 0.2% ± 0.4%,
euclidean) — far below the old method's 71.8%/45.5%. Where EGD DOES succeed, its paths
are measurably less plausible (KDE significantly worse, p<0.001 both cohorts) and
require dramatically more clinical effort (UCI: 10.3 vs. old's 0.2, a ~50x increase;
p<0.001) than the old method's paths. `FINDINGS.md` and `ASSESSMENT.md` state the
resulting verdict without softening it.
