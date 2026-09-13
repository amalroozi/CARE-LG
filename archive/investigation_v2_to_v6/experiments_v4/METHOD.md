# METHOD.md — Decode-Safe Recourse (DSR): formal description

## 0. The problem DSR targets

The prior method (`experiments_v2`/`experiments_v3`, "latent pruning" below) builds a
k-NN graph over encoded patients `z_i = enc(x_i)` and prunes an edge `(i,j)` when a
constraint predicate `C` — immutability of `sex`, non-decreasing `age`, directional
risk-factor rules — evaluates True on the nodes' **recorded** feature values `x_i, x_j`.
The path returned by Dijkstra is reported to the patient as a sequence of **decoded**
states `x̂_k = f_θ(z_k)`, and `f_θ(enc(x)) ≠ x` in general. `experiments_v3/FINDINGS.md`
measured this gap directly: 78–92% of paths that are provably constraint-satisfying on
`x` violate the same constraints once re-checked on `x̂`. The guarantee is made about a
representation (`x`) the patient never sees, and is silent about the one they do (`x̂`).

DSR is three components, each independently switchable so their contributions are
attributable (`experiments_v4/run_dsr.py::METHOD_CELLS`).

## 1. Component 1 — decoded-space edge pruning

Every graph node is decoded once, `X̂ = f_θ(Z)` (`experiments_v4/lib/constraints_ext.py
::decode_nodes_batch`, O(N), computed once per (dataset, seed) in
`dsr_graph.py::build_dsr_edge_table`, never inside the edge loop). The predicate `C` is
then evaluated on `(x̂_i, x̂_j)` instead of `(x_i, x_j)`:

```
prune(i, j)  ⟺  C(x̂_i, x̂_j) = True
```

using the same sub-conditions as the original codebase (immutable Δ > 1e-4 after
rounding binary features to {0,1}; age decrease > 0.1yr; single-step age horizon >
3.05yr; directional epsilons per feature — `experiments_v4/lib/dsr_rules.py`, mode
`'decoded_naive'`). This is the entirety of Component 1: the pruning criterion is
realigned with the representation the patient is shown.

## 2. Component 2 — calibrated tolerance bands and the identity rule

A decoded check is a check against a *noisy estimate* of `x`, not `x` itself, so
Component 1 alone is not a certified guarantee. Component 2 adds two things.

### 2a. Conformal tolerance bands (continuous, mutable features)

On a calibration split `X_calib` the VAE and classifier never trained on (§4), the
per-feature reconstruction error is measured and its `q`-quantile taken:

```
τ_k(q) = quantile_q( | unscale(f_θ(enc(x)))_k − unscale(x)_k | ,  x ∈ X_calib )
```

(`experiments_v4/lib/calibration.py::calibrate`). This is a distribution-free,
conformal-style empirical quantile — no Gaussian or other parametric assumption on the
error distribution.

**Coverage claim, stated exactly and no more strongly**: for a single feature `k` and a
node exchangeable with the calibration set, `P(|x̂_k − x_k| ≤ τ_k(q)) ≥ q`. A
constraint compares **two** nodes, so its true error is the difference of two such
residuals; the rule below applies a single `τ_k`, which is the literal rule this
project's brief specifies for the non-decreasing case. A fully worst-case two-endpoint
bound would use `2τ_k`; we did not implement that stricter variant, and say so here
rather than letting the single-`τ_k` rule imply a tighter guarantee than what was
tested. This is stated as a scope limit in §5, not glossed over.

Each threshold is tightened by `τ_k(q)` in the conservative direction
(`experiments_v4/lib/dsr_rules.py::violation_breakdown`, mode `'decoded_banded'`):

- **Non-decreasing (age)**: require `x̂_j[age] − x̂_i[age] ≥ +τ_age(q)` (replacing the
  original `≥ −0.1`) — the brief's own worked example, implemented literally.
- **Age horizon**: require `x̂_j[age] − x̂_i[age] ≤ 3.05 − τ_age(q)` (upper bound
  tightened downward by the same margin, since decoder error could make the true delta
  up to `τ_age` larger than shown).
- **Directional reduce** (e.g. cholesterol should not rise): require
  `x̂_j[k] − x̂_i[k] ≤ ε_k − τ_k(q)` (the original small epsilon `ε_k`, shrunk by the
  calibrated margin). **A real, measured consequence, not a bug**: for several
  continuous features (cholesterol, oldpeak on UCI; cholesterol, BMI on real NHANES),
  `τ_k(q)` exceeds `ε_k` even at the loosest quantile tested (q=0.05) —
  `experiments_v4/results/tau_vs_eps_diagnostic.csv` — because the decoder's actual
  reconstruction error on those features is intrinsically larger than the tight
  tolerance the original codebase assumed for measurement noise. The threshold goes
  negative, and the honest reading of a negative threshold is not "an error" — it is a
  correct statement that no confidently-certified single-hop change is currently
  achievable for that feature on this decoder. §5 and `FINDINGS.md` report this plainly
  rather than loosening the rule to hide it.
- **Directional increase**: mirror image, `x̂_j[k] − x̂_i[k] ≥ −ε_k + τ_k(q)`.

### 2b. Immutability by node identity (binary/categorical features)

A calibrated band is the wrong instrument for a binary immutable like `sex`: decoded
sex accuracy measured in `experiments_v3` is 67.5% (UCI) / 68.2% (real NHANES) — close
to chance — so no band width makes a decoded sex comparison informative; a band wide
enough to cover the error is wide enough to admit any transition at all.

Every node that is a **real patient** — every graph node, and the query — has an exact
recorded value for its immutable features. DSR checks immutables on that recorded
value, not the decoded reconstruction, and the constraint holds **exactly**, with no
probabilistic qualifier:

```
prune_immutable(i, j)  ⟺  x_i[immutable] ≠ x_j[immutable]     (exact, not decoded)
```

**Scope of this guarantee, stated precisely**: this is a guarantee about which
*transitions the graph permits* (a node whose true sex differs from the source's is
never an eligible next hop), not a claim that the decoder reconstructs sex correctly.
A patient shown the decoded reconstruction of an accepted step may still see a decoded
`sex` value that rounds to the "wrong" class — DSR guarantees the step was not
*selected* on the basis of a sex change, not that its decoded display is faithful. This
distinction is real and is the reason mutable and immutable features are treated
differently in this method: a mutable feature's decoded value *is* the actionable
recommendation ("aim for roughly this blood pressure"), so its error must be bounded and
carried into the constraint (§2a); an immutable feature is never an action item, it
exists purely to gate feasibility, and identity gives a strictly better answer than a
band can, at zero cost, wherever an identity is available.

`experiments_v4/run_dsr.py` also runs `dsr_bands_no_identity` (bands applied everywhere,
including immutables on decoded values) as an ablation-of-the-ablation, isolating how
much of Component 2's benefit is the bands versus the identity rule.

### 2c. A discrepancy found and resolved: bands as pre-filter vs. bands as repair standard

The brief's literal component ordering applies Component 2's bands as a hard PRUNING
pre-filter (a stricter version of Component 1). We built this exactly
(`prune_mode='decoded_banded'`, cell `dsr_c1c2`) and it collapses the graph to zero
surviving interior edges at **every** tested `τ` quantile from 0.05 to 0.99 on UCI, and
to near-zero on real NHANES, because a directional violation is an OR over all
directional features — the single feature with `τ_k > ε_k` (§2a) poisons every edge
regardless of how the other features behave. This is reported as `dsr_c1c2`'s own
result (Fig 2), not hidden.

**Resolution used for `dsr_full`** (documented per Hard Rule 6, most defensible
interpretation): the GRAPH is pruned with Component 1 only (so a path can exist at
all), and Component 2's bands are applied at the Component 3 verify-and-repair stage,
where a violation can be *repaired* rather than making the edge's existence impossible.
`dsr_c1c2` (bands as a pure pre-filter, no repair) is kept as its own cell precisely
because its collapse is the direct evidence motivating this choice.

## 3. Component 3 — verify-and-repair projection

After Dijkstra returns path `π = (q, v_1, …, v_m)` on the Component-1 graph, every node
is decoded and every consecutive step is checked against the Component 2 banded
predicate (`experiments_v4/lib/repair.py::verify_and_repair`). A violating step is
repaired:

1. **Project**: clip the violating feature(s) of the decoded target state to the
   constraint boundary plus band (e.g. age set to `source_age + τ_age(q)` if it fell
   short) — `repair.py::_project`.
2. **Preserve immutables by construction**: the repaired state is copied from the
   query's own recorded immutable values, not reconstructed — a repaired state is
   synthetic (not a real graph node), so §2b's identity argument does not apply to it;
   the recourse trajectory is by definition a sequence of states for **one** person, so
   its immutables are simply copied.
3. **Re-encode, re-decode**: `z' = enc(x_proj)`, `x̂' = dec(z')`. This round trip is
   necessary because the object the method traffics in is a latent code — a clipped
   feature vector that is never pushed back through enc/dec is a value the model never
   actually predicts.
4. **Re-verify**: repeat up to 3 times; if still violating, the step — and therefore the
   path — is rejected.
5. **Global checks on the repaired path**: monotone risk reduction along every step
   (classifier risk must not increase, tolerance 1e-3) and a genuinely low-risk terminal
   state. Failing either rejects the path.

**If repair cannot produce a satisfying, risk-monotone, low-risk path, the query
ABSTAINS.** DSR never returns a path it cannot verify.

### A finding from building this component, reported because it is real and it matters

Re-encoding does **not** reliably preserve a targeted single-feature edit for every
feature. A controlled test (20 UCI train nodes, `+2yr` requested age edit,
`experiments_v3`'s type-aware decoder): the round trip achieves **103.8%** of the
requested delta on average (std 1.63, i.e. steerable, because age has a dedicated,
loss-upweighted decoder head — Phase B of `experiments_v3`). The same test on
cholesterol (a standard, non-upweighted continuous head, architecturally unchanged since
the original decoder): a requested `−3.36 mg/dL` clip, after one re-encode/decode cycle,
achieves `240.55 → 240.35 → 240.34` over three iterations — the decoder's 4-dimensional
latent bottleneck pulls the reconstruction back toward its own fixed point for that
neighborhood, not toward the requested target, regardless of iteration count. **Repair
is only as effective as the decoder head it is asking to move**, and inherits Phase B's
architectural asymmetry: it reliably fixes age violations and reliably fails to fix
violations in features that never got a dedicated head. This is why `dsr_full`'s
achievable operating points (§5, Fig 3) exist only at looser `τ` quantiles on real
NHANES and not at all on UCI (whose smaller calibration set and feature mix make even
the loosest band unrepairable for the features involved) — a decoder-quality limit
propagating through the method, not a flaw specific to the projection step's logic.

## 4. The three-way data split (why the old method is re-run, not quoted from v3)

Component 2 requires a calibration split the decoder never saw. `experiments_v4/lib
/pipeline_v4.py` keeps `experiments_v3`'s exact 80/20 train/test split (identical test
patients per seed, for comparability), then re-splits the 80% train partition 85/15 into
graph-train/calibration, stratified and seeded. The VAE and classifier are fit on
graph-train only. Consequence: the old (latent-pruning) method, re-run inside this same
pipeline for a controlled comparison, sees ~15% fewer training rows than in
`experiments_v3` and its numbers here are not bit-identical to the published v3 table —
this is why Table 1's "old" row is a fresh run under this session's protocol, not a copy
of v3's number.

## 5. Scope of the guarantee — what is and is not claimed

**Guaranteed** (by construction, not probabilistically): no accepted graph transition
changes a node's recorded immutable feature (§2b). No path is returned unless every
step, checked against the calibrated band, and the terminal risk, verify (§3) — i.e.
whatever DSR returns has already been re-checked in decoded space at the point of
return; it is not "hopefully correct."

**Calibrated, at a stated confidence, not guaranteed absolutely**: continuous mutable
feature constraints hold with margin `τ_k(q)`, valid under exchangeability of the
calibration and deployment populations, using a single-endpoint (not doubled) margin
(§2a) — a real, stated limitation.

**Not guaranteed at all, and not claimed to be**: (a) that the *decoded display* of an
accepted state exactly matches what a hypothetical "true" state at that point would be
— only that constraint *selection* was not based on decoder error exceeding the band;
(b) validity for features whose calibrated `τ_k(q)` exceeds the original tolerance
`ε_k` at every tested quantile (§2a) — DSR does not claim to certify these, and the
`dsr_c1c2` and `tau_vs_eps_diagnostic.csv` results are the direct evidence that it
cannot, on this decoder, for these specific features; (c) that repair succeeds — its
success is architecture-dependent (§3) and is reported, per (dataset, feature), as an
empirical rate, never assumed.
