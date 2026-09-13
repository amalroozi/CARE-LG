# PRIOR_ART_COUNTERFLOWNET.md — close read, Phase 0

Read via arXiv HTML (`arxiv.org/html/2602.17244v1`), not just the abstract, per the
brief's instruction. Two adjacent papers also checked in this phase for the same
questions: FCEGAN (arXiv:2502.17613) and FastDCFlow (arXiv:2404.13224).

## CounterFlowNet (arXiv:2602.17244)

**"CounterFlowNet: From Minimal Changes to Meaningful Counterfactual Explanations."**
Furman, Marszałek, Masłowski, Gaiński, Zięba, Śmieja.

**Mechanism**: a conditional GFlowNet samples counterfactuals as a *sequence of discrete
feature-edit actions* `a = (d, v)` (edit feature index `d` to value `v`) over a
quantized state space — continuous features are binned offline into a finite set of
values before training. There is **no VAE, no encoder/decoder, no latent space**. Quote
(Sec. 3.3): *"we quantize continuous features into a finite number of bins... the
conditioned CounterFlowNet operates over a discrete and countable state space."*

**Constraint enforcement**: action masking on the policy logits, at inference time, no
retraining required (Sec. 3.5): *"we apply a mask ℳ(sₜ, x₀) to the logits, setting
cⱼ = −∞ for indices j that are either immutable by domain constraints or have already
been modified in the current trajectory."* This is architectural/exact (a masked action
has probability exactly zero, not merely a soft penalty), and the abstract states it
covers **both immutability and monotonicity**: *"actionability constraints, such as
immutability and monotonicity of features, can be enforced at inference time via action
masking, without retraining."* Monotonicity is presumably enforced by masking out the
subset of bin-transition actions that would move a constrained feature in the disallowed
direction — the paper does not need a continuous residual/softplus construction because
its action space is already discrete and enumerable.

**"Full satisfaction of the given constraints" — operationally**: the paper does not
explicitly separate this into "conditional on returning a CF" vs. "unconditional over
all queries," but the evidence (Table 3, validity ~99% even under tight constraint sets)
is consistent with: among the counterfactuals the method DOES return, constraint
satisfaction is 100%; it is not a claim that literally every query receives a
constrained, valid counterfactual.

**Infeasibility — the load-bearing question for this session's novelty claim**:
**absent from the paper.** No passage distinguishes "no CF returned because none exists"
from "no CF returned because the method failed to find one that exists." No blindspot
analysis, no exhaustive feasibility check, no analog of `experiments_v3`/`v4`'s
certified-infeasible/blindspot/unresolved decomposition. The paper's evaluation protocol
implicitly assumes a valid counterfactual exists for every test instance and reports
validity/sparsity/proximity/plausibility among returned results.

**Datasets**: German Credit, Adult Income, Graduate Admission, Student Performance
(Protocol A); Lending Club, Give Me Some Credit, Bank Marketing, Credit Default
(Protocol B). **Zero clinical or healthcare datasets.**

## FCEGAN (arXiv:2502.17613) — "Flexible Counterfactual Explanations with Generative Models"

Hellemans, Algaba, Verboven, Ginis. WGAN-GP generator (no VAE). Immutability enforced by
**explicit post-hoc resetting**: *"After each gradient step, immutable features are
reverted back to their original values from the counterfactual template"* — the same
"copy the source value through" idea this project's own identity rule (`experiments_v4`
Component 2b) and CounterFlowNet's masking both use, independently arrived at three
times now. **No monotonicity/directional mechanism is discussed at all.** No
infeasibility discussion. Uses two healthcare datasets (a heart disease risk dataset and
a diabetes/BRFSS dataset) — **not NHANES**, and with no graph/shortest-path/latent-metric
structure of any kind; it is a direct feature-space GAN.

## FastDCFlow (arXiv:2404.13224) — "Model-Based Counterfactual Explanations Incorporating Feature Space Attributes for Tabular Data"

Sumiya, Shouno. Normalizing-flow generator (RealNVP; also not a VAE). Both immutability
and monotonicity are enforced as **soft training-time penalties**, not architectural
guarantees: immutable features get up-weighted in the proximity loss (e.g. weight 3.0 vs
1.0), and monotonicity uses an explicit hinge loss,
`L_mon = (1/|D_mon| N) Σ max(x_source − x_cf, 0)`, penalizing but not preventing
violations. This is *weaker* than either CounterFlowNet's or FCEGAN's mechanism — a
statistical patch, not a structural guarantee — and is architecturally closer to this
project's own `experiments_v2`/`v3` soft-penalty ablation cell than to what this
session's EGD is meant to build. No infeasibility discussion.

## What this means for the EGD novelty claim (see `NOVELTY_CLAIM.md`)

1. **Exact/architectural constraint satisfaction is NOT a novel idea.** CounterFlowNet
   already achieves it, for both immutability and monotonicity, via discrete action
   masking in a GFlowNet — a different but equally "exact" mechanism to the one this
   session's brief proposes (masked/residual decoder heads in a VAE). FCEGAN achieves it
   for immutability only, via post-hoc value resetting. Both predate this session.
2. **No VAE-latent-graph-shortest-path method with architecturally exact constraint
   heads was found.** All three exact/near-exact mechanisms found (CounterFlowNet's
   masking, FCEGAN's resetting, and this project's own `experiments_v4` identity rule)
   operate directly on discrete actions or feature values, not through a continuous
   decoder that must be redesigned to make a residual/monotone-safe output structurally
   inevitable. This narrower architectural niche is where EGD's actual contribution, if
   any, must be located — not "inventing exact guarantees."
3. **Infeasibility characterization is absent from every paper checked in this search.**
   This is the strongest, most load-bearing part of the session's premise, and it
   holds: pairing an exact-guarantee generator (whichever specific mechanism) with
   rigorous, exhaustive, multi-seed certified-infeasibility/blindspot decomposition
   does not appear anywhere in this literature.
4. **Real clinical data (NHANES specifically) with this combination is also absent.**
   FCEGAN uses two other healthcare datasets but no graph structure and no infeasibility
   analysis; CounterFlowNet uses no clinical data at all.
