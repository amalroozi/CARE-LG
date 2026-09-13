# NOVELTY_CLAIM.md — the precise claim, checked against evidence, before building anything

Written after `PRIOR_ART_COUNTERFLOWNET.md`'s close read, per the brief's explicit
instruction to stop here if the claim doesn't hold. **Verdict: the claim holds, but
narrower than the brief's framing suggested — recalibrated below, not stopped.**

## The brief's original framing, and what changes

The brief characterized EGD as "the combination that nobody has published: exact-
guarantee generation when an answer exists, paired with exhaustive provable
certification of infeasibility when it doesn't." Phase 0 confirms the **second half**
of this (certified infeasibility pairing) is genuinely absent from the five closest
papers checked (CounterFlowNet, FCEGAN, FastDCFlow, SharpShooter, ECCCo —
`experiments_v4/ASSESSMENT.md` + this session's `PRIOR_ART_COUNTERFLOWNET.md`).

The **first half** needs correction: "exact-guarantee generation" is **not** a novel
capability in the field generally — CounterFlowNet already achieves it (both
immutability and monotonicity, via discrete action masking in a GFlowNet with no VAE at
all), and FCEGAN achieves it for immutability alone (via post-hoc value resetting, GAN-
based). The brief's framing that this is "the closest and most threatening piece of
prior art" was correct, and reading it changes the claim this session can honestly make.

## The recalibrated claim

**What is new**: not "exact constraint guarantees for tabular counterfactuals" (that
already exists, in at least two published mechanisms) but (a) achieving an exact
guarantee **specifically within a continuous VAE-latent-graph shortest-path recourse
architecture** — the class of method this whole project (`experiments_v2`→`v5`) has
built and measured, which none of the exact-guarantee papers use (CounterFlowNet:
discrete GFlowNet, no VAE; FCEGAN: GAN, no graph) — via a decoder-head design specific
to that setting (masked identity for immutables, signed-residual through a
strictly-signed activation for monotone/directional features, requiring sequential
decoding along a path since each hop's "source" is the previous hop's own output); and
(b) **pairing that guarantee, in that architecture, with the exhaustive, multi-seed
certified-infeasibility/blindspot/unresolved decomposition this project already built
and validated in `v3`/`v4`** — a combination not found in any of the five closest
papers, all of which either don't discuss infeasibility (four of five) or don't have an
exact-guarantee mechanism at all (FastDCFlow).

**What is explicitly NOT claimed**: that architecturally-exact constraint satisfaction
for tabular counterfactuals is a new idea (it is not — cite CounterFlowNet and FCEGAN
directly and early in any paper draft); that this session's specific decoder-head
mechanism (masked identity + signed-residual-softplus) is meaningfully different in
KIND from CounterFlowNet's discrete masking — both remove the possibility of violation
by construction, and a reviewer who wants to could reasonably call EGD "CounterFlowNet's
masking idea, re-implemented for a continuous VAE decoder instead of a discrete action
space." The response to that specific objection is: the re-implementation is not free
(§ below, and see how Phase 1 turns out), and the infeasibility pairing is the part that
was actually absent, not the masking idea itself.

## Why this is still worth building (not stopping here)

The certified-infeasibility pairing is real, checked, load-bearing evidence for
proceeding — it is the one part of the brief's premise that survived a hostile,
literature-grounded check rather than an assumption. Phases 1-4 should proceed with this
corrected framing: EGD's contribution is the union of (exact guarantee, in this
architecture) × (paired with rigorous infeasibility certification), not either half
alone. `ASSESSMENT.md` (written last, per instructions) will restate this precisely
against whatever Phases 1-4 actually produce — including, honestly, if EGD's specific
decoder-head mechanism turns out to be worse in practice (plausibility, effort, success
rate) than the calibrated/repair-based DSR from `v4`, which is a real possible outcome
this document does not pre-empt.
