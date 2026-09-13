# DEEP_AUDIT.md

Verbatim preservation of the "what am I missing about the entire thing" analysis
delivered at the end of the prior session, before any resolution work began. This
is the source document Phase 0 of the follow-up brief asked to be preserved.

---

I dug through the repo end-to-end (git history, requirements, src/ vs experiments/,
seeding, dependency wiring) rather than just recapping prior sessions. Here's what I
think you're actually missing — ordered by how much it matters.

## 1. None of this is in `src/` — it's all a parallel research trail

`src/recourse/search.py::find_recourse_path` takes a `source_idx` that must already
be a node in a pre-built graph. It has **no query-attachment concept at all** — no
ephemeral node, no entry gate. That whole mechanism (and therefore the B1 bug, and
therefore its fix) was invented by the `experiments_v2`+ evaluation harness to
support proper train/test splitting. It never existed in, and was never merged back
into, `src/`.

Net effect: `src/graph/calg.py` and `src/recourse/search.py` — the actual importable
library — are untouched since the initial commits. The corrected pipeline, the
neurosymbolic layer, the three-layer explanations, preference elicitation, the UI —
all of it lives only in `experiments_v6/`, a self-contained rebuild that imports
almost nothing from `src/` except the classifier and the raw constraint-checking
functions. If you or anyone else imports `src/` to actually deploy this, you get the
original, never-B1/B2-audited, no-explanation version. Six sessions of fixes are
real, verified, and orphaned from the thing they're fixing.

## 2. Six sessions of work exist only as untracked directories

```
?? experiments_v2/  ?? experiments_v3/  ?? experiments_v4/
?? experiments_v5/  ?? experiments_v6/  ?? experiments_v6_audit/
```
None of these are gitignored (`git check-ignore` confirms it) — they're just never
`git add`ed, per the hard rule that I never commit without being asked. That rule is
about *me* not committing; it isn't a reason for *this work* to stay uncommitted
indefinitely. Right now `git clean -fdx` or a bad `rm -rf` would erase the entire
investigation — the audit, the B1/B2 discovery, all of v6. Worth a deliberate commit
(or at least a backup) on your own schedule.

## 3. `requirements.txt` doesn't know about half of v6

`clingo`, `shap`, `fastapi`, `uvicorn` — all installed this session, all
load-bearing for the neurosymbolic layer, explanations, and UI — aren't in
`requirements.txt`. A fresh `pip install -r requirements.txt` on a clean clone
builds an environment where `experiments_v6` can't import. (Also: `carla-recourse`
is listed but never imported anywhere — dead weight in the other direction.)

## 4. The classifier itself has never been audited

I grepped every FINDINGS/ASSESSMENT doc across all six sessions for AUC, accuracy,
calibration, ROC. Every hit is about *decoder* reconstruction accuracy or
*conformal calibration* of DSR's bands — never the `RiskClassifier` MLP's own
predictive validity. There's no AUC, no Brier score, no calibration curve, no
subgroup check (e.g. does it perform equally well by sex — the one immutable
feature every constraint check treats as sacred). Every downstream claim — 0% CVR,
certified infeasibility, "this patient is high-risk" — is built on a risk score
whose own trustworthiness was simply never checked.

Related: there are **two unreconciled threshold systems**. The data label
(`cvd_risk_flag`) is built from a 7.5% 10-year PCE/ASCVD risk cutoff. The recourse
pipeline's own high/low-risk gate uses the classifier's output probability at
0.55/0.45. Nobody has verified that a classifier score of 0.45 actually corresponds
to anything like "under 7.5% ASCVD risk" — they're different scales that happen to
share the same code path.

## 5. The success-rate cost may be undersold relative to how it's framed

37-38% success, both cohorts, is the honest number — but it's consistently
presented as the *cost* of the headline achievement (0.00% CVR), not as its own
finding. Read the other way: **6 in 10 high-risk patients now get no plan at all.**
That's a first-order clinical-utility problem, not a footnote to a correctness win.
Also worth noting: UCI's success rate swings from 21.7% to 55.0% across 5 seeds
(±13.1pt std) — on a 20-25-patient high-risk cohort, that's small-sample noise
large enough that any single seed's number is close to meaningless on its own.

## 6. The core method's central assumption is never stated as an assumption

Every recourse plan ends at *another real recorded patient's feature vector*. The
implicit claim is "if patient #225 achieved this state, it's a realistic target for
you too." That elides everything the graph doesn't model — medication regimen,
genetics, adherence, comorbidities, time elapsed. It's a defensible design choice
(it's *why* CALG beats pure perturbation methods on plausibility), but I don't see
it written down anywhere as a stated limitation. Right now a reader could think
"recourse to a real patient's values" is validated as achievable; it's really
validated as *recorded*, which isn't the same thing.

## 7. Certified infeasibility has been checked four times, but only one way

v3, v4, v5, and now v6 each ran the identical probe style (2×k-NN, +500 VAE-prior
synthetic nodes) and each found zero blindspots. That's a genuinely strong,
well-replicated result — but it's four repetitions of *the same instrument*, not
four independent methods. "Our own search method can't find a counterexample to our
own search method's claim" is weaker than the phrase "confirmed four times"
suggests on first read.

## 8. Built but never wired in: `G-STATIN` / `G-BP-TARGET-DM/NODM`

I grepped for `statin_indication` and `bp_target` outside `neurosymbolic.py` — zero
hits. These two guideline annotations (the ones citing the actual ACC/AHA statin
threshold and diabetes-differentiated BP targets — arguably the most clinically
"real" content in the whole neurosymbolic layer) are never called from the UI, the
explanations, or any report. The regression-verified rule base is real and correct;
the two rules that would make it feel clinically substantive rather than a
re-encoding of six boolean checks are decorative.

## 9. `CLINICAL_EFFORT_WEIGHTS` are unexplained magic numbers

`resting_bp: 1.5, cholesterol: 2.0, oldpeak: 2.5, exercise_angina: 3.0, ...`
(`configs/dataset_config.py`) — no citation, no derivation, present since before
v2. Every effort number in every session, and now the entire Phase 6
preference-elicitation mechanism, is built by reweighting these already-arbitrary
constants. Nobody's flagged that the foundation being reweighted was never itself
validated.

## 10. Nothing stops B1 from happening again

There's no test anywhere that asserts "the entry gate and the interior-edge gate
use the same predicate." That's exactly the invariant that silently broke for three
sessions before the audit caught it by hand. It would be a five-line regression
test; it doesn't exist.

---

None of this contradicts what's in `FINDINGS.md` — it's all consistent with it.
It's the layer *above* the findings: what the project's own throughline (0% CVR,
certified infeasibility, the neurosymbolic layer) quietly assumes but has never
gone back to check.
