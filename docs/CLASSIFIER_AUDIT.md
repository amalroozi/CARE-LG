# CLASSIFIER_AUDIT.md

The first independent audit, across the whole project's history, of the
`RiskClassifier` MLP's own predictive validity (originally Phase 2 of
`archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md`'s
resolution, issue #4). Every prior claim in this project (0% CVR, certified
infeasibility, "this patient is high-risk") assumed this classifier's output was
trustworthy without ever checking it directly. All numbers below were
originally produced by `run_classifier_audit.py`, 5 seeds, both cohorts,
held-out test set; the canonical, current version of that script is
`scripts/run_classifier_audit.py`, importing directly from `src/`. Raw data:
`results/tables/classifier_audit.json`.

## 1. Discrimination: AUC-ROC and Brier score

| Dataset | AUC-ROC (mean ± std, 5 seeds) | Brier score (mean ± std) |
|---|---|---|
| UCI | 0.8342 ± 0.0406 | 0.1817 ± 0.0099 |
| Real NHANES | 0.9595 ± 0.0044 | 0.0779 ± 0.0060 |

Both are respectable discrimination numbers for an MLP on tabular clinical data
(NHANES's much larger, more homogeneous cohort gives a tighter, higher AUC and
lower variance across seeds than UCI's small 237-patient training set). Neither
number had ever been reported anywhere in this project before this session.

## 2. Calibration curves

10-bin calibration curves (predicted probability vs. observed positive frequency)
are stored per-seed in `classifier_audit.json` under `calibration_curve`. Not
reproduced in full here (10 bins × 5 seeds × 2 datasets); the practically important
summary is the empirical threshold-crossing analysis in section 3, which is
derived directly from the same binning.

## 3. Subgroup performance by sex

| Dataset | Subgroup | n (mean) | AUC (mean ± std) | Brier (mean ± std) |
|---|---|---:|---|---|
| UCI | male (sex=1) | ~41-46 | 0.8357 ± 0.0346 | 0.1830 ± 0.0065 |
| UCI | female (sex=0) | ~14-19 | 0.8698 ± 0.0924 | 0.1767 ± 0.0276 |
| Real NHANES | male (sex=1) | ~435-473 | 0.9569 ± 0.0066 | 0.0794 ± 0.0067 |
| Real NHANES | female (sex=0) | ~435-473 | 0.9540 ± 0.0088 | 0.0763 ± 0.0077 |

**Real NHANES shows no meaningful fairness gap** (AUC difference ~0.003, well
within seed-to-seed noise, on large, comparably-sized subgroups). **UCI's female
subgroup is too small to draw a conclusion either way** (n≈14-19 per seed; AUC std
0.092 vs. 0.035 for the male subgroup, and the point estimate is actually slightly
*higher* for female, not lower) — this reads as small-sample noise, not evidence of
a disparity, but the sample is genuinely too small to rule one out.

## 4. Reconciling the two threshold systems

The project's data label is built from a real clinical cutoff: 10-year ASCVD risk
≥7.5% (2013 ACC/AHA Pooled Cohort Equations — see `archive/investigation_v2_to_v6/experiments_v3/DATA_PROVENANCE.md`
for real NHANES; UCI's own `heart_disease_risk` label is a different, non-PCE
construct — angiographic disease presence — so the 7.5% comparison below is only
literally clinically grounded for NHANES, and is reported for UCI purely as an
analogous general-calibration diagnostic, not a clinical claim). The RECOURSE
PIPELINE's own operational gate uses the classifier's raw output PROBABILITY at
0.55 (high-risk / eligible for a search) and 0.45 (low-risk / eligible as a
recourse target) — a completely different, never-previously-reconciled scale.

**Empirical crossing point** — the classifier-probability bin (of 20, width 0.05)
at which observed test-set outcome frequency first reaches 7.5%:

| Dataset | Empirical 7.5%-crossing (classifier probability) | Pipeline's operational gate |
|---|---|---|
| UCI | 0.245 ± 0.040 (5/5 seeds) | 0.55 (high) / 0.45 (low) |
| Real NHANES | **0.075 ± 0.000 (5/5 seeds, exact across every seed)** | 0.55 (high) / 0.45 (low) |

**This is a real, large, precisely-quantified discrepancy, not just "different
scales":**
- On Real NHANES, the pipeline's 0.55 "high-risk" cutoff sits at **~7.3×** the
  classifier-probability level where observed outcomes actually reach the clinical
  7.5% threshold (0.55 / 0.075 ≈ 7.3). The 0.45 "low-risk" target-eligibility cutoff
  sits at **~6×** that level.
- On UCI, the ratio is smaller but still substantial: 0.55 / 0.245 ≈ **2.24×**.

**What this means in plain terms**: the "high-risk" cohort this entire project's
recourse search operates on (classifier probability > 0.55) is not "patients at or
above the 7.5% ASCVD guideline threshold" — it is a much more severely-scored
subset of patients, several times further out on the classifier's own probability
scale than the clinical threshold the data label itself was built from. Likewise,
the pool of training patients used as valid recourse TARGETS (probability < 0.45)
is not merely "below the clinical risk threshold" — it is well below it. Both gates
have operated, unexamined, at a much stricter operating point than the clinical
threshold underlying the project's own data label, for the entire six-session
history of this project.

**On the NHANES crossing being IDENTICAL across all 5 independently-trained
models** (0.075 in every single seed, zero variance): this is not a coincidence of
one lucky bin — it happening identically across five separately-trained classifiers
is itself informative: the crossing point is a robust, reproducible property of the
data/label relationship at this bin resolution (bin width 0.05), not an artifact of
one particular training run.

## 5. Should the pipeline's threshold be recalibrated?

**Not implemented this session.** Per the brief's own instruction (Phase 2 step 3:
"only implement if it's a small, safe change") and the working-style requirement to
stop and flag rather than silently touch an already-cited number: **this is not a
small or safe change.** The 0.45/0.55 gate determines, from the ground up:
- which test patients are even considered "high-risk" and enter the recourse
  search at all (`high_risk_test_idx` in every script in this project),
- which training patients are valid recourse targets (`low_risk_mask`),
- and therefore every previously-published success rate, real-value CVR, and
  certified-infeasibility percentage in `experiments_v2` through `experiments_v6`'s
  `FINDINGS.md` files, all of which are conditioned on this exact gate.

Recalibrating it would not fix a bug in the existing numbers — the existing numbers
are internally consistent given the gate they used — but it WOULD mean every one of
those numbers describes a different, non-comparable patient population than a
recalibrated version would. That is exactly the kind of change this brief's working
style section says to stop and report, not silently apply. **Recommendation, not
implemented**: a dedicated future session should decide, deliberately, whether
"high-risk" in this project's operational sense should mean "above the clinical
7.5% guideline threshold" (which would sharply widen the high-risk cohort and very
likely raise the success rate, since more patients — many closer to the low-risk
population already in the graph — would now qualify for a search) or continue to
mean "flagged by the classifier at a much stricter, empirically-higher-severity
level than the guideline threshold" (the status quo, which this audit shows was
never a deliberate choice — it was an unexamined inherited constant). Either choice
is defensible; what is not defensible is that it was never previously stated as a
choice at all.
