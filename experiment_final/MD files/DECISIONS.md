# DECISIONS.md: judgement calls in CARE-LG

Every non-obvious choice made while building and running CARE-LG, in the order it came up. Each entry gives what was decided, why, and where in the code it lives. Numbers quoted here come from the run outputs listed in `SUMMARY.md`.

## Scope, rules and repo hygiene

- **D-GIT.** No git writes of any kind: no commit, push, branch or PR (`../CLAUDE.md`). Every file in `experiment_final/` is left for the user to review and commit. `src/`, `scripts/`, `experiments_v8-v10/`, `experiments_caren/` and all earlier results were not edited. They are imported, or copied with attribution in the docstring.
- **D-NS. Package name clash.** Both the repo and experiment_final have a top-level `src` package. experiment_final code imports itself as `experiment_final.*` (a namespace package) and the CARE-LG code as `src.*`. Tests use `--import-mode=importlib` and put the repo root first on `sys.path` (`pytest.ini`, `tests/conftest.py`).
- **D-OVERWRITE. Never overwrite.** Every run goes into a new `results/runs/<timestamp>/`. Every table, figure and report is written through `common.new_path`, which adds `_v2`, `_v3`, … rather than replacing a file. Multi-run aggregation uses a symlink view (`runs/combined_<timestamp>/`) and never copies results.
- **D-PRIMARY. Primary dataset changed mid-run.** At 20:07, two minutes before the first full launch, the user changed `config.yaml` (adding `nhanes_pooled` as the primary dataset and `graph.k.nhanes_pooled: 120`), `CLAUDE_CODE_PROMPT.md` and `NHANES_POOLED.md`, and added a stub adapter. As a result, the first run (`20261004_200902`) scheduled `nhanes_pooled` jobs without the patient cap. I stopped those jobs and that run's orchestrator. Its finished NHANES-2017 and cardio seeds are kept and used. UCI (`20261004_204411`), pooled NHANES (`20261004_203823`) and the NHANES-2017 ontology-as-features variant (`20261004_203826`) ran as separate runs. Final tables come from a combined view of the folders marked `DONE`.

## Data (P1)

- **D-XPT. Extra NHANES 2017-2020 files.** `P_RXQ_RX`, `P_MCQ` and `P_KIQ_U` were not in `data/raw/nhanes_extra/`. I downloaded them from the public CDC site into that gitignored folder, which is the location the prompt names. The adapter still works without them (fewer concepts, no medications).
- **D-POOLED-DL. Pooled NHANES download.** All 11 continuous cycles (1999-2000 … 2021-2023) were fetched: every component plus its `.htm` codebook, and the 2019 Linked Mortality Files. All succeeded. Download log: `data/raw/nhanes_pooled/download_log.csv`. NHANES III (optional) was not used.
- **D-POOLED-CHECK. Pooled slice reproduces the 2017-2020 cohort exactly.** The pooled 2017-2020 slice reproduces `data/nhanes_real_full.csv` exactly: the same 4,539 SEQNs, and identical features, flags, PCE% and label. The PCE function is imported unchanged from `archive/.../build_real_nhanes.py`. This is enforced by a test (`test_adapter_contract.py`).
- **D-POOLED-HARM. Harmonisation choices.** The full list is in `DATA_PROVENANCE_pooled.md`. In brief:
  - Treated BP comes from `BPQ050A` (1999-2020) or `BPQ150` (2021-2023, which has no `BPQ050A`).
  - BP is the mean of the available readings; diastolic readings of 0 are treated as missing.
  - Older cycles use `LBDHDL`/`LBXHDD` for HDL.
  - Rx drug names are `RXD240B` (1999-2002) or `RXDDRUG` (2003-2020). 2021-2023 has no drug names, so its patients have no medications recorded.
  - Kidney: `KIQ020` in 1999-2000, `KIQ022`/`KIQ025` afterwards.
  - COPD: `MCQ160P` from 2017; before that, `MCQ160G` (emphysema) and `MCQ160K` (chronic bronchitis).
- **D-MORT. Mortality link.** The 2019 public LMF links 1999-2018. The pre-pandemic `P_` files assign new SEQNs that do not match the 2017-2018 (J) LMF, so 2017-2020 and 2021-2023 participants have no outcome. 24,518 participants are linked, with 4,544 deaths, of which 1,329 are CVD deaths (UCOD 001 or 005).
- **D-CARDIO. Cardio sample.** The cohort is a stratified subsample of 8,000 rows (the v10 loader default) with a fixed `random_state=12345`, so all seeds see the same cohort; seeds change the split and the models. I did not add the subsample size to `config.yaml`, which is locked.
- **D-UNWEIGHTED.** All analyses are unweighted (methods paper).

## Knowledge layer (P2)

- **D-RULES. Rule tiers.**
  - HARD is the CARE-LG admissibility predicate: immutables equal; age never decreases (0.1 y tolerance); total age horizon ≤ 3.05 y for tier 1; directional features never worsen beyond the per-feature noise tolerance. Keeping it unchanged means tier-1 certificates and the SRA guarantee stay comparable with v6-v10.
  - SOFT is the patient-specific guideline target range (resolved down ICD-10-CM: the deepest node wins, with an envelope on ties) plus the max-plausible change.
  - FILTER is contraindications (inherited down ICD-10-CM) and indications.
  - Consequence: inheritance changes targets, guidance, indications and contraindications, but never the hard gate.
- **D-INT. Integer arithmetic.** All rule arithmetic uses integers scaled ×1000 in both NumPy and clingo, so parity is exact rather than approximate. Parity is 100% on all four datasets, with and without inheritance (`test_rule_parity.py`). Against the original float32 `src` gate, any disagreement can only fall exactly on a tolerance boundary; the test checks this.
- **D-EPS. Per-dataset gate tolerances.** The CARE-LG gate takes per-feature noise tolerances from the adapter. For UCI and NHANES they equal `src/graph/constraints_ext.py::_DIRECTIONAL_EPS`. For cardio's binary features I set 0.5. The CARE-LG default of 1.0 for `increase` features would have let `active` drop from 1 to 0 without being flagged; v10 had this gap.
- **D-SCHEMA.** Knowledge tables are keyed by canonical clinical concepts (`systolic_bp`, `hba1c`, …). Adapters map their columns to concepts (`feature_meta.concept`), so `resting_bp` and `ap_hi` share the systolic-BP rules. I added `rule_id`, `min_meaningful_change` (feature rules) and `first_line`, `typical_effect`, `indication`, `indication_fallback` (interventions) to the stub CSV headers; the escalation planner and the indication filter need them.
- **D-EXTRA-CLASSES.** I added `alcohol_moderation` (cardio has an alcohol feature; HTN2017 Table 15) and `beta_blocker_nonselective`. The latter is not first-line for hypertension, but without it the prompt's example contraindication (non-selective beta-blocker with asthma, J45) could never fire. The planner ranks first-line classes first.
- **D-DIABETES-TYPE.** NHANES does not record diabetes type, so doctor-told diabetes maps to E11 (type 2).
- **D-MAXCHG.** `rules.soft_change_percentile: 95` is computed as the 95th percentile of |Δ| over admissible real CALG edges (`setup.json: data_maxchg_p95`). It is used only for features whose CSV `max_plausible_change` is blank. Every mutable feature has a literature-based value, so the percentile is reported but does not bind.
- **D-REVIEW.** All guideline rows are marked `verify=1`, because the citations were written from memory. Self-review notes are in `REVIEW_rules.md`. Codebook items were checked against the downloaded `.htm` files (MCQ230 code 37 = Thyroid; KIQ022/KIQ025/MCQ160G/MCQ160K wording).

## Models, latent space, graph (P3-P6)

- **D-SPLIT.** 80/20 stratified split with `random_state=seed`. The inner 80/20 split of train is used to fit (`fit`) and select (`val`) the risk models. The VAE and graph use the full train set.
- **D-ORACLE.** MLP = the existing CARE-LG `RiskClassifier`, trained with `train_blackbox_model`, unchanged. XGBoost uses fixed hyper-parameters (200 trees, depth 3, lr 0.05) with no tuning on val, so the val-AUC comparison is fair. The oracle is whichever has the higher val AUC; both are reported. Gradient methods (DiCE, Wachter, Riemannian NG) use the MLP as a differentiable surrogate when XGBoost is the oracle; validity is always judged by the oracle.
- **D-RECORDED.** Every rule check on a real patient uses the recorded values from the adapter, never values unscaled from the model space (CARE-LG B2).

## Tier 1 / certification (P7-P8)

- **D-TARGET. Tier-1 target rule.** Targets are training patients whose oracle risk is below the age-bracket cut-off (config `age_stratified_targets: true`; bottom risk quartile within the patient's age quartile) **and** below the patient's own risk. Without the second condition, Dijkstra can pick a "target" with higher risk than the source. This happens because NHANES's oldest bracket cut-off is 0.97-0.98, an artefact of the age-dominated PCE label. For transparency the outputs also report the share of targets below the global 0.45 and a threshold/target-definition sensitivity.
- **D-SRA-IDENTITY.** Under SRA with full query attachment, tier-1 success holds exactly when some target is admissible directly from x0. Graph-limited failures therefore cannot occur, and tier-1 failure is itself an exhaustive certificate over every training patient. The identity is checked patient by patient (`certification.csv: sra_identity_violations = 0`). The exhaustive 2-hop check (config `exhaustive_two_hop_check: true`) is still run, under hop-wise semantics, to show what a non-SRA search could add.
- **D-TIER1-SOFT.** Tier 1 enforces only the HARD tier, so a real endpoint can overshoot a soft guideline floor. Example: an SBP endpoint of 88 mmHg against a 90 mmHg floor. I did not change tier-1 semantics mid-run. Instead the soft-violation count of the target is reported for every arm (`soft_violations_true_rules`), and the `soft_as_hard` ablation shows the cost of enforcing soft rules as vetoes. Listed as a limitation.

## Tier 2 / Tier 3 (P9)

- **D-T2-COPY.** The CARE-N v2 generator is copied and adapted rather than imported, because it is bound to eICU state rows and APACHE strata. The hard tier is enforced by construction: immutables and age are frozen ("now" recourse, as in the CARE-LG stage-2 search), directional features move one way only, discrete features are snapped to their grid, and values stay within physiological bounds.
- **D-T2-ALL.** Tier 2 runs for **every** high-risk patient, so its quality can be compared with the baselines on the same population. The tiered method uses it only after a tier-1 failure, and its card uses the best valid counterfactual.
- **D-NG.** The Riemannian natural-gradient variant (`config.tier2.variant`) is reported as its own method (`tier2_riemannian_ng`); the tiered method uses the guided GA (`config.tier2.method`).
- **D-TIER3.** Tier 3 means two things: certified no real-patient recourse (exhaustive), and no valid synthetic counterfactual from the guided search. For tier-3 patients a validity-only search with double the budget and no guidance (from CARE-N's `ceiling_v2.py`) separates a *model ceiling* (not even that search reaches the goal) from a guided-search miss. The synthetic part of tier 3 is search-based, not a proof; the report says so.
- **D-MAD.** Binary features have MAD = 0. For them the proximity scale falls back to the SD, then to 1.

## Recommendations / explanations (P10-P11)

- **D-ESC. Escalation planner.** For each feature, allowed lifestyle classes are applied first, assuming additive typical effects; then the best single drug (first-line first, then by effect); then a second drug (combination). A drug is "overtreatment" if lifestyle alone would cover **every** feature it addresses. Escalation appropriateness is computed over cards with ≥1 recommendation, and is high for the main arm by construction. The `no_escalation` ablation (rank by risk drop, as CARE-N v2 does) is the contrast.
- **D-INDICATION.** Classes need an indication. Antihypertensives require baseline BP above the **patient-specific** target. Statins require PCE ≥7.5%, diabetes, ASCVD or above-target cholesterol. Diabetes drugs require a diabetes concept, or an above-normal glucose category in cardio. If no indication token can be assessed on a dataset, `indication_fallback` decides: weight management and lifestyle classes default to indicated, drugs do not.
- **D-SKIP-MEDS.** Classes the patient already takes are skipped and listed as "already taking" (intensification is not modelled). 2021-2023 has no drug names, so nothing is skipped there.
- **D-PRECEDENT.** Precedent is descriptive: the share of training patients for whom a class is indicated who already take it. Datasets without medications report none.
- **D-EXPLAIN.** Explanations with per-patient SHAP and full tag tracing are generated for the first 60 main-arm patients of each seed, to bound the cost of KernelSHAP. Every tag is machine-checked by `explain.trace`, and the traceability test reads them all.

## Baselines (P12)

- **D-CARLA.** CARLA is installed but fails to import (it needs TensorFlow), and installing TensorFlow would change the shared venv. Wachter and C-CHVAE are therefore faithful reimplementations; deviations are listed in `src/baselines.py`:
  - Wachter: y′ = 0; immutables and age frozen, as for DiCE here; a fixed λ schedule.
  - C-CHVAE: the CARE-LG type-aware VAE replaces the conditional HVAE; immutables are clamped after decoding.
- **D-REACH.** The authors' `reachml` 0.2.0a1 pins numpy 2.1, pandas 2.2 and a SCIP solver; the venv has pandas 3.0, so installing it would break other experiments. Instead I implemented an exhaustive discrete reachable set:
  - Action grid per feature: steps of `min_meaningful_change` in the admissible direction, up to `max_plausible_change`, within physiological bounds. Binary and ordinal features move one level at a time. Immutables and age are fixed.
  - Logistic regression is fitted on the same `fit` split. "Precluded" means no reachable point has LR risk below min(τ_p, LR risk).
  - This is exact on the grid, which corresponds to enumeration in the paper.
- **D-BASE-TARGET.** All methods get the same per-patient goal τ = min(τ_p, r0) and are judged by the same oracle. FACE uses the same tier-1 target set and k. Growing Spheres moves all features (no actionability), as the method specifies, so its hard violations are expected.

## Compute scoping

- **D-CAP. Pooled patient sample.** Pooled NHANES (26k training nodes; full query attachment needs a Riemannian distance to every node for each patient) evaluates a seeded uniform random sample of **1,000 high-risk test patients per seed**, out of about 3,000. Models, VAE, graph, targets and baselines still use every patient. The ontology-as-features variants evaluate 300 patients per seed.
- **D-FAST.** `resolve_rule_set` was sped up about 150× (cached per-concept rows) before the pooled run. Semantics are unchanged; parity was re-verified at 100%.
- **D-ESC-METRIC.** The escalation metric definition changed at aggregation level only (above); no rerun was needed.

## Not done / deferred

- The eICU demo adapter is an explicit stub (not in `config.datasets`). The MIMIC-IV adapter is planned.
- The UI integration (`../ui/` is outside experiment_final and was not edited) and a github.io publication (which would need a push) are not done. `single_patient.py` gives the per-patient tiered output from the command line, and `results/report.html` is the self-contained page that could be published.

## Final aggregation

- **D-REBUILD.** The final aggregation (`runs/combined_20261004_215313`, a symlink view of the four runs) was built twice. The first build exposed figure-layout problems: overlapping legend, labels and ticks, and an uninformative GA-validity panel that was replaced by soft-rule adherence. I did not delete the first build. Per the never-overwrite rule, the second build wrote `report_v2.html`, `examples_v2.md` and `figures/*_v2.*`, and **these `_v2` files are the final versions**. Tables were not re-saved: each recomputed table was checked identical to the saved CSV (all 19 matched), and the saved CSVs were reused. The superseded first-build files (`report.html`, `examples.md`, unsuffixed figures) can be deleted by the user.
- **D-TEST-SCOPE.** The results-level guarantee tests (zero contraindicated cards, clean traces) apply to the main method only. Ablation-variant folders (`<dataset>__ontology_as_features`) deliberately run without the symbolic filter and are excluded from them. A separate test checks that, in that ablation, the tracer flags every contraindicated recommendation, and nothing else.

## Rename

- **D-RENAME (2026-10-05).** At the user's request the folder `experiment_final/` was renamed to `experiment_final/`. Python imports (`experiment_final.*` → `experiment_final.*`) and `pytest.ini` were updated, and the symlinks in `runs/combined_20261004_215313/` were repointed. Historical records still mention the old name, deliberately unchanged: run logs, `setup.json`, build manifests, `CLAUDE_CODE_PROMPT.md`, and the entries above. All 26 tests pass after the rename.
- **D-REPORT-LAYOUT (2026-10-05).** At the user's request, `results/report_v2.html` was replaced in place by a condensed report in the layout of `../results/final_report.html`, built by `src/report_final.py`. The order is Final Output (one real pooled-NHANES patient: seed 2, test #139, Tier 1, chosen because it exercises inheritance, a contraindication, skipped current meds and escalation to combination), then the Architecture Walkthrough of that same patient, then Final Results (one headline table, five numbered figures, one supporting table, each number stated once). The long-form tables remain in `results/tables/` and the narrative in `SUMMARY.md`.
- **D-NEWFIGS (2026-10-05).** Figures 11-18 and `results/report_v3.html` (`src/new_figures.py`).
  - Figures 12, 13, 14, 17 and 18 and the PCE panel of 15 use only `results/tables` and the cohort file.
  - Figures 11, 15 (model-gate panel) and 16 need data the runs did not store, so pooled-NHANES seed 2 was re-run with the identical code path. The re-run matched the stored run exactly on: oracle, 3,147,360 edges, 99.22141096% pruned, age-bracket thresholds, patient #139's risk, and her target counts (5,334 / 394). All 6,000 baseline patient-method valid / hard-violation flags matched `methods.csv` (0 mismatches). Verification record: `data/processed/new_figures_seed2.json`.
  - Kaplan-Meier is implemented directly because `lifelines` is not installed. CVD death = UCOD 001/005; other deaths are censored.
  - The first render of Fig. 12 had a shared-axis label bug: method names came from another panel. Figures 11-18 and `report_v3.html`, all created by this same request, were regenerated in place after the fix. No pre-existing file was overwritten.

## Report v4 (2026-10-05)

- **D-V4-NORERUN.** A full rerun (all high-risk patients; full 68,590-row Kaggle data) was started and then stopped at the user's request. `results/run_full_nhanes/` holds two unfinished seeds (checkpoints at 400 patients each) and is **not** used anywhere. `report_v4.html`, `paper/main_v4.tex`, `results/tables_v4/` and `results/figures_v4/` are built only from the report_v3 results (`runs/combined_20261004_215313`) and the verified seed-2 caches. No experiment was rerun.
- **D-V4-WACHTER.** Wachter was removed by re-aggregating the existing per-patient results without its rows. Every mean, SD and count in the headline, ablations, methods, certification and fairness tables is identical to report_v3. Holm-adjusted p-values change because the comparison family is one method smaller. Bootstrap CI bounds shift slightly because the shared resampling stream changes; the report shows no CIs.
- **D-V4-ACCURACY.** Classifier accuracy (threshold 0.5) comes from `tables_v4/accuracy_backfill_v3runs.csv`. The report_v3 models were re-fitted with identical code and seeds, and each accuracy was accepted only because the re-fitted validation and test AUCs equal the stored ones to 1e-9 (all 40 model-seeds).
- **D-V4-NAMES.** Display names: "NHANES 1999-2023", "NHANES 2017-2020", "Kaggle CVD-8k (8,000-patient stratified subsample of the Kaggle Cardiovascular Disease dataset, Ulianova 2019)". Internal dataset IDs in the id-keyed CSVs are unchanged; the human-readable copies are in `tables_v4/labelled/`.
- **D-V4-EXAMPLES.** The Example-output patients are chosen by rule, not by hand. The rule requires: an explanation exists; at least 3 recommendations; an inherited contraindication; a target below the global 0.45; and no change below a guideline floor. Among those, it prefers an inherited target rule, then the fewest soft violations, then the smallest relative change.
- **D-V4-FIGS.** Removed: the sensitivity/scale figure and the fairness figure; the fairness result is now 2 sentences and a table. Figures are renumbered 1-10. The architecture figure has no adapter box and its steps are numbered 1-7. In-place rebuilds were made only of files created by this same request (`fig01_architecture`, `report_v4.html`).
- **D-V4-TODO.** None: every number requested for v4 exists in the report_v3 results (`tables_v4/TODO_v4.json` = []).
