# src/ — final pipeline code

Core pipeline modules (`MD files/PIPELINE.md` phases P0-P15) plus the build scripts. Run from the repo root as `.venv/bin/python -m experiment_final.src.<module>`.

- `__init__.py` — package marker.
- `ablations.py` — Phase 13: tier1-only, no ontology, no inheritance, no guidance, soft-as-hard, no escalation, Euclidean, no SRA, ontology-as-features.
- `baselines.py` — Phase 12: FACE, DiCE, Growing Spheres, PACE-style, C-CHVAE, Wachter, reachable-set recourse.
- `calg.py` — fig_calg data: reproduces the seed-2 setup, verifies it against the stored run, caches the 40 neighbours of patient #139.
- `calg_col.py` — fig_calg_col: the CALG patient figure redrawn at IEEE one-column width (3.5 in) for the paper.
- `calg_draw.py` — draws fig_calg (SVG + PDF, 6.5 in) from the cache written by `calg.py compute`.
- `certify.py` — Phase 8: exhaustive 1-hop + 2-hop checks, certified infeasible vs graph-limited, ceiling diagnostic, closest miss, blocking-rule frequency.
- `common.py` — Shared paths, config, seeding, run provenance and never-overwrite helpers (PIPELINE P0).
- `evaluate.py` — Phase 14: metrics, stats (bootstrap CI, Wilcoxon+Holm), sensitivity, robustness to retraining, fairness tier-split, failure cases.
- `explain.py` — Phase 11: SHAP, guideline derivation, semifactual, statin/BP annotations, four-part template + traceability.
- `narrate.py` — template-only (no LLM) stepwise plan for a stored card: tagged text, `explain.trace()` + recomputation of expected values, then sentences, grey chips and numbered references; `render_html` for the report.
- `final_figs.py` — 2x3 ablation grid, retraining and rule-blocking figures, title-free architecture figure and the ICD-10-CM inheritance figure.
- `graph.py` — Phase 6: CALG k-NN latent graph, rule-pruned edges (reuse src/graph/query_attachment.py).
- `knowledge_layer.py` — Phase 2: patient concepts -> rule set (hard/soft/filter) with inheritance; clingo certificates + NumPy mirror.
- `latent.py` — Phase 5: type-aware VAE (reuse care_lg/src/vae/type_aware.py) + Riemannian metric (src/graph/riemannian_batch.py).
- `model.py` — Phase 4: MLP + XGBoost risk models, calibration, gate audit vs true risk, subgroup AUC, SHAP.
- `new_figures.py` — verified seed-2 reproduction + the Kaplan-Meier, baseline-failure-type and ontology-vs-rules figures used by `build_results.py`.
- `pooled_studies.py` — pooled-NHANES studies: learning curve, cycle-held-out validation, mortality validation.
- `prep.py` — Phase 3: 80/20 stratified split, scaler fit on train only, unscaled (recorded) copies kept.
- `recommend.py` — Phase 10: acts_on mapping, contraindication filter, skip current meds, escalation-tier ranking, precedent, recommendation cards.
- `report_final.py` — patient-card HTML and CSS helpers reused by `build_report.py`.
- `review_rules.py` — P2 deliverable: results/REVIEW_rules.md -- every knowledge row for human review + self-review notes.
- `run_seed.py` — Per-(dataset, seed[, variant]) driver for phases P3-P12 (+ the per-patient parts of P13/P14).
- `tier1.py` — Phase 7: certified real-patient recourse (gate-symmetric attachment, SRA, age-stratified targets, Dijkstra, recorded-value check).
- `tier2.py` — Phase 9: ontology-guided synthetic recourse (genetic search + guidance; Riemannian natural-gradient variant).
- `build_results.py` — aggregation + figures: NHANES 1999-2023 from `results/run_full_nhanes`, other datasets from the archived report_v3 runs; writes `results/tables`, `results/figures`.
- `rerun_changes.py` — writes `MD files/FULL_RERUN_CHANGES.md` (every changed NHANES 1999-2023 number, old → new, plus Tier-1 / contraindication flags).
- `build_report.py` — builds `results/report_final.html` (example card for patient #139, architecture, CALG figure, final results).
- `visualize.py` — palette, labels, `plaus_validity` and `inheritance` figure code used by the figure scripts.

Build order for the final outputs: `build_results aggregate` → `build_results figures` → `calg compute` + `calg draw` + `calg_col` + `final_figs` → `rerun_changes` → `build_report`. The paper (`paper/main_final.tex`) is edited by hand from these tables and figures.
Paths to the archived report_v3/v4 inputs are defined once in `common.py` (`OLD_RUNS`, `OLD_TABLES`, `OLD_TABLES_V4`).
