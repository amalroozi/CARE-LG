# CARE-LG pipeline
Tags: [LG] CARE-LG core, [SRA] v10 fix, [N2] CARE-N v2, [v9], [NEW].
Contributions: C1 tiered recourse (certified real -> ontology-guided synthetic -> certified none);
C2 patient-specific rules via ontology inheritance; C3 symbolic guidance of generation + escalation-aware recommendations;
C4 fairness of infeasibility (tier split by subgroup).

P0 Repro: new folder; seeds 0-4; SHA+config per run; never overwrite results.
P1 Data (adapters/): NHANES primary (+P_RXQ_RX reason codes, P_MCQ, P_KIQ_U) [NEW]; PCE >=7.5% label [LG]; UCI + Cardio as no-ontology robustness [LG][v10]; cohort flow.
P2 Knowledge: ICD-10-CM hierarchy [N2]; NHANES flags + Rx reason codes -> concepts [NEW]; ancestors; feature rules w/ patient-specific targets (BP <130/80 if diabetes/CKD; HbA1c by diabetes) [NEW]; intervention classes + acts_on + escalation tier [NEW]; contraindications inherited [N2]; skip current meds [NEW]; hard/soft/filter tiers [N2]; clingo + NumPy mirror, parity 100% [LG].
P3 Preprocess: 80/20 stratified; scaler on train; unscaled copy [LG].
P4 Model: MLP + XGBoost; AUC/Brier/calibration; gate vs true ASCVD%; subgroup AUC; 0.55/0.45; SHAP [LG][N2].
P5 Latent: type-aware VAE; Riemannian metric [LG].
P6 Graph: CALG k-NN, rule-pruned, Riemannian weights (Euclidean ablation) [LG].
P7 Tier 1: patient rule set; gate-symmetric attachment (B1); SRA; age-stratified targets; Dijkstra; recorded values (B2); certificate [LG][SRA][v9].
P8 Certify: exhaustive 1-hop + 2-hop; certified infeasible vs graph-limited; ceiling diagnostic; closest miss; blocking-rule frequency [LG][v9].
P9 Tier 2: ontology-guided genetic search (validity+proximity+sparsity+soft-rule+plausibility-diversity), guidance via allowed interventions, Riemannian natural-gradient variant, 3 diverse CFs; Tier 3 = certified none [N2][v9].
P10 Recommend: acts_on mapping; contraindication filter; skip current meds; rank escalation -> risk drop -> coverage; precedent (descriptive); card with tier label [N2][NEW].
P11 Explain: SHAP; guideline derivation; semifactual; statin/BP annotations; four-part template + traceability [LG][N2].
P12 Baselines: FACE, DiCE, Growing Spheres, PACE-style, C-CHVAE, Wachter, reachable-set verification (Kothari et al. 2024, linear model) [LG][NEW].
P13 Ablations: tier1-only, no ontology, no inheritance, no guidance, soft-as-hard, no escalation, Euclidean, no SRA, ontology-as-features.
P14 Evaluate: coverage, tier split, tier-1 violations=0, soft adherence, LOF/kNN plausibility, MAD proximity, sparsity, diversity, actionability, escalation appropriateness, contraindicated cards=0, latency; bootstrap CI, Wilcoxon+Holm; sensitivity; robustness to retraining; fairness tier split; failure cases.
P15 Outputs: unified HTML report; UI + single-patient on SRA/tiered; tests; github.io; paper.
