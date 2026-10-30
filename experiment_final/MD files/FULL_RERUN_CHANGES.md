# Full NHANES 1999-2023 rerun: what changed from report_v3

Old = `archive/experiment_final_prev/v3/tables/` (report_v3: 1,000 randomly sampled high-risk test patients per seed; ontology-as-features capped at 300). New = `results/tables/` from `results/run_full_nhanes/` (all high-risk test patients per seed, no cap, 5 seeds, all ablation arms). Wachter excluded (removed from the report). Kaggle CVD-8k, NHANES 2017-2020 and UCI Heart were not rerun and are unchanged. Only cells that differ are listed; machine-readable copy: `results/tables/full_rerun_changes.csv`.

High-risk test patients evaluated per seed (new): 3,073, 2,931, 3,010, 3,016, 2,972 (total 15,002; report_v3: 5,000).

## Flags: Tier-1 violations and contraindicated cards

**Non-zero values on arms that must be zero (main method and ordinary variants): NONE.**

| scope | quantity | value |
|---|---|---|
| nhanes_pooled | Tier-1 real-value violations (headline) | 0 |
| nhanes_pooled | contraindicated cards (headline) | 0 |
| nhanes_pooled | Tier-1 paths with a real-value violation at any node / endpoint (of 9,471 paths) | 0 / 0 |
| nhanes_pooled / arm euclidean | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |
| nhanes_pooled / arm main | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |
| nhanes_pooled / arm no_escalation | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |
| nhanes_pooled / arm no_guidance | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |
| nhanes_pooled / arm no_inheritance | contraindicated recommendations / Tier-1 paths with a real violation | 69 / 0 |
| nhanes_pooled / arm no_ontology | contraindicated recommendations / Tier-1 paths with a real violation | 67 / 0 |
| nhanes_pooled / arm no_sra | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 113 |
| nhanes_pooled / arm ontology_as_features | contraindicated recommendations / Tier-1 paths with a real violation | 86 / 0 |
| nhanes_pooled / arm soft_as_hard | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |
| nhanes_pooled / arm tier1_only | contraindicated recommendations / Tier-1 paths with a real violation | 0 / 0 |

Non-zero values on the `no_sra`, `no_inheritance`, `no_ontology`, `ontology_as_features` ablation arms are the intended cost of removing the component, not failures of the full method.

## Coverage, Tier 1/2/3, risk drop, guarantees (`headline_tiered.csv`)

| dataset | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|
| nhanes_pooled | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier1_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | tier1_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | tier2_mean | 33.02 | 33.24 | +0.2171 |
| nhanes_pooled | tier2_sd | 9.58 | 10.13 | +0.5487 |
| nhanes_pooled | tier3_mean | 3.72 | 3.63 | -0.0909 |
| nhanes_pooled | tier3_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | covered_tier12_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | covered_tier12_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | card_covered_f_mean | 94.90 | 95.12 | +0.2245 |
| nhanes_pooled | card_covered_f_sd | 8.72 | 8.16 | -0.5617 |
| nhanes_pooled | actionability_mean | 0.9363 | 0.9352 | -0.0011 |
| nhanes_pooled | actionability_sd | 0.008204 | 0.009054 | +0.0009 |
| nhanes_pooled | risk_drop_mean | 0.3171 | 0.3159 | -0.0012 |
| nhanes_pooled | risk_drop_sd | 0.04126 | 0.04711 | +0.0058 |
| nhanes_pooled | global_low_mean | 44.14 | 44.20 | +0.0575 |
| nhanes_pooled | global_low_sd | 7.52 | 8.36 | +0.8342 |
| nhanes_pooled | n_recs_mean | 4.42 | 4.43 | +0.0109 |
| nhanes_pooled | n_recs_sd | 0.3947 | 0.384 | -0.0107 |
| nhanes_pooled | n_high_risk_total | 5,000 | 15,002 | +10,002 |

## Baselines and CARE-LG (all methods) (`methods_comparison.csv`)

| dataset | method | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|
| nhanes_pooled | care_lg_tiered | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | care_lg_tiered | valid_f_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | care_lg_tiered | valid_f_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | care_lg_tiered | valid_and_admissible_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | care_lg_tiered | valid_and_admissible_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | care_lg_tiered | prox_mad_l1_mean | 9.11 | 9.18 | +0.0711 |
| nhanes_pooled | care_lg_tiered | prox_mad_l1_sd | 0.4704 | 0.4358 | -0.0346 |
| nhanes_pooled | care_lg_tiered | l0_mean | 4.26 | 4.26 | -0.0022 |
| nhanes_pooled | care_lg_tiered | l0_sd | 0.2467 | 0.2619 | +0.0153 |
| nhanes_pooled | care_lg_tiered | plaus_lof_mean | 1.30 | 1.31 | +0.0045 |
| nhanes_pooled | care_lg_tiered | plaus_lof_sd | 0.07964 | 0.08614 | +0.0065 |
| nhanes_pooled | care_lg_tiered | plaus_knn_z_mean | 0.7724 | 0.7751 | +0.0027 |
| nhanes_pooled | care_lg_tiered | plaus_knn_z_sd | 0.09958 | 0.1068 | +0.0072 |
| nhanes_pooled | care_lg_tiered | soft_violations_mean | 2.62 | 2.62 | -0.0011 |
| nhanes_pooled | care_lg_tiered | soft_violations_sd | 0.2513 | 0.2577 | +0.0064 |
| nhanes_pooled | care_lg_tiered | diversity_mean | 2.10 | 2.13 | +0.0213 |
| nhanes_pooled | care_lg_tiered | diversity_sd | 0.3887 | 0.3334 | -0.0552 |
| nhanes_pooled | care_lg_tiered | actionability_mean | 0.9363 | 0.9352 | -0.0011 |
| nhanes_pooled | care_lg_tiered | actionability_sd | 0.008215 | 0.009057 | +0.0008 |
| nhanes_pooled | care_lg_tiered | latency_s_mean | 0.08555 | 0.04398 | -0.0416 |
| nhanes_pooled | care_lg_tiered | latency_s_sd | 0.009213 | 0.008315 | -0.0009 |
| nhanes_pooled | cchvae | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | cchvae | valid_f_mean | 49.30 | 49.15 | -0.1504 |
| nhanes_pooled | cchvae | valid_f_sd | 6.23 | 6.45 | +0.2197 |
| nhanes_pooled | cchvae | valid_and_admissible_mean | 10.02 | 9.92 | -0.0988 |
| nhanes_pooled | cchvae | valid_and_admissible_sd | 1.82 | 1.93 | +0.1102 |
| nhanes_pooled | cchvae | hard_violation_f_mean | 79.53 | 79.70 | +0.1764 |
| nhanes_pooled | cchvae | hard_violation_f_sd | 3.78 | 3.88 | +0.105 |
| nhanes_pooled | cchvae | prox_mad_l1_mean | 6.74 | 6.86 | +0.1219 |
| nhanes_pooled | cchvae | prox_mad_l1_sd | 0.3132 | 0.2816 | -0.0316 |
| nhanes_pooled | cchvae | l0_mean | 5.97 | 5.96 | -0.0016 |
| nhanes_pooled | cchvae | l0_sd | 0.01102 | 0.003873 | -0.0071 |
| nhanes_pooled | cchvae | plaus_lof_mean | 1.01 | 1.01 | +0.0001 |
| nhanes_pooled | cchvae | plaus_lof_sd | 0.004571 | 0.004725 | +0.0002 |
| nhanes_pooled | cchvae | plaus_knn_z_mean | 0.4693 | 0.4648 | -0.0045 |
| nhanes_pooled | cchvae | plaus_knn_z_sd | 0.01916 | 0.01697 | -0.0022 |
| nhanes_pooled | cchvae | soft_violations_mean | 2.43 | 2.43 | -0.0022 |
| nhanes_pooled | cchvae | soft_violations_sd | 0.1654 | 0.1624 | -0.003 |
| nhanes_pooled | cchvae | actionability_mean | 0.7026 | 0.7058 | +0.0032 |
| nhanes_pooled | cchvae | actionability_sd | 0.01578 | 0.01775 | +0.002 |
| nhanes_pooled | cchvae | latency_s_mean | 0.01248 | 0.006682 | -0.0058 |
| nhanes_pooled | cchvae | latency_s_sd | 0.006934 | 0.003565 | -0.0034 |
| nhanes_pooled | dice | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | dice | valid_f_mean | 81.28 | 81.38 | +0.0987 |
| nhanes_pooled | dice | valid_f_sd | 13.71 | 13.22 | -0.4912 |
| nhanes_pooled | dice | valid_and_admissible_mean | 11.74 | 11.97 | +0.2267 |
| nhanes_pooled | dice | valid_and_admissible_sd | 3.65 | 3.39 | -0.2578 |
| nhanes_pooled | dice | hard_violation_f_mean | 85.36 | 85.12 | -0.2456 |
| nhanes_pooled | dice | hard_violation_f_sd | 4.22 | 3.91 | -0.3059 |
| nhanes_pooled | dice | prox_mad_l1_mean | 11.70 | 11.76 | +0.0687 |
| nhanes_pooled | dice | prox_mad_l1_sd | 0.3516 | 0.3776 | +0.026 |
| nhanes_pooled | dice | l0_mean | 5.98 | 5.98 | +0 |
| nhanes_pooled | dice | l0_sd | 0.006992 | 0.00217 | -0.0048 |
| nhanes_pooled | dice | plaus_lof_mean | 2.32 | 2.34 | +0.0169 |
| nhanes_pooled | dice | plaus_lof_sd | 0.1012 | 0.1142 | +0.013 |
| nhanes_pooled | dice | plaus_knn_z_mean | 2.11 | 2.12 | +0.0167 |
| nhanes_pooled | dice | plaus_knn_z_sd | 0.1018 | 0.1149 | +0.0132 |
| nhanes_pooled | dice | soft_violations_mean | 3.99 | 4.00 | +0.0099 |
| nhanes_pooled | dice | soft_violations_sd | 0.04846 | 0.07737 | +0.0289 |
| nhanes_pooled | dice | diversity_mean | 9.65 | 9.68 | +0.0284 |
| nhanes_pooled | dice | diversity_sd | 0.4343 | 0.375 | -0.0593 |
| nhanes_pooled | dice | actionability_mean | 0.7266 | 0.7268 | +0.0003 |
| nhanes_pooled | dice | actionability_sd | 0.02359 | 0.03005 | +0.0065 |
| nhanes_pooled | dice | latency_s_mean | 0.2171 | 0.1117 | -0.1054 |
| nhanes_pooled | dice | latency_s_sd | 0.00225 | 0.00663 | +0.0044 |
| nhanes_pooled | face | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | face | valid_and_admissible_mean | 0.48 | 0.4191 | -0.0609 |
| nhanes_pooled | face | valid_and_admissible_sd | 0.1483 | 0.1535 | +0.0051 |
| nhanes_pooled | face | hard_violation_f_mean | 99.52 | 99.58 | +0.0609 |
| nhanes_pooled | face | hard_violation_f_sd | 0.1483 | 0.1535 | +0.0051 |
| nhanes_pooled | face | prox_mad_l1_mean | 4.40 | 4.39 | -0.015 |
| nhanes_pooled | face | prox_mad_l1_sd | 0.2231 | 0.1603 | -0.0628 |
| nhanes_pooled | face | l0_mean | 6.03 | 6.04 | +0.0027 |
| nhanes_pooled | face | l0_sd | 0.01869 | 0.02367 | +0.005 |
| nhanes_pooled | face | plaus_lof_mean | 1.06 | 1.06 | +0.0013 |
| nhanes_pooled | face | plaus_lof_sd | 0.004766 | 0.00286 | -0.0019 |
| nhanes_pooled | face | plaus_knn_z_mean | 0.4924 | 0.4933 | +0.0009 |
| nhanes_pooled | face | plaus_knn_z_sd | 0.01159 | 0.005394 | -0.0062 |
| nhanes_pooled | face | soft_violations_mean | 2.24 | 2.22 | -0.0152 |
| nhanes_pooled | face | soft_violations_sd | 0.03232 | 0.01423 | -0.0181 |
| nhanes_pooled | face | actionability_mean | 0.5971 | 0.5957 | -0.0015 |
| nhanes_pooled | face | actionability_sd | 0.01432 | 0.012 | -0.0023 |
| nhanes_pooled | face | latency_s_mean | 0.07302 | 0.03308 | -0.0399 |
| nhanes_pooled | face | latency_s_sd | 0.0004765 | 0.004967 | +0.0045 |
| nhanes_pooled | growing_spheres | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | growing_spheres | valid_f_mean | 99.18 | 99.36 | +0.1808 |
| nhanes_pooled | growing_spheres | valid_f_sd | 0.2775 | 0.1514 | -0.1261 |
| nhanes_pooled | growing_spheres | valid_and_admissible_mean | 1.32 | 1.57 | +0.2546 |
| nhanes_pooled | growing_spheres | valid_and_admissible_sd | 2.95 | 3.50 | +0.5508 |
| nhanes_pooled | growing_spheres | hard_violation_f_mean | 98.67 | 98.42 | -0.2555 |
| nhanes_pooled | growing_spheres | hard_violation_f_sd | 2.97 | 3.52 | +0.5525 |
| nhanes_pooled | growing_spheres | prox_mad_l1_mean | 2.44 | 2.48 | +0.0432 |
| nhanes_pooled | growing_spheres | prox_mad_l1_sd | 0.6144 | 0.6008 | -0.0136 |
| nhanes_pooled | growing_spheres | plaus_lof_mean | 1.23 | 1.23 | +0.0034 |
| nhanes_pooled | growing_spheres | plaus_lof_sd | 0.0746 | 0.07602 | +0.0014 |
| nhanes_pooled | growing_spheres | plaus_knn_z_mean | 0.8795 | 0.8856 | +0.0061 |
| nhanes_pooled | growing_spheres | plaus_knn_z_sd | 0.1067 | 0.1042 | -0.0025 |
| nhanes_pooled | growing_spheres | soft_violations_mean | 2.32 | 2.33 | +0.0055 |
| nhanes_pooled | growing_spheres | soft_violations_sd | 0.04206 | 0.01984 | -0.0222 |
| nhanes_pooled | growing_spheres | actionability_mean | 0.6833 | 0.6831 | -0.0002 |
| nhanes_pooled | growing_spheres | actionability_sd | 0.04857 | 0.03231 | -0.0163 |
| nhanes_pooled | growing_spheres | latency_s_mean | 0.007406 | 0.004257 | -0.0031 |
| nhanes_pooled | growing_spheres | latency_s_sd | 0.00492 | 0.00261 | -0.0023 |
| nhanes_pooled | pace_style | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | pace_style | valid_f_mean | 67.26 | 66.44 | -0.8219 |
| nhanes_pooled | pace_style | valid_f_sd | 5.37 | 5.55 | +0.181 |
| nhanes_pooled | pace_style | valid_and_admissible_mean | 67.26 | 66.44 | -0.8219 |
| nhanes_pooled | pace_style | valid_and_admissible_sd | 5.37 | 5.55 | +0.181 |
| nhanes_pooled | pace_style | prox_mad_l1_mean | 4.20 | 4.22 | +0.0253 |
| nhanes_pooled | pace_style | prox_mad_l1_sd | 0.8403 | 0.8477 | +0.0074 |
| nhanes_pooled | pace_style | l0_mean | 5.87 | 5.87 | +0.0016 |
| nhanes_pooled | pace_style | l0_sd | 0.04007 | 0.04383 | +0.0038 |
| nhanes_pooled | pace_style | plaus_lof_mean | 1.32 | 1.32 | +0.0012 |
| nhanes_pooled | pace_style | plaus_lof_sd | 0.09403 | 0.09517 | +0.0011 |
| nhanes_pooled | pace_style | plaus_knn_z_mean | 0.89 | 0.8883 | -0.0017 |
| nhanes_pooled | pace_style | plaus_knn_z_sd | 0.09597 | 0.09597 | +0 |
| nhanes_pooled | pace_style | soft_violations_mean | 2.12 | 2.12 | -0.0055 |
| nhanes_pooled | pace_style | soft_violations_sd | 0.1177 | 0.1342 | +0.0164 |
| nhanes_pooled | pace_style | actionability_mean | 0.954 | 0.9551 | +0.0012 |
| nhanes_pooled | pace_style | actionability_sd | 0.0071 | 0.006442 | -0.0007 |
| nhanes_pooled | pace_style | latency_s_mean | 0.009157 | 0.005283 | -0.0039 |
| nhanes_pooled | pace_style | latency_s_sd | 0.00318 | 0.001535 | -0.0016 |
| nhanes_pooled | reachable_set_lr | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | reachable_set_lr | valid_f_mean | 27.82 | 28.13 | +0.3133 |
| nhanes_pooled | reachable_set_lr | valid_f_sd | 7.08 | 7.59 | +0.5064 |
| nhanes_pooled | reachable_set_lr | valid_and_admissible_mean | 27.82 | 28.13 | +0.3133 |
| nhanes_pooled | reachable_set_lr | valid_and_admissible_sd | 7.08 | 7.59 | +0.5064 |
| nhanes_pooled | reachable_set_lr | prox_mad_l1_mean | 4.10 | 4.28 | +0.1807 |
| nhanes_pooled | reachable_set_lr | prox_mad_l1_sd | 0.7117 | 0.7148 | +0.0031 |
| nhanes_pooled | reachable_set_lr | l0_mean | 1.91 | 1.96 | +0.0494 |
| nhanes_pooled | reachable_set_lr | l0_sd | 0.2261 | 0.202 | -0.0241 |
| nhanes_pooled | reachable_set_lr | plaus_lof_mean | 1.36 | 1.37 | +0.0105 |
| nhanes_pooled | reachable_set_lr | plaus_lof_sd | 0.08662 | 0.09399 | +0.0074 |
| nhanes_pooled | reachable_set_lr | plaus_knn_z_mean | 0.9177 | 0.9305 | +0.0129 |
| nhanes_pooled | reachable_set_lr | plaus_knn_z_sd | 0.0833 | 0.08435 | +0.0011 |
| nhanes_pooled | reachable_set_lr | soft_violations_mean | 1.89 | 1.90 | +0.0082 |
| nhanes_pooled | reachable_set_lr | soft_violations_sd | 0.09843 | 0.1019 | +0.0035 |
| nhanes_pooled | reachable_set_lr | actionability_mean | 0.8683 | 0.8675 | -0.0008 |
| nhanes_pooled | reachable_set_lr | actionability_sd | 0.03454 | 0.02548 | -0.0091 |
| nhanes_pooled | reachable_set_lr | latency_s_mean | 0.006291 | 0.003153 | -0.0031 |
| nhanes_pooled | reachable_set_lr | latency_s_sd | 0.0001491 | 0.0003062 | +0.0002 |
| nhanes_pooled | tier1_real_only | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier1_real_only | valid_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | tier1_real_only | valid_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | tier1_real_only | valid_and_admissible_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | tier1_real_only | valid_and_admissible_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | tier1_real_only | prox_mad_l1_mean | 10.18 | 10.22 | +0.0388 |
| nhanes_pooled | tier1_real_only | prox_mad_l1_sd | 0.683 | 0.6935 | +0.0105 |
| nhanes_pooled | tier1_real_only | l0_mean | 5.29 | 5.28 | -0.0137 |
| nhanes_pooled | tier1_real_only | l0_sd | 0.01215 | 0.01633 | +0.0042 |
| nhanes_pooled | tier1_real_only | plaus_lof_mean | 1.16 | 1.16 | +0.0038 |
| nhanes_pooled | tier1_real_only | plaus_lof_sd | 0.01689 | 0.02238 | +0.0055 |
| nhanes_pooled | tier1_real_only | plaus_knn_z_mean | 0.5595 | 0.5603 | +0.0008 |
| nhanes_pooled | tier1_real_only | plaus_knn_z_sd | 0.02165 | 0.02691 | +0.0053 |
| nhanes_pooled | tier1_real_only | soft_violations_mean | 2.61 | 2.60 | -0.012 |
| nhanes_pooled | tier1_real_only | soft_violations_sd | 0.1995 | 0.2191 | +0.0196 |
| nhanes_pooled | tier1_real_only | actionability_mean | 0.9185 | 0.9167 | -0.0019 |
| nhanes_pooled | tier1_real_only | actionability_sd | 0.002076 | 0.001566 | -0.0005 |
| nhanes_pooled | tier1_real_only | latency_s_mean | 0.001503 | 0.0007024 | -0.0008 |
| nhanes_pooled | tier1_real_only | latency_s_sd | 6.398e-05 | 6.512e-05 | +0 |
| nhanes_pooled | tier2_guided_ga | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_guided_ga | valid_f_mean | 96.18 | 96.26 | +0.0842 |
| nhanes_pooled | tier2_guided_ga | valid_f_sd | 8.37 | 8.11 | -0.2663 |
| nhanes_pooled | tier2_guided_ga | valid_and_admissible_mean | 96.18 | 96.26 | +0.0842 |
| nhanes_pooled | tier2_guided_ga | valid_and_admissible_sd | 8.37 | 8.11 | -0.2663 |
| nhanes_pooled | tier2_guided_ga | prox_mad_l1_mean | 5.67 | 5.72 | +0.0425 |
| nhanes_pooled | tier2_guided_ga | prox_mad_l1_sd | 0.7498 | 0.7568 | +0.0071 |
| nhanes_pooled | tier2_guided_ga | l0_mean | 2.04 | 2.05 | +0.0149 |
| nhanes_pooled | tier2_guided_ga | l0_sd | 0.03936 | 0.04708 | +0.0077 |
| nhanes_pooled | tier2_guided_ga | plaus_lof_mean | 1.35 | 1.35 | +0.0017 |
| nhanes_pooled | tier2_guided_ga | plaus_lof_sd | 0.09575 | 0.09828 | +0.0025 |
| nhanes_pooled | tier2_guided_ga | plaus_knn_z_mean | 0.8998 | 0.8997 | -0.0001 |
| nhanes_pooled | tier2_guided_ga | plaus_knn_z_sd | 0.1055 | 0.1064 | +0.0009 |
| nhanes_pooled | tier2_guided_ga | soft_violations_mean | 2.47 | 2.47 | -0.0053 |
| nhanes_pooled | tier2_guided_ga | soft_violations_sd | 0.2128 | 0.2262 | +0.0134 |
| nhanes_pooled | tier2_guided_ga | diversity_mean | 1.96 | 1.95 | -0.011 |
| nhanes_pooled | tier2_guided_ga | diversity_sd | 0.2031 | 0.2119 | +0.0088 |
| nhanes_pooled | tier2_guided_ga | actionability_mean | 0.9767 | 0.978 | +0.0012 |
| nhanes_pooled | tier2_guided_ga | actionability_sd | 0.00712 | 0.007613 | +0.0005 |
| nhanes_pooled | tier2_guided_ga | latency_s_mean | 0.2224 | 0.1146 | -0.1079 |
| nhanes_pooled | tier2_guided_ga | latency_s_sd | 0.005135 | 0.01051 | +0.0054 |
| nhanes_pooled | tier2_riemannian_ng | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_riemannian_ng | valid_f_mean | 25.64 | 24.65 | -0.9916 |
| nhanes_pooled | tier2_riemannian_ng | valid_f_sd | 2.01 | 1.77 | -0.237 |
| nhanes_pooled | tier2_riemannian_ng | valid_and_admissible_mean | 25.64 | 24.65 | -0.9916 |
| nhanes_pooled | tier2_riemannian_ng | valid_and_admissible_sd | 2.01 | 1.77 | -0.237 |
| nhanes_pooled | tier2_riemannian_ng | prox_mad_l1_mean | 4.74 | 4.96 | +0.2206 |
| nhanes_pooled | tier2_riemannian_ng | prox_mad_l1_sd | 0.3226 | 0.2658 | -0.0568 |
| nhanes_pooled | tier2_riemannian_ng | l0_mean | 2.75 | 2.78 | +0.0337 |
| nhanes_pooled | tier2_riemannian_ng | l0_sd | 0.12 | 0.09562 | -0.0244 |
| nhanes_pooled | tier2_riemannian_ng | plaus_lof_mean | 1.03 | 1.03 | -0.0009 |
| nhanes_pooled | tier2_riemannian_ng | plaus_lof_sd | 0.002711 | 0.002624 | -0.0001 |
| nhanes_pooled | tier2_riemannian_ng | plaus_knn_z_mean | 0.5264 | 0.5169 | -0.0094 |
| nhanes_pooled | tier2_riemannian_ng | plaus_knn_z_sd | 0.01381 | 0.01283 | -0.001 |
| nhanes_pooled | tier2_riemannian_ng | soft_violations_mean | 2.47 | 2.45 | -0.0167 |
| nhanes_pooled | tier2_riemannian_ng | soft_violations_sd | 0.1148 | 0.1235 | +0.0087 |
| nhanes_pooled | tier2_riemannian_ng | actionability_mean | 0.9686 | 0.9706 | +0.002 |
| nhanes_pooled | tier2_riemannian_ng | actionability_sd | 0.007212 | 0.006803 | -0.0004 |
| nhanes_pooled | tier2_riemannian_ng | latency_s_mean | 0.05577 | 0.02805 | -0.0277 |
| nhanes_pooled | tier2_riemannian_ng | latency_s_sd | 0.00289 | 0.001939 | -0.001 |
| nhanes_pooled | tier2_unguided_ga | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_unguided_ga | valid_f_mean | 94.48 | 94.47 | -0.0096 |
| nhanes_pooled | tier2_unguided_ga | valid_f_sd | 8.82 | 8.79 | -0.0356 |
| nhanes_pooled | tier2_unguided_ga | valid_and_admissible_mean | 94.48 | 94.47 | -0.0096 |
| nhanes_pooled | tier2_unguided_ga | valid_and_admissible_sd | 8.82 | 8.79 | -0.0356 |
| nhanes_pooled | tier2_unguided_ga | prox_mad_l1_mean | 5.19 | 5.24 | +0.048 |
| nhanes_pooled | tier2_unguided_ga | prox_mad_l1_sd | 0.7766 | 0.7992 | +0.0226 |
| nhanes_pooled | tier2_unguided_ga | l0_mean | 2.04 | 2.04 | +0.0087 |
| nhanes_pooled | tier2_unguided_ga | l0_sd | 0.04714 | 0.06252 | +0.0154 |
| nhanes_pooled | tier2_unguided_ga | plaus_lof_mean | 1.36 | 1.36 | +0.0016 |
| nhanes_pooled | tier2_unguided_ga | plaus_lof_sd | 0.09794 | 0.1022 | +0.0042 |
| nhanes_pooled | tier2_unguided_ga | plaus_knn_z_mean | 0.9335 | 0.9322 | -0.0013 |
| nhanes_pooled | tier2_unguided_ga | plaus_knn_z_sd | 0.115 | 0.1185 | +0.0035 |
| nhanes_pooled | tier2_unguided_ga | soft_violations_mean | 2.49 | 2.48 | -0.0128 |
| nhanes_pooled | tier2_unguided_ga | soft_violations_sd | 0.2401 | 0.2513 | +0.0112 |
| nhanes_pooled | tier2_unguided_ga | diversity_mean | 2.16 | 2.16 | +0.0067 |
| nhanes_pooled | tier2_unguided_ga | diversity_sd | 0.2676 | 0.2907 | +0.0231 |
| nhanes_pooled | tier2_unguided_ga | actionability_mean | 0.9756 | 0.9766 | +0.001 |
| nhanes_pooled | tier2_unguided_ga | actionability_sd | 0.008467 | 0.008796 | +0.0003 |
| nhanes_pooled | tier2_unguided_ga | latency_s_mean | 0.2193 | 0.113 | -0.1063 |
| nhanes_pooled | tier2_unguided_ga | latency_s_sd | 0.00522 | 0.01008 | +0.0049 |

## Ablation arms (`ablations.csv`)

| dataset | arm | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|
| nhanes_pooled | euclidean | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | euclidean | covered_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | euclidean | covered_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | euclidean | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | euclidean | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | euclidean | actionability_mean | 0.9365 | 0.9352 | -0.0013 |
| nhanes_pooled | euclidean | actionability_sd | 0.007589 | 0.009312 | +0.0017 |
| nhanes_pooled | euclidean | soft_violations_true_rules_mean | 2.62 | 2.62 | -0.0064 |
| nhanes_pooled | euclidean | soft_violations_true_rules_sd | 0.2649 | 0.2815 | +0.0166 |
| nhanes_pooled | euclidean | risk_drop_mean | 0.3107 | 0.3101 | -0.0005 |
| nhanes_pooled | euclidean | risk_drop_sd | 0.04249 | 0.04831 | +0.0058 |
| nhanes_pooled | main | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | main | covered_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | main | covered_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | main | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | main | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | main | actionability_mean | 0.9363 | 0.9352 | -0.0011 |
| nhanes_pooled | main | actionability_sd | 0.008204 | 0.009054 | +0.0009 |
| nhanes_pooled | main | soft_violations_true_rules_mean | 2.64 | 2.64 | -0.0011 |
| nhanes_pooled | main | soft_violations_true_rules_sd | 0.2684 | 0.2754 | +0.007 |
| nhanes_pooled | main | risk_drop_mean | 0.3171 | 0.3159 | -0.0012 |
| nhanes_pooled | main | risk_drop_sd | 0.04126 | 0.04711 | +0.0058 |
| nhanes_pooled | no_escalation | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | no_escalation | covered_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | no_escalation | covered_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | no_escalation | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | no_escalation | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | no_escalation | esc_ok_mean | 48.39 | 49.11 | +0.7279 |
| nhanes_pooled | no_escalation | esc_ok_sd | 2.22 | 1.61 | -0.6102 |
| nhanes_pooled | no_escalation | over_mean | 4.15 | 3.99 | -0.1614 |
| nhanes_pooled | no_escalation | over_sd | 0.6275 | 0.206 | -0.4214 |
| nhanes_pooled | no_escalation | actionability_mean | 0.9363 | 0.9352 | -0.0011 |
| nhanes_pooled | no_escalation | actionability_sd | 0.008204 | 0.009054 | +0.0009 |
| nhanes_pooled | no_escalation | soft_violations_true_rules_mean | 2.64 | 2.64 | -0.0011 |
| nhanes_pooled | no_escalation | soft_violations_true_rules_sd | 0.2684 | 0.2754 | +0.007 |
| nhanes_pooled | no_escalation | risk_drop_mean | 0.3171 | 0.3159 | -0.0012 |
| nhanes_pooled | no_escalation | risk_drop_sd | 0.04126 | 0.04711 | +0.0058 |
| nhanes_pooled | no_guidance | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | no_guidance | covered_mean | 95.44 | 95.38 | -0.0644 |
| nhanes_pooled | no_guidance | covered_sd | 8.30 | 8.10 | -0.1991 |
| nhanes_pooled | no_guidance | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | no_guidance | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | no_guidance | actionability_mean | 0.9354 | 0.9339 | -0.0016 |
| nhanes_pooled | no_guidance | actionability_sd | 0.007441 | 0.0088 | +0.0014 |
| nhanes_pooled | no_guidance | soft_violations_true_rules_mean | 2.64 | 2.63 | -0.0067 |
| nhanes_pooled | no_guidance | soft_violations_true_rules_sd | 0.2724 | 0.2737 | +0.0013 |
| nhanes_pooled | no_guidance | risk_drop_mean | 0.3143 | 0.3129 | -0.0014 |
| nhanes_pooled | no_guidance | risk_drop_sd | 0.04281 | 0.04936 | +0.0066 |
| nhanes_pooled | no_inheritance | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | no_inheritance | covered_mean | 96.32 | 96.41 | +0.0908 |
| nhanes_pooled | no_inheritance | covered_sd | 8.17 | 7.86 | -0.3173 |
| nhanes_pooled | no_inheritance | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | no_inheritance | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | no_inheritance | contra_card_mean | 0.56 | 0.4524 | -0.1076 |
| nhanes_pooled | no_inheritance | contra_card_sd | 0.1817 | 0.1178 | -0.0639 |
| nhanes_pooled | no_inheritance | actionability_mean | 0.9335 | 0.9318 | -0.0018 |
| nhanes_pooled | no_inheritance | actionability_sd | 0.00947 | 0.01048 | +0.001 |
| nhanes_pooled | no_inheritance | soft_violations_true_rules_mean | 2.66 | 2.66 | +0.0033 |
| nhanes_pooled | no_inheritance | soft_violations_true_rules_sd | 0.2721 | 0.279 | +0.0069 |
| nhanes_pooled | no_inheritance | risk_drop_mean | 0.3172 | 0.3162 | -0.001 |
| nhanes_pooled | no_inheritance | risk_drop_sd | 0.04104 | 0.04679 | +0.0057 |
| nhanes_pooled | no_inheritance | contraindicated_recs_total | 28.00 | 69.00 | +41 |
| nhanes_pooled | no_ontology | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | no_ontology | covered_mean | 96.32 | 96.41 | +0.0908 |
| nhanes_pooled | no_ontology | covered_sd | 8.17 | 7.86 | -0.3173 |
| nhanes_pooled | no_ontology | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | no_ontology | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | no_ontology | contra_card_mean | 0.52 | 0.4392 | -0.0808 |
| nhanes_pooled | no_ontology | contra_card_sd | 0.1304 | 0.1037 | -0.0267 |
| nhanes_pooled | no_ontology | actionability_mean | 0.9334 | 0.9317 | -0.0016 |
| nhanes_pooled | no_ontology | actionability_sd | 0.009383 | 0.01049 | +0.0011 |
| nhanes_pooled | no_ontology | soft_violations_true_rules_mean | 2.66 | 2.66 | +0.0035 |
| nhanes_pooled | no_ontology | soft_violations_true_rules_sd | 0.2721 | 0.2788 | +0.0067 |
| nhanes_pooled | no_ontology | risk_drop_mean | 0.3172 | 0.3162 | -0.001 |
| nhanes_pooled | no_ontology | risk_drop_sd | 0.04104 | 0.04679 | +0.0057 |
| nhanes_pooled | no_ontology | contraindicated_recs_total | 26.00 | 67.00 | +41 |
| nhanes_pooled | no_sra | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | no_sra | covered_mean | 96.28 | 96.37 | +0.0909 |
| nhanes_pooled | no_sra | covered_sd | 8.21 | 7.93 | -0.2807 |
| nhanes_pooled | no_sra | tier1_f_mean | 63.82 | 63.74 | -0.0841 |
| nhanes_pooled | no_sra | tier1_f_sd | 2.13 | 2.60 | +0.4725 |
| nhanes_pooled | no_sra | actionability_mean | 0.936 | 0.9347 | -0.0012 |
| nhanes_pooled | no_sra | actionability_sd | 0.008309 | 0.009042 | +0.0007 |
| nhanes_pooled | no_sra | soft_violations_true_rules_mean | 2.66 | 2.66 | -0.0019 |
| nhanes_pooled | no_sra | soft_violations_true_rules_sd | 0.2748 | 0.2808 | +0.006 |
| nhanes_pooled | no_sra | real_viol_mean | 0.72 | 0.7553 | +0.0353 |
| nhanes_pooled | no_sra | real_viol_sd | 0.531 | 0.5556 | +0.0246 |
| nhanes_pooled | no_sra | risk_drop_mean | 0.3171 | 0.3159 | -0.0012 |
| nhanes_pooled | no_sra | risk_drop_sd | 0.04136 | 0.04722 | +0.0059 |
| nhanes_pooled | no_sra | real_violation_paths_total | 36.00 | 113 | +77 |
| nhanes_pooled | ontology_as_features | n_patient_rows | 1,500 | 16,074 | +14,574 |
| nhanes_pooled | ontology_as_features | covered_mean | 96.27 | 95.97 | -0.296 |
| nhanes_pooled | ontology_as_features | covered_sd | 1.77 | 0.6088 | -1.1613 |
| nhanes_pooled | ontology_as_features | tier1_f_mean | 13.00 | 14.55 | +1.5526 |
| nhanes_pooled | ontology_as_features | tier1_f_sd | 2.13 | 1.24 | -0.8959 |
| nhanes_pooled | ontology_as_features | contra_card_mean | 0.5333 | 0.5285 | -0.0049 |
| nhanes_pooled | ontology_as_features | contra_card_sd | 0.5578 | 0.1051 | -0.4527 |
| nhanes_pooled | ontology_as_features | actionability_mean | 0.9234 | 0.9209 | -0.0025 |
| nhanes_pooled | ontology_as_features | actionability_sd | 0.01377 | 0.009667 | -0.0041 |
| nhanes_pooled | ontology_as_features | soft_violations_true_rules_mean | 2.88 | 2.82 | -0.0588 |
| nhanes_pooled | ontology_as_features | soft_violations_true_rules_sd | 0.08848 | 0.03922 | -0.0493 |
| nhanes_pooled | ontology_as_features | risk_drop_mean | 0.4238 | 0.4174 | -0.0065 |
| nhanes_pooled | ontology_as_features | risk_drop_sd | 0.03204 | 0.01079 | -0.0213 |
| nhanes_pooled | ontology_as_features | contraindicated_recs_total | 8.00 | 86.00 | +78 |
| nhanes_pooled | soft_as_hard | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | soft_as_hard | covered_mean | 30.12 | 30.63 | +0.5077 |
| nhanes_pooled | soft_as_hard | covered_sd | 3.09 | 1.90 | -1.1883 |
| nhanes_pooled | soft_as_hard | tier1_f_mean | 9.38 | 9.22 | -0.1568 |
| nhanes_pooled | soft_as_hard | tier1_f_sd | 1.41 | 1.42 | +0.0086 |
| nhanes_pooled | soft_as_hard | actionability_mean | 0.9315 | 0.934 | +0.0025 |
| nhanes_pooled | soft_as_hard | actionability_sd | 0.007046 | 0.007336 | +0.0003 |
| nhanes_pooled | soft_as_hard | risk_drop_mean | 0.2776 | 0.2703 | -0.0073 |
| nhanes_pooled | soft_as_hard | risk_drop_sd | 0.05027 | 0.06369 | +0.0134 |
| nhanes_pooled | tier1_only | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier1_only | covered_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | tier1_only | covered_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | tier1_only | tier1_f_mean | 63.26 | 63.13 | -0.1262 |
| nhanes_pooled | tier1_only | tier1_f_sd | 2.46 | 2.78 | +0.3203 |
| nhanes_pooled | tier1_only | actionability_mean | 0.9185 | 0.9167 | -0.0019 |
| nhanes_pooled | tier1_only | actionability_sd | 0.002076 | 0.001566 | -0.0005 |
| nhanes_pooled | tier1_only | soft_violations_true_rules_mean | 2.61 | 2.60 | -0.012 |
| nhanes_pooled | tier1_only | soft_violations_true_rules_sd | 0.1995 | 0.2191 | +0.0196 |
| nhanes_pooled | tier1_only | risk_drop_mean | 0.3194 | 0.3162 | -0.0032 |
| nhanes_pooled | tier1_only | risk_drop_sd | 0.04524 | 0.0552 | +0.01 |

## Tier-3 certification (`certification.csv`)

| dataset | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|
| nhanes_pooled | n_high_risk | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | n_tier1_fail | 1,837 | 5,531 | +3,694 |
| nhanes_pooled | pct_blocked_only_by_age_horizon | 5.82 | 5.42 | -0.4007 |
| nhanes_pooled | pct_two_hop_chain_hopwise | 2.89 | 3.15 | +0.2608 |
| nhanes_pooled | pct_age_ceiling | 10.89 | 10.61 | -0.2744 |
| nhanes_pooled | pct_data_scarcity | 89.11 | 89.39 | +0.2744 |
| nhanes_pooled | n_tier3 | 186 | 546 | +360 |
| nhanes_pooled | pct_tier3_model_ceiling | 98.92 | 98.17 | -0.7562 |

## Tier-1 guarantees (`tier1_guarantees.csv`)

| dataset | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|
| nhanes_pooled | n | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier1_paths | 3,163 | 9,471 | +6,308 |
| nhanes_pooled | certificates_ok | 3,163 | 9,471 | +6,308 |
| nhanes_pooled | mean_hops | 1.05 | 1.05 | +0.0022 |

## Tier-2 variants (`tier2_variants.csv`)

| dataset | method | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|
| nhanes_pooled | tier2_guided_ga | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_guided_ga | valid_f_mean | 96.18 | 96.26 | +0.0842 |
| nhanes_pooled | tier2_guided_ga | valid_f_sd | 8.37 | 8.11 | -0.2663 |
| nhanes_pooled | tier2_guided_ga | prox_mad_l1_mean | 5.90 | 5.95 | +0.0527 |
| nhanes_pooled | tier2_guided_ga | prox_mad_l1_sd | 0.3011 | 0.3092 | +0.0081 |
| nhanes_pooled | tier2_guided_ga | l0_mean | 2.10 | 2.11 | +0.0144 |
| nhanes_pooled | tier2_guided_ga | l0_sd | 0.123 | 0.1085 | -0.0145 |
| nhanes_pooled | tier2_guided_ga | plaus_lof_mean | 1.36 | 1.36 | +0.0012 |
| nhanes_pooled | tier2_guided_ga | plaus_lof_sd | 0.07428 | 0.0787 | +0.0044 |
| nhanes_pooled | tier2_guided_ga | soft_violations_mean | 2.46 | 2.46 | +0.002 |
| nhanes_pooled | tier2_guided_ga | soft_violations_sd | 0.2369 | 0.2373 | +0.0004 |
| nhanes_pooled | tier2_guided_ga | latency_s_mean | 0.2224 | 0.1146 | -0.1079 |
| nhanes_pooled | tier2_guided_ga | latency_s_sd | 0.005135 | 0.01051 | +0.0054 |
| nhanes_pooled | tier2_riemannian_ng | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_riemannian_ng | valid_f_mean | 25.64 | 24.65 | -0.9916 |
| nhanes_pooled | tier2_riemannian_ng | valid_f_sd | 2.01 | 1.77 | -0.237 |
| nhanes_pooled | tier2_riemannian_ng | prox_mad_l1_mean | 5.13 | 5.20 | +0.0675 |
| nhanes_pooled | tier2_riemannian_ng | prox_mad_l1_sd | 0.2668 | 0.1938 | -0.073 |
| nhanes_pooled | tier2_riemannian_ng | l0_mean | 2.79 | 2.79 | -0.001 |
| nhanes_pooled | tier2_riemannian_ng | l0_sd | 0.0992 | 0.08846 | -0.0107 |
| nhanes_pooled | tier2_riemannian_ng | plaus_lof_mean | 1.02 | 1.02 | -0.0004 |
| nhanes_pooled | tier2_riemannian_ng | plaus_lof_sd | 0.001249 | 0.00162 | +0.0004 |
| nhanes_pooled | tier2_riemannian_ng | soft_violations_mean | 2.11 | 2.10 | -0.0126 |
| nhanes_pooled | tier2_riemannian_ng | soft_violations_sd | 0.09584 | 0.08775 | -0.0081 |
| nhanes_pooled | tier2_riemannian_ng | latency_s_mean | 0.05577 | 0.02805 | -0.0277 |
| nhanes_pooled | tier2_riemannian_ng | latency_s_sd | 0.00289 | 0.001939 | -0.001 |
| nhanes_pooled | tier2_unguided_ga | n_patient_rows | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | tier2_unguided_ga | valid_f_mean | 94.48 | 94.47 | -0.0096 |
| nhanes_pooled | tier2_unguided_ga | valid_f_sd | 8.82 | 8.79 | -0.0356 |
| nhanes_pooled | tier2_unguided_ga | prox_mad_l1_mean | 5.56 | 5.60 | +0.0402 |
| nhanes_pooled | tier2_unguided_ga | prox_mad_l1_sd | 0.3902 | 0.4054 | +0.0152 |
| nhanes_pooled | tier2_unguided_ga | l0_mean | 2.12 | 2.13 | +0.0078 |
| nhanes_pooled | tier2_unguided_ga | l0_sd | 0.1059 | 0.09042 | -0.0155 |
| nhanes_pooled | tier2_unguided_ga | plaus_lof_mean | 1.38 | 1.38 | +0.0011 |
| nhanes_pooled | tier2_unguided_ga | plaus_lof_sd | 0.07954 | 0.08343 | +0.0039 |
| nhanes_pooled | tier2_unguided_ga | soft_violations_mean | 2.52 | 2.52 | -0.0062 |
| nhanes_pooled | tier2_unguided_ga | soft_violations_sd | 0.2792 | 0.2758 | -0.0034 |
| nhanes_pooled | tier2_unguided_ga | latency_s_mean | 0.2193 | 0.113 | -0.1063 |
| nhanes_pooled | tier2_unguided_ga | latency_s_sd | 0.00522 | 0.01008 | +0.0049 |

## Fairness table (`fairness_tier_split.csv`)

| dataset | subgroup | value | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|---|
| nhanes_pooled | age_band | 40-49 | n | 71.00 | 182 | +111 |
| nhanes_pooled | age_band | 40-49 | tier1_pct | 84.51 | 83.52 | -0.9906 |
| nhanes_pooled | age_band | 40-49 | tier2_pct | 9.86 | 10.44 | +0.5804 |
| nhanes_pooled | age_band | 40-49 | tier3_pct | 5.63 | 6.04 | +0.4102 |
| nhanes_pooled | age_band | 50-59 | n | 637 | 1,962 | +1,325 |
| nhanes_pooled | age_band | 50-59 | tier1_pct | 76.61 | 75.48 | -1.1249 |
| nhanes_pooled | age_band | 50-59 | tier2_pct | 17.11 | 16.97 | -0.139 |
| nhanes_pooled | age_band | 50-59 | tier3_pct | 6.28 | 7.54 | +1.2639 |
| nhanes_pooled | age_band | 60-69 | n | 2,328 | 6,942 | +4,614 |
| nhanes_pooled | age_band | 60-69 | tier1_pct | 62.41 | 61.94 | -0.4723 |
| nhanes_pooled | age_band | 60-69 | tier2_pct | 31.57 | 32.61 | +1.0409 |
| nhanes_pooled | age_band | 60-69 | tier3_pct | 6.01 | 5.45 | -0.5686 |
| nhanes_pooled | age_band | 70-79 | n | 1,964 | 5,916 | +3,952 |
| nhanes_pooled | age_band | 70-79 | tier1_pct | 59.16 | 59.80 | +0.639 |
| nhanes_pooled | age_band | 70-79 | tier2_pct | 40.73 | 40.04 | -0.6892 |
| nhanes_pooled | age_band | 70-79 | tier3_pct | 0.1018 | 0.1521 | +0.0503 |
| nhanes_pooled | cycle | 1999-2000 | n | 366 | 1,117 | +751 |
| nhanes_pooled | cycle | 1999-2000 | tier1_pct | 65.03 | 62.76 | -2.2699 |
| nhanes_pooled | cycle | 1999-2000 | tier2_pct | 31.97 | 33.30 | +1.3363 |
| nhanes_pooled | cycle | 1999-2000 | tier3_pct | 3.01 | 3.94 | +0.9337 |
| nhanes_pooled | cycle | 1999-2000 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2001-2002 | n | 341 | 1,060 | +719 |
| nhanes_pooled | cycle | 2001-2002 | tier1_pct | 64.22 | 66.23 | +2.0035 |
| nhanes_pooled | cycle | 2001-2002 | tier2_pct | 32.84 | 30.66 | -2.1842 |
| nhanes_pooled | cycle | 2001-2002 | tier3_pct | 2.93 | 3.11 | +0.1807 |
| nhanes_pooled | cycle | 2001-2002 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2003-2004 | n | 385 | 1,118 | +733 |
| nhanes_pooled | cycle | 2003-2004 | tier1_pct | 63.12 | 63.06 | -0.0578 |
| nhanes_pooled | cycle | 2003-2004 | tier2_pct | 32.99 | 33.18 | +0.1972 |
| nhanes_pooled | cycle | 2003-2004 | tier3_pct | 3.90 | 3.76 | -0.1394 |
| nhanes_pooled | cycle | 2003-2004 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2005-2006 | n | 317 | 952 | +635 |
| nhanes_pooled | cycle | 2005-2006 | tier1_pct | 53.31 | 56.09 | +2.7801 |
| nhanes_pooled | cycle | 2005-2006 | tier2_pct | 43.22 | 40.86 | -2.3563 |
| nhanes_pooled | cycle | 2005-2006 | tier3_pct | 3.47 | 3.05 | -0.4238 |
| nhanes_pooled | cycle | 2005-2006 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2007-2008 | n | 507 | 1,461 | +954 |
| nhanes_pooled | cycle | 2007-2008 | tier1_pct | 64.69 | 65.02 | +0.3297 |
| nhanes_pooled | cycle | 2007-2008 | tier2_pct | 32.35 | 31.55 | -0.7934 |
| nhanes_pooled | cycle | 2007-2008 | tier3_pct | 2.96 | 3.42 | +0.4637 |
| nhanes_pooled | cycle | 2007-2008 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2009-2010 | n | 450 | 1,403 | +953 |
| nhanes_pooled | cycle | 2009-2010 | tier1_pct | 65.78 | 62.72 | -3.055 |
| nhanes_pooled | cycle | 2009-2010 | tier2_pct | 31.11 | 34.00 | +2.8875 |
| nhanes_pooled | cycle | 2009-2010 | tier3_pct | 3.11 | 3.28 | +0.1676 |
| nhanes_pooled | cycle | 2009-2010 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2011-2012 | n | 385 | 1,186 | +801 |
| nhanes_pooled | cycle | 2011-2012 | tier1_pct | 66.23 | 62.06 | -4.1764 |
| nhanes_pooled | cycle | 2011-2012 | tier2_pct | 29.35 | 33.90 | +4.5448 |
| nhanes_pooled | cycle | 2011-2012 | tier3_pct | 4.42 | 4.05 | -0.3684 |
| nhanes_pooled | cycle | 2011-2012 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2013-2014 | n | 419 | 1,318 | +899 |
| nhanes_pooled | cycle | 2013-2014 | tier1_pct | 58.71 | 59.71 | +1.0005 |
| nhanes_pooled | cycle | 2013-2014 | tier2_pct | 36.52 | 36.65 | +0.1309 |
| nhanes_pooled | cycle | 2013-2014 | tier3_pct | 4.77 | 3.64 | -1.1314 |
| nhanes_pooled | cycle | 2013-2014 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2015-2016 | n | 492 | 1,381 | +889 |
| nhanes_pooled | cycle | 2015-2016 | tier1_pct | 61.99 | 62.78 | +0.7887 |
| nhanes_pooled | cycle | 2015-2016 | tier2_pct | 34.55 | 33.31 | -1.2436 |
| nhanes_pooled | cycle | 2015-2016 | tier3_pct | 3.46 | 3.91 | +0.4549 |
| nhanes_pooled | cycle | 2015-2016 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2017-2020 | n | 710 | 2,127 | +1,417 |
| nhanes_pooled | cycle | 2017-2020 | tier1_pct | 65.07 | 64.88 | -0.1903 |
| nhanes_pooled | cycle | 2017-2020 | tier2_pct | 30.00 | 30.70 | +0.7005 |
| nhanes_pooled | cycle | 2017-2020 | tier3_pct | 4.93 | 4.42 | -0.5102 |
| nhanes_pooled | cycle | 2017-2020 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | cycle | 2021-2023 | n | 628 | 1,879 | +1,251 |
| nhanes_pooled | cycle | 2021-2023 | tier1_pct | 64.01 | 65.41 | +1.3944 |
| nhanes_pooled | cycle | 2021-2023 | tier2_pct | 32.64 | 31.51 | -1.1372 |
| nhanes_pooled | cycle | 2021-2023 | tier3_pct | 3.34 | 3.09 | -0.2572 |
| nhanes_pooled | cycle | 2021-2023 | chi2_p | 0.04697 | 4.457e-05 | -0.0469 |
| nhanes_pooled | race | Mexican American | n | 756 | 2,286 | +1,530 |
| nhanes_pooled | race | Mexican American | tier1_pct | 67.46 | 64.96 | -2.4997 |
| nhanes_pooled | race | Mexican American | tier2_pct | 28.31 | 30.27 | +1.9643 |
| nhanes_pooled | race | Mexican American | tier3_pct | 4.23 | 4.77 | +0.5353 |
| nhanes_pooled | race | Mexican American | chi2_p | 0.01502 | 3.648e-09 | -0.015 |
| nhanes_pooled | race | Non-Hispanic Black | n | 1,026 | 3,160 | +2,134 |
| nhanes_pooled | race | Non-Hispanic Black | tier1_pct | 64.13 | 64.46 | +0.3295 |
| nhanes_pooled | race | Non-Hispanic Black | tier2_pct | 32.55 | 32.31 | -0.2435 |
| nhanes_pooled | race | Non-Hispanic Black | tier3_pct | 3.31 | 3.23 | -0.086 |
| nhanes_pooled | race | Non-Hispanic Black | chi2_p | 0.01502 | 3.648e-09 | -0.015 |
| nhanes_pooled | race | Non-Hispanic White | n | 2,426 | 7,130 | +4,704 |
| nhanes_pooled | race | Non-Hispanic White | tier1_pct | 61.25 | 61.33 | +0.0793 |
| nhanes_pooled | race | Non-Hispanic White | tier2_pct | 35.12 | 35.25 | +0.1259 |
| nhanes_pooled | race | Non-Hispanic White | tier3_pct | 3.63 | 3.42 | -0.2052 |
| nhanes_pooled | race | Non-Hispanic White | chi2_p | 0.01502 | 3.648e-09 | -0.015 |
| nhanes_pooled | race | Other Hispanic | n | 420 | 1,285 | +865 |
| nhanes_pooled | race | Other Hispanic | tier1_pct | 67.86 | 69.03 | +1.1701 |
| nhanes_pooled | race | Other Hispanic | tier2_pct | 28.57 | 27.78 | -0.7893 |
| nhanes_pooled | race | Other Hispanic | tier3_pct | 3.57 | 3.19 | -0.3808 |
| nhanes_pooled | race | Other Hispanic | chi2_p | 0.01502 | 3.648e-09 | -0.015 |
| nhanes_pooled | race | Other/Multiracial | n | 372 | 1,141 | +769 |
| nhanes_pooled | race | Other/Multiracial | tier1_pct | 60.22 | 60.39 | +0.1706 |
| nhanes_pooled | race | Other/Multiracial | tier2_pct | 35.22 | 35.23 | +0.0172 |
| nhanes_pooled | race | Other/Multiracial | tier3_pct | 4.57 | 4.38 | -0.1878 |
| nhanes_pooled | race | Other/Multiracial | chi2_p | 0.01502 | 3.648e-09 | -0.015 |
| nhanes_pooled | sex | Female | n | 1,841 | 5,415 | +3,574 |
| nhanes_pooled | sex | Female | tier1_pct | 88.21 | 89.51 | +1.2977 |
| nhanes_pooled | sex | Female | tier2_pct | 11.79 | 10.49 | -1.2977 |
| nhanes_pooled | sex | Male | n | 3,159 | 9,587 | +6,428 |
| nhanes_pooled | sex | Male | tier1_pct | 48.72 | 48.23 | -0.486 |
| nhanes_pooled | sex | Male | tier2_pct | 45.39 | 46.07 | +0.6787 |
| nhanes_pooled | sex | Male | tier3_pct | 5.89 | 5.70 | -0.1927 |

## Robustness to oracle retraining (`robustness_retraining.csv`)

| dataset | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|
| nhanes_pooled | still_valid_pct_tier1 | 85.07 | 84.86 | -0.2111 |
| nhanes_pooled | still_valid_pct_tier2 | 64.76 | 63.47 | -1.2855 |

## Reachable-set head-to-head (`reachable_set_head_to_head.csv`)

| dataset | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|
| nhanes_pooled | n | 5,000 | 15,002 | +10,002 |
| nhanes_pooled | pct_lr_precluded | 15.64 | 15.74 | +0.1046 |
| nhanes_pooled | pct_carelg_tier1_infeasible | 36.74 | 36.87 | +0.1284 |
| nhanes_pooled | pct_carelg_tier3 | 3.72 | 3.64 | -0.0805 |
| nhanes_pooled | pct_lr_precluded_but_carelg_tier1 | 7.52 | 7.55 | +0.0257 |
| nhanes_pooled | pct_lr_precluded_but_carelg_tier2 | 6.56 | 6.60 | +0.0391 |
| nhanes_pooled | pct_lr_feasible_but_carelg_tier3 | 2.16 | 2.04 | -0.1203 |
| nhanes_pooled | pct_lr_cf_valid_under_oracle | 27.82 | 28.12 | +0.3029 |

## Classifier: accuracy and AUC

| dataset | model | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|
| nhanes_pooled | oracle | test_accuracy (mean of 5 seeds) | 0.8956 | 0.8956 | +0 |
| nhanes_pooled | oracle | test_auc (mean of 5 seeds) | 0.9643 | 0.9643 | +0 |
| nhanes_pooled | non-oracle model | test_accuracy (mean of 5 seeds) | 0.8958 | 0.8958 | +0 |
| nhanes_pooled | non-oracle model | test_auc (mean of 5 seeds) | 0.9642 | 0.9642 | +0 |

## Learning curve: full-data point

| dataset | n_patients | metric | old (report_v3) | new (full rerun) | change |
|---|---|---|---|---|---|
| nhanes_pooled | 32786.0 | tier1_coverage_pct (full-data point, mean of 5 seeds) | 63.13 | 63.13 | +0 |
| nhanes_pooled | 32786.0 | tier1_coverage_pct (full-data point, SD over seeds) | 2.78 | 2.78 | +0 |

## Appendix: bootstrap confidence-interval bounds that changed (374 cells)

Listed in `results/tables/full_rerun_changes.csv` (column `ci` = True); CI bounds change with the data and with the shared bootstrap RNG stream.
