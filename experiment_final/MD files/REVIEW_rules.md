# REVIEW_rules.md -- knowledge tables for human review (CARE-LG, PIPELINE P2)

Every row below is used by the pipeline. `verify=1` means the citation/value was written from memory and must be checked against the source document before any clinical use. Citation keys resolve in `knowledge/citations.csv`. ICD-10-CM nodes were checked to exist in the ICD-10-CM 2027 tabular XML (fail-safe in `src/knowledge_layer.py`).

## Self-review notes (author)

- **F-SBP / F-DBP (ROOT 140/90)**: Generic target uses the JNC8 / HTN2017-stage-2 threshold; HTN2017 itself recommends <130/80 for most adults with hypertension and >=10% ASCVD risk, and JNC8 allows <150/90 at age >=60. The ROOT row is a deliberate middle ground so that the patient-specific <130/80 rows (diabetes, CKD, ischaemic heart disease, stroke, HF) actually differ from the default. Reviewer: confirm.
- **F-A1C-DM lower bound 6.0%**: ADA gives an upper goal (<7%) and warns about hypoglycaemia; the 6.0% floor is an author choice used only as a SOFT rule.
- **F-TC (<200 mg/dL)**: Total-cholesterol target is from ATP III (2002); CHOL2018 targets LDL-C, which NHANES/UCI here do not provide.
- **Typical effects (interventions.csv)**: Additive lifestyle effects and the drug effects (LAW2009 ~9/5.5 mmHg per standard-dose class, statin -25% TC, metformin -1.1% HbA1c, GLP-1 RA BMI -2.0) are approximations used only to choose the escalation step; they never change validity.
- **beta_blocker_nonselective**: Not a first-line antihypertensive (HTN2017). Included so the asthma contraindication (C-BB-J45) is exercised; carvedilol and labetalol are matched as non-selective (they also block alpha-1).
- **C-GLP1-C73**: The FDA boxed warning concerns medullary thyroid carcinoma; ICD-10-CM C73 covers all thyroid cancers, so blocking at C73 over-blocks (conservative).
- **C-THZ-M10, C-STATIN-K746, C-ACEARB-T783**: Relative contraindications treated as blocking (gout with thiazides; cirrhosis with statins - compensated cirrhosis may tolerate statins; ACE-inhibitor angioedema applied to the ACEi/ARB class although ARB cross-reactivity is low). Conservative.
- **diabetic -> E11**: NHANES does not record diabetes type; mapping all doctor-told diabetes to E11 means C-SGLT2-E10 only fires for patients with an explicit E10 Rx reason code.
- **Indication tokens**: Antihypertensives are indicated when the baseline BP is above the PATIENT-SPECIFIC target (so inheritance changes indication); statins by PCE >= 7.5%, diabetes, ASCVD or above-target cholesterol; diabetes drugs need a diabetes concept (or above-normal cardio glucose category).

## Feature rules (HARD direction from the adapter; SOFT targets/magnitude, inherited down ICD-10-CM) -- `knowledge/feature_rules.csv` (30 rows, 26 marked verify=1)

| rule_id | feature | applies_to_icd10_node | direction | target_low | target_high | max_plausible_change | min_meaningful_change | phys_min | phys_max | citation | verify | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F-AGE | age | ROOT | non_decreasing |  |  |  |  | 18 | 100 | STRUCT | 0 | Cannot decrease (0.1 y tolerance); tier-1 total path horizon <= 3.05 y (CARE-LG R-AGE-HORIZON). |
| F-SEX | sex | ROOT | equal |  |  |  |  | 0 | 1 | STRUCT | 0 | Immutable. |
| F-FBS | fasting_glucose_flag | ROOT | equal |  |  |  |  | 0 | 1 | CARELG_META | 0 | UCI fbs>120 mg/dL flag; immutable in the CARE-LG schema (configs/dataset_config.py). |
| F-SBP | systolic_bp | ROOT | reduce | 90 | 140 | 40 | 5 | 70 | 250 | JNC8;HTN2017 | 1 | Generic adult target <140/90 (JNC8 general population; HTN2017 stage-2 threshold). max_plausible_change ~ lifestyle + 2-drug combination (author approximation from LAW2009 effects). |
| F-SBP-DM | systolic_bp | E08-E13 | reduce | 90 | 130 |  |  |  |  | ADA2024;HTN2017 | 1 | Diabetes: BP target <130/80 mmHg if it can be safely attained. |
| F-SBP-CKD | systolic_bp | N18 | reduce | 90 | 130 |  |  |  |  | HTN2017;KDIGO2021BP | 1 | CKD: target <130/80 (HTN2017); KDIGO 2021 suggests SBP <120 with standardised office BP when tolerated (not used: different measurement protocol). |
| F-SBP-ASCVD | systolic_bp | I20-I25 | reduce | 90 | 130 |  |  |  |  | HTN2017 | 1 | Established ischaemic heart disease: target <130/80. |
| F-SBP-STROKE | systolic_bp | Z86.73 | reduce | 90 | 130 |  |  |  |  | HTN2017 | 1 | Prior stroke/TIA (secondary prevention): target <130/80. |
| F-SBP-HF | systolic_bp | I50 | reduce | 90 | 130 |  |  |  |  | HTN2017;HF2022 | 1 | Heart failure with hypertension: target <130/80. |
| F-DBP | diastolic_bp | ROOT | reduce | 60 | 90 | 25 | 3 | 40 | 150 | JNC8;HTN2017 | 1 | Generic adult target <90 mmHg. |
| F-DBP-DM | diastolic_bp | E08-E13 | reduce | 60 | 80 |  |  |  |  | ADA2024;HTN2017 | 1 | Diabetes: <80 mmHg. |
| F-DBP-CKD | diastolic_bp | N18 | reduce | 60 | 80 |  |  |  |  | HTN2017 | 1 | CKD: <80 mmHg. |
| F-DBP-ASCVD | diastolic_bp | I20-I25 | reduce | 60 | 80 |  |  |  |  | HTN2017 | 1 | Ischaemic heart disease: <80 mmHg. |
| F-DBP-STROKE | diastolic_bp | Z86.73 | reduce | 60 | 80 |  |  |  |  | HTN2017 | 1 | Prior stroke/TIA: <80 mmHg. |
| F-DBP-HF | diastolic_bp | I50 | reduce | 60 | 80 |  |  |  |  | HTN2017;HF2022 | 1 | Heart failure with hypertension: <80 mmHg. |
| F-TC | total_cholesterol | ROOT | reduce |  | 200 | 90 | 10 | 70 | 450 | NCEP2002;CHOL2018 | 1 | Desirable total cholesterol <200 mg/dL (ATP III); CHOL2018 targets LDL-C which these datasets do not record. max change ~ high-intensity statin effect on TC (author approximation, verify). |
| F-BMI | bmi | ROOT | reduce | 18.5 | 25 | 5 | 0.5 | 14 | 80 | WHO_BMI;OBESITY2013 | 1 | Normal BMI 18.5-24.9; max change ~15% body weight at BMI 33 (author approximation from WILDING2021, verify). |
| F-A1C | hba1c | ROOT | reduce | 4.0 | 5.7 | 2.0 | 0.3 | 3.5 | 17 | ADA2024 | 1 | No diabetes: normal HbA1c <5.7% (5.7-6.4 = prediabetes). |
| F-A1C-DM | hba1c | E08-E13 | reduce | 6.0 | 7.0 |  |  |  |  | ADA2024;ADA2025 | 1 | Diabetes: HbA1c goal <7% for many non-pregnant adults; lower bound 6.0% is an author-chosen hypoglycaemia guard (soft rule), verify. |
| F-MAXHR | max_heart_rate | ROOT | increase |  |  | 30 | 5 | 60 | 220 | CARELG_META | 1 | UCI exercise-test maximum heart rate; direction-only rule from the CARE-LG schema (no guideline target). |
| F-STDEP | st_depression | ROOT | reduce | 0 | 0.5 | 2 | 0.5 | 0 | 7 | CARELG_META | 1 | UCI oldpeak (exercise-induced ST depression); direction-only in CARE-LG; 0.5 mm upper bound is an author choice (verify). |
| F-ANGINA | exercise_angina | ROOT | reduce | 0 | 0 | 1 | 1 | 0 | 1 | CARELG_META | 1 | UCI exercise-induced angina flag. |
| F-SLOPE | st_slope | ROOT | reduce | 1 | 1 | 2 | 1 | 1 | 3 | CARELG_META | 1 | UCI ST-segment slope (1 upsloping, 2 flat, 3 downsloping); upsloping treated as target (author choice, verify). |
| F-WEIGHT | body_weight | ROOT | reduce |  |  | 15 | 2 | 35 | 200 | OBESITY2013 | 1 | Cardio dataset body weight (kg); no height-free target. max change ~15% weight loss (author approximation, verify). |
| F-CHOLCAT | cholesterol_category | ROOT | reduce | 1 | 1 | 2 | 1 | 1 | 3 | CARDIO_KAGGLE;NCEP2002 | 1 | Cardio ordinal cholesterol (1 normal, 2 above normal, 3 well above normal); target = normal. |
| F-GLUCCAT | glucose_category | ROOT | reduce | 1 | 1 | 2 | 1 | 1 | 3 | CARDIO_KAGGLE;ADA2024 | 1 | Cardio ordinal glucose (1 normal, 2 above normal, 3 well above normal); target = normal. |
| F-SMOKE | smoking | ROOT | reduce | 0 | 0 | 1 | 1 | 0 | 1 | LE8_2022;PREV2019 | 1 | Non-smoking is the target. |
| F-ALCO | alcohol | ROOT | reduce | 0 | 0 | 1 | 1 | 0 | 1 | HTN2017 | 1 | Cardio self-reported alcohol intake flag; moderation/abstinence target. |
| F-ACTIVE | physical_activity | ROOT | increase | 1 | 1 | 1 | 1 | 0 | 1 | LE8_2022;PREV2019 | 1 | Physically active is the target. |
| F-CONCEPT-FLAG | concept_flag | ROOT | equal |  |  |  |  | 0 | 1 | STRUCT | 0 | Ontology-as-features ablation only: ICD-10-CM block indicator appended to the model inputs; immutable. |

## Intervention classes (escalation tier, acts_on, typical effect, indication) -- `knowledge/interventions.csv` (13 rows, 13 marked verify=1)

| intervention_class | escalation_tier | first_line | acts_on_features | expected_direction | typical_effect | indication | indication_fallback | nhanes_rx_match_pattern | citation | verify | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dash_sodium | lifestyle | 1 | systolic_bp;diastolic_bp | -;- | -11;-5.5 | always | 1 |  | HTN2017 | 1 | DASH diet ~-11 mmHg SBP in hypertensive adults (HTN2017 Table 15); DBP effect is an author approximation (verify). |
| weight_management | lifestyle | 1 | bmi;body_weight;systolic_bp;diastolic_bp;hba1c | -;-;-;-;- | -1.5;-5%;-5;-2.5;-0.3 | above_target:bmi | 1 |  | OBESITY2013;HTN2017;ADA2024 | 1 | 5-10% weight loss recommended for BMI>=25 (OBESITY2013); ~1 mmHg SBP per kg (HTN2017 Table 15); effect sizes are approximations (verify). |
| exercise | lifestyle | 1 | systolic_bp;diastolic_bp;hba1c;physical_activity | -;-;-;+ | -5;-3;-0.6;+1 | always | 1 |  | HTN2017;UMPIERRE2011;PREV2019 | 1 | Aerobic exercise ~-5 to -8 mmHg SBP (HTN2017 Table 15); structured exercise ~-0.67% HbA1c in T2D (UMPIERRE2011). |
| smoking_cessation | lifestyle | 1 | smoking | - | -1 | above_target:smoking|concept:Z72.0|concept:F17 | 0 |  | PREV2019;LE8_2022 | 1 | All adults who use tobacco should be assisted to quit (PREV2019). |
| alcohol_moderation | lifestyle | 1 | alcohol;systolic_bp;diastolic_bp | -;-;- | -1;-4;-3 | above_target:alcohol | 0 |  | HTN2017 | 1 | Alcohol reduction ~-4 mmHg SBP (HTN2017 Table 15); DBP author approximation (verify). Not in the prompt's list; added for the cardio alcohol feature. |
| statin | single_drug | 1 | total_cholesterol;cholesterol_category | -;- | -25%;-1 | ascvd_ge:7.5|above_target:total_cholesterol|above_target:cholesterol_category|concept:I20-I25|concept:Z86.73|concept:E08-E13 | 0 | STATIN | CHOL2018 | 1 | Moderate-intensity statin lowers LDL-C 30-49% (CHOL2018); total-cholesterol effect ~-25% is an author approximation (verify). |
| acei_arb | single_drug | 1 | systolic_bp;diastolic_bp | -;- | -9;-5.5 | above_target:systolic_bp|above_target:diastolic_bp | 0 | PRIL\b|SARTAN | HTN2017;LAW2009 | 1 | First-line antihypertensive class (HTN2017); standard-dose effect ~-9/-5.5 mmHg (LAW2009). |
| thiazide | single_drug | 1 | systolic_bp;diastolic_bp | -;- | -9;-5.5 | above_target:systolic_bp|above_target:diastolic_bp | 0 | HYDROCHLOROTHIAZIDE|CHLORTHALIDONE|INDAPAMIDE|METOLAZONE|CHLOROTHIAZIDE | HTN2017;LAW2009 | 1 | First-line antihypertensive class (HTN2017). |
| ccb | single_drug | 1 | systolic_bp;diastolic_bp | -;- | -9;-5.5 | above_target:systolic_bp|above_target:diastolic_bp | 0 | AMLODIPINE|NIFEDIPINE|FELODIPINE|ISRADIPINE|NICARDIPINE|NISOLDIPINE | HTN2017;LAW2009 | 1 | Dihydropyridine CCB; first-line antihypertensive class (HTN2017). |
| beta_blocker_nonselective | single_drug | 0 | systolic_bp;diastolic_bp | -;- | -9;-5.5 | above_target:systolic_bp|above_target:diastolic_bp | 0 | PROPRANOLOL|NADOLOL|TIMOLOL|PINDOLOL|SOTALOL|CARVEDILOL|LABETALOL | HTN2017;LAW2009 | 1 | Not first-line for hypertension without a compelling indication (HTN2017); included so the asthma contraindication is exercised. |
| metformin | single_drug | 1 | hba1c;glucose_category | -;- | -1.1;-1 | concept:E08-E13|above_target:glucose_category | 0 | METFORMIN | ADA2024 | 1 | First-line glucose-lowering therapy in T2D (high efficacy; ~1-1.5% HbA1c). |
| sglt2i | single_drug | 1 | hba1c;systolic_bp;bmi;glucose_category | -;-;-;- | -0.7;-4;-0.7;-1 | concept:E08-E13|concept:N18|concept:I50|above_target:glucose_category | 0 | GLIFLOZIN | ADA2024;KDIGO2024;HF2022 | 1 | Intermediate HbA1c efficacy; recommended for CKD (eGFR>=20) and HF irrespective of HbA1c (KDIGO2024, HF2022). |
| glp1ra | single_drug | 1 | hba1c;bmi;body_weight;systolic_bp;glucose_category | -;-;-;-;- | -1.2;-2.0;-8%;-3;-1 | concept:E08-E13|concept:E66|above_target:glucose_category | 0 | GLUTIDE|EXENATIDE|TIRZEPATIDE | ADA2024;WILDING2021 | 1 | High HbA1c and weight efficacy (ADA2024); semaglutide 2.4 mg ~-15% weight (WILDING2021); BMI/weight effects here are conservative approximations (verify). |

## Contraindications (FILTER tier; inherited down ICD-10-CM) -- `knowledge/contraindications.csv` (18 rows, 18 marked verify=1)

| rule_id | intervention_class | contraindicated_icd10_node | node_label | reason | citation | verify |
|---|---|---|---|---|---|---|
| C-MET-N184 | metformin | N18.4 | Chronic kidney disease, stage 4 (severe) | Metformin contraindicated at eGFR <30 mL/min/1.73m2 (lactic acidosis risk) | FDA_METFORMIN2016;ADA2024 | 1 |
| C-MET-N185 | metformin | N18.5 | Chronic kidney disease, stage 5 | Metformin contraindicated at eGFR <30 mL/min/1.73m2 | FDA_METFORMIN2016;ADA2024 | 1 |
| C-MET-N186 | metformin | N18.6 | End stage renal disease | Metformin contraindicated in end-stage renal disease | FDA_METFORMIN2016;ADA2024 | 1 |
| C-MET-Z992 | metformin | Z99.2 | Dependence on renal dialysis | Metformin contraindicated on dialysis | FDA_METFORMIN2016;KDIGO2022DM | 1 |
| C-SGLT2-Z992 | sglt2i | Z99.2 | Dependence on renal dialysis | SGLT2i not initiated in patients on dialysis | KDIGO2024;KDIGO2022DM | 1 |
| C-SGLT2-N186 | sglt2i | N18.6 | End stage renal disease | SGLT2i initiation not recommended at eGFR <20 / kidney failure | KDIGO2024 | 1 |
| C-SGLT2-E10 | sglt2i | E10 | Type 1 diabetes mellitus | SGLT2i not approved in type 1 diabetes (diabetic ketoacidosis risk) | ADA2024 | 1 |
| C-GLP1-C73 | glp1ra | C73 | Malignant neoplasm of thyroid gland | Boxed warning: contraindicated with personal history of medullary thyroid carcinoma (C73 covers all thyroid cancers; conservative, verify) | FDA_GLP1;ADA2024 | 1 |
| C-GLP1-E3122 | glp1ra | E31.2 | Multiple endocrine neoplasia | Contraindicated with MEN 2 (attached at E31.2 so MEN 2A/2B inherit) | FDA_GLP1 | 1 |
| C-ACEARB-Z331 | acei_arb | Z33.1 | Pregnant state, incidental | ACE inhibitors / ARBs contraindicated in pregnancy (fetotoxicity) | HTN2017 | 1 |
| C-ACEARB-T783 | acei_arb | T78.3 | Angioneurotic edema | History of angioedema contraindicates ACE inhibitors (class-level here; ARB cross-reactivity is low but conservative) | HTN2017 | 1 |
| C-BB-J45 | beta_blocker_nonselective | J45 | Asthma | Non-selective beta-blockers contraindicated in asthma (bronchospasm) | GINA2024;HTN2017 | 1 |
| C-BB-I442 | beta_blocker_nonselective | I44.2 | Atrioventricular block, complete | Beta-blockers contraindicated in high-degree AV block without a pacemaker | HTN2017 | 1 |
| C-THZ-M10 | thiazide | M10 | Gout | History of gout: thiazides raise uric acid; avoid unless on uric-acid-lowering therapy (relative; treated as blocking) | HTN2017 | 1 |
| C-THZ-E871 | thiazide | E87.1 | Hypo-osmolality and hyponatremia | Thiazides worsen hyponatraemia | HTN2017 | 1 |
| C-STATIN-K72 | statin | K72 | Hepatic failure, not elsewhere classified | Statins contraindicated in acute liver failure / decompensated liver disease | CHOL2018 | 1 |
| C-STATIN-K746 | statin | K74.6 | Other and unspecified cirrhosis of liver | Decompensated cirrhosis: statin contraindicated (conservative; compensated cirrhosis may tolerate statins) | CHOL2018 | 1 |
| C-STATIN-Z331 | statin | Z33.1 | Pregnant state, incidental | Statins should be discontinued in pregnancy | CHOL2018 | 1 |

## Non-blocking guideline annotations -- `knowledge/annotations.csv` (10 rows, 9 marked verify=1)

| rule_id | attached_icd10_node | condition | description | citation | verify |
|---|---|---|---|---|---|
| A-STATIN-PCE | ROOT | ascvd_ge:7.5 | 10-year ASCVD risk >= 7.5% (PCE): moderate-intensity statin is reasonable after a clinician-patient risk discussion (intermediate risk 7.5-19.9%; high-intensity if >=20%). | CHOL2018;PCE2013 | 1 |
| A-STATIN-DM | E08-E13 |  | Diabetes, age 40-75: moderate-intensity statin regardless of estimated 10-year risk. | CHOL2018;ADA2024 | 1 |
| A-STATIN-ASCVD | I20-I25 |  | Clinical ASCVD (ischaemic heart disease): high-intensity statin indicated (secondary prevention); the PCE label is not validated in this group. | CHOL2018 | 1 |
| A-STATIN-STROKE | Z86.73 |  | Prior stroke/TIA: high-intensity statin indicated (secondary prevention); the PCE label is not validated in this group. | CHOL2018 | 1 |
| A-BP-DM | E08-E13 |  | Diabetes: blood-pressure target <130/80 mmHg replaces the generic <140/90. | ADA2024;HTN2017 | 1 |
| A-BP-CKD | N18 |  | CKD: blood-pressure target <130/80 mmHg (HTN2017); KDIGO 2021 suggests SBP <120 with standardised measurement. | HTN2017;KDIGO2021BP | 1 |
| A-SGLT2-CKD | N18 |  | CKD: SGLT2 inhibitor recommended for kidney/cardiovascular protection when eGFR >= 20 irrespective of HbA1c. | KDIGO2024 | 1 |
| A-SMOKING | Z72.0 |  | Tobacco use: offer cessation counselling plus pharmacotherapy at every visit. | PREV2019 | 1 |
| A-HF-SGLT2 | I50 |  | Heart failure: SGLT2 inhibitor is part of guideline-directed medical therapy. | HF2022 | 1 |
| A-HORIZON | ROOT |  | Tier-1 (real-patient) recourse targets are at most 3.05 years older than the patient (follow-up interval for risk recalculation). | STRUCT | 0 |

## NHANES item -> ICD-10-CM concept map -- `knowledge/nhanes_condition_map.csv` (21 rows)

| nhanes_variable | value | icd10_code | label |
|---|---|---|---|
| diabetic | True | E11 | Doctor-told diabetes (DIQ010=1). NHANES does not record diabetes type; mapped to type 2 (about 90-95% of adult diabetes). VERIFY. |
| treated_bp | True | I10 | Taking prescribed medicine for high blood pressure (BPQ050A=1; BPQ150=1 in 2021-2023, which has no BPQ050A) implies diagnosed hypertension. |
| BPQ020 | 1 | I10 | Ever told by a doctor you had high blood pressure. |
| BPQ080 | 1 | E78.5 | Ever told by a doctor your blood cholesterol was high (hyperlipidaemia, unspecified). |
| smoker | True | Z72.0 | Current cigarette smoker (SMQ020=1 and SMQ040 in {1;2}); tobacco use rather than dependence (no dependence assessment). |
| MCQ160B | 1 | I50.9 | Ever told congestive heart failure (heart failure, unspecified). |
| MCQ160C | 1 | I25.10 | Ever told coronary heart disease (atherosclerotic heart disease of native coronary artery). |
| MCQ160E | 1 | I25.2 | Ever told heart attack (old myocardial infarction). |
| MCQ160F | 1 | Z86.73 | Ever told stroke (personal history of cerebral infarction without residual deficits). |
| MCQ160P (2017+) | 1 | J44.9 | Ever told COPD, emphysema or chronic bronchitis (COPD, unspecified). |
| MCQ160G (1999-2016) | 1 | J43.9 | Ever told you had emphysema (codebook verified); emphysema, unspecified. |
| MCQ160K (1999-2016) | 1 | J42 | Ever told you had chronic bronchitis (codebook verified); unspecified chronic bronchitis. |
| MCQ035 | 1 | J45.909 | Still have asthma (unspecified asthma, uncomplicated). |
| MCQ170L | 1 | K76.9 | Still have a liver condition (liver disease, unspecified). |
| MCQ230A-D | 37 | C73 | Cancer type code 37 = Thyroid (verified in the P_MCQ.htm codebook frequency table); malignant neoplasm of thyroid gland. |
| KIQ022 | 1 | N18.9 | Ever told you had weak/failing kidneys (codebook KIQ_U_D wording verified); chronic kidney disease, unspecified. |
| KIQ020 (1999-2000 only) | 1 | N18.9 | Same item in the 1999-2000 KIQ file ('Ever told you had weak/failing kidneys', codebook verified). |
| KIQ025 | 1 | N18.6;Z99.2 | Received dialysis in past 12 months (end-stage renal disease; dependence on renal dialysis). |
| RIDEXPRG | 1 | Z33.1 | Pregnancy status positive at exam. |
| bmi | >=30 | E66.9 | Derived from the baseline BMI >= 30 kg/m2 (obesity, unspecified; WHO_BMI). |
| RXDRSC1-3 | <code> | <code> | ICD-10-CM reason-for-use codes reported with each prescription; trailing NCHS 'P' suffix stripped and resolved to the most specific existing ICD-10-CM 2027 node; '99999' (unknown) dropped. |

## Citations -- `knowledge/citations.csv` (27 rows, 24 marked verify=1)

| key | reference | verify |
|---|---|---|
| STRUCT | Structural constraint (not a guideline): immutable attributes and age monotonicity cannot be changed by an intervention. | 0 |
| AUTHOR | Author-chosen engineering parameter (not a guideline value); disclosed in results/DECISIONS.md. | 0 |
| CARELG_META | CARE-LG feature classification inherited from configs/dataset_config.py (v6+) and experiments_v10/data_cardio.py; structural, not a guideline. | 0 |
| HTN2017 | Whelton PK, Carey RM, Aronow WS, et al. 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. Hypertension. 2018;71(6):e13-e115. | 1 |
| JNC8 | James PA, Oparil S, Carter BL, et al. 2014 Evidence-Based Guideline for the Management of High Blood Pressure in Adults (JNC 8). JAMA. 2014;311(5):507-520. | 1 |
| CHOL2018 | Grundy SM, Stone NJ, Bailey AL, et al. 2018 AHA/ACC/AACVPR/AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the Management of Blood Cholesterol. Circulation. 2019;139(25):e1082-e1143. | 1 |
| PREV2019 | Arnett DK, Blumenthal RS, Albert MA, et al. 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease. Circulation. 2019;140(11):e596-e646. | 1 |
| PCE2013 | Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk. Circulation. 2014;129(25 Suppl 2):S49-S73. | 1 |
| NCEP2002 | National Cholesterol Education Program Expert Panel. Third Report of the NCEP Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (ATP III) Final Report. Circulation. 2002;106(25):3143-3421. | 1 |
| ADA2024 | American Diabetes Association Professional Practice Committee. Standards of Care in Diabetes-2024. Diabetes Care. 2024;47(Suppl 1):S1-S321. | 1 |
| ADA2025 | American Diabetes Association Professional Practice Committee. Standards of Care in Diabetes-2025. Diabetes Care. 2025;48(Suppl 1):S1-S352. | 1 |
| KDIGO2024 | Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney Int. 2024;105(4S):S117-S314. | 1 |
| KDIGO2022DM | KDIGO Diabetes Work Group. KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease. Kidney Int. 2022;102(5S):S1-S127. | 1 |
| KDIGO2021BP | KDIGO Blood Pressure Work Group. KDIGO 2021 Clinical Practice Guideline for the Management of Blood Pressure in Chronic Kidney Disease. Kidney Int. 2021;99(3S):S1-S87. | 1 |
| GINA2024 | Global Initiative for Asthma. Global Strategy for Asthma Management and Prevention, 2024 update. www.ginasthma.org. | 1 |
| HF2022 | Heidenreich PA, Bozkurt B, Aguilar D, et al. 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure. Circulation. 2022;145(18):e895-e1032. | 1 |
| OBESITY2013 | Jensen MD, Ryan DH, Apovian CM, et al. 2013 AHA/ACC/TOS Guideline for the Management of Overweight and Obesity in Adults. Circulation. 2014;129(25 Suppl 2):S102-S138. | 1 |
| WHO_BMI | World Health Organization. Obesity: preventing and managing the global epidemic. WHO Technical Report Series 894. Geneva: WHO; 2000. | 1 |
| LAW2009 | Law MR, Morris JK, Wald NJ. Use of blood pressure lowering drugs in the prevention of cardiovascular disease: meta-analysis of 147 randomised trials. BMJ. 2009;338:b1665. | 1 |
| UMPIERRE2011 | Umpierre D, Ribeiro PAB, Kramer CK, et al. Physical activity advice only or structured exercise training and association with HbA1c levels in type 2 diabetes. JAMA. 2011;305(17):1790-1799. | 1 |
| WILDING2021 | Wilding JPH, Batterham RL, Calanna S, et al. Once-weekly semaglutide in adults with overweight or obesity (STEP 1). N Engl J Med. 2021;384(11):989-1002. | 1 |
| FDA_GLP1 | US FDA prescribing information for GLP-1 receptor agonists (e.g. semaglutide, liraglutide): boxed warning for thyroid C-cell tumours; contraindicated with personal/family history of medullary thyroid carcinoma or MEN 2. | 1 |
| FDA_METFORMIN2016 | US FDA Drug Safety Communication (8 April 2016): FDA revises warnings regarding use of metformin in certain patients with reduced kidney function (contraindicated when eGFR < 30 mL/min/1.73 m2). | 1 |
| LE8_2022 | Lloyd-Jones DM, Allen NB, Anderson CAM, et al. Life's Essential 8: Updating and Enhancing the American Heart Association's Construct of Cardiovascular Health. Circulation. 2022;146(5):e18-e43. | 1 |
| NHANES_DOC | National Center for Health Statistics. NHANES 2017-March 2020 Pre-pandemic data documentation (P_DEMO, P_BPQ, P_DIQ, P_MCQ, P_KIQ_U, P_RXQ_RX). https://wwwn.cdc.gov/nchs/nhanes/. | 1 |
| UCI1989 | Detrano R, Janosi A, Steinbrunn W, et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. Am J Cardiol. 1989;64(5):304-310 (UCI Heart Disease dataset). | 1 |
| CARDIO_KAGGLE | Ulianova S. Cardiovascular Disease dataset (70,000 patient records). Kaggle; 2019. https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset. | 1 |
