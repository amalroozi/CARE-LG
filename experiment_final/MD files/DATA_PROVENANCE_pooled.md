# DATA_PROVENANCE_pooled.md -- pooled NHANES 1999-2023 (CARE-LG primary dataset)

Built by `adapters/nhanes_pooled.py` (download: `adapters/nhanes_pooled_download.py`, log in `data/raw/nhanes_pooled/download_log.csv`). All 11 continuous cycles 1999-2000 ... Aug 2021-2023 were downloaded from wwwn.cdc.gov (every component + its .htm codebook) and the 2019 public-use Linked Mortality Files from ftp.cdc.gov. NHANES III (optional) was not used. No cycle failed.

## N per cycle at every filter

| cycle | DEMO participants | present in DEMO, BMX, BP exam, TCHOL, HDL, GHB, BPQ, DIQ, SMQ | ages 40-79 (PCE-valid) | complete case on model features + HDL | positive cholesterol/HDL/SBP (PCE log terms) | final (PCE label computed) |
|---|---|---|---|---|---|---|
| 1999-2000 | 9965 | 4444 | 2553 | 2284 | 2284 | 2284 |
| 2001-2002 | 11039 | 5027 | 2782 | 2421 | 2421 | 2421 |
| 2003-2004 | 10122 | 4742 | 2637 | 2329 | 2329 | 2329 |
| 2005-2006 | 10348 | 5891 | 2576 | 2328 | 2328 | 2328 |
| 2007-2008 | 10149 | 6307 | 3474 | 3113 | 3113 | 3113 |
| 2009-2010 | 10537 | 6717 | 3630 | 3260 | 3260 | 3260 |
| 2011-2012 | 9756 | 5919 | 3115 | 2730 | 2730 | 2730 |
| 2013-2014 | 10175 | 6266 | 3376 | 3084 | 3084 | 3084 |
| 2015-2016 | 9971 | 6063 | 3267 | 3004 | 3004 | 3004 |
| 2017-2020 | 15560 | 9445 | 5373 | 4539 | 4539 | 4539 |
| 2021-2023 | 11933 | 6615 | 4163 | 3694 | 3694 | 3694 |
| TOTAL | 119555 | 67436 | 36946 | 32786 | 32786 | 32786 |

## Final cohort per cycle (label, mortality link)

| cycle | n | label_pos | with_mortality_link | deaths |
|---|---|---|---|---|
| 1999-2000 | 2284 | 1223 | 2281 | 931 |
| 2001-2002 | 2421 | 1180 | 2419 | 741 |
| 2003-2004 | 2329 | 1229 | 2322 | 707 |
| 2005-2006 | 2328 | 1122 | 2328 | 508 |
| 2007-2008 | 3113 | 1589 | 3112 | 606 |
| 2009-2010 | 3260 | 1548 | 3254 | 423 |
| 2011-2012 | 2730 | 1306 | 2725 | 280 |
| 2013-2014 | 3084 | 1443 | 3080 | 215 |
| 2015-2016 | 3004 | 1491 | 2997 | 133 |
| 2017-2020 | 4539 | 2289 | 0 | 0 |
| 2021-2023 | 3694 | 1940 | 0 | 0 |
| TOTAL | 32786 | 16360 | 24518 | 4544 |

Pooled final N = 32786; label prevalence (PCE >= 7.5%) = 49.9%; mortality-linked = 24518; deaths = 4544; CVD deaths (UCOD 001 heart disease or 005 cerebrovascular) = 1329.

## Harmonisation choices

1. `cycle` column added; cycle-held-out validation trains on 1999-2016 and tests on 2017-2020 and 2021-2023 (`tables/pooled_cycle_held_out.csv`).
2. Blood pressure: mean of available readings. Auscultatory BPXSY1-4/BPXDI1-4 (1999-2016); oscillometric BPXOSY1-3/BPXODI1-3 (2017-2020 `P_BPXO`, 2021-2023 `BPXO_L`). Diastolic readings of 0 (NHANES codes for an inaudible Korotkoff phase V) are treated as missing. Results with vs without oscillometric cycles: `tables/pooled_oscillometric_sensitivity.csv`.
3. Ages 40-79 only (PCE-valid; also avoids the 80/85 age top-codes).
4. Lab-method changes (total cholesterol, HDL, HbA1c assays changed across cycles) are not re-calibrated; noted as a limitation.
5. Rx: drug names RXD240B (1999-2002) / RXDDRUG (2003-2020) -> intervention classes (interventions.csv patterns); reason codes RXDRSC1-3 only 2013-2020; 2021-2023 `RXQ_RX_L` contains no drug names (only counts), so current_meds are empty there.
6. Complete case on the 7 model features + HDL (PCE input); N dropped per cycle above. Non-model fields are not imputed.
7. Unweighted analysis (methods paper); survey weights are not used.
8. Treated-BP flag: BPQ050A (1999-2020); 2021-2023 has no BPQ050A, so BPQ150 ('Taking high blood pressure medication', same skip pattern after BPQ020/030) is used.
9. PCE label computed with `archive/investigation_v2_to_v6/experiments_v3/build_real_nhanes.py::pce_10yr_risk` (imported unchanged). Race for PCE: RIDRETH1 = 4 (non-Hispanic Black) -> Black equation, all others -> White equation (same rule as docs/DATA_PROVENANCE.md). Check: the pooled 2017-2020 slice reproduces `data/nhanes_real_full.csv` exactly (same 4,539 SEQNs; every feature, flag, PCE% and label identical).
10. Mortality: LMF 2019 linkage is available for 1999-2018 only; the pre-pandemic `P_` files (2017-March 2020) assign new SEQNs that do not match the 2017-2018 (J) LMF, so 2017-2020 and 2021-2023 participants have no linked outcome. Fields: eligstat, mortstat, ucod_leading, permth_exm (months from exam).
11. Kidney: KIQ022 (2001+) / KIQ020 (1999-2000) 'weak/failing kidneys' -> N18.9; dialysis KIQ025 (2001+) -> N18.6 + Z99.2. COPD: MCQ160P (2017+) -> J44.9; before 2017 emphysema MCQ160G -> J43.9 and chronic bronchitis MCQ160K -> J42. Asthma 'still have' MCQ035 is absent in 1999-2000.

## Variable map (canonical name x cycle; full table `knowledge/nhanes_variable_map.csv`)

| canonical_name | 1999-2000 | 2001-2002 | 2003-2004 | 2005-2006 | 2007-2008 | 2009-2010 | 2011-2012 | 2013-2014 | 2015-2016 | 2017-2020 | 2021-2023 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 | DIQ010 |
| MCQ035 | - | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 | MCQ035 |
| MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B | MCQ160B |
| MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C | MCQ160C |
| MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E | MCQ160E |
| MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F | MCQ160F |
| MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | MCQ160G | - | - |
| MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | MCQ160K | - | - |
| MCQ160P | - | - | - | - | - | - | - | - | - | MCQ160P | MCQ160P |
| MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L | MCQ170L |
| MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A | MCQ230A |
| MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B | MCQ230B |
| MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C | MCQ230C |
| MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D | MCQ230D |
| SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 | SMQ020 |
| SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 | SMQ040 |
| age | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR | RIDAGEYR |
| bmi | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI | BMXBMI |
| bp_meds | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ050A | BPQ150 |
| cholesterol | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC | LBXTC |
| dbp1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXDI1 | BPXODI1 | BPXODI1 |
| dbp2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXDI2 | BPXODI2 | BPXODI2 |
| dbp3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXDI3 | BPXODI3 | BPXODI3 |
| dbp4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | BPXDI4 | - | - |
| dialysis | - | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 | KIQ025 |
| exam_status | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR | RIDSTATR |
| glycemic_hba1c | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH | LBXGH |
| hdl | LBDHDL | LBDHDL | LBXHDD | LBDHDD | LBDHDD | LBDHDD | LBDHDD | LBDHDD | LBDHDD | LBDHDD | LBDHDD |
| lipid_meds | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ100D | BPQ101D |
| pregnant | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG | RIDEXPRG |
| race1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 | RIDRETH1 |
| rx_drug_name | RXD240B | RXD240B | RXDDRUG | RXDDRUG | RXDDRUG | RXDDRUG | RXDDRUG | RXDDRUG | RXDDRUG | RXDDRUG | - |
| rx_reason_codes | - | - | - | - | - | - | - | RXDRSC1;RXDRSC2;RXDRSC3 | RXDRSC1;RXDRSC2;RXDRSC3 | RXDRSC1;RXDRSC2;RXDRSC3 | - |
| sbp1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXSY1 | BPXOSY1 | BPXOSY1 |
| sbp2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXSY2 | BPXOSY2 | BPXOSY2 |
| sbp3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXSY3 | BPXOSY3 | BPXOSY3 |
| sbp4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | BPXSY4 | - | - |
| sex_raw | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR | RIAGENDR |
| told_chol | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 | BPQ080 |
| told_hbp | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 | BPQ020 |
| weak_kidneys | KIQ020 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 | KIQ022 |
