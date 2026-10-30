# Pooled NHANES (largest possible open cohort)

Goal: every continuous NHANES cycle 1999–2023 (+ optional NHANES III 1988–94), harmonised into one DatasetBundle via `adapters/nhanes_pooled.py`.

| Cycle | suffix | Lab file names (chol / HDL / HbA1c) | BP file | Rx | Mortality link |
|---|---|---|---|---|---|
| 1999–2000 | (none) | LAB13 / LAB13 / LAB10 | BPX | RXQ_RX | yes |
| 2001–02 | _B | L13_B / L13_B / L10_B | BPX_B | RXQ_RX_B | yes |
| 2003–04 | _C | L13_C / L13_C / L10_C | BPX_C | RXQ_RX_C | yes |
| 2005–06 | _D | TCHOL_D / HDL_D / GHB_D | BPX_D | RXQ_RX_D | yes |
| 2007–08 | _E | TCHOL_E / HDL_E / GHB_E | BPX_E | RXQ_RX_E | yes |
| 2009–10 | _F | TCHOL_F / HDL_F / GHB_F | BPX_F | RXQ_RX_F | yes |
| 2011–12 | _G | TCHOL_G / HDL_G / GHB_G | BPX_G | RXQ_RX_G | yes |
| 2013–14 | _H | TCHOL_H / HDL_H / GHB_H | BPX_H | RXQ_RX_H | yes |
| 2015–16 | _I | TCHOL_I / HDL_I / GHB_I | BPX_I | RXQ_RX_I | yes |
| 2017–Mar 2020 | P_ | P_TCHOL / P_HDL / P_GHB | P_BPXO (oscillometric) | P_RXQ_RX | 2017–18 part only (via J LMF) |
| Aug 2021–2023 | _L | TCHOL_L / HDL_L / GHB_L | BPXO_L (oscillometric) | RXQ_RX_L | no (too recent) |
| NHANES III 1988–94 (optional) | — | different format (ASCII + SAS code) | — | — | yes |

Per cycle also: DEMO, BMX, BPQ, DIQ, SMQ, MCQ, KIQ_U (kidney; name varies), plus mortality `NHANES_<years>_MORT_2019_PUBLIC.dat` from https://ftp.cdc.gov/pub/health_statistics/nchs/datalinkage/linked_mortality/

Variable names change across cycles: verify every variable against that cycle's codebook (`.htm` next to each `.xpt`) and record the mapping in `knowledge/nhanes_variable_map.csv` (cycle, file, variable, canonical_name).

Harmonisation (record every choice in data provenance):
1. Add `cycle` column; split train/test by cycle as a secondary external validation (older -> newer).
2. BP: auscultatory BPX (<=2015–16 and 2017–18 if used) vs oscillometric BPXO (P_, _L): mean of available readings; report results with/without oscillometric cycles.
3. Ages 40–79 for the PCE label (age top-coded at 80/85 in some cycles).
4. HbA1c/cholesterol lab method changes: note; optional cycle-level sensitivity.
5. Rx reason codes (RXDRSC*) exist only in later cycles; earlier cycles map drug name -> intervention class.
6. Complete-case on model features; report counts dropped per cycle. Optional: impute non-model fields only.
7. Unweighted analysis (methods paper); state it. Optional weighted prevalence tables.
Expected size: ~25,000–35,000 adults 40–79 with complete features (more with NHANES III).
