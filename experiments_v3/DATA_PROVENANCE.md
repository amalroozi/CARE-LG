# DATA_PROVENANCE.md — real NHANES cohort construction

## 1. Cycle and files

**Cycle used: NHANES August 2017 – March 2020 Pre-pandemic ("P_" prefix files).** This is
the most recent *complete* pre-pandemic cycle: NHANES 2019–2020 data collection was
interrupted by COVID-19 in March 2020 and was never released as a standalone cycle: CDC
instead combined the partial 2019–2020 collection with the full 2017–2018 collection into
this single 2017–March 2020 "pre-pandemic" file set, which is the standard recommendation
for analyses needing the most recent complete pre-pandemic NHANES data.

**A real network/URL-path obstacle was hit and resolved, not worked around with fake data.**
The URL pattern `https://wwwn.cdc.gov/Nchs/Nhanes/2017-2020/P_DEMO.XPT` (the pattern used by
most older tutorials/blog posts, and initially attempted here) returns HTTP 200 but with an
HTML "Page Not Found" body — CDC migrated the actual file paths since. The correct, currently
live path pattern, discovered by fetching CDC's own `search/datapage.aspx` listing pages and
confirmed by binary XPT content (not HTML) on download, is:
```
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{FILE}.xpt
```
All 9 files below were downloaded from this corrected path and verified as real binary SAS
XPORT data (`file` command reports "data", not "HTML document") before use. Raw files live
in `experiments_v3/data_raw/` (not deleted, kept for audit).

| File | Component | Size | Variables used |
|---|---|---:|---|
| `P_DEMO.xpt` | Demographics | 3,614,720 B | `SEQN`, `RIAGENDR` (sex), `RIDAGEYR` (age), `RIDRETH3` (race/ethnicity), `RIDSTATR` (interview/exam status) |
| `P_BPXO.xpt` | Blood Pressure – Oscillometric Measurements | 1,039,840 B | `BPXOSY1-3`, `BPXODI1-3` (up to 3 oscillometric readings; this cycle switched from the older auscultatory `BPX`/`BPXSY` protocol to the oscillometric `BPXO` device) |
| `P_BMX.xpt` | Body Measures | 2,520,640 B | `BMXBMI` |
| `P_TCHOL.xpt` | Total Cholesterol | 294,000 B | `LBXTC` (mg/dL) |
| `P_HDL.xpt` | HDL Cholesterol | 294,000 B | `LBDHDD` (mg/dL) |
| `P_GHB.xpt` | Glycohemoglobin | 167,600 B | `LBXGH` (HbA1c, %) |
| `P_BPQ.xpt` | Blood Pressure & Cholesterol Questionnaire | 899,520 B | `BPQ020` (ever told high BP), `BPQ050A` (currently taking BP medication) |
| `P_DIQ.xpt` | Diabetes Questionnaire | 3,361,520 B | `DIQ010` (ever told diabetes) |
| `P_SMQ.xpt` | Smoking – Cigarette Use | 1,428,560 B | `SMQ020` (smoked ≥100 cigarettes lifetime), `SMQ040` (currently smokes: every day/some days/not at all) |

## 2. Feature schema mapping (matched one-for-one to the existing synthetic generator)

The existing synthetic NHANES generator (`scripts/download_nhanes.py`) and
`configs/dataset_config.py`'s `DATASET_CONFIGS['nhanes']` produce exactly 7 model features +
1 target: `age, sex, systolic_bp, diastolic_bp, cholesterol, bmi, glycemic_hba1c,
cvd_risk_flag`. The real cohort matches this **exactly**, one-for-one, for the model-facing
CSV (`experiments_v3/data/nhanes_real.csv`):

| Model feature | NHANES source | Transform |
|---|---|---|
| `age` | `RIDAGEYR` | none (already years) |
| `sex` | `RIAGENDR` | `1.0` if male (`RIAGENDR==1`), else `0.0` — matches the existing repo's convention (see `format_clinical_explanation_report`'s `"Male (1.0)"` check) |
| `systolic_bp` | mean of `BPXOSY1, BPXOSY2, BPXOSY3` | row-wise mean, ignoring missing readings |
| `diastolic_bp` | mean of `BPXODI1, BPXODI2, BPXODI3` | row-wise mean, ignoring missing readings |
| `cholesterol` | `LBXTC` | none (already mg/dL, total cholesterol — same clinical quantity as the UCI/synthetic `cholesterol` feature) |
| `bmi` | `BMXBMI` | none |
| `glycemic_hba1c` | `LBXGH` | none (already %) |

**No feature in the existing schema had to be dropped or substituted** — NHANES has a
direct source for every one of the 7 model features.

Additional NHANES variables (`RIDRETH3`, `BPQ050A`, `DIQ010`, `SMQ020`/`SMQ040`, `LBDHDD`)
are used **only** to compute the risk label (§3) and are **not** included as model/VAE/
classifier input features, preserving exact architectural compatibility with the existing
pipeline (input dimension, feature ordering, `FEATURE_METADATA` constraint classification —
all identical to the existing `'nhanes'` config entry, reused verbatim, not re-derived).

## 3. Label definition: 2013 ACC/AHA Pooled Cohort Equations, ≥7.5% cutoff

The existing synthetic generator's `cvd_risk_flag` was produced from an ad hoc, uncited
logistic function of risk factors (see `experiments_v2/REPO_MAP.md`) — not a validated score.
The original MLP classifier in this repo is trained to predict that flag directly (a binary
"elevated cardiovascular risk" label), not a diagnosis flag — so the **risk-score framing**,
not a self-reported-diagnosis framing, is what the existing pipeline actually expects,
and is preserved here.

**Definition used:** the 2013 ACC/AHA/ACC Pooled Cohort Equations (PCE) 10-year risk of a
first atherosclerotic cardiovascular disease (ASCVD) event, computed per the coefficients
published in:

> Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. "2013 ACC/AHA Guideline on the Assessment
> of Cardiovascular Risk: A Report of the American College of Cardiology/American Heart
> Association Task Force on Practice Guidelines." *J Am Coll Cardiol.* 2014;63(25 Pt
> B):2935–2959. PMCID: PMC4700825.

`cvd_risk_flag = 1` if 10-year ASCVD risk **≥ 7.5%**, else `0`. The 7.5% cutoff is the
guideline's own threshold for recommending statin therapy consideration — the standard,
citable "elevated risk" operating point used throughout the clinical literature (it is not
an arbitrary choice made for this project).

**Coefficients used** (race- and sex-specific; race categories: "White" uses the White
equation, "Black" uses the African-American equation; per the guideline's own stated
approach, all non-Black race/ethnicity categories in NHANES — `RIDRETH3` values other than
4 — are scored with the White equation as the guideline-recommended approximation for
populations without a dedicated equation):

Risk = `1 − S0^exp(Σ(coefficient × ln-transformed or binary covariate) − mean_xb)`

| Term | White Women | Black Women | White Men | Black Men |
|---|---:|---:|---:|---:|
| ln(Age) | −29.799 | 17.1141 | 12.344 | 2.469 |
| ln(Age)² | 4.884 | 0 | 0 | 0 |
| ln(Total Chol) | 13.54 | 0.9396 | 11.853 | 0.302 |
| ln(Age)×ln(Total Chol) | −3.114 | 0 | −2.664 | 0 |
| ln(HDL) | −13.578 | −18.9196 | −7.99 | −0.307 |
| ln(Age)×ln(HDL) | 3.149 | 4.4748 | 1.769 | 0 |
| ln(Treated SBP) | 2.019 | 29.2907 | 1.797 | 1.916 |
| ln(Age)×ln(Treated SBP) | 0 | −6.4321 | 0 | 0 |
| ln(Untreated SBP) | 1.957 | 27.8197 | 1.764 | 1.809 |
| ln(Age)×ln(Untreated SBP) | 0 | −6.0873 | 0 | 0 |
| Current smoker | 7.574 | 0.6908 | 7.837 | 0.549 |
| ln(Age)×Smoker | −1.665 | 0 | −1.795 | 0 |
| Diabetes | 0.661 | 0.8738 | 0.658 | 0.645 |
| Mean (Σ coef×mean) | −29.1817 | 86.6081 | 61.1816 | 19.5425 |
| Baseline survival S0 (10-yr) | 0.96652 | 0.95334 | 0.91436 | 0.89536 |

These exact numeric values (to 4-5 decimal places) were cross-checked against the
open-source reference implementation at
`https://github.com/cerner/ascvd-risk-calculator/blob/master/app/load_fhir_data.js`
(Cerner's archived, publicly released SMART-on-FHIR ASCVD risk calculator, itself a direct
transcription of the Goff et al. 2014 published table), since the primary journal source
(ahajournals.org) returned HTTP 403 to automated fetches from this environment. The two
sources agree to the precision each publishes.

**Inputs to the equation and their NHANES source:**
- Age, sex, race group: `RIDAGEYR`, `RIAGENDR`, `RIDRETH3` (as above)
- Total cholesterol, HDL: `LBXTC`, `LBDHDD`
- Systolic BP: mean of `BPXOSY1-3` (same value used as the model feature)
- Treated-BP flag: `BPQ050A == 1` ("Yes" to currently taking BP medication)
- Current smoker flag: `SMQ020 == 1` (smoked ≥100 cigarettes lifetime) **AND**
  `SMQ040 in {1, 2}` (currently smokes every day or some days) — i.e. a genuine *current*
  smoker, not merely a lifetime-ever smoker
- Diabetes flag: `DIQ010 == 1` ("Yes" to ever told by a doctor you have diabetes)

**Documented simplifications / deviations from a fully faithful clinical PCE application:**
- The PCE is designed for **primary prevention** (people without prior ASCVD). We did **not**
  exclude respondents with a prior heart attack/stroke/CHD diagnosis (would require the MCQ
  questionnaire component, not otherwise needed by this schema) — a real simplification,
  stated here rather than silently omitted. Including prevalent-disease cases likely
  modestly inflates the high-risk group beyond a strict incident-risk cohort.
  - Following clinical convention, the PCE is **only validated for ages 40–79**. This forces
    a real restriction of the modeling cohort to that age range (see §5) — a direct,
    documented consequence of choosing this label definition, different from the synthetic
    generator's wider 30–85 age range and from the UCI cohort's ~29–77 range.

## 4. SEQN merge and sample size at each step

All 9 components were inner-joined on `SEQN` (the unique NHANES respondent ID) — i.e. a
respondent must appear in **every** component (DEMO, BPXO, BMX, TCHOL, HDL, GHB, BPQ, DIQ,
SMQ) to be retained, since NHANES draws different lab/exam subsamples per component (e.g.
fasting glucose only from a fasting subsample; not everyone completes the MEC exam).

| Step | N remaining | N dropped |
|---|---:|---:|
| DEMO (all ages, all respondents) | 15,560 | — |
| Inner-join across all 9 components on `SEQN` | 9,445 | 6,115 |
| Restrict to `RIDSTATR==2` (interviewed AND MEC-examined) | 9,445 | 0 (already satisfied by the join) |
| Restrict to PCE-valid age range [40, 79] | 5,373 | 4,072 |
| Complete-case exclusion (drop any remaining missing value in a needed field) | 4,539 | 834 |
| Exclude non-positive cholesterol/HDL/SBP (required for PCE log terms) | 4,539 | 0 |
| **Final N** | **4,539** | |

## 5. Missingness (measured after the 9-way component join, before the age/complete-case filters)

| Feature | N missing | % missing (of 9,445) |
|---|---:|---:|
| age | 0 | 0.00% |
| sex | 0 | 0.00% |
| systolic_bp | 988 | 10.46% |
| diastolic_bp | 988 | 10.46% |
| cholesterol | 719 | 7.61% |
| bmi | 181 | 1.92% |
| glycemic_hba1c | 540 | 5.72% |
| race_group / treated_bp / smoker / diabetic | 0 | 0.00% (derived from questionnaire responses already required by the join) |

Full table: `experiments_v3/data/missingness_report.csv`.

**Handling rule: complete-case exclusion, not imputation.** Rows with any missing value in a
needed field are dropped rather than imputed. This is a deliberate, documented choice: BP,
cholesterol, and HbA1c are all continuous features that also serve as the *decision
variables* for the recourse task itself (constraint checks, effort computation); imputing
them would inject synthetic values into precisely the quantities this whole project is
trying to reason about honestly, which conflicts with Hard Rule 2 ("never fabricate ...
backfill a result"). The complete-case final N (4,539) is reported as the actual, real
cohort size — larger than either the UCI cohort (237 train) or the prior synthetic NHANES
stand-in (2,000 train), i.e. still a genuinely "dense" cohort even after every restriction.

## 6. Final cohort characteristics

N = 4,539. Age range 40–79 (mean 58.2, std 10.5). Sex: 50.0% male. Mean systolic BP 128.1
mmHg, mean total cholesterol 190.5 mg/dL, mean BMI 30.5, mean HbA1c 6.07%. `cvd_risk_flag`
positive rate: **50.4%** (2,289 / 4,539) — i.e. essentially balanced, which is a real,
unforced property of thresholding a 40–79-year-old general population at the clinical 7.5%
statin-consideration cutoff (roughly half of US adults in this age range cross that
threshold under the PCE — consistent with published PCE validation cohorts), not something
tuned to look balanced.

Full model-facing CSV: `experiments_v3/data/nhanes_real.csv` (7 features + target, exactly
matching the existing schema). Full CSV with provenance columns (race group, computed
ASCVD%, smoker/diabetic/treated-BP flags) retained for audit: `experiments_v3/data/nhanes_real_full.csv`.

## 7. Train/test split protocol

The existing `src/data/loader.py::get_dataloaders()` is reused UNCHANGED: an 80/20
stratified `train_test_split(random_state=seed)` on `cvd_risk_flag`, executed fresh for
each of the 5 seeds (0–4) used throughout this study, with the `StandardScaler` fit on the
train partition only (no leakage) — identical protocol to every other dataset in this
project. No model touches the test partition before this split is drawn.

## 8. Synthetic pipeline preserved

`scripts/download_nhanes.py`, `data/nhanes.csv`, and `configs/dataset_config.py`'s
`DATASET_CONFIGS['nhanes']` entry are **all left completely untouched** and remain fully
runnable exactly as in `experiments_v2`. The new real cohort is registered under a
**separate** dataset key, `'nhanes_real'` (see `experiments_v3/lib/dataset_registration.py`),
at runtime, without editing `configs/dataset_config.py`. Every output filename, config
label, and figure/table row in `experiments_v3/` says `nhanes_real` or "real NHANES," never
bare `"nhanes"` or "NHANES" without qualification, to avoid ever re-creating the original
mislabeling.
