# HOW_TO_RUN.md — experiments_v6

No screenshots in this document or elsewhere in this session's deliverables,
per the brief (explicitly deferred to a later session).

## Prerequisites

From the repo root, using the project's existing virtualenv:

```bash
.venv/bin/pip install clingo shap fastapi "uvicorn[standard]"
```

(`clingo` and `shap` were installed this session for Phases 4-5; `fastapi`/
`uvicorn` for Phase 7's UI. All four installed cleanly with no compiler step.)

## Reproducing the experiment results (Phases 1, 2, 4, 6, 8)

Run from the repo root. Each writes its own metadata JSON (git SHA, seeds,
config) and raw CSV(s) under `experiments_v6/results/`.

```bash
# Phase 1: corrected baseline grid (B1+B2 fix), 5 seeds, both datasets
.venv/bin/python -m experiments_v6.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4

# Phase 2: certified infeasibility vs. blindspot decomposition
.venv/bin/python -m experiments_v6.run_blindspot --datasets uci nhanes_real --seeds 0 1 2 3 4

# Phase 4: neurosymbolic regression check (symbolic reasoner vs. hardcoded predicate)
.venv/bin/python -m experiments_v6.run_neurosymbolic_regression

# Phase 6: preference-elicitation demo (small, illustrative)
.venv/bin/python -m experiments_v6.run_preference_demo --datasets uci nhanes_real --seeds 0 --n_patients 5

# Phase 8: FACE rerun (corrected protocol) and the PACE-style reimplementation
.venv/bin/python -m experiments_v6.run_face_rerun --datasets uci nhanes_real --seeds 0 1 2 3 4
.venv/bin/python -m experiments_v6.run_pace_reimpl --datasets uci nhanes_real --seeds 0 1 2 3 4
```

Approximate wall-clock this session (CPU, per `experiments_v6/results/run_*_log.txt`):
Phase 1 grid ~9 min total (UCI seconds, real NHANES ~80s/seed x 5); Phase 2
blindspot ~5 min (NHANES's densification probes dominate, ~50-65s/seed);
Phase 4 regression ~1 min; Phase 6/8 scripts each well under a minute.

## Running the UI (Phase 7)

A minimal local FastAPI backend serves both the JSON API and the static
frontend. From the repo root:

```bash
.venv/bin/python -m uvicorn experiments_v6.ui.server:app --host 127.0.0.1 --port 8731
```

Then open `http://127.0.0.1:8731/` in a browser. The first request for a
given (dataset, seed) pair trains a fresh VAE + classifier in-process
(a few seconds for UCI, ~3-5s for real NHANES) and caches it for the rest of
that server's lifetime; subsequent requests for the same (dataset, seed) are
fast.

**What you'll see** (five panels, per the brief):
1. **Patient selector** — dataset, seed, and a dropdown of real high-risk
   test-cohort patients (loaded from `run_dataset_seed`'s actual test split,
   never fabricated) plus a preference-profile selector.
2. **Risk gauge** — classifier risk before the plan and (if successful)
   after reaching the selected target patient.
3. **Preference control** — `no_preference` / `exercise_averse` /
   `medication_averse` (Phase 6); changes the edge COST only, never
   admissibility.
4. **Step-by-step plan** — each hop's real feature deltas plus its Phase-4
   guideline derivation (which rules were checked and passed, or which
   blocked it).
5. **Abstention panel** — shown instead of the plan when no admissible path
   exists: the Phase-2 certification (certified-infeasible / confirmed-
   blindspot / unresolved, looked up from that patient's actual precomputed
   `experiments_v6/results/{dataset}_blindspot.csv` row when available) and
   the Phase-5 semifactual explanation, grounded in a live recomputation
   against every low-risk training candidate — not a generic error message.

To stop the server, `Ctrl+C` in its terminal (or find and kill the
`uvicorn` process if it was backgrounded).

### API surface (for reference / scripting against the backend directly)

- `GET /api/datasets` → `{"datasets": ["uci", "nhanes_real"]}`
- `GET /api/patients?dataset=uci&seed=0&limit=60` → real test-cohort patients
  with their classifier risk and high-risk flag, plus the available
  preference-profile names.
- `POST /api/recourse` — body `{"dataset", "seed", "patient_idx", "profile"}`
  → on success: `risk_before`/`risk_after`, path length, cost, effort, and
  per-step `feature_deltas` + `guideline_derivation`; on abstention:
  `certification`, `semifactual_explanation`, the closest-miss candidate and
  its violation count, and the blocking-rule frequency table. Both branches
  include `shap_top_features` (Phase 5 layer 1).

## Auto-generated single-patient HTML report (no server needed)

For a single real patient, this writes one self-contained HTML file
(patient-facing result + a plain-language architecture walkthrough of every
pipeline stage using that patient's own real numbers) and opens it in the
default browser automatically — no UI server required:

```bash
.venv/bin/python -m experiments_v6.run_single_patient --dataset uci --seed 0 --patient_idx 0
# writes + opens experiments_v6/results/uci_seed0_patient0_report.html

# pass --no-open to just write the file without launching a browser
.venv/bin/python -m experiments_v6.run_single_patient --dataset uci --seed 0 --patient_idx 1 --no-open
```

Two example outputs (one success, one abstention) are already saved at
`experiments_v6/results/uci_seed0_patient0_report.html` and
`..._patient1_report.html` for review. The same generation code is also
reachable from the running UI server: pass `"generate_report": true` in the
`POST /api/recourse` body and it writes + opens the same report as a side
effect of that call, using `src.recourse.single_patient.run_single_patient`
— the same underlying function the CLI script calls, so both entry points
produce identical reports for the same (dataset, seed, patient_idx, profile).
Named interventions (e.g. "DASH diet + sodium restriction" for blood
pressure) are looked up from `configs/interventions.json`, cited against the
same guideline registry ([HTN2017], [CHOL2018], [ADA2023]) used by the
neurosymbolic layer.

## Notes

- The UI backend reuses the exact same corrected library code
  (`experiments_v6/lib/`) as the experiment scripts above — it is not a
  separate reimplementation, so anything shown in the UI is reproducible
  from the command line the same way.
- `v4` (DSR) and `v5` (EGD) code is untouched and not wired into this UI or
  any v6 script, per the brief (`RECOMMENDATION.md`'s conclusion that
  neither is needed once B1/B2 are fixed).
