# CARE-LG: Clinical Algorithmic Recourse via Latent Geometry

**CARE-LG** finds clinically admissible recourse plans for patients flagged
high-risk for cardiovascular disease. Given a real patient, it encodes them
into a learned latent space, builds a graph over real training patients
weighted by a Riemannian metric derived from the VAE decoder's own geometry,
and searches for the cheapest path to a real, recorded low-risk patient —
filtering every step through a neurosymbolic layer of cited clinical
guideline rules (ACC/AHA, ADA). Patients with no admissible path are not
given a generic error — they receive a certified-infeasibility
classification and a semifactual explanation grounded in an exhaustive
check of every real alternative.

This repository went through a multi-session investigation (see
`docs/HISTORY.md`) that found and fixed two code-level bugs which had let
early versions silently accept clinically unsafe plans. **`src/` is the
current, corrected implementation** — the investigation that produced it is
preserved in `archive/` as historical record, not as active code.

## Current results (real data, 5 seeds — see `docs/FINDINGS.md` for full detail)

| Dataset | Success rate | Real-world constraint violation | Certified infeasible (of abstentions) |
|---|---|---|---|
| UCI Heart | 38.6% ± 13.1% | **0.00% ± 0.00%** | 87.9% |
| Real NHANES (PCE-labeled) | 37.7% ± 3.4% | **0.00% ± 0.00%** | 98.5% |

Read plainly: when CARE-LG finds a plan, it is real-world constraint-safe
with no exceptions found across 4,500+ evaluations. It does not find a plan
for roughly 6 in 10 high-risk patients — and for the large majority of
those, an exhaustive check confirms no safe single-step plan actually
exists in the data, rather than the search having simply failed to find one.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

**Single-patient report** (no server; trains models, writes and opens an
HTML report explaining every pipeline stage for that one real patient):
```bash
python -m scripts.run_single_patient --dataset uci --seed 0 --patient_idx 0
```

**Interactive UI** (patient selector, risk gauge, preference control,
step-by-step plan with guideline citations, abstention explanations):
```bash
python -m uvicorn ui.server:app --host 127.0.0.1 --port 8731
# open http://127.0.0.1:8731/
```

**Reproduce the findings** (5 seeds, both datasets — ~10 min):
```bash
python -m scripts.run_grid --datasets uci nhanes_real --seeds 0 1 2 3 4
python -m scripts.run_blindspot --datasets uci nhanes_real --seeds 0 1 2 3 4
python -m scripts.run_classifier_audit
python -m pytest tests/
```

## Where things are

- **`src/`** — the current implementation: `classifier/` (risk model),
  `vae/` (type-aware VAE), `graph/` (CALG construction + Riemannian metric
  + neurosymbolic admissibility layer), `recourse/` (search, explanations,
  preference elicitation), `reporting/` (single-patient HTML report).
- **`scripts/`** — entry points that reproduce every result in `docs/FINDINGS.md`.
- **`ui/`** — the local interactive UI.
- **`tests/`** — regression tests, including the entry-gate/interior-gate
  symmetry check (`test_gate_symmetry.py`) that guards against the bug this
  project's central fix addressed.
- **`docs/`** — `FINDINGS.md` (current results), `METHOD.md` (method
  description, including the neurosymbolic rule base with citations),
  `HISTORY.md` (how the project got here, in plain language),
  `CLASSIFIER_AUDIT.md`, `DATA_PROVENANCE.md`.
- **`results/`** — current, reproducible output (tables, figures, the full
  HTML report, two example single-patient reports).
- **`archive/investigation_v2_to_v6/`** — the full six-session
  investigation that led to `src/`, preserved as historical record. Not
  meant to be run from its new location; read `docs/HISTORY.md` first.

## Requirements

Python 3.11, PyTorch 2.0+. `clingo` (ASP solver for the neurosymbolic
layer), `shap` (explanations), and `fastapi`/`uvicorn` (UI) are all in
`requirements.txt` and install cleanly with no compiler step.
