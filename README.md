# CARE-LG: Clinical Algorithmic Recourse via Latent Geometry

### **[▶ Open the final report](https://htmlpreview.github.io/?https://github.com/amalroozi/CARE-LG/blob/main/experiment_final/results/report_final.html)**

The report shows one patient's recommendation plan, the CARE-LG architecture, the patient in the rule-pruned latent graph, and all final results.

## What CARE-LG does

For every high-risk cardiovascular patient, CARE-LG returns one of three outcomes, together with an ordered, contraindication-free treatment plan:

1. **Tier 1: certified real.** A safe path to a real, recorded lower-risk patient, found by Dijkstra on a rule-pruned latent graph and certified by a clingo logic program.
2. **Tier 2: guided synthetic.** If no real patient is reachable, an ontology-guided genetic search for a new, valid, plausible profile.
3. **Tier 3: certified none.** An exhaustive proof that no safe real target exists, with the closest miss.

A learned risk model and latent graph are constrained and guided by medical rules (an ICD-10-CM ontology with inherited guideline rules): the neurosymbolic part.

## Results (5 seeds)

| Dataset | High-risk test patients | Admissible recourse (Tier 1 + 2) | Tier 1 (certified real) | Tier 3 (certified none) | Tier-1 real-value violations | Contraindicated cards |
|---|---|---|---|---|---|---|
| NHANES 1999-2023 | 15,002 | 96.4% | 63.1% | 3.6% | 0 | 0 |
| NHANES 2017-2020 | 2,151 | 83.7% | 40.0% | 16.3% | 0 | 0 |
| Kaggle CVD-8k | 3,296 | 99.9% | 98.8% | 0.1% | 0 | 0 |
| UCI Heart | 115 | 95.8% | 26.0% | 4.2% | 0 | 0 |

Kaggle CVD-8k is an 8,000-patient stratified subsample of the Kaggle Cardiovascular Disease dataset (Ulianova 2019). All numbers are means over 5 seeds.

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

From the repo root:
```bash
.venv/bin/python -m pytest experiment_final                  # tests
.venv/bin/python -m experiment_final.src.build_report        # rebuild the final report
.venv/bin/python -m experiment_final.src.narrate --tier 2    # print one patient's plan (tiers 1, 2, 3)
open experiment_final/results/report_final.html              # open the report locally
```
Commands for showing a different patient are at the top of `RUNNING_THE_PROJECT.md`.

## Where things are

- **`experiment_final/`**: the final project (`src/` pipeline, `adapters/` datasets, `knowledge/` clinical rule and intervention tables, `tests/`, `results/`, `MD files/` documentation). Start with its `README.md`.
- **`src/`**: the VAE, graph and baseline code that the final project builds on.
- **`RUNNING_THE_PROJECT.md`**: all commands.
