# CARE-LG: Cost-Aware Latent Geometry for Explainable AI in Heart Disease

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**CARE-LG** is an algorithmic recourse framework for cardiovascular risk prediction. By combining Tabular Variational Autoencoders (VAEs), Riemannian metric pullback tensors, and asymmetric graph masking, CARE-LG generates realistic, step-by-step patient recourse trajectories with a **mathematically guaranteed 0% Constraint Violation Rate (CVR)**.

---

## Key Highlights

- **0% Constraint Violations:** Asymmetric graph masking ($W_{ij} = \infty$) prevents impossible steps like decreasing age ($\Delta \text{age} < 0$) or altering biological sex.
- **Riemannian Latent Geometry:** Measures true clinical effort using the pullback metric tensor $G(z) = J_g(z)^T J_g(z)$ over the VAE latent space.
- **Ultra-Fast Recourse:** Solves shortest-path trajectories over the Clinical Adaptive Latent Graph (CALG) via Dijkstra search in **0.0001 seconds**.

---

## Experimental Results (UCI Cleveland Dataset)

Evaluated across 100 high-risk query instances ($N=297$, $d=13$ clinical attributes):

| Method | Success Rate (%) | Constraint Violation Rate (CVR %) | Manifold Density (KDE) | Latency (sec) |
| :--- | :---: | :---: | :---: | :---: |
| **DiCE** | 86.36% | 100.00% | -13.32 | 0.0513s |
| **FACE** | 100.00% | 86.36% | -7.18 | 0.0040s |
| **Growing Spheres** | 63.64% | 100.00% | -16.48 | 0.0148s |
| **CARE-LG (Ours)** | **72.73%** | **0.00%** | **-7.38** | **0.0001s** |

---

## Quickstart

```bash
# Clone & install dependencies
git clone [https://github.com/your-username/care_lg.git](https://github.com/your-username/care_lg.git)
cd care_lg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run single recourse test
python scripts/test_day2.py

# Run full benchmark evaluation & generate figures
python benchmarks/run_benchmarks.py
python scripts/generate_paper_artifacts.py
