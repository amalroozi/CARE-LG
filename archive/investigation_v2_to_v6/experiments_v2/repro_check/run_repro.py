"""
Read-only reproduction check: reruns the EXISTING benchmarks/run_benchmarks.py pipeline
end-to-end (fresh model training, fresh CALG build) for both datasets, redirecting all
output paths to experiments_v2/repro_check/ so nothing under benchmarks/ or models/ is
touched. Uses the exact same functions as the original pipeline (no logic changes).
"""
import sys, json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import benchmarks.run_benchmarks as rb

REPRO_ROOT = Path(__file__).resolve().parent
rb.PROJECT_ROOT = REPRO_ROOT  # redirect models_dir / output_dir construction only

for dataset in ["uci", "nhanes"]:
    t0 = time.perf_counter()
    rb.run_all_benchmarks(dataset_name=dataset, num_query_instances=100, seed=42)
    t1 = time.perf_counter()
    print(f"[repro] {dataset} total wall time: {t1-t0:.1f}s")
