"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/pipeline.py."""
from src.pipeline import DatasetRun, run_dataset_seed, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD, K_NEIGHBORS

__all__ = ["DatasetRun", "run_dataset_seed", "HIGH_RISK_THRESHOLD", "LOW_RISK_THRESHOLD", "K_NEIGHBORS"]
