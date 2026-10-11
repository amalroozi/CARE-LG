"""Adapter: eicu_demo -> DatasetBundle (see DATASET_ADAPTER.md). Optional; not in config.datasets.

Planned: static first-24h snapshot of the eICU-CRD demo (archive/experiments_caren/data/processed/
state_table.parquet, window 0) with diagnosis-derived ICD-10-CM concepts. Not implemented:
the run is restricted to the three datasets in config.yaml. Kept as an explicit stub so the
adapter-contract test documents (and skips) it rather than silently ignoring it.
"""
from experiment_final.adapters.base import DatasetBundle


def load(config: dict) -> DatasetBundle:
    raise NotImplementedError("eicu_demo adapter is planned but not part of the final run (see DECISIONS.md)")
