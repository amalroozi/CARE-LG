"""Dataset adapters: `get(name).load(config) -> DatasetBundle` (DATASET_ADAPTER.md)."""
import importlib


def get(name: str):
    """Adapter module by name; the only place a dataset name is resolved to code (P1)."""
    return importlib.import_module(f"experiment_final.adapters.{name}")
