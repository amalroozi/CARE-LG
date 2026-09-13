"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/vae/type_aware.py."""
from src.vae.type_aware import (
    ColumnTypeInfo,
    detect_column_types,
    TabularVAETypeAware,
    loss_function_type_aware,
    train_vae_type_aware,
)

__all__ = [
    "ColumnTypeInfo", "detect_column_types", "TabularVAETypeAware",
    "loss_function_type_aware", "train_vae_type_aware",
]
