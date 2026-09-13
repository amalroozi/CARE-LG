"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/graph/constraints_ext.py."""
from src.graph.constraints_ext import (
    violates,
    count_violations,
    decode_node,
    decode_nodes_batch,
    check_decoded_violations,
    _DIRECTIONAL_EPS,
    _AGE_DECREASE_TOL,
    _AGE_HORIZON_CAP,
    _IMMUTABLE_TOL,
)

__all__ = [
    "violates", "count_violations", "decode_node", "decode_nodes_batch",
    "check_decoded_violations", "_DIRECTIONAL_EPS", "_AGE_DECREASE_TOL",
    "_AGE_HORIZON_CAP", "_IMMUTABLE_TOL",
]
