"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/graph/riemannian_batch.py."""
from src.graph.riemannian_batch import batched_riemannian_distances

__all__ = ["batched_riemannian_distances"]
