"""
Compatibility shim. The actual implementation moved to `src/graph/query_attachment.py`
as part of Phase 1 of experiments_v6_audit/DEEP_AUDIT.md's resolution (issue #1:
"none of this is in src/") -- see experiments_v6_audit/DEEP_AUDIT_RESOLUTION.md.
Re-exported here, unmodified, so every existing `from experiments_v6.lib.graph_build
import ...` in this project's scripts continues to work without any changes and
produces byte-identical results (it is literally the same code, just relocated).
"""
from src.graph.query_attachment import (
    unscale_matrix,
    vectorized_violation_components,
    vectorized_violation_counts,
    EdgeTable,
    build_edge_table,
    QueryEdges,
    build_query_edges,
    make_cell_matrix,
    make_query_row,
    augmented_matrix,
)

__all__ = [
    "unscale_matrix", "vectorized_violation_components", "vectorized_violation_counts",
    "EdgeTable", "build_edge_table", "QueryEdges", "build_query_edges",
    "make_cell_matrix", "make_query_row", "augmented_matrix",
]
