"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/recourse/preference.py."""
from src.recourse.preference import (
    PROFILE_MULTIPLIER,
    preference_profiles,
    make_cell_matrix_preference,
    make_query_row_preference,
)

__all__ = [
    "PROFILE_MULTIPLIER", "preference_profiles",
    "make_cell_matrix_preference", "make_query_row_preference",
]
