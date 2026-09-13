"""Compatibility shim -- see experiments_v6/lib/graph_build.py's shim docstring
for the full explanation. Canonical implementation: src/data/dataset_registration.py
(a side-effect-only module: importing it registers the 'nhanes_real' dataset key).
Importing THIS shim triggers that same side effect via the import below."""
import src.data.dataset_registration  # noqa: F401
