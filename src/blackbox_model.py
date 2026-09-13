"""Compatibility shim -- the RiskClassifier moved to src/classifier/blackbox_model.py
as part of the repo consolidation (see CHANGES.md). Re-exported here, unmodified,
so every existing `from src.blackbox_model import ...` (scripts/, benchmarks/)
keeps working without any changes."""
from src.classifier.blackbox_model import RiskClassifier, get_device, train_blackbox_model

__all__ = ["RiskClassifier", "get_device", "train_blackbox_model"]
