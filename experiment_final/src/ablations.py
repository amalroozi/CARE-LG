"""Phase 13: tier1-only, no ontology, no inheritance, no guidance, soft-as-hard, no escalation, Euclidean, no SRA, ontology-as-features.

Each arm = (tier-1 arm, tier-2 setting, rule-set source, recommendation mode, tier-2 enabled). The
`ontology_as_features` arm is a separate pipeline variant (concept-block indicator columns appended
to the model inputs as immutable binary features; generic population rules; no symbolic filter).
Arms that remove the ontology are evaluated for harm against the TRUE (full-inheritance) rule set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from experiment_final.src.tier1 import Tier1Arm

ONTOLOGY_ARMS = {"no_ontology", "no_inheritance", "ontology_as_features"}


@dataclass
class Arm:
    name: str
    tier1: Tier1Arm
    rules: str = "full"            # full | no_inheritance | generic
    guidance: bool = True
    soft_as_hard: bool = False
    rec_mode: str = "escalation"   # escalation | risk_only
    tier2: bool = True
    apply_filter: bool = True


def arms(cfg: dict, has_ontology: bool) -> Dict[str, Arm]:
    """Main method + every ablation in config.ablations applicable to this dataset (P13)."""
    t_main = Tier1Arm("main")
    out = {"main": Arm("main", t_main)}
    spec = {
        "tier1_only": Arm("tier1_only", t_main, tier2=False),
        "no_ontology": Arm("no_ontology", t_main, rules="generic"),
        "no_inheritance": Arm("no_inheritance", t_main, rules="no_inheritance"),
        "no_guidance": Arm("no_guidance", t_main, guidance=False),
        "soft_as_hard": Arm("soft_as_hard", Tier1Arm("soft_as_hard", soft_as_hard=True), soft_as_hard=True),
        "no_escalation": Arm("no_escalation", t_main, rec_mode="risk_only"),
        "euclidean": Arm("euclidean", Tier1Arm("euclidean", metric="euclidean")),
        "no_sra": Arm("no_sra", Tier1Arm("no_sra", sra=False)),
    }
    for a in cfg["ablations"]:
        if a in spec and (has_ontology or a not in ONTOLOGY_ARMS):
            out[a] = spec[a]
    return out


def concept_feature_matrix(concepts: List[List[str]], onto, min_prev: float = 0.05) -> pd.DataFrame:
    """ICD-10-CM block indicators (prevalence >= 5%) for the ontology-as-features arm (NEURON-style) (P13)."""
    blocks = [sorted({a for c in cs if c in onto for a in onto.ancestors(c) if "-" in a}) for cs in concepts]
    prev = pd.Series([b for bl in blocks for b in bl]).value_counts() / len(concepts)
    keep = sorted(prev[prev >= min_prev].index)
    return pd.DataFrame({f"blk_{b.replace('-', '_').lower()}": [float(b in bl) for bl in blocks] for b in keep})
