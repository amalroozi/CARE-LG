"""Shared fixtures: knowledge tables, adapter bundles, the newest finished run (if any)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))  # repo root first so `src` is CARE-LG's src, not experiment_final/src

from experiment_final import adapters  # noqa: E402
from experiment_final.src.common import OLD_RUNS as RUNS, load_config  # noqa: E402
from experiment_final.src.knowledge_layer import load_knowledge, schema_from_meta  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def K():
    return load_knowledge()


@pytest.fixture(scope="session")
def bundles(cfg):
    return {n: adapters.get(n).load(cfg) for n in cfg["datasets"]}


@pytest.fixture(scope="session")
def schemas(bundles, K):
    return {n: schema_from_meta(b.feature_meta, K) for n, b in bundles.items()}


def finished_seed_dirs(include_variants: bool = False):
    """Finished <run>/<dataset>/seed<k> folders (results-level tests skip if none). Ablation-variant folders
    (<dataset>__<variant>, e.g. ontology_as_features, which deliberately runs without the symbolic filter)
    are excluded unless asked for: the guarantees are claimed for the main method only."""
    return sorted(d for d in RUNS.glob("2*/*/seed*") if (d / "DONE").exists() and (include_variants or "__" not in d.parent.name))
