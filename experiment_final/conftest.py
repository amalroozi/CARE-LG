"""pytest path setup: the repo root must be importable so both `src.*` (CARE-LG) and
`experiment_final.*` (this tree) resolve without the two `src` packages colliding."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))  # repo root first so `src` is CARE-LG's src, not experiment_final/src
