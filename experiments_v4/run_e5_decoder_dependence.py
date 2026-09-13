"""
E5: decoder-dependence check. Runs the main method cells (at the q=0.95
operating point, no tau-sweep) with the v2 (untyped) decoder instead of v3
(type-aware), to test whether DSR's decoded-CVR reduction is a property of
the METHOD or is mostly inherited from Phase B's better decoder.

Writes to its own output-suffix'd files (`_v2decoder`) so it never collides
with the main run's ({dataset}_per_patient.csv from run_dsr.py).

Usage:
  .venv/bin/python -m experiments_v4.run_e5_decoder_dependence --datasets uci nhanes_real --seeds 0 1 2 3 4 [--smoke]
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_v4.run_dsr import main as run_dsr_main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["uci", "nhanes_real"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    forwarded = ["--datasets", *args.datasets, "--seeds", *[str(s) for s in args.seeds],
                 "--decoders", "v2", "--output-suffix", "_v2decoder"]
    if args.smoke:
        forwarded.append("--smoke")
    sys.argv = [sys.argv[0]] + forwarded
    run_dsr_main()
