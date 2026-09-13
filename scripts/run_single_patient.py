"""
Runs the corrected src/ pipeline for exactly ONE real patient, writes a
single self-contained HTML report (results/{dataset}_seed{seed}_patient{idx}_report.html),
and opens it in the default browser immediately -- no server, no manual step.

Usage:
  .venv/bin/python -m experiments_v6.run_single_patient --dataset uci --seed 0 --patient_idx 0
  .venv/bin/python -m experiments_v6.run_single_patient --dataset uci --seed 0 --patient_idx 1 --no-open
"""
import argparse
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recourse.single_patient import run_single_patient
from src.reporting.report_html import write_report

RESULTS_DIR = REPO_ROOT / "results"  # write_report() writes the standalone HTML report here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["uci", "nhanes_real"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patient_idx", type=int, required=True)
    ap.add_argument("--profile", default="no_preference")
    ap.add_argument("--no-open", action="store_true", help="write the report but skip opening a browser")
    args = ap.parse_args()

    print(f"[single-patient] running {args.dataset} seed={args.seed} patient #{args.patient_idx} "
          f"(profile={args.profile})...", flush=True)
    result = run_single_patient(args.dataset, args.seed, args.patient_idx, args.profile)
    outcome = "SUCCESS" if result["success"] else f"ABSTAINED ({result['certification']})"
    print(f"[single-patient] result: {outcome}", flush=True)

    path = write_report(result, RESULTS_DIR)
    print(f"[single-patient] wrote {path}", flush=True)

    if not args.no_open:
        webbrowser.open(f"file://{path.resolve()}")
        print("[single-patient] opened in default browser", flush=True)


if __name__ == "__main__":
    main()
