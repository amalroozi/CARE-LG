"""Single-patient view of the tiered method (PIPELINE P15), read from a finished run.

  .venv/bin/python experiment_final/single_patient.py --dataset nhanes_pooled --seed 0 [--test-pos 123] [--tier 2]

Prints the patient's tier, recourse target, recommendation card (escalation order, excluded classes with
their ICD-10-CM derivation, unmapped changes), tier-1 clingo certificate and the traceable explanation.
Without --test-pos it picks the first patient of the requested tier. Patient-level output stays local.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from experiment_final.src.common import OLD_RUNS as RUNS  # noqa: E402


def find_seed_dir(dataset: str, seed: int) -> Path:
    full = EXP_DIR / "results" / "run_full_nhanes" / dataset / f"seed{seed}"      # full NHANES 1999-2023 rerun
    c = sorted((d for d in RUNS.glob(f"*/{dataset}/seed{seed}") if (d / "DONE").exists()), key=lambda p: p.stat().st_mtime)
    if (full / "DONE").exists():
        return full
    if not c:
        sys.exit(f"no finished run for {dataset} seed {seed} under {RUNS} or {full.parent}")
    return c[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes_pooled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-pos", type=int)
    ap.add_argument("--tier", type=int, choices=[1, 2, 3])
    a = ap.parse_args()
    d = find_seed_dir(a.dataset, a.seed)
    feats = json.loads((d / "setup.json").read_text())["features"]
    cards = [json.loads(l) for l in open(d / "cards.jsonl")]
    expl = {json.loads(l)["test_pos"]: json.loads(l) for l in open(d / "explanations.jsonl")}
    sel = [c for c in cards if (a.test_pos is None or c["test_pos"] == a.test_pos) and (a.tier is None or c["tier"] == a.tier)]
    if not sel:
        sys.exit("no matching patient")
    c = sel[0]
    print(f"run: {d}\npatient test_pos={c['test_pos']}  TIER {c['tier']}")
    print("concepts:", ", ".join(c["concepts"]) or "none", "| current meds:", ", ".join(c["current_meds"]) or "none")
    print("baseline:", ", ".join(f"{f}={v:.1f}" for f, v in zip(feats, c["x0"])))
    if c["target"] is not None:
        print("target:  ", ", ".join(f"{f}={v:.1f}" for f, v in zip(feats, c["target"])))
    card = c["card"]
    for it in card.get("recommendations", []):
        print(f"  {it['rank']}. {it['class']:<26} [{it['step']}] features={it['features']} risk_drop={it['risk_drop']:+.3f} cite={it['citation']}")
    for e in card.get("excluded", []):
        print(f"  excluded {e['class']}: {e['reason']} {e.get('derivation') or ''}")
    for u in card.get("unmapped", []):
        print(f"  unmapped {u['feature']}: {u['reason']}")
    if c.get("certificate"):
        print("certificate: all_hard_ok =", c["certificate"]["all_hard_ok"], "| passed:", c["certificate"]["passed_checks_endpoint"])
    if c["test_pos"] in expl:
        print("\n" + expl[c["test_pos"]]["text"] + f"\ntrace failures: {expl[c['test_pos']]['trace_failures'] or 'none'}")


if __name__ == "__main__":
    main()
