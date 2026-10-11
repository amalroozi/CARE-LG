"""Full rerun (no patient caps, full cleaned Kaggle data), one dataset at a time, <= 2 seeds in parallel,
resumable: a seed folder with DONE is skipped; an unfinished seed resumes from its 200-patient checkpoint.

  .venv/bin/python experiment_final/run_full.py nhanes      -> results/run_full_nhanes/  (nhanes_pooled + ontology_as_features)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PY = str(REPO / ".venv" / "bin" / "python")
SEEDS = [0, 1, 2, 3, 4]
WORKERS, THREADS = int(os.environ.get("RUN_FULL_WORKERS", 2)), 2
COMMON = {"skip_baselines": ["wachter"]}          # Wachter removed from the report (user request)
PLANS = {
    "nhanes": ("run_full_nhanes", [("nhanes_pooled", ""), ("nhanes_pooled", "ontology_as_features")], {}),
}
JOB = ("import sys; sys.path.insert(0, {repo!r}); from pathlib import Path; from experiment_final.src.run_seed import run; "
       "run({ds!r}, {seed}, Path({rd!r}), variant={var!r}, max_patients=0, overrides={ov!r})")


def job(rd: Path, ds: str, var: str, seed: int, ov: dict) -> dict:
    tag = f"{ds}{'__' + var if var else ''}_s{seed}"
    out = rd / (ds + (f"__{var}" if var else "")) / f"seed{seed}"
    if (out / "DONE").exists():
        return {"job": tag, "status": "already done"}
    t0 = time.time()
    with open(rd / "logs" / f"{tag}.log", "a") as fh:
        rc = subprocess.call([PY, "-W", "ignore", "-c", JOB.format(repo=str(REPO), ds=ds, seed=seed, rd=str(rd), var=var, ov=ov)],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
                             env={"CARELG_THREADS": str(THREADS), "OMP_NUM_THREADS": str(THREADS), "PATH": "/usr/bin:/bin"})
    return {"job": tag, "returncode": rc, "seconds": round(time.time() - t0, 1)}


def main(plan: str) -> None:
    folder, jobs, ov = PLANS[plan]
    ov = {**COMMON, **ov}
    rd = ROOT / "results" / folder
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(ds, var, s) for ds, var in jobs for s in SEEDS]
    log = rd / "run_log.json"
    hist = json.loads(log.read_text()) if log.exists() else {"plan": plan, "overrides": ov, "attempts": []}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(lambda j: job(rd, j[0], j[1], j[2], ov), todo))
    hist["attempts"].append({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res})
    log.write_text(json.dumps(hist, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
