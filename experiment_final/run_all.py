"""Run all phases for every dataset in config.datasets via its adapter.

  .venv/bin/python experiment_final/run_all.py run        [--workers 7] [--datasets ...] [--seeds ...]
  .venv/bin/python experiment_final/run_all.py aggregate  --run-dir experiment_final/results/runs/<stamp>
  .venv/bin/python experiment_final/run_all.py all

`run` executes P3-P12 per (dataset, seed) in separate processes (plus the NHANES ontology-as-features
variant, P13) into a fresh, timestamped results/runs/<stamp>/ (never overwritten). `aggregate` builds
the P13-P14 tables (results/tables). Figures, the report and the paper tables come from src/build_results.py + src/build_report.py (see src/README.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
sys.path.insert(0, str(REPO))

from experiment_final.src.common import RUNS, git_sha, load_config, run_stamp  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")
JOB = ("import sys; sys.path.insert(0, {repo!r}); from pathlib import Path; "
       "from experiment_final.src.run_seed import run; run({ds!r}, {seed}, Path({rd!r}), variant={var!r}, max_patients={cap})")
# Disclosed compute scoping (DECISIONS.md D-CAP): high-risk test patients evaluated per seed (0 = all).
PATIENT_CAP = {"nhanes_pooled": 1000}
VARIANT_CAP = 300


def _has_ontology(ds: str) -> bool:
    import importlib
    return bool(getattr(importlib.import_module(f"experiment_final.adapters.{ds}"), "HAS_ONTOLOGY", False))


def _job(ds: str, seed: int, run_dir: Path, variant: str = "") -> dict:
    """One (dataset, seed[, variant]) process with its own log (P0)."""
    tag = f"{ds}{'__' + variant if variant else ''}_s{seed}"
    log = run_dir / "logs" / f"{tag}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-W", "ignore", "-c", JOB.format(repo=str(REPO), ds=ds, seed=seed, rd=str(run_dir), var=variant,
                                                       cap=VARIANT_CAP if variant else PATIENT_CAP.get(ds, 0))],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), env={"CARELG_THREADS": "1", "PATH": "/usr/bin:/bin",
                                                                                      "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return {"job": tag, "returncode": rc, "seconds": round(time.time() - t0, 1)}


def stage_run(args) -> Path:
    """P0: provenance + parallel per-seed jobs into a fresh run directory."""
    cfg = load_config()
    run_dir = RUNS / run_stamp()
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    order = sorted(datasets, key=lambda d: -PATIENT_CAP.get(d, 0))
    jobs = [] if args.variants_only else [(d, s, "") for d in order for s in seeds]
    if "ontology_as_features" in cfg["ablations"]:
        jobs += [(d, s, "ontology_as_features") for d in order if _has_ontology(d) for s in seeds]
    (run_dir / "run_log.json").write_text(json.dumps({"git_sha": git_sha(), "started": run_stamp(), "config": cfg,
                                                      "jobs": [list(j) for j in jobs]}, indent=2))
    print(f"[run_all] {len(jobs)} jobs -> {run_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda j: _job(j[0], j[1], run_dir, j[2]), jobs))
    for r in res:
        print(f"[run_all] {r['job']}: rc={r['returncode']} {r['seconds']}s", flush=True)
    log = json.loads((run_dir / "run_log.json").read_text())
    log.update({"finished": run_stamp(), "results": res})
    (run_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    return run_dir


def merge_runs(run_dirs) -> Path:
    """Combine finished <dataset>/seed<k> folders of several runs into one view (symlinks; later runs win
    only for folders that contain DONE). Never copies or overwrites results (P0)."""
    view = RUNS / f"combined_{run_stamp()}"
    view.mkdir(parents=True)
    for rd in run_dirs:
        for d in sorted(Path(rd).glob("*/seed*")):
            if (d / "DONE").exists():
                tgt = view / d.parent.name / d.name
                tgt.parent.mkdir(exist_ok=True)
                if tgt.exists() or tgt.is_symlink():
                    tgt.unlink()
                tgt.symlink_to(d.resolve())
    (view / "SOURCES.json").write_text(json.dumps([str(r) for r in run_dirs], indent=2))
    return view


def stage_aggregate(run_dirs) -> None:
    """P13-P15 from one or more finished run directories."""
    from experiment_final.src.evaluate import aggregate
    rd = Path(run_dirs[0]) if len(run_dirs) == 1 else merge_runs(run_dirs)
    aggregate(rd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "aggregate", "all"])
    ap.add_argument("--run-dir", type=str, nargs="*")
    ap.add_argument("--variants-only", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    if args.stage in ("run", "all"):
        rd = stage_run(args)
        if args.stage == "all":
            stage_aggregate([rd])
    else:
        stage_aggregate(args.run_dir)


if __name__ == "__main__":
    main()
