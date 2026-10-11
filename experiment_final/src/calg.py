"""fig_calg: one patient, one story (NHANES 1999-2023, seed 2, test patient #139 = the Example-output card).

Stage `compute` reproduces the seed-2 setup with the identical code path (same as new_figures.reproduce), verifies it
against the stored run (oracle, edge count, % pruned, age-bracket thresholds, patient risk, target counts, Tier-1 path)
and caches the ~40 nearest training patients with their real values, risk, edge admissibility and the rule that blocks
each removed edge. Stage `draw` renders SVG + PDF only from that cache. Nothing is simulated.

  .venv/bin/python -m experiment_final.src.calg compute [stored_run_dir]
  .venv/bin/python -m experiment_final.src.calg draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_final.src.common import EXAMPLE_DATASET, EXAMPLE_POS, EXAMPLE_SEED, EXP_DIR, OLD_RUNS, RESULTS

DS, SEED, POS, NNEAR = EXAMPLE_DATASET, EXAMPLE_SEED, EXAMPLE_POS, 40
CACHE = EXP_DIR / "data" / "processed" / f"calg_seed{SEED}_pos{POS}.npz"
OUT = RESULTS / "figures"
DEFAULT_RUN = sorted(OLD_RUNS.glob("combined_*"))[-1] / DS / f"seed{SEED}"
UNIT = {"systolic_bp": "mmHg", "diastolic_bp": "mmHg", "cholesterol": "mg/dL", "bmi": "kg/m²", "glycemic_hba1c": "%", "age": "y"}
NAME = {"systolic_bp": "SBP", "diastolic_bp": "DBP", "cholesterol": "total cholesterol", "bmi": "BMI", "glycemic_hba1c": "HbA1c", "age": "age"}
RULE_WORDS = {("age_decrease", "age"): "age cannot decrease", ("age_horizon", "age"): "age gap too large",
              ("immutable", "sex"): "sex cannot change"}


def rule_words(kind: str, f: str) -> str:
    if (kind, f) in RULE_WORDS:
        return RULE_WORDS[(kind, f)]
    if kind == "direction":
        return f"{NAME.get(f, f)} cannot rise"
    return f"{kind}: {f}"


def compute(run_dir: Path = DEFAULT_RUN) -> None:
    import torch
    from experiment_final.src.graph import query_edges
    from experiment_final.src.knowledge_layer import hard_set
    from experiment_final.src.latent import encode
    from experiment_final.src.model import patient_threshold
    from experiment_final.src.run_seed import setup
    from experiment_final.src.tier1 import target_mask
    from experiment_final.src.common import load_config
    from src.seeding import set_all_seeds
    torch.set_num_threads(1)
    cfg = load_config(); set_all_seeds(SEED)
    S = setup(DS, SEED, "", cfg)
    su = json.loads((run_dir / "setup.json").read_text())
    sp, sch, orc, vae, b = S["split"], S["schema"], S["oracle"], S["vae"], S["b"]
    pat = pd.read_csv(run_dir / "patients.csv").set_index("test_pos")
    test_risk = orc.predict(sp.Xte)
    checks = {"oracle": (orc.name, su["oracle"]), "n_edges": (len(S["et"].edge_i), su["n_edges"]),
              "pct_pruned": (round(100 * S["et"].violates.mean(), 9), round(su["pct_edges_pruned"], 9)),
              "r0": (round(float(test_risk[POS]), 9), round(float(pat.loc[POS, "r0"]), 9))}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"reproduction mismatch, aborting: {bad}")
    card = next(c for c in map(json.loads, open(run_dir / "cards.jsonl")) if c["test_pos"] == POS)
    x0 = sp.Xte_raw[POS]; z0 = encode(vae, sp.Xte[POS:POS + 1])[0]
    Z = S["Ztr"]
    qe = query_edges(vae, z0, x0, Z, sp.Xtr_raw, sch)
    age = float(x0[sch.idx(S["age_f"])])
    tau_p = patient_threshold(age, S["bracket_edges"], S["bracket_thr"])
    tm = target_mask(S["train_risk"], tau_p, float(test_risk[POS]))
    if int((tm & ~qe.violates).sum()) != int(pat.loc[POS, "tier1_one_hop_targets"]) or int(tm.sum()) != int(pat.loc[POS, "n_targets"]):
        raise SystemExit("target counts differ from stored run")
    path = [int(j) for j in card["path"]]
    tgt = path[-1]
    near = [int(j) for j in np.argsort(qe.dist_riemannian)[:NNEAR]]
    if tgt not in near:
        near = near[:NNEAR - 1] + [tgt]
    nodes = np.array(near)
    # latent layout: PCA of the VAE latent means of the patient + these training patients (2 components)
    L = np.vstack([z0, Z[nodes]]); mu = L.mean(0)
    _, _, Vt = np.linalg.svd(L - mu, full_matrices=False)
    P = (L - mu) @ Vt[:2].T
    # patient -> neighbour edges (the gate), with the rules that block each removed one
    blk = [";".join(sorted(rule_words(k, f) for k, f in hard_set(sch, x0, sp.Xtr_raw[j:j + 1], True)[0])) for j in nodes]
    # graph edges among the shown training patients, with their pruned flag
    pos = {int(j): n for n, j in enumerate(nodes)}
    ee = [(pos[int(a)], pos[int(c)], bool(v)) for a, c, v in zip(S["et"].edge_i, S["et"].edge_j, S["et"].violates) if int(a) in pos and int(c) in pos]
    ee_rules = []
    for a, c, _ in ee:
        s = hard_set(sch, sp.Xtr_raw[nodes[a]], sp.Xtr_raw[nodes[c]:nodes[c] + 1], True)[0]
        ee_rules.append(";".join(sorted(rule_words(k, f) for k, f in s)))
    ctx = b.context
    pce = (ctx["ascvd_10yr_risk_pct"].to_numpy(float) if "ascvd_10yr_risk_pct" in ctx else b.y.astype(float))
    feats = list(sch.features)
    meta = {"checks": {k: [str(a), str(b)] for k, (a, b) in checks.items()}, "features": feats, "pid": str(b.patient_id.iloc[sp.te[POS]]),
            "target_train_idx": tgt, "target_pid": str(b.patient_id.iloc[sp.tr[tgt]]), "r0": float(test_risk[POS]),
            "target_risk": float(S["train_risk"][tgt]), "card_changes": card["card"]["changes"], "path": path,
            "tau_patient": float(tau_p), "n_edges_shown": len(ee), "target_in_nearest": bool(tgt in np.argsort(qe.dist_riemannian)[:NNEAR]),
            "blocked_rules": blk, "edge_rules": ee_rules, "edges": [[a, c, v] for a, c, v in ee]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, pce_nodes=pce[sp.tr[nodes]], pce0=float(pce[sp.te[POS]]), y_nodes=np.asarray(b.y)[sp.tr[nodes]], y0=int(np.asarray(b.y)[sp.te[POS]]), P=P, x0=x0, Xn=sp.Xtr_raw[nodes], risk=S["train_risk"][nodes], r0=float(test_risk[POS]),
             adm=~qe.violates[nodes], is_target=(nodes == tgt), dist=qe.dist_riemannian[nodes], tm=tm[nodes], nodes=nodes)
    CACHE.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("context columns:", list(ctx.columns)[:30]); print("verified:", checks, "| target in nearest-40:", meta["target_in_nearest"], "| shown edges:", len(ee), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RUN)
    else:
        from experiment_final.src.calg_draw import draw
        draw()
