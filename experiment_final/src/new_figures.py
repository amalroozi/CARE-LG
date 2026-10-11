"""Verified seed-2 reproduction (reproduce) and the three figures that need per-patient data the runs did not store:
Kaplan-Meier CVD-death survival (fig_km), baseline failure types (fig_failures)).
Used by build_results.py (`repro_figures`), which points RUN / CACHE / TABLES at the full NHANES 1999-2023 rerun.

reproduce() re-runs the seed-2 setup with the identical code path and aborts unless the oracle, edge count, % pruned, age-bracket
thresholds, patient risk, target count and every baseline's per-patient valid / hard-violation flag match the stored run.
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiment_final.src import baselines as B  # noqa: E402
from experiment_final.src.common import EXP_DIR, RESULTS, load_config  # noqa: E402
from experiment_final.src.visualize import (DS_LABEL, GRID, INK, INK2, METHOD_LABEL, MUTED, OURS_C, TIER_C, TIER_L,  # noqa: E402
                                            save)

from experiment_final.src.common import EXAMPLE_DATASET as DS, EXAMPLE_POS as SHOW_POS, EXAMPLE_SEED as SEED  # noqa: E402
RUN = RESULTS / "run_full_nhanes" / DS / f"seed{SEED}"                        # stored run that reproduce() must match
CACHE = EXP_DIR / "data" / "processed" / "new_figures_seed2_full.npz"
TABLES = RESULTS / "tables"
REPRO_THREADS = 2          # must equal the thread count of the run being reproduced (run_full.py uses 2)
SKIP = {"wachter"}         # baselines not run in the reproduced run
DS_NAME = "NHANES 1999-2023"  # display name used inside figure titles


def t(name):
    return pd.read_csv(TABLES / name)


# ------------------------------------------------------------------ reproduction (verified)

def reproduce():
    """Re-run seed-2 setup exactly as run_seed.run did, verify, compute what figures 1/5/6 need."""
    from experiment_final.src.graph import query_edges
    from experiment_final.src.knowledge_layer import hard_set, resolve_rule_set
    from experiment_final.src.latent import encode
    from experiment_final.src.model import patient_threshold
    from experiment_final.src.prep import mad_vector
    from experiment_final.src.run_seed import setup
    from experiment_final.src.tier1 import target_mask
    from src.baselines.face import build_face_graph
    from src.seeding import set_all_seeds
    torch.set_num_threads(REPRO_THREADS)
    cfg = load_config()
    set_all_seeds(SEED)
    S = setup(DS, SEED, "", cfg)
    su = json.loads((RUN / "setup.json").read_text())
    sp, sch, orc, vae = S["split"], S["schema"], S["oracle"], S["vae"]
    checks = {"oracle": (orc.name, su["oracle"]), "n_edges": (len(S["et"].edge_i), su["n_edges"]),
              "pct_pruned": (round(100 * S["et"].violates.mean(), 9), round(su["pct_edges_pruned"], 9)),
              "brackets": ([round(v, 9) for v in S["bracket_thr"].values()], [round(v, 9) for v in su["bracket_thresholds"].values()])}
    pat = pd.read_csv(RUN / "patients.csv").set_index("test_pos")
    test_risk = orc.predict(sp.Xte)
    checks["r0_139"] = (round(float(test_risk[SHOW_POS]), 9), round(float(pat.loc[SHOW_POS, "r0"]), 9))
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"reproduction mismatch, aborting: {bad}")
    print("reproduction verified:", checks, flush=True)
    cards = {c["test_pos"]: c for c in map(json.loads, open(RUN / "cards.jsonl"))}
    # ---- (1a) PCA of latent means; admissible (unpruned) edges
    Z = S["Ztr"]; mu = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    P2 = lambda A: (np.atleast_2d(A) - mu) @ Vt[:2].T  # noqa: E731
    keep = ~S["et"].violates
    ei, ej = S["et"].edge_i[keep], S["et"].edge_j[keep]
    # ---- (1b) showcase patient
    x0 = sp.Xte_raw[SHOW_POS]; z0 = encode(vae, sp.Xte[SHOW_POS:SHOW_POS + 1])[0]
    qe = query_edges(vae, z0, x0, Z, sp.Xtr_raw, sch)
    age = float(x0[sch.idx(S["age_f"])])
    tau_p = patient_threshold(age, S["bracket_edges"], S["bracket_thr"])
    tm = target_mask(S["train_risk"], tau_p, float(test_risk[SHOW_POS]))
    adm = ~qe.violates
    if int((tm & adm).sum()) != int(pat.loc[SHOW_POS, "tier1_one_hop_targets"]) or int(tm.sum()) != int(pat.loc[SHOW_POS, "n_targets"]):
        raise SystemExit("target counts differ from stored run")
    path = cards[SHOW_POS]["path"]
    rs = resolve_rule_set(cards[SHOW_POS]["concepts"], sch, S["k"], True, True)
    order = np.argsort(qe.dist_riemannian)
    blocked = [int(j) for j in order if qe.violates[j]][:3]
    blk = []
    for j in blocked:
        rules = sorted({rs.rules[f].rule_ids[0] for _, f in hard_set(sch, x0, sp.Xtr_raw[j:j + 1], True)[0]})
        blk.append({"node": j, "rules": rules})
    # second patient: a seed-2 Tier-2 patient with a valid synthetic target, nearest in latent space to the showcase
    t2 = [c for c in cards.values() if c["tier"] == 2 and c["target"] is not None]
    zt2 = {c["test_pos"]: encode(vae, sp.Xte[c["test_pos"]:c["test_pos"] + 1])[0] for c in t2}
    c2 = min(t2, key=lambda c: np.linalg.norm(zt2[c["test_pos"]] - z0))
    z2src = zt2[c2["test_pos"]]; z2tgt = encode(vae, sp.scale(np.asarray(c2["target"])[None, :]))[0]
    # ---- (5) model gate on linked test participants
    ctx = S["b"].context.iloc[sp.te].reset_index(drop=True)
    # ---- (6) baseline rule-level breakdown on the same sampled patients
    high = pat.index.to_numpy()
    zmu = sp.Xtr_raw.mean(0)
    bctx = {"split": sp, "features": sp.features, "legacy_meta": sp.legacy_meta, "Xtr_s": sp.Xtr, "Xtr_raw": sp.Xtr_raw, "oracle": orc,
            "grad_clf": orc.torch_module if orc.name == "mlp" else S["models"]["mlp"].torch_module, "vae": vae, "schema": sch,
            "mad_raw": mad_vector(sp.Xtr_raw), "mad_s": np.maximum(mad_vector(sp.Xtr), 1e-3), "face_graph": build_face_graph(sp.Xtr, S["kk"])[0]}
    stored = pd.read_csv(RUN / "methods.csv").set_index(["test_pos", "method"])
    rows, mism = [], 0
    for n, i in enumerate(high):
        x0i, x0s, r0 = sp.Xte_raw[i], sp.Xte[i], float(test_risk[i])
        tau_pi = patient_threshold(float(x0i[sch.idx(S["age_f"])]), S["bracket_edges"], S["bracket_thr"]); tau = min(tau_pi, r0)
        tmi = target_mask(S["train_risk"], tau_pi, r0)
        fns = {"face": lambda: B.face(bctx, x0s, tmi), "dice": lambda: B.dice(bctx, x0s, tau, SEED), "growing_spheres": lambda: B.growing_spheres(bctx, x0s, tau, SEED),
               "pace_style": lambda: B.pace_style(bctx, x0s, tau), "cchvae": lambda: B.cchvae(bctx, x0s, tau, SEED), "wachter": lambda: B.wachter(bctx, x0s, tau, SEED)}
        out = {m: f() for m, f in fns.items() if m not in SKIP}
        for m, res in out.items():
            if m in SKIP:
                continue
            X = res["X"]
            if not len(X):
                valid, viol = False, []
            else:
                p = orc.predict(sp.scale(X)); v = p < tau; valid = bool(v.any())
                Xv = X[v] if valid else X[:1]
                viol = sorted(set().union(*hard_set(sch, x0i, Xv, False)))
            st = stored.loc[(i, m)]
            hv = bool(viol) if len(X) else None
            if valid != bool(st["valid"]) or (len(X) and hv != bool(st["hard_violation"])):
                mism += 1
            rows.append({"test_pos": int(i), "method": m, "valid": valid, "violations": ";".join(f"{k}:{f}" for k, f in viol)})
        if (n + 1) % 100 == 0:
            print(f"baselines {n + 1}/{len(high)} (mismatches so far {mism})", flush=True)
    if mism:
        raise SystemExit(f"baseline reproduction differs from stored methods.csv on {mism} patient-method pairs")
    pd.DataFrame(rows).to_csv(CACHE.with_suffix(".baselines.csv"), index=False)
    np.savez(CACHE, P_train=P2(Z), risk=S["train_risk"], ei=ei, ej=ej, P0=P2(z0), adm=adm, tm=tm, path=np.array(path),
             blocked=np.array([b["node"] for b in blk]), P2src=P2(z2src), P2tgt=P2(z2tgt), test_risk=test_risk,
             mort_cvd=ctx["mort_cvd"].to_numpy(float), fu=ctx["followup_months"].to_numpy(float))
    (CACHE.with_suffix(".json")).write_text(json.dumps({"checks": {k: [str(a), str(b)] for k, (a, b) in checks.items()}, "blocked": blk,
                                                         "second_patient": c2["test_pos"], "second_target_risk": float(orc.predict(sp.scale(np.asarray(c2["target"])[None, :]))[0]),
                                                         "baseline_mismatches": mism, "n_patients": int(len(high))}, indent=2))


# ------------------------------------------------------------------ figures

def km(time, event):
    """Kaplan-Meier estimate (step function) (P14)."""
    o = np.argsort(time); time, event = time[o], event[o]
    ts, S = [0.0], [1.0]
    for u in np.unique(time):
        d = int(event[time == u].sum()); n = int((time >= u).sum())
        if d:
            ts.append(u); S.append(S[-1] * (1 - d / n))
    return np.array(ts), np.array(S)


def fig_km(store):
    c = pd.read_parquet(EXP_DIR / "data" / "processed" / "nhanes_pooled_cohort.parquet")
    c = c[c.eligstat == 1]
    tm = c.permth_exm.to_numpy(float); ev = ((c.mortstat == 1) & c.ucod_leading.isin(["001", "005"])).to_numpy(int)
    d = np.load(CACHE); ok = ~np.isnan(d["mort_cvd"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (title, groups) in zip(axes, [
        (f"PCE label, all linked participants 1999–2016 (n={len(c):,})", [("PCE ≥ 7.5% (label high-risk)", c.cvd_risk_flag.to_numpy() == 1, TIER_C[2]),
                                                                           ("PCE < 7.5%", c.cvd_risk_flag.to_numpy() == 0, TIER_C[1])]),
        (f"Oracle gate, seed-2 linked test participants (n={int(ok.sum()):,})", [("predicted risk > 0.55", ok & (d["test_risk"] > 0.55), TIER_C[2]),
                                                                                  ("predicted risk < 0.45", ok & (d["test_risk"] < 0.45), TIER_C[1])])]):
        for lab, msk, col in groups:
            if title.startswith("PCE"):
                T, E = tm[msk], ev[msk]
            else:
                T, E = d["fu"][msk], d["mort_cvd"][msk].astype(int)
            x, y = km(T, E)
            ax.step(x / 12, y, where="post", color=col, lw=2, label=f"{lab}: n={msk.sum():,}, CVD deaths={int(E.sum()):,}")
        ax.set_title(title, loc="left", fontsize=9, color=INK); ax.set_xlabel("years since exam"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        ax.set_ylim(0.8, 1.005)
    axes[0].set_ylabel("Kaplan–Meier freedom from CVD death")
    fig.suptitle("CVD-death survival by risk group (NHANES Linked Mortality Files 2019; other deaths censored)", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, "fig15_km_cvd_survival", store)


def fig_failures(store):
    b = pd.read_csv(CACHE.with_suffix(".baselines.csv"))
    b = b[b.valid & b.violations.notna() & (b.violations != "")]
    cats = lambda v: sorted({("immutable (sex)" if s.startswith("immutable") else "age decreases" if s.startswith("age_decrease")  # noqa: E731
                              else "age horizon" if s.startswith("age_horizon") else "worsens " + s.split(":")[1]) for s in v.split(";")})
    rows = []
    for m, g in b.groupby("method"):
        cnt = Counter(c for v in g.violations for c in cats(v))
        for k, n in cnt.items():
            rows.append({"method": m, "rule": k, "pct": 100 * n / len(g), "n": len(g)})
    df = pd.DataFrame(rows)
    order = df.groupby("rule").pct.sum().sort_values(ascending=False).index.tolist()
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    meths = [m for m in ["face", "growing_spheres", "dice", "cchvae", "wachter"] if m in set(df.method) and m not in SKIP]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, m in enumerate(meths):
        left = 0
        for j, r in enumerate(order):
            v = df[(df.method == m) & (df.rule == r)].pct.sum()
            ax.barh(i, v, left=left, color=cols[j % 8], height=0.6, edgecolor="white", linewidth=1.2, label=r if i == 0 else None); left += v
        ax.text(left + 2, i, f"n={int(df[df.method == m].n.iloc[0])} rule-breaking valid CFs", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(meths))); ax.set_yticklabels([METHOD_LABEL[m] for m in meths]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of rule-breaking valid counterfactuals breaking each hard rule (a CF can break several; bars sum > 100)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    meta = json.loads(CACHE.with_suffix(".json").read_text())
    fig.suptitle(f"Baseline failure types: which hard rule is broken ({DS_NAME}, seed 2, {meta['n_patients']:,} patients; PACE-style and reachable sets break none)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "fig16_baseline_failure_types", store)
