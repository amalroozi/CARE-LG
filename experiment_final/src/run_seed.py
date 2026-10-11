"""Per-(dataset, seed[, variant]) driver for phases P3-P12 (+ the per-patient parts of P13/P14).

Writes into <run_dir>/<dataset>[__variant]/seed<k>/ :
  model_metrics.json, shap_global.csv, setup.json      (P4-P6)
  patients.csv     one row per high-risk test patient, main arm (P7-P11)
  arms.csv         one row per patient x ablation arm (P13)
  methods.csv      one row per patient x method incl. baselines (P12/P14)
  cards.jsonl, explanations.jsonl, certify.jsonl        (patient-level; stay local)
  sensitivity_*.csv, robustness.csv                     (seed 0 / all seeds, P14)
Nothing after the adapter references a dataset by name.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from experiment_final import adapters
from experiment_final.src import baselines as B
from experiment_final.src.ablations import Arm, arms as make_arms, concept_feature_matrix
from experiment_final.src.certify import certify_failure, exhaustive_one_hop
from experiment_final.src.common import git_sha, load_config
from experiment_final.src.explain import explanation, trace
from experiment_final.src.graph import build_edge_table, edge_change_percentile, query_edges
from experiment_final.src.knowledge_layer import (certificate, hard_violates, indicated_classes, load_knowledge,
                                                 resolve_rule_set, schema_from_meta, soft_degrees)
from experiment_final.src.latent import encode, fit_vae
from experiment_final.src.model import (age_bracket_thresholds, evaluate_models, fit_mlp, fit_xgb, global_shap,
                                       patient_shap, patient_threshold)
from experiment_final.src.prep import mad_vector, make_split
from experiment_final.src.recommend import build_card, precedent_table
from experiment_final.src.tier1 import Tier1Arm, run_arm, target_mask
from experiment_final.src.tier2 import GenConfig, Problem, ceiling_search, describe, generate_ga, generate_ng
from src.graph.query_attachment import make_cell_matrix
from src.seeding import set_all_seeds
from src.baselines.face import build_face_graph

N_EXPLAIN = 60


def _jsonl(path: Path, rows):
    with open(path, "w") as fh:   # written once per finished seed (resume-safe)
        for r in rows:
            fh.write(json.dumps(r, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))) + "\n")


def apply_variant(bundle, variant: str, k):
    """ontology_as_features: append ICD-10-CM block indicators as immutable model inputs; drop concepts from rules (P13)."""
    if variant != "ontology_as_features":
        return bundle
    F = concept_feature_matrix(bundle.concepts, k.onto)
    bundle.X = pd.concat([bundle.X.reset_index(drop=True), F], axis=1)
    for c in F.columns:
        bundle.feature_meta[c] = dict(type="binary", **{"class": "immutable"}, direction="none", unit="0/1",
                                      phys_min=0, phys_max=1, concept="concept_flag")
    return bundle


def setup(dataset: str, seed: int, variant: str, cfg: dict):
    """P3-P6 for one seed: split, models, oracle, VAE, CALG, thresholds (P3-P6)."""
    k = load_knowledge()
    b = apply_variant(adapters.get(dataset).load(cfg), variant, k)
    schema = schema_from_meta(b.feature_meta, k)
    sp = make_split(b, seed, cfg["split"]["test_size"])
    Xfit, yfit = sp.Xtr[sp.fit], sp.ytr[sp.fit]
    models = {"mlp": fit_mlp(Xfit, yfit, seed), "xgboost": fit_xgb(Xfit, yfit, seed)}
    hi, lo = cfg["thresholds"]["high"], cfg["thresholds"]["low"]
    metrics = evaluate_models(models, sp, b, hi, lo)
    oracle = models[metrics["oracle"]]
    vae = fit_vae(sp, seed, cfg["latent"]["dim"], cfg["latent"]["epochs"], cfg["latent"]["age_loss_weight"])
    Ztr = encode(vae, sp.Xtr)
    kk = cfg["graph"]["k"][dataset]
    et = build_edge_table(vae, Ztr, sp.Xtr_raw, schema, kk)
    train_risk = oracle.predict(sp.Xtr)
    age_f = next(f for f in schema.features if schema.kind[f] == "non_decreasing")
    edges, thr = age_bracket_thresholds(sp.Xtr_raw[:, schema.idx(age_f)], train_risk)
    return dict(k=k, b=b, schema=schema, split=sp, models=models, metrics=metrics, oracle=oracle, vae=vae, Ztr=Ztr,
                et=et, train_risk=train_risk, age_f=age_f, bracket_edges=edges, bracket_thr=thr, kk=kk)


def run(dataset: str, seed: int, run_dir: Path, variant: str = "", max_patients: int = 0, sensitivity: bool = None,
        overrides: dict = None, checkpoint_every: int = 200):
    """Full per-seed pipeline; returns the output directory (P3-P14)."""
    torch.set_num_threads(int(os.environ.get("CARELG_THREADS", "1")))
    cfg = load_config()
    cfg.update(overrides or {})   # per-run overrides (e.g. cardio_n_subsample=0, skip_baselines); recorded in setup.json
    set_all_seeds(seed)
    T0 = time.perf_counter()
    S = setup(dataset, seed, variant, cfg)
    k, b, schema, sp, oracle, vae = S["k"], S["b"], S["schema"], S["split"], S["oracle"], S["vae"]
    out = Path(run_dir) / (dataset + (f"__{variant}" if variant else "")) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    hi_thr = cfg["thresholds"]["high"]
    age_strat = cfg["thresholds"]["age_stratified_targets"]
    has_onto = b.has_ontology and variant != "ontology_as_features"
    A = make_arms(cfg, has_onto) if not variant else {"main": Arm("main", Tier1Arm("main"), rules="generic")}
    data_maxchg = edge_change_percentile(S["et"], sp.Xtr_raw, schema, cfg["rules"]["soft_change_percentile"])
    # ---- shared per-seed structures
    Xtr_raw, Xte_raw = sp.Xtr_raw, sp.Xte_raw
    Xte_s, Xtr_s = sp.Xte, sp.Xtr
    test_risk = oracle.predict(Xte_s)
    high = np.where(test_risk > hi_thr)[0]
    if max_patients and len(high) > max_patients:
        # disclosed compute scoping (DECISIONS.md D-CAP): uniform random sample of high-risk test patients
        high = np.sort(np.random.default_rng(seed + 424242).choice(high, max_patients, replace=False))
    base_csr = {m: make_cell_matrix(S["et"], m, "hard") for m in ("riemannian", "euclidean")}
    t1ctx = {"X_raw": Xtr_raw, "N": len(Xtr_raw), "schema": schema, "base_csr": base_csr, "train_risk": S["train_risk"]}
    mad_raw = mad_vector(Xtr_raw)
    zmu, zsd = Xtr_raw.mean(0), np.where(Xtr_raw.std(0) > 1e-9, Xtr_raw.std(0), 1.0)
    Zs = (Xtr_raw - zmu) / zsd
    knn = NearestNeighbors(n_neighbors=5).fit(Zs)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Zs)
    grad_clf = oracle.torch_module if oracle.name == "mlp" else S["models"]["mlp"].torch_module
    concepts_te = [b.concepts[i] for i in sp.te]
    concepts_tr = [b.concepts[i] for i in sp.tr]
    meds_te = [set(b.current_meds[i]) for i in sp.te]
    ctx_te = b.context.iloc[sp.te].reset_index(drop=True)
    sub_te = {g: s.to_numpy()[sp.te] for g, s in b.subgroups.items()}

    @lru_cache(maxsize=4096)
    def rs_for(concepts: tuple, mode: str):
        if mode == "generic":
            return resolve_rule_set([], schema, k, True, False, data_maxchg)
        return resolve_rule_set(list(concepts), schema, k, mode != "no_inheritance", True, data_maxchg)

    prec = precedent_table(k, schema, lambda c: rs_for(c, "full"), Xtr_raw,
                           [dict(r) for _, r in b.context.iloc[sp.tr].iterrows()] if "ascvd_10yr_risk_pct" in b.context else [None] * len(sp.tr),
                           [set(b.current_meds[i]) for i in sp.tr], concepts_tr)
    bctx = {"split": sp, "features": sp.features, "legacy_meta": sp.legacy_meta, "Xtr_s": Xtr_s, "Xtr_raw": Xtr_raw,
            "oracle": oracle, "grad_clf": grad_clf, "vae": vae, "schema": schema, "mad_raw": mad_raw,
            "mad_s": np.maximum(mad_vector(Xtr_s), 1e-3), "face_graph": build_face_graph(Xtr_s, S["kk"])[0]}
    lin = B.fit_linear(bctx)
    risk_fn = lambda X: oracle.predict(sp.scale(X))  # noqa: E731
    setup_info = {"dataset": dataset, "variant": variant, "seed": seed, "git_sha": git_sha(), "config": cfg,
                  "n_train": len(sp.tr), "n_test": len(sp.te), "n_high_risk_test": int(len(high)),
                  "oracle": oracle.name, "k": S["kk"], "n_edges": int(len(S["et"].edge_i)),
                  "pct_edges_pruned": float(100 * S["et"].violates.mean()), "bracket_edges": S["bracket_edges"].tolist(),
                  "bracket_thresholds": S["bracket_thr"], "data_maxchg_p95": data_maxchg, "precedent": prec,
                  "linear_test_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(sp.yte, lin.predict_proba(Xte_s)[:, 1])),
                  "cohort_flow": b.cohort_flow, "features": sp.features, "has_ontology": has_onto,
                  "label_definition": b.label_definition}
    (out / "setup.json").write_text(json.dumps(setup_info, indent=2, default=str))
    (out / "model_metrics.json").write_text(json.dumps(S["metrics"], indent=2, default=float))
    global_shap(oracle, sp, seed=seed).to_csv(out / "shap_global.csv", index=False)

    P_rows, A_rows, M_rows, cards, expl, certs = [], [], [], [], [], []
    ckpt = out / "partial_checkpoint.pkl"
    done = set()
    if ckpt.exists():   # resume after a crash: setup is deterministic, so already-finished patients are skipped
        P_rows, A_rows, M_rows, cards, expl, certs = pickle.loads(ckpt.read_bytes())
        done = {r["test_pos"] for r in P_rows}
        print(f"[{dataset} s{seed}] resuming from checkpoint: {len(done)} patients already done", flush=True)
    skip_b = set(cfg.get("skip_baselines", []))
    for n_done, i in enumerate(high):
        if int(i) in done:
            continue
        x0, x0s, r0 = Xte_raw[i], Xte_s[i], float(test_risk[i])
        age = float(x0[schema.idx(S["age_f"])])
        tau_p = patient_threshold(age, S["bracket_edges"], S["bracket_thr"]) if age_strat else cfg["thresholds"]["low"]
        tau = min(tau_p, r0)
        tm = target_mask(S["train_risk"], tau_p, r0)
        qe = query_edges(vae, encode(vae, x0s[None, :])[0], x0, S["Ztr"], Xtr_raw, schema)
        ctxd = ctx_te.iloc[i].to_dict()
        rs_true = rs_for(tuple(concepts_te[i]), "full")
        base = {"dataset": dataset, "variant": variant, "seed": seed, "test_pos": int(i), "pid": str(b.patient_id.iloc[sp.te[i]]),
                "r0": r0, "tau_p": tau_p, "tau": tau, "n_targets": int(tm.sum()), "age": age,
                **{f"sub_{g}": str(v[i]) for g, v in sub_te.items()},
                "n_concepts": len(concepts_te[i]), "n_current_meds": len(meds_te[i])}
        if "ascvd_10yr_risk_pct" in ctx_te:
            base["true_pce_pct"] = float(ctxd["ascvd_10yr_risk_pct"])
        # ---------------- tier 1 arms (shared query edges)
        t1 = {}
        for an in {a.tier1.name: a.tier1 for a in A.values()}.values():
            t1[an.name] = run_arm(an, t1ctx, qe, tm, x0, r0, rs_true)
        main1 = t1["main"]
        cert, cf_rec = None, None
        if main1["tier1_success"]:
            cert = certificate(concepts_te[i], schema, k, x0, Xtr_raw[main1["path"]], True, data_maxchg)
        else:
            cf_rec = certify_failure(schema, x0, Xtr_raw, tm, qe.dist_riemannian, risk_fn, tau, cfg["tier1"]["exhaustive_two_hop_check"])
            certs.append({**base, **{kk: v for kk, v in cf_rec.items()}})
        one_hop = exhaustive_one_hop(schema, x0, Xtr_raw, tm)
        # ---------------- tier 2 (all high-risk; tiered method uses it only after a tier-1 failure)
        rng_seed = seed * 1_000_003 + int(i)

        def problem(rs, guidance=True, hard_soft=False):
            ind = indicated_classes(rs, schema, k, dict(zip(schema.features, x0)), ctxd)
            allowed = [c for c, t in ind.items() if t and c not in rs.contra_classes]
            gc = GenConfig(guidance=guidance, hard_soft=hard_soft, k=cfg["tier2"]["n_counterfactuals"], seed=seed)
            return Problem(x0, r0, tau_p, schema, rs, k, oracle, sp, knn, zmu, zsd, mad_raw, allowed, sorted(rs.contra_classes), gc), ind

        t2 = {}
        main_key = f"{A['main'].rules}|{A['main'].guidance}|{A['main'].soft_as_hard}"
        settings = {main_key: (A["main"].rules, A["main"].guidance, A["main"].soft_as_hard)}
        for a in A.values():
            if a.tier2:
                settings.setdefault(f"{a.rules}|{a.guidance}|{a.soft_as_hard}", (a.rules, a.guidance, a.soft_as_hard))
        if has_onto or not b.has_ontology:
            settings.setdefault("full|False|False", ("full", False, False))
        for key, (rules, gd, sh) in settings.items():
            rs = rs_for(tuple(concepts_te[i]), rules)
            P, ind = problem(rs, gd, sh)
            g = generate_ga(P, np.random.default_rng(rng_seed))
            t2[key] = {"P": P, "ind": ind, "rs": rs, "res": g, "desc": describe(P, g["Xm"], lof)}
        t2["main"] = t2[main_key]
        Pm = t2["main"]["P"]
        ng = generate_ng(Pm, vae, grad_clf)
        ng_desc = describe(Pm, ng["Xm"], lof)
        ceil = None
        if not main1["tier1_success"] and not t2["main"]["res"]["valid"].any():
            ceil = ceiling_search(Pm, np.random.default_rng(rng_seed + 7))
        # ---------------- tiered outcome + cards per arm
        for an, a in A.items():
            r1 = t1[a.tier1.name]
            rs_a = rs_for(tuple(concepts_te[i]), a.rules)
            ind_a = indicated_classes(rs_a, schema, k, dict(zip(schema.features, x0)), ctxd)
            if r1["tier1_success"]:
                tier, target, rt = 1, Xtr_raw[r1["endpoint"]], r1["endpoint_risk"]
            elif a.tier2:
                key = f"{a.rules}|{a.guidance}|{a.soft_as_hard}"
                g = t2[key]
                v = np.where(g["res"]["valid"])[0]
                if len(v):
                    tier, target, rt = 2, g["P"].full(g["res"]["Xm"][v[:1]])[0], float(g["res"]["p"][v[0]])
                else:
                    tier, target, rt = 3, None, None
            else:
                tier, target, rt = 3, None, None
            card = None
            if target is not None:
                card = build_card(schema, k, rs_a, x0, target, r0, risk_fn, ind_a, meds_te[i] if cfg["recommendation"]["skip_current_meds"] else set(),
                                  a.rec_mode, a.apply_filter, precedent=prec)
            contra_rec = sorted(set(card["recommended_classes"]) & rs_true.contra_classes) if card else []
            soft_t = None
            if target is not None:
                deg = soft_degrees(rs_true, schema, x0, target[None, :])
                soft_t = int(sum(int((d[0] > 0).sum()) for d in deg.values()))
            row = {**base, "arm": an, "tier": tier, "tier1_success": r1["tier1_success"], "target_risk": rt,
                   "risk_drop": (r0 - rt) if rt is not None else np.nan,
                   "reaches_global_low": bool(rt is not None and rt < cfg["thresholds"]["low"]),
                   "real_violation_endpoint": r1.get("real_violation_endpoint") if tier == 1 else None,
                   "real_violation_any_node": r1.get("real_violation_any_node") if tier == 1 else None,
                   "n_hops": r1.get("n_hops") if tier == 1 else None, "tier1_latency_s": r1["latency_s"],
                   "card_covered": bool(card and card["covered"]), "n_recs": len(card["recommendations"]) if card else 0,
                   "contraindicated_recs": len(contra_rec), "contraindicated_classes": ";".join(contra_rec),
                   "escalation_appropriate": card["escalation_appropriate"] if card else None,
                   "overtreatment": card["overtreatment"] if card else None,
                   "actionability": card["actionability"] if card else np.nan,
                   "n_unmapped": len(card["unmapped"]) if card else 0, "soft_violations_true_rules": soft_t,
                   "top_class": card["recommendations"][0]["class"] if card and card["recommendations"] else "",
                   "top_step": card["recommendations"][0]["step"] if card and card["recommendations"] else ""}
            A_rows.append(row)
            if an == "main":
                prow = dict(row)
                prow.update({"tier1_one_hop_targets": one_hop["n_admissible_targets"], "cert_ok": cert["all_hard_ok"] if cert else None,
                             "tier2_valid_any": bool(t2["main"]["res"]["valid"].any()), "tier2_n_valid": int(t2["main"]["res"]["valid"].sum()),
                             "ng_valid": bool(ng["valid"][0]), "ceiling_reachable": ceil["ceiling_reachable"] if ceil else None,
                             "ceiling_min_risk": ceil["ceiling_min_risk"] if ceil else None,
                             "certified_infeasible_real": cf_rec["certified_infeasible_real"] if cf_rec else False,
                             "blocked_only_by_horizon": cf_rec["blocked_only_by_horizon"] if cf_rec else False,
                             "two_hop_chain_exists": cf_rec.get("two_hop_chain_exists") if cf_rec else None,
                             "ceiling_category": cf_rec["ceiling_category"] if cf_rec else None,
                             "best_case_risk": cf_rec["best_case_risk"] if cf_rec else None})
                P_rows.append(prow)
                if card is None:
                    card = {"recommendations": [], "excluded": [], "changes": [],
                            "semifactual": {"best_case_risk": cf_rec["best_case_risk"] if cf_rec else r0,
                                            "blocking": (cf_rec.get("closest_miss") or {}).get("blocking", []) if cf_rec else []}}
                cards.append({**{kk: base[kk] for kk in ("dataset", "seed", "test_pos", "pid")}, "tier": tier, "card": card,
                              "path": r1.get("path"), "certificate": cert, "target": target, "x0": x0,
                              "concepts": concepts_te[i], "current_meds": sorted(meds_te[i])})
                if n_done < N_EXPLAIN:
                    sh = patient_shap(oracle, sp, x0s, seed)
                    risks = [r0, tau] + ([rt] if rt is not None else []) + ([card.get("semifactual", {}).get("best_case_risk")] if tier == 3 else [])
                    txt = explanation(tier, schema, k, rs_true, x0, target, r0, rt if rt is not None else r0, tau, sh, card, ctxd, cert)
                    fails = trace(txt, k, schema, rs_true, x0, target, [r for r in risks if r is not None], card, rs_true.contra_classes)
                    expl.append({**{kk: base[kk] for kk in ("dataset", "seed", "test_pos")}, "tier": tier, "text": txt,
                                 "trace_failures": fails, "shap": sh})
        # ---------------- method comparison (P12/P14)
        def mrow(method, X, latency, extra=None):
            X = np.atleast_2d(X) if len(X) else np.zeros((0, len(x0)))
            r = {**{kk: base[kk] for kk in ("dataset", "variant", "seed", "test_pos", "r0", "tau")}, "method": method,
                 "n_cf": len(X), "latency_s": latency, **(extra or {})}
            if len(X):
                p = risk_fn(X)
                valid = p < tau
                r.update({"valid": bool(valid.any()), "n_valid": int(valid.sum())})
                Xv = X[valid] if valid.any() else X[:1]
                hv = hard_violates(schema, x0, Xv, horizon=False)
                D = Xv - x0
                deg = soft_degrees(rs_true, schema, x0, Xv)
                soft_n = sum((d > 0).sum(1) for d in deg.values())
                card = build_card(schema, k, rs_true, x0, Xv[0], r0, risk_fn, t2["main"]["ind"], meds_te[i], "escalation", True)
                r.update({"hard_violation": bool(hv.any()), "hard_violation_rate": float(hv.mean()),
                          "prox_mad_l1": float((np.abs(D) / mad_raw).sum(1).mean()), "l0": float((np.abs(D) > 1e-9).sum(1).mean()),
                          "plaus_lof": float((-lof.score_samples((Xv - zmu) / zsd)).mean()),
                          "plaus_knn_z": float(knn.kneighbors((Xv - zmu) / zsd)[0].mean()),
                          "soft_violations": float(soft_n.mean()), "risk_after": float(p[valid].min() if valid.any() else p.min()),
                          "diversity": float(np.mean([np.abs(a - c).dot(1 / mad_raw) for ii, a in enumerate(Xv) for c in Xv[ii + 1:]])) if len(Xv) > 1 else np.nan,
                          "actionability": card["actionability"], "contraindicated_recs_if_carded": len(set(card["recommended_classes"]) & rs_true.contra_classes)})
            else:
                r.update({"valid": False, "n_valid": 0})
            return r

        main_row = next(r for r in A_rows if r["arm"] == "main" and r["test_pos"] == i)
        if main_row["tier"] == 1:
            M_rows.append(mrow("care_lg_tiered", Xtr_raw[[main1["endpoint"]]], main1["latency_s"], {"tier": 1}))
        elif main_row["tier"] == 2:
            g = t2["main"]
            M_rows.append(mrow("care_lg_tiered", g["P"].full(g["res"]["Xm"]), main1["latency_s"] + g["res"]["latency_s"], {"tier": 2}))
        else:
            M_rows.append(mrow("care_lg_tiered", np.zeros((0, 0)), main1["latency_s"] + t2["main"]["res"]["latency_s"], {"tier": 3}))
        M_rows.append(mrow("tier1_real_only", Xtr_raw[[main1["endpoint"]]] if main1["tier1_success"] else np.zeros((0, 0)), main1["latency_s"]))
        M_rows.append(mrow("tier2_guided_ga", t2["main"]["P"].full(t2["main"]["res"]["Xm"]), t2["main"]["res"]["latency_s"]))
        M_rows.append(mrow("tier2_riemannian_ng", Pm.full(ng["Xm"]), ng["latency_s"]))
        if "full|False|False" in t2:
            g = t2["full|False|False"]
            M_rows.append(mrow("tier2_unguided_ga", g["P"].full(g["res"]["Xm"]), g["res"]["latency_s"]))
        for name, fn in (("face", lambda: B.face(bctx, x0s, tm)), ("dice", lambda: B.dice(bctx, x0s, tau, seed)),
                         ("growing_spheres", lambda: B.growing_spheres(bctx, x0s, tau, seed)), ("pace_style", lambda: B.pace_style(bctx, x0s, tau)),
                         ("cchvae", lambda: B.cchvae(bctx, x0s, tau, seed)), ("wachter", lambda: B.wachter(bctx, x0s, tau, seed))):
            if name in skip_b:
                continue
            res = fn()
            M_rows.append(mrow(name, res["X"], res["latency_s"]))
        rsv = B.reachable_set_verify(bctx, lin, x0, rs_true, tau_p)
        M_rows.append(mrow("reachable_set_lr", rsv.pop("X"), rsv["latency_s"],
                           {kk: v for kk, v in rsv.items() if kk.startswith("rs_")}))
        if checkpoint_every and (n_done + 1) % checkpoint_every == 0:
            tmp = ckpt.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps((P_rows, A_rows, M_rows, cards, expl, certs)))
            tmp.replace(ckpt)   # atomic
        if (n_done + 1) % 50 == 0:
            print(f"[{dataset}{'/' + variant if variant else ''} s{seed}] {n_done + 1}/{len(high)} patients ({time.perf_counter() - T0:.0f}s)", flush=True)

    pd.DataFrame(P_rows).to_csv(out / "patients.csv", index=False)
    pd.DataFrame(A_rows).to_csv(out / "arms.csv", index=False)
    pd.DataFrame(M_rows).to_csv(out / "methods.csv", index=False)
    _jsonl(out / "cards.jsonl", cards)
    _jsonl(out / "explanations.jsonl", expl)
    _jsonl(out / "certify.jsonl", certs)
    if not variant:
        robustness(S, cfg, cards, out)
        if sensitivity if sensitivity is not None else seed == 0:
            sensitivity_runs(S, cfg, out, data_maxchg)
    (out / "DONE").write_text(f"{time.perf_counter() - T0:.1f}s\n")
    print(f"[{dataset}{'/' + variant if variant else ''} s{seed}] done: {len(high)} high-risk in {time.perf_counter() - T0:.0f}s", flush=True)
    return out


def robustness(S, cfg, cards, out: Path):
    """Robustness to retraining: re-fit the oracle family with seed+1000 on the same data; share of
    tier-1/tier-2 targets still below each patient's tau under the retrained model (P14)."""
    sp, name = S["split"], S["oracle"].name
    seed2 = S["split"].seed + 1000
    m2 = (fit_mlp if name == "mlp" else fit_xgb)(sp.Xtr[sp.fit], sp.ytr[sp.fit], seed2)
    rows = []
    taus = pd.read_csv(out / "patients.csv").set_index("test_pos")["tau"]
    for c in cards:
        if c["target"] is None:
            continue
        p2 = float(m2.predict(sp.scale(np.asarray(c["target"])[None, :]))[0])
        rows.append({"test_pos": c["test_pos"], "tier": c["tier"], "risk_retrained": p2, "tau": float(taus[c["test_pos"]]),
                     "still_valid": p2 < float(taus[c["test_pos"]])})
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)


def sensitivity_runs(S, cfg, out: Path, data_maxchg):
    """Seed-0 sensitivity: thresholds x target definition (tier-1 coverage via the exhaustive identity),
    graph k (path length), GA weights (tier-2 quality on a sub-sample) (P14)."""
    sp, schema, oracle, k = S["split"], S["schema"], S["oracle"], S["k"]
    risk_te = oracle.predict(sp.Xte)
    age_te = sp.Xte_raw[:, schema.idx(S["age_f"])]
    rows = []
    for hi, lo in ((0.5, 0.4), (0.55, 0.45), (0.6, 0.5), (0.65, 0.55)):
        for strat in (True, False):
            H = np.where(risk_te > hi)[0]
            ok = []
            for i in H:
                tau_p = patient_threshold(age_te[i], S["bracket_edges"], S["bracket_thr"]) if strat else lo
                tm = target_mask(S["train_risk"], tau_p, risk_te[i])
                ok.append(bool((tm & ~hard_violates(schema, sp.Xte_raw[i], sp.Xtr_raw, horizon=True)).any()))
            rows.append({"high": hi, "low": lo, "age_stratified": strat, "n_high": len(H),
                         "tier1_coverage_pct": 100 * float(np.mean(ok)) if ok else np.nan})
    pd.DataFrame(rows).to_csv(out / "sensitivity_thresholds.csv", index=False)
    # GA weights on up to 40 high-risk patients
    knn = NearestNeighbors(n_neighbors=5).fit((sp.Xtr_raw - sp.Xtr_raw.mean(0)) / np.where(sp.Xtr_raw.std(0) > 1e-9, sp.Xtr_raw.std(0), 1))
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit((sp.Xtr_raw - sp.Xtr_raw.mean(0)) / np.where(sp.Xtr_raw.std(0) > 1e-9, sp.Xtr_raw.std(0), 1))
    zmu, zsd = sp.Xtr_raw.mean(0), np.where(sp.Xtr_raw.std(0) > 1e-9, sp.Xtr_raw.std(0), 1)
    mad = mad_vector(sp.Xtr_raw)
    b = S["b"]
    H = np.where(risk_te > cfg["thresholds"]["high"])[0][:40]
    grid = [("w_prox", v) for v in (0.25, 1.0, 4.0)] + [("w_rule", v) for v in (0.0, 1.0, 4.0)] + [("w_plaus", v) for v in (0.0, 0.3, 1.0)]
    rows = []
    for param, val in grid:
        for i in H:
            x0 = sp.Xte_raw[i]
            tau_p = patient_threshold(age_te[i], S["bracket_edges"], S["bracket_thr"])
            rs = resolve_rule_set(b.concepts[sp.te[i]], schema, k, True, True, data_maxchg)
            ind = indicated_classes(rs, schema, k, dict(zip(schema.features, x0)), b.context.iloc[sp.te[i]].to_dict())
            allowed = [c for c, t in ind.items() if t and c not in rs.contra_classes]
            gc = replace(GenConfig(seed=0), **{param: val})
            P = Problem(x0, float(risk_te[i]), tau_p, schema, rs, k, oracle, sp, knn, zmu, zsd, mad, allowed, sorted(rs.contra_classes), gc)
            g = generate_ga(P, np.random.default_rng(int(i)))
            d = describe(P, g["Xm"], lof)
            v = [x for x in d if x["valid"]]
            rows.append({"param": param, "value": val, "test_pos": int(i), "valid": bool(v),
                         "prox_mad_l1": np.mean([x["prox_mad_l1"] for x in v]) if v else np.nan,
                         "soft_adherence": np.mean([x["soft_adherence"] for x in v]) if v else np.nan,
                         "plaus_lof": np.mean([x["plaus_lof"] for x in v]) if v else np.nan,
                         "l0": np.mean([x["l0"] for x in v]) if v else np.nan})
    pd.DataFrame(rows).to_csv(out / "sensitivity_ga_weights.csv", index=False)
    # graph k: path length / cost of the main arm for k/2, k, 2k on up to 60 patients (coverage is k-invariant under SRA)
    rows = []
    for kk in (max(5, S["kk"] // 2), S["kk"], 2 * S["kk"]):
        et = S["et"] if kk == S["kk"] else build_edge_table(S["vae"], S["Ztr"], sp.Xtr_raw, schema, kk)
        ctx = {"X_raw": sp.Xtr_raw, "N": len(sp.Xtr_raw), "schema": schema, "train_risk": S["train_risk"],
               "base_csr": {"riemannian": make_cell_matrix(et, "riemannian", "hard")}}
        for i in np.where(risk_te > cfg["thresholds"]["high"])[0][:60]:
            x0 = sp.Xte_raw[i]
            tau_p = patient_threshold(age_te[i], S["bracket_edges"], S["bracket_thr"])
            qe = query_edges(S["vae"], encode(S["vae"], sp.Xte[i][None, :])[0], x0, S["Ztr"], sp.Xtr_raw, schema)
            r = run_arm(Tier1Arm("main"), ctx, qe, target_mask(S["train_risk"], tau_p, risk_te[i]), x0, float(risk_te[i]))
            rows.append({"k": kk, "test_pos": int(i), "success": r["tier1_success"], "n_hops": r.get("n_hops"), "cost": r.get("cost"),
                         "risk_drop": r.get("risk_drop")})
    pd.DataFrame(rows).to_csv(out / "sensitivity_k.csv", index=False)
