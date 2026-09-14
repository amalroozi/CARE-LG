"""
ADDITIONAL EXPERIMENT -- clearly separate from and does NOT replace the
project's main, published results. Recalibrates the pipeline's operational
high/low-risk gate to the empirically-measured clinical threshold, instead
of the original, never-derived 0.55/0.45 constants.

Background (docs/CLASSIFIER_AUDIT.md): the project's data label is built
from a real 7.5% 10-year ASCVD/PCE risk cutoff, but the pipeline's own
0.55/0.45 gate was an inherited constant, never checked against it. The
audit measured where the classifier's OWN output probability actually
crosses real-world 7.5% risk: UCI 0.245 +/- 0.040, Real NHANES 0.075 +/-
0.000 (both already-published numbers, not recomputed here).

This experiment recenters the SAME 0.1-wide buffer the original gate used
(0.55 - 0.45 = 0.10) on that empirical crossing point instead of on 0.5:
  UCI:          HIGH=0.295, LOW=0.195   (0.245 +/- 0.05)
  Real NHANES:  HIGH=0.125, LOW=0.025   (0.075 +/- 0.05)

Everything else -- the graph, the B1/B2-fixed gate, the search, the
certified-infeasibility protocol -- is EXACTLY the same code as the main
results (src/graph/query_attachment.py, src/recourse/search_augmented.py).
Only which patients count as "high-risk" (the ones evaluated) and
"low-risk" (the valid recourse targets) changes.

Writes results/recalibrated_threshold_experiment.html (self-contained,
auto-opens) plus raw data in results/tables/recalibrated_*.

Usage:
  .venv/bin/python -m scripts.run_recalibrated_threshold_experiment
"""
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table, build_query_edges, make_cell_matrix, make_query_row, augmented_matrix
from src.recourse.search_augmented import find_path_augmented
from src.graph.constraints_ext import check_decoded_violations, violates as violates_fn
from src.seeding import set_all_seeds

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["uci", "nhanes_real"]
METRIC = "riemannian"
M_SYNTHETIC = 500

# Original (published, unchanged elsewhere in the project)
ORIGINAL_THRESHOLDS = {"uci": (0.55, 0.45), "nhanes_real": (0.55, 0.45)}
# Recalibrated to the empirically-measured 7.5%-crossing point (docs/CLASSIFIER_AUDIT.md),
# same 0.10 buffer width as the original.
RECALIBRATED_THRESHOLDS = {"uci": (0.295, 0.195), "nhanes_real": (0.125, 0.025)}
EMPIRICAL_CROSSING = {"uci": "0.245 +/- 0.040 (5 seeds)", "nhanes_real": "0.075 +/- 0.000 (5 seeds, exact)"}


def try_search(run, edge_table, base_csr, pidx, low_risk_mask):
    x0 = run.X_test[pidx]
    with torch.no_grad():
        z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
    N = edge_table.N
    qedges = build_query_edges(run.vae, z0, x0, run.Z_train, run.X_train, run.feature_metadata,
                                run.scaler, run.feature_cols, k=run.k_neighbors)
    qrow = make_query_row(qedges, METRIC, "hard", N)
    aug = augmented_matrix(base_csr, qrow)
    target_mask = np.concatenate([low_risk_mask, [False]])
    try:
        path, cost = find_path_augmented(aug, N, target_mask)
    except ValueError:
        return None
    real_points = [x0 if n == N else run.X_train[n] for n in path]
    any_hop = False
    for a, b in zip(real_points[:-1], real_points[1:]):
        v, _ = check_decoded_violations(a, b, run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=True)
        any_hop = any_hop or v
    endpoint_v, _ = check_decoded_violations(real_points[0], real_points[-1], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)
    return {"success": True, "real_cvr": bool(any_hop or endpoint_v)}


def certified_infeasibility_check(run, edge_table, base_hard, high_risk_idx, low_risk_mask, seed, abstained_pidx_z0, low_thr):
    """Exhaustive check + 2xk / +500-synthetic densification probes, same protocol as scripts/run_blindspot.py."""
    if not abstained_pidx_z0:
        return []
    low_risk_idxs = np.where(low_risk_mask)[0]
    set_all_seeds(seed + 100000)
    edge_table_2k = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                      run.scaler, run.feature_cols, k=2 * run.k_neighbors)
    base_hard_2k = make_cell_matrix(edge_table_2k, METRIC, "hard")
    with torch.no_grad():
        z_synth = torch.randn(M_SYNTHETIC, run.vae.latent_dim)
        x_synth = run.vae.decode(z_synth).numpy()
        risk_synth = run.classifier(torch.tensor(x_synth, dtype=torch.float32)).numpy().squeeze()
    synth_low_risk = risk_synth < low_thr
    Z_combined = np.vstack([run.Z_train, z_synth.numpy()])
    X_combined = np.vstack([run.X_train, x_synth])
    low_risk_combined = np.concatenate([low_risk_mask, synth_low_risk])
    edge_table_synth = build_edge_table(run.vae, Z_combined, X_combined, run.feature_metadata,
                                        run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard_synth = make_cell_matrix(edge_table_synth, METRIC, "hard")

    rows = []
    for pidx, z0 in abstained_pidx_z0:
        x0 = run.X_test[pidx]
        valid_targets = [j for j in low_risk_idxs
                         if not violates_fn(x0, run.X_train[j], run.feature_metadata, run.scaler, run.feature_cols, check_step_horizon=False)]
        certified_infeasible = len(valid_targets) == 0
        resolved_2k = resolved_synth = None
        if not certified_infeasible:
            resolved_2k = try_search(run, edge_table_2k, base_hard_2k, pidx, low_risk_mask) is not None
            resolved_synth = try_search(run, edge_table_synth, base_hard_synth, pidx, low_risk_combined) is not None
        rows.append({
            "certified_infeasible": certified_infeasible,
            "confirmed_blindspot": (not certified_infeasible) and bool(resolved_2k or resolved_synth),
            "unresolved": (not certified_infeasible) and not bool(resolved_2k or resolved_synth),
        })
    return rows


def run_one(dataset, threshold_label, high_thr, low_thr, seed):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)
    base_hard = make_cell_matrix(edge_table, METRIC, "hard")

    high_risk_idx = np.where(run.test_risk > high_thr)[0]
    low_risk_mask = run.train_risk < low_thr

    rows, abstained = [], []
    for pidx in high_risk_idx:
        x0 = run.X_test[pidx]
        with torch.no_grad():
            z0 = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0].squeeze(0).numpy()
        r = try_search(run, edge_table, base_hard, pidx, low_risk_mask)
        if r is None:
            abstained.append((pidx, z0))
            rows.append({"patient_idx": int(pidx), "success": False, "real_cvr": None})
        else:
            rows.append({"patient_idx": int(pidx), "success": True, "real_cvr": r["real_cvr"]})

    cert_rows = certified_infeasibility_check(run, edge_table, base_hard, high_risk_idx, low_risk_mask, seed, abstained, low_thr)

    n = len(high_risk_idx)
    n_success = sum(r["success"] for r in rows)
    n_cvr = sum(1 for r in rows if r["success"] and r["real_cvr"])
    n_cert = sum(1 for c in cert_rows if c["certified_infeasible"])
    n_blind = sum(1 for c in cert_rows if c["confirmed_blindspot"])
    n_unres = sum(1 for c in cert_rows if c["unresolved"])

    return {
        "dataset": dataset, "threshold_label": threshold_label, "seed": seed,
        "high_thr": high_thr, "low_thr": low_thr,
        "n_high_risk": n, "n_low_risk_train": int(low_risk_mask.sum()),
        "n_success": n_success, "success_pct": 100.0 * n_success / max(1, n),
        "real_cvr_pct": 100.0 * n_cvr / max(1, n_success) if n_success else float("nan"),
        "n_abstain": len(abstained),
        "certified_infeasible_pct": 100.0 * n_cert / max(1, len(cert_rows)) if cert_rows else float("nan"),
        "confirmed_blindspot_pct": 100.0 * n_blind / max(1, len(cert_rows)) if cert_rows else float("nan"),
        "unresolved_pct": 100.0 * n_unres / max(1, len(cert_rows)) if cert_rows else float("nan"),
    }


def bar_svg(labels, values, title, width=520, height=220, color="#2563eb", fmt="{:.1f}%", ymax=100):
    n = len(labels)
    bw = (width - 60) / n
    bars = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = (v / ymax) * (height - 50) if ymax else 0
        x = 40 + i * bw
        y = height - 30 - h
        bars.append(f'<rect x="{x+8:.1f}" y="{y:.1f}" width="{bw-16:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" font-size="11" text-anchor="middle" fill="#111">{fmt.format(v)}</text>')
        bars.append(f'<text x="{x+bw/2:.1f}" y="{height-12:.1f}" font-size="10" text-anchor="middle" fill="#555">{lab}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
            f'<line x1="40" y1="{height-30}" x2="{width-20}" y2="{height-30}" stroke="#ccc"/>'
            f'{"".join(bars)}</svg>')


def build_html(df, meta):
    def cell(dataset, label, col):
        d = df[(df.dataset == dataset) & (df.threshold_label == label)]
        return d[col].mean(), d[col].std()

    sections = []
    for dataset in DATASETS:
        orig_succ, orig_succ_sd = cell(dataset, "original", "success_pct")
        new_succ, new_succ_sd = cell(dataset, "recalibrated", "success_pct")
        orig_cvr, _ = cell(dataset, "original", "real_cvr_pct")
        new_cvr, _ = cell(dataset, "recalibrated", "real_cvr_pct")
        orig_cert, _ = cell(dataset, "original", "certified_infeasible_pct")
        new_cert, _ = cell(dataset, "recalibrated", "certified_infeasible_pct")
        orig_n, _ = cell(dataset, "original", "n_high_risk")
        new_n, _ = cell(dataset, "recalibrated", "n_high_risk")

        succ_svg = bar_svg(["orig 0.55/0.45", "recalib."], [orig_succ, new_succ], "Success rate")
        cvr_svg = bar_svg(["orig 0.55/0.45", "recalib."], [orig_cvr, new_cvr], "Real-value CVR")
        cert_svg = bar_svg(["orig 0.55/0.45", "recalib."], [orig_cert, new_cert], "Certified infeasible (of abstentions)")

        d_orig = df[(df.dataset == dataset) & (df.threshold_label == "original")]
        d_new = df[(df.dataset == dataset) & (df.threshold_label == "recalibrated")]
        thr = RECALIBRATED_THRESHOLDS[dataset]
        rows_table = "".join(
            f"<tr><td>{s}</td><td>{ro.n_high_risk}</td><td>{ro.n_low_risk_train}</td><td>{ro.success_pct:.1f}%</td>"
            f"<td>{ro.real_cvr_pct:.1f}%</td><td>{ro.certified_infeasible_pct:.1f}%</td></tr>"
            for s, ro in zip(d_orig.seed, d_orig.itertuples())
        )
        rows_table_new = "".join(
            f"<tr><td>{s}</td><td>{rn.n_high_risk}</td><td>{rn.n_low_risk_train}</td><td>{rn.success_pct:.1f}%</td>"
            f"<td>{rn.real_cvr_pct:.1f}%</td><td>{rn.certified_infeasible_pct:.1f}%</td></tr>"
            for s, rn in zip(d_new.seed, d_new.itertuples())
        )

        sections.append(f"""
        <section>
          <h2>{dataset}</h2>
          <p class="note">Empirical 7.5%-crossing: {EMPIRICAL_CROSSING[dataset]} &middot; recalibrated gate: HIGH&gt;{thr[0]}, LOW&lt;{thr[1]} (vs. original HIGH&gt;0.55, LOW&lt;0.45)</p>
          <div class="charts">{succ_svg}{cvr_svg}{cert_svg}</div>
          <h3>Original (0.55/0.45) &mdash; per seed</h3>
          <table><thead><tr><th>seed</th><th>n high-risk</th><th>n low-risk targets</th><th>success</th><th>real CVR</th><th>certified infeasible</th></tr></thead>
          <tbody>{rows_table}</tbody></table>
          <h3>Recalibrated ({thr[0]}/{thr[1]}) &mdash; per seed</h3>
          <table><thead><tr><th>seed</th><th>n high-risk</th><th>n low-risk targets</th><th>success</th><th>real CVR</th><th>certified infeasible</th></tr></thead>
          <tbody>{rows_table_new}</tbody></table>
        </section>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Recalibrated Threshold Experiment</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#f7f7f9; margin:0; padding:24px; color:#111; }}
.wrap {{ max-width: 900px; margin:0 auto; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; }} h3 {{ font-size:13px; color:#555; margin-top:16px; }}
section {{ background:#fff; border:1px solid #e2e2e6; border-radius:10px; padding:18px 22px; margin-bottom:18px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:6px 0; }}
th,td {{ border:1px solid #e2e2e6; padding:4px 8px; text-align:left; }}
th {{ background:rgba(37,99,235,.08); }}
.charts {{ display:flex; flex-wrap:wrap; gap:10px; }}
.charts svg {{ flex:1; min-width:200px; }}
.note {{ font-size:12px; color:#666; }}
.badge {{ background:#fef3c7; color:#92400e; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:700; }}
</style></head><body><div class="wrap">
<h1>Recalibrated Threshold Experiment <span class="badge">ADDITIONAL &mdash; does not replace main results</span></h1>
<p>The pipeline's operational high/low-risk gate (originally 0.55/0.45, never derived from anything) recentered on the
empirically-measured point where the classifier's own probability actually crosses the real 7.5% ASCVD/PCE clinical
threshold (see docs/CLASSIFIER_AUDIT.md). Same 0.10-wide buffer, same graph, same B1/B2-fixed search code &mdash;
only which patients count as high/low risk changes. Git SHA: {meta['git_sha']} &middot; seeds: {meta['seeds']}</p>
{''.join(sections)}
<section><h2>Raw data</h2><p class="note">results/tables/recalibrated_threshold_per_seed.csv</p></section>
</div></body></html>"""
    return html


def main():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"

    all_rows = []
    for dataset in DATASETS:
        for label, thr_map in [("original", ORIGINAL_THRESHOLDS), ("recalibrated", RECALIBRATED_THRESHOLDS)]:
            high_thr, low_thr = thr_map[dataset]
            for seed in SEEDS:
                t0 = time.perf_counter()
                r = run_one(dataset, label, high_thr, low_thr, seed)
                all_rows.append(r)
                print(f"[recalib] {dataset:12s} {label:12s} seed={seed}: n_high={r['n_high_risk']:4d} "
                      f"success={r['success_pct']:5.1f}% real_cvr={r['real_cvr_pct']:5.1f}% "
                      f"certified_infeasible={r['certified_infeasible_pct']:5.1f}% ({time.perf_counter()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "recalibrated_threshold_per_seed.csv", index=False)
    meta = {"git_sha": sha, "seeds": SEEDS, "original_thresholds": ORIGINAL_THRESHOLDS,
            "recalibrated_thresholds": RECALIBRATED_THRESHOLDS, "empirical_crossing": EMPIRICAL_CROSSING,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(TABLES_DIR / "recalibrated_threshold_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    html = build_html(df, meta)
    out_path = RESULTS_DIR / "recalibrated_threshold_experiment.html"
    out_path.write_text(html)
    print(f"\n[recalib] wrote {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")
    print("[recalib] opened in default browser")


if __name__ == "__main__":
    main()
