"""
Fig. 13: a whole-graph overview of the Clinical Adaptive Latent Graph (CALG)
-- distinct from Fig. 12 (one query's local neighborhood). Real patient
nodes (a random sample, since the full graph is too dense to read), laid
out by PCA of their actual latent codes, colored by real predicted risk,
with real admissible k-NN edges drawn between sampled nodes. All data from
an actual trained (dataset, seed) run -- no illustrative/fake layout.

Usage:
  .venv/bin/python -m scripts.build_fig13_calg_graph
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA

import src.data.dataset_registration  # noqa: F401
from src.pipeline import run_dataset_seed
from src.graph.query_attachment import build_edge_table

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
_SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'

N_SAMPLE = 70
RNG_SEED = 0


def main():
    run = run_dataset_seed("uci", 0)
    edge_table = build_edge_table(run.vae, run.Z_train, run.X_train, run.feature_metadata,
                                  run.scaler, run.feature_cols, k=run.k_neighbors)

    rng = np.random.default_rng(RNG_SEED)
    N = len(run.Z_train)
    sample = rng.choice(N, size=min(N_SAMPLE, N), replace=False)
    sample_set = set(sample.tolist())

    pca = PCA(n_components=2, random_state=0)
    coords_all = pca.fit_transform(run.Z_train)
    coords = coords_all[sample]

    width, height = 640, 560
    margin = 60
    xs, ys = coords[:, 0], coords[:, 1]
    x_lo, x_hi = xs.min(), xs.max()
    y_lo, y_hi = ys.min(), ys.max()
    def sx(v): return margin + (v - x_lo) / (x_hi - x_lo + 1e-9) * (width - 2 * margin)
    def sy(v): return margin + (v - y_lo) / (y_hi - y_lo + 1e-9) * (height - 2 * margin - 60)

    pos = {int(node): (sx(coords[i, 0]), sy(coords[i, 1])) for i, node in enumerate(sample)}

    # admissible edges between sampled nodes only (real edge_table data)
    edge_lines = []
    mask = np.isin(edge_table.edge_i, sample) & np.isin(edge_table.edge_j, sample) & (~edge_table.violates)
    ei, ej = edge_table.edge_i[mask], edge_table.edge_j[mask]
    seen = set()
    for a, b in zip(ei.tolist(), ej.tolist()):
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        xa, ya = pos[a]
        xb, yb = pos[b]
        edge_lines.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="#9ca3af" stroke-width="0.8" opacity="0.55"/>')

    # nodes, colored by real predicted risk
    node_circles = []
    for node in sample.tolist():
        x, y = pos[node]
        risk = float(run.train_risk[node])
        if risk > 0.55:
            color = "#dc2626"
        elif risk < 0.45:
            color = "#16a34a"
        else:
            color = "#9ca3af"
        node_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="white" stroke-width="1"/>')

    legend_y = height - 45
    legend = [
        f'<circle cx="20" cy="{legend_y}" r="5.5" fill="#dc2626"/><text x="32" y="{legend_y+4}" font-size="10.5" fill="#333">High risk (&gt;0.55)</text>',
        f'<circle cx="190" cy="{legend_y}" r="5.5" fill="#9ca3af"/><text x="202" y="{legend_y+4}" font-size="10.5" fill="#333">Borderline</text>',
        f'<circle cx="330" cy="{legend_y}" r="5.5" fill="#16a34a"/><text x="342" y="{legend_y+4}" font-size="10.5" fill="#333">Low risk (&lt;0.45)</text>',
        f'<line x1="470" y1="{legend_y}" x2="500" y2="{legend_y}" stroke="#9ca3af" stroke-width="1.2"/><text x="506" y="{legend_y+4}" font-size="10.5" fill="#333">Admissible edge</text>',
    ]

    svg = (f'<svg {_SVG_NS} viewBox="0 0 {width} {height}" width="100%" height="{height}">'
           f'<rect width="{width}" height="{height}" fill="white"/>'
           f'<text x="{width/2}" y="24" font-size="13" text-anchor="middle" fill="#111" font-weight="700">'
           f'CALG structure (UCI, seed 0) &#8212; {len(sample)}-node sample, PCA projection of latent codes</text>'
           f'{"".join(edge_lines)}{"".join(node_circles)}{"".join(legend)}</svg>')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "fig13_calg_graph.svg"
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n{svg}')
    print(f"[fig13] wrote {path} ({len(sample)} nodes, {len(edge_lines)} admissible edges drawn)")


if __name__ == "__main__":
    main()
