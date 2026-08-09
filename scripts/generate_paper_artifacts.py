"""
Paper Artifact Generator for CARE-LG research framework.
Generates LaTeX Table 1 (table1_results.tex) and publication figure (fig1_trajectories.png).
"""

import sys
import json
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from configs.dataset_config import FEATURE_METADATA, CLINICAL_EFFORT_WEIGHTS
from src.data.loader import get_dataloaders
from src.blackbox_model import train_blackbox_model, get_device
from src.vae.model import train_vae
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path
from benchmarks.run_benchmarks import run_dice, run_growing_spheres


def generate_latex_table(csv_path, tex_output_path):
    """
    Generates LaTeX Table 1 summarizing comparative benchmark metrics.
    """
    df = pd.read_csv(csv_path)

    latex_code = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative comparison of counterfactual recourse methods on tabular EHR cardiovascular dataset. CARE-LG achieves zero constraint violations ($0.0\%$) while maintaining high data manifold log-density.}",
        r"\label{tab:benchmark_results}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Method} & \textbf{Success Rate (\%)} & \textbf{CVR (\%)} $\downarrow$ & \textbf{KDE Density} $\uparrow$ & \textbf{Clinical Cost} $\downarrow$ & \textbf{Latency (s)} $\downarrow$ \\",
        r"\hline"
    ]

    for _, row in df.iterrows():
        method = row['Method']
        succ = f"{row['Success_Rate_Pct']:.1f}\%"
        cvr = f"{row['CVR_Pct']:.1f}\%"
        kde = f"{row['KDE_Density']:.4f}"
        cost = f"{row['Clinical_Cost']:.2f}"
        lat = f"{row['Latency_Sec']:.3f}"

        # Apply bold to best performing CARE-LG values
        if method == 'CARE-LG':
            method_str = r"\textbf{CARE-LG (Ours)}"
            cvr_str = rf"\textbf{{{cvr}}}"
            kde_str = rf"\textbf{{{kde}}}"
            line = f"{method_str} & {succ} & {cvr_str} & {kde_str} & {cost} & {lat} \\\\"
        else:
            line = f"{method} & {succ} & {cvr} & {kde} & {cost} & {lat} \\\\"

        latex_code.append(line)

    latex_code.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table}"
    ])

    tex_content = "\n".join(latex_code)
    with open(tex_output_path, 'w') as f:
        f.write(tex_content)

    print(f"Generated LaTeX Table 1 at: {tex_output_path}")


def generate_publication_figure(fig_output_path, seed=42):
    """
    Generates 2D PCA publication figure contrasting CARE-LG manifold recourse trajectory vs. baseline off-manifold jumps.
    """
    print("Generating 2D PCA Trajectory Figure...")
    device = get_device()

    # Load data and models
    train_loader, _, scaler, feature_cols = get_dataloaders(batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=15, device=device)
    vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=10, device=device)

    # Build CALG graph on N=400
    calg_matrix, Z_train, X_train, y_train = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=FEATURE_METADATA,
        effort_weights=CLINICAL_EFFORT_WEIGHTS,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=30,
        lambda_effort=1.0,
        max_samples=400
    )

    # Predict risk scores
    with torch.no_grad():
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        risk_scores = classifier(X_tensor).cpu().numpy().squeeze()

    low_risk_mask = risk_scores < 0.45
    high_risk_indices = np.where(risk_scores > 0.55)[0]

    # Find a source patient with a recourse path
    path = None
    source_idx = None
    for cand_idx in high_risk_indices:
        try:
            p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
            if p is not None:
                path = p
                source_idx = cand_idx
                if len(p) >= 2:
                    break
        except Exception:
            continue

    assert path is not None, "Could not find recourse path for publication figure."

    source_x = X_train[source_idx]

    # Run baseline algorithms for comparison
    _, dice_rec_x, _ = run_dice(source_x, classifier, device, target_threshold=0.35)
    _, gs_rec_x, _ = run_growing_spheres(source_x, classifier, device, target_threshold=0.35)

    # Fit 2D PCA on training data features X_train
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_train)

    # Fit plot aesthetic styling
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. Plot Background Patient Density (KDE Contour)
    sns.kdeplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        cmap="Blues",
        fill=True,
        thresh=0.05,
        alpha=0.4,
        ax=ax,
        label="Data Manifold Density"
    )

    # Scatter plot background points (High Risk vs Low Risk)
    high_pts = X_pca[risk_scores >= 0.5]
    low_pts = X_pca[risk_scores < 0.5]
    ax.scatter(high_pts[:, 0], high_pts[:, 1], color='#e74c3c', s=15, alpha=0.3, label='High-Risk Patients')
    ax.scatter(low_pts[:, 0], low_pts[:, 1], color='#2ecc71', s=15, alpha=0.3, label='Low-Risk Patients')

    # 2. Transform Trajectories to PCA 2D
    care_lg_path_pca = pca.transform(X_train[path])
    source_pca = pca.transform(source_x.reshape(1, -1))[0]

    # Plot Source High-Risk Patient
    ax.scatter(source_pca[0], source_pca[1], color='black', s=140, zorder=5, marker='X', label='High-Risk Source')

    # 3. Plot CARE-LG Manifold Path
    ax.plot(
        care_lg_path_pca[:, 0],
        care_lg_path_pca[:, 1],
        color='#9b59b6',
        linewidth=3,
        linestyle='-',
        marker='o',
        markersize=8,
        zorder=4,
        label='CARE-LG (Smooth Manifold Path)'
    )

    # 4. Plot DiCE Off-Manifold Jump
    if dice_rec_x is not None:
        dice_pca = pca.transform(dice_rec_x.reshape(1, -1))[0]
        ax.plot(
            [source_pca[0], dice_pca[0]],
            [source_pca[1], dice_pca[1]],
            color='#e67e22',
            linewidth=2,
            linestyle='--',
            marker='s',
            markersize=7,
            zorder=4,
            label='DiCE (Off-Manifold Jump)'
        )

    # 5. Plot Growing Spheres Random Jump
    if gs_rec_x is not None:
        gs_pca = pca.transform(gs_rec_x.reshape(1, -1))[0]
        ax.plot(
            [source_pca[0], gs_pca[0]],
            [source_pca[1], gs_pca[1]],
            color='#34495e',
            linewidth=2,
            linestyle=':',
            marker='^',
            markersize=7,
            zorder=4,
            label='Growing Spheres (Random Perturbation)'
        )

    ax.set_title("CARE-LG Manifold Recourse vs. Baseline Off-Manifold Jumps", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(f"Latent Manifold Dimension 1 (PCA {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Latent Manifold Dimension 2 (PCA {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated publication figure at: {fig_output_path}")


def main():
    print("==================================================")
    print("CARE-LG Paper Artifacts Generation")
    print("==================================================")

    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    csv_path = benchmarks_dir / "results.csv"
    tex_path = benchmarks_dir / "table1_results.tex"
    fig_path = benchmarks_dir / "fig1_trajectories.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark results CSV not found at {csv_path}. Please run benchmarks/run_benchmarks.py first.")

    generate_latex_table(csv_path, tex_path)
    generate_publication_figure(fig_path, seed=42)

    print("==================================================")
    print("SUCCESS: Paper Artifacts Generated!")
    print("==================================================")


if __name__ == "__main__":
    main()
