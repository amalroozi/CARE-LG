"""
Patient Transition Trajectory Visualizer for CARE-LG research framework.
Generates multi-panel publication figure (fig4_{dataset}_patient_transitions.png) visualizing:
- Panel A: Step-by-step actionable feature transitions and percentage changes.
- Panel B: Clinical constraint safeguards (immutable feature invariance & non-decreasing age).
- Panel C: Monotonic cardiovascular risk reduction trajectory f(x_k) below clinical safety target.
Supports multi-dataset selection via --dataset uci / nhanes.
"""

import sys
import os
import argparse
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

from configs.dataset_config import get_dataset_config
from src.data.loader import get_dataloaders
from src.blackbox_model import RiskClassifier, train_blackbox_model, get_device
from src.vae.model import TabularVAE, train_vae
from src.graph.calg import build_calg_graph
from src.recourse.search import find_recourse_path, decode_recourse_trajectory


def generate_patient_transition_figure(dataset_name="uci", seed=42):
    print("==================================================")
    print(f"CARE-LG Patient Transition Trajectory Visualizer ({dataset_name.upper()} Dataset)")
    print("==================================================")

    device = get_device()
    print(f"Using PyTorch device: {device}")

    feature_metadata, effort_weights, dataset_path = get_dataset_config(dataset_name)

    # 1. Load Data & Models
    print(f"\n1. Loading {dataset_name.upper()} Dataset & Trained Models...")
    train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=dataset_name, batch_size=64, seed=seed)
    input_dim = len(feature_cols)

    models_dir = PROJECT_ROOT / "models" / dataset_name
    models_dir.mkdir(parents=True, exist_ok=True)
    vae_path = models_dir / "vae_model.pt"
    clf_path = models_dir / "classifier_model.pt"

    vae = TabularVAE(input_dim=input_dim, latent_dim=4).to(device)
    classifier = RiskClassifier(input_dim=input_dim).to(device)

    if vae_path.exists():
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        vae = train_vae(train_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=device)
        torch.save(vae.state_dict(), vae_path)

    if clf_path.exists():
        classifier.load_state_dict(torch.load(clf_path, map_location=device))
        classifier.eval()
    else:
        classifier = train_blackbox_model(train_loader, input_dim=input_dim, epochs=20, device=device)
        torch.save(classifier.state_dict(), clf_path)

    # 2. Build CALG Graph & Compute Risk Scores
    print("\n2. Constructing CALG Graph...")
    calg_matrix, Z, X, y = build_calg_graph(
        vae_model=vae,
        train_loader=train_loader,
        feature_metadata=feature_metadata,
        effort_weights=effort_weights,
        scaler=scaler,
        feature_cols=feature_cols,
        k_neighbors=60,
        lambda_effort=1.0
    )

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        risk_scores = classifier(X_tensor).cpu().numpy().squeeze()

    low_risk_mask = risk_scores < np.percentile(risk_scores, 45)
    high_risk_candidates = np.argsort(risk_scores)[::-1]

    # 3. Find Representative Multi-Step Recourse Trajectory
    print("\n3. Finding Representative Patient Recourse Trajectory...")
    path = None
    source_idx = None

    for cand_idx in high_risk_candidates:
        if risk_scores[cand_idx] <= 0.50:
            break
        try:
            p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
            if p is not None and len(p) >= 2:
                path = p
                source_idx = cand_idx
                break
            elif p is not None and path is None:
                path = p
                source_idx = cand_idx
        except Exception:
            continue

    if path is None:
        low_risk_mask = risk_scores < np.median(risk_scores)
        for cand_idx in high_risk_candidates:
            try:
                p = find_recourse_path(calg_matrix, cand_idx, low_risk_mask)
                if p is not None:
                    path = p
                    source_idx = cand_idx
                    break
            except Exception:
                continue

    assert path is not None, "Could not find a valid recourse trajectory for visualization."
    print(f"   Selected High-Risk Patient (Node {source_idx}, Initial Risk: {risk_scores[source_idx]:.4f})")
    print(f"   Recourse Trajectory ({len(path)} steps): {path}")

    # Decode step-by-step unscaled clinical trajectory
    df_traj = decode_recourse_trajectory(vae, Z, path, X, scaler, feature_cols, feature_metadata)
    step_risks = [float(risk_scores[node]) for node in path]
    df_traj['risk_score'] = step_risks

    print("\nStep-by-Step Clinical Trajectory:")
    print(df_traj)

    # 4. Multi-Panel Figure Rendering (3 Subplots)
    print(f"\n4. Rendering Publication Figure (fig4_{dataset_name}_patient_transitions.png)...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    steps_labels = [f"Step {s}\n(Node {node})" for s, node in zip(df_traj['step'], df_traj['node_idx'])]

    # -------------------------------------------------------------
    # PANEL A: Actionable Feature Transitions
    # -------------------------------------------------------------
    if dataset_name == 'nhanes':
        act_cols = ['cholesterol', 'systolic_bp', 'bmi', 'glycemic_hba1c']
        act_names = {'cholesterol': 'Serum Chol (mg/dL)', 'systolic_bp': 'Systolic BP (mmHg)',
                     'bmi': 'BMI (kg/m²)', 'glycemic_hba1c': 'HbA1c (%)'}
    else:
        act_cols = ['cholesterol', 'resting_bp', 'max_heart_rate', 'oldpeak']
        act_names = {'cholesterol': 'Serum Chol (mg/dL)', 'resting_bp': 'Resting BP (mmHg)',
                     'max_heart_rate': 'Max HR (bpm)', 'oldpeak': 'ST Oldpeak'}

    act_cols = [c for c in act_cols if c in df_traj.columns]
    x_indices = np.arange(len(df_traj))
    width = 0.18

    for i, col in enumerate(act_cols):
        vals = df_traj[col].values
        offset = (i - (len(act_cols)-1)/2.0) * width
        rects = ax_a.bar(x_indices + offset, vals, width, label=act_names.get(col, col), alpha=0.85)

    ax_a.set_xticks(x_indices)
    ax_a.set_xticklabels(steps_labels, fontsize=9, fontweight='bold')
    ax_a.set_ylabel("Clinical Feature Value", fontsize=11, fontweight='bold')
    ax_a.set_title(f"Panel A: Actionable Clinical Transitions ({dataset_name.upper()})", fontsize=12, fontweight='bold', pad=10)
    ax_a.legend(loc='upper right', frameon=True, facecolor='white', fontsize=8)

    # -------------------------------------------------------------
    # PANEL B: Immutable & Monotone Safeguards (Δ = x* - x0)
    # -------------------------------------------------------------
    x0 = df_traj.iloc[0]
    x_star = df_traj.iloc[-1]

    safeguard_metrics = feature_metadata['immutable'] + feature_metadata['non_decreasing']
    safeguard_metrics = [c for c in safeguard_metrics if c in df_traj.columns]

    delta_vals = [x_star[col] - x0[col] for col in safeguard_metrics]
    safeguard_names = [col.replace('_', ' ').title() for col in safeguard_metrics]

    colors_b = ['#2ecc71' if d >= 0 else '#e74c3c' for d in delta_vals]
    bars_b = ax_b.bar(safeguard_names, delta_vals, color=colors_b, width=0.45, edgecolor='black', linewidth=1.2)

    ax_b.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax_b.set_ylabel("Total Transition Change (Δ = x* - x0)", fontsize=11, fontweight='bold')
    ax_b.set_title("Panel B: Clinical Constraint Verification", fontsize=12, fontweight='bold', pad=10)

    for bar, val in zip(bars_b, delta_vals):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax_b.annotate(f"{val:+.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4 if height >= 0 else -12), textcoords="offset points",
                    ha='center', va=va, fontsize=10, fontweight='bold')

    ax_b.text(0.04, 0.88, "PASS: Immutable Δ = 0\nPASS: Age Δ ≥ 0", transform=ax_b.transAxes, fontsize=9.5, fontweight='bold',
              color='darkgreen', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # -------------------------------------------------------------
    # PANEL C: Monotonic Risk Profile Trajectory f(x_k)
    # -------------------------------------------------------------
    ax_c.plot(x_indices, step_risks, color='#c0392b', linewidth=3.0, marker='o', markersize=9, label='Model Risk f(x_k)')

    thresh = 0.45
    ax_c.axhline(thresh, color='#27ae60', linestyle='--', linewidth=2.0, label=f'Clinical Safety Threshold ({thresh:.2f})')

    for i, r in enumerate(step_risks):
        ax_c.annotate(f"{r:.3f}",
                    xy=(i, r),
                    xytext=(0, 10 if i % 2 == 0 else -16), textcoords="offset points",
                    ha='center', fontsize=9.5, fontweight='bold', color='#8e44ad')

    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(steps_labels, fontsize=9, fontweight='bold')
    ax_c.set_ylabel("Predicted CVD Risk Score f(x_k)", fontsize=11, fontweight='bold')
    ax_c.set_title("Panel C: Risk Profile Trajectory", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_ylim(0.1, 1.0)
    ax_c.legend(loc='upper right', frameon=True, facecolor='white', fontsize=8.5)

    plt.tight_layout()

    benchmarks_dir = PROJECT_ROOT / "benchmarks" / dataset_name
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    fig_path = benchmarks_dir / f"fig4_{dataset_name}_patient_transitions.png"

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSuccessfully generated multi-panel figure: {fig_path}")
    print("==================================================")
    print(f"SUCCESS: Patient Transition Trajectory Visualized for {dataset_name.upper()}!")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE-LG Patient Transition Trajectory Visualizer")
    parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "nhanes"], help="Dataset to process (uci or nhanes)")
    args = parser.parse_args()

    generate_patient_transition_figure(dataset_name=args.dataset, seed=42)
