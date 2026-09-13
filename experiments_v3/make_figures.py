"""
Phase 6: figures A-E, written as PNG (300dpi) + vector PDF into
experiments_v3/figures/. matplotlib only, no seaborn.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

import experiments_v3.lib.dataset_registration  # noqa: F401 -- registers 'nhanes_real'
from experiments_v3.lib.pipeline_v3 import run_dataset_seed

RESULTS_DIR = REPO_ROOT / "experiments_v3" / "results"
FIG_DIR = REPO_ROOT / "experiments_v3" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"riemannian": "#9b59b6", "euclidean": "#e67e22"}

DISPLAY_NAME = {"uci": "UCI", "nhanes_real": "Real NHANES"}


def disp(dataset: str) -> str:
    return DISPLAY_NAME.get(dataset, dataset.upper())


def savefig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {name}.png / {name}.pdf")


def fig_a_ablation_bars():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    cells = ["riemannian_hard", "riemannian_soft_lam1.0", "euclidean_hard", "euclidean_soft_lam1.0"]
    labels = ["R+hard", "R+soft(λ=1)", "E+hard", "E+soft(λ=1)"]
    for col, dataset in enumerate(["uci", "nhanes_real"]):
        df = pd.read_csv(RESULTS_DIR / f"seedcell_{dataset}.csv")
        kde_m, kde_s, eff_m, eff_s = [], [], [], []
        for c in cells:
            g = df[df["cell"] == c]
            kde_m.append(g["kde_mean"].mean()); kde_s.append(g["kde_mean"].std())
            eff_m.append(g["effort_mean"].mean()); eff_s.append(g["effort_mean"].std())

        ax = axes[0, col]
        bar_colors = ["#9b59b6", "#c9a0dc", "#e67e22", "#f0b27a"]
        ax.bar(labels, kde_m, yerr=kde_s, color=bar_colors, edgecolor="black", capsize=4)
        ax.set_title(f"{disp(dataset)}: KDE log-density (↑ better)", fontweight="bold")
        ax.set_ylabel("Mean trajectory KDE log-density")
        ax.tick_params(axis="x", rotation=20)

        ax = axes[1, col]
        ax.bar(labels, eff_m, yerr=eff_s, color=bar_colors, edgecolor="black", capsize=4)
        ax.set_title(f"{disp(dataset)}: Clinical effort (↓ better)", fontweight="bold")
        ax.set_ylabel("Weighted L1 effort (decoded endpoints)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Fig A — Ablation: KDE plausibility and clinical effort across the four main cells\n"
                  "(mean ± std over 5 seeds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "figA_ablation_bars")


def fig_b_abstention_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dataset in zip(axes, ["uci", "nhanes_real"]):
        path = RESULTS_DIR / f"{dataset}_blindspot_seed_summary.csv"
        df = pd.read_csv(path)
        if df["n_abstain"].sum() == 0:
            ax.text(0.5, 0.5, f"{disp(dataset)}:\nzero R+hard abstentions\nacross all 5 seeds\n(nothing to decompose)",
                    ha="center", va="center", fontsize=11, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{disp(dataset)} abstention decomposition", fontweight="bold")
            continue
        means = [df["pct_certified_infeasible"].mean(), df["pct_confirmed_blindspot"].mean(), df["pct_unresolved"].mean()]
        stds = [df["pct_certified_infeasible"].std(), df["pct_confirmed_blindspot"].std(), df["pct_unresolved"].std()]
        cats = ["Certified\ninfeasible", "Confirmed\nblindspot\n(densification-resolved)", "Unresolved /\nunknown"]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        bottom = 0
        x = [0]
        for m, s, c, col in zip(means, stds, cats, colors):
            ax.bar(x, [m], bottom=[bottom], color=col, edgecolor="black", label=c, yerr=[s] if m > 0 else None, capsize=4, width=0.5)
            if m > 1:
                ax.text(0, bottom + m / 2, f"{m:.0f}%", ha="center", va="center", fontweight="bold")
            bottom += m
        ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_ylabel("% of R+hard abstentions")
        ax.set_title(f"{disp(dataset)} (mean n={df['n_abstain'].mean():.1f} abstentions/seed)", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8)
    fig.suptitle("Fig B — R+hard abstention decomposition (mean ± std over 5 seeds)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "figB_abstention_decomposition")


def fig_c_path_geometry(dataset="uci", seed=0, n_examples=3):
    run = run_dataset_seed(dataset, seed, device=torch.device("cpu"))
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(run.Z_train)

    df = pd.read_csv(RESULTS_DIR / f"{dataset}_per_patient.csv")
    df_seed = df[df["seed"] == seed]

    candidates = []
    for pidx in run.high_risk_test_idx:
        rh = df_seed[(df_seed["cell"] == "riemannian_hard") & (df_seed["patient_idx"] == pidx) & (df_seed["success"] == True)]
        eh = df_seed[(df_seed["cell"] == "euclidean_hard") & (df_seed["patient_idx"] == pidx) & (df_seed["success"] == True)]
        if len(rh) and len(eh):
            candidates.append((pidx, json.loads(rh.iloc[0]["path"]), json.loads(eh.iloc[0]["path"])))
        if len(candidates) >= n_examples:
            break

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=["#2ecc71" if lr else "#e74c3c" for lr in run.low_risk_mask],
               s=18, alpha=0.35, label=None)
    ax.scatter([], [], c="#2ecc71", label="Low-risk train patient")
    ax.scatter([], [], c="#e74c3c", label="Not-yet-low-risk train patient")

    N = len(run.Z_train)
    for i, (pidx, path_r, path_e) in enumerate(candidates):
        x0 = run.X_test[pidx]
        with torch.no_grad():
            mu, _ = run.vae.encode(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))
        z0 = mu.squeeze(0).numpy()
        z0_pca = pca.transform(z0.reshape(1, -1))[0]

        def path_to_pca(path):
            pts = [z0_pca]
            for node in path[1:]:
                pts.append(Z_pca[node])
            return np.array(pts)

        pr = path_to_pca(path_r)
        pe = path_to_pca(path_e)
        ax.plot(pr[:, 0], pr[:, 1], "-o", color=COLORS["riemannian"], linewidth=2.5, markersize=6,
                label="R+hard path" if i == 0 else None, zorder=4)
        ax.plot(pe[:, 0], pe[:, 1], "--s", color=COLORS["euclidean"], linewidth=2.5, markersize=6,
                label="E+hard path" if i == 0 else None, zorder=4)
        ax.scatter(*z0_pca, color="black", marker="X", s=140, zorder=5, label="High-risk source" if i == 0 else None)

    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Fig C — R+hard vs E+hard path geometry, {disp(dataset)} seed {seed}\n"
                 f"({len(candidates)} example patients where both metrics succeeded)", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    savefig(fig, "figC_path_geometry")
    return len(candidates)


def fig_d_soft_violation_curve():
    fig, ax = plt.subplots(figsize=(7, 5))
    lambdas = [0.1, 1.0, 10.0, 100.0]
    for dataset, color in [("uci", "#3498db"), ("nhanes_real", "#e74c3c")]:
        df = pd.read_csv(RESULTS_DIR / f"seedcell_{dataset}.csv")
        means, stds = [], []
        for lam in lambdas:
            g = df[df["cell"] == f"riemannian_soft_lam{lam}"]
            means.append(g["decoded_cvr_endpoint_rate"].mean())
            stds.append(g["decoded_cvr_endpoint_rate"].std())
        # also plot the hard-mode reference as a horizontal line
        gh = df[df["cell"] == "riemannian_hard"]
        hard_mean = gh["decoded_cvr_endpoint_rate"].mean()
        ax.errorbar(lambdas, means, yerr=stds, marker="o", color=color, label=f"{disp(dataset)} R+soft(λ)", capsize=4)
        ax.axhline(hard_mean, color=color, linestyle=":", alpha=0.7, label=f"{disp(dataset)} R+hard (reference)")
    ax.set_xscale("log")
    ax.set_xlabel("λ (soft-penalty weight, log scale)")
    ax.set_ylabel("Decoded-space CVR, source→terminal (%)")
    ax.set_title("Fig D — Does the soft penalty converge to hard pruning's guarantee?\n(it does not: decoded-space CVR stays high at every λ)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "figD_soft_violation_curve")


def fig_e_seed_variability():
    fig, ax = plt.subplots(figsize=(6, 5))
    df = pd.read_csv(RESULTS_DIR / "seedcell_uci.csv")
    g = df[df["cell"] == "riemannian_hard"].sort_values("seed")
    ax.plot(g["seed"], g["success_rate"], "o-", color="#9b59b6", markersize=10, linewidth=2)
    mean = g["success_rate"].mean()
    ax.axhline(mean, color="gray", linestyle="--", label=f"mean = {mean:.1f}%")
    ax.axhline(47.62, color="red", linestyle=":", label="original paper point estimate (47.6%, seed=42)")
    for _, row in g.iterrows():
        ax.annotate(f"{row['success_rate']:.1f}%", (row["seed"], row["success_rate"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel("R+hard success rate (%)")
    ax.set_xticks(g["seed"])
    ax.set_title("Fig E — UCI R+hard success rate varies substantially by seed\n"
                 "(unseeded VAE/classifier training, REPO_MAP.md D4)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "figE_seed_variability")


def main():
    fig_a_ablation_bars()
    fig_b_abstention_decomposition()
    n = fig_c_path_geometry("uci", seed=0, n_examples=3)
    print(f"[figures] Fig C used {n} example patients")
    fig_d_soft_violation_curve()
    fig_e_seed_variability()


if __name__ == "__main__":
    main()
