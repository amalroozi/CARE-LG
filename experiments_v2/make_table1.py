"""
Builds the final Table 1 (brief's row/column spec) as CSV + LaTeX from the
already-aggregated table1_{dataset}.csv files.

Row order: R+hard, R+soft(lam=1), R+soft(lam=10), E+hard, E+soft(lam=1),
E+none, DiCE, FACE, GrowingSpheres (substituting for REVISE -- see
CHANGES.md: REVISE is not implemented anywhere in this codebase and was not
reimplemented from scratch, per the brief's own instruction for absent
baselines).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

RESULTS_DIR = REPO_ROOT / "experiments_v2" / "results"

ROW_SPEC = [
    ("riemannian_hard", "R+hard (CARE-LG)"),
    ("riemannian_soft_lam1.0", "R+soft (λ=1)"),
    ("riemannian_soft_lam10.0", "R+soft (λ=10)"),
    ("euclidean_hard", "E+hard (≈FACE-in-latent-space)"),
    ("euclidean_soft_lam1.0", "E+soft (λ=1)"),
    ("euclidean_none", "E+none"),
    ("DiCE", "DiCE"),
    ("FACE", "FACE"),
    ("GrowingSpheres", "GrowingSpheres (substitutes REVISE — see CHANGES.md)"),
]


def fmt(m, s, digits=1, pct=False):
    if pd.isna(m):
        return "--"
    suffix = "\\%" if pct else ""
    if pd.isna(s):
        return f"{m:.{digits}f}{suffix}"
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}{suffix}"


def build_dataset_table(dataset):
    df = pd.read_csv(RESULTS_DIR / f"table1_{dataset}.csv").set_index("cell")
    rows = []
    for cell, label in ROW_SPEC:
        if cell not in df.index:
            continue
        r = df.loc[cell]
        cvr_col = "decoded_cvr_endpoint" if "decoded_cvr_endpoint_mean" in r else "cvr_hard"
        rows.append({
            "Method": label,
            "Success (%)": fmt(r["success_rate_mean"], r["success_rate_std"]),
            "Decoded CVR (%)": fmt(r.get("decoded_cvr_endpoint_mean", r.get("cvr_hard_mean")), r.get("decoded_cvr_endpoint_std", r.get("cvr_hard_std"))),
            "KDE": fmt(r["kde_mean"], r["kde_std"], digits=3),
            "Effort": fmt(r["effort_mean"], r["effort_std"], digits=2),
            "Latency (s)": fmt(r["latency_mean"], r["latency_std"], digits=4),
        })
    return pd.DataFrame(rows)


def to_latex(df, dataset):
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{CARE-LG ablation grid vs. baselines on {dataset.upper()}. Mean$\pm$std over 5 seeds. "
        r"``Decoded CVR'' is the full hard+directional constraint check re-run on VAE-DECODED trajectory endpoints "
        r"(Phase 3), not the original repo's hard-only CVR\%.}",
        rf"\label{{tab:table1_{dataset}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"\textbf{Method} & \textbf{Success (\%)} & \textbf{Decoded CVR (\%)} $\downarrow$ & \textbf{KDE} $\uparrow$ & \textbf{Effort} $\downarrow$ & \textbf{Latency (s)} $\downarrow$ \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(f"{row['Method']} & {row['Success (%)']} & {row['Decoded CVR (%)']} & {row['KDE']} & {row['Effort']} & {row['Latency (s)']} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines)


def main():
    for dataset in ["uci", "nhanes"]:
        df = build_dataset_table(dataset)
        df.to_csv(RESULTS_DIR / f"table1_final_{dataset}.csv", index=False)
        tex = to_latex(df, dataset)
        with open(RESULTS_DIR / f"table1_{dataset}.tex", "w") as f:
            f.write(tex)
        print(f"=== {dataset.upper()} ===")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
